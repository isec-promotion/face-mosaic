#!/usr/bin/env python3
"""
// face-mosaic-youtube.py
監視カメラ映像の顔モザイク処理（YouTube配信専用版）

概要:
    RTSPストリームから監視カメラ映像を取得し、YOLOv8で人物検出を行い、
    顔部分にモザイクを適用してYouTube Liveに配信するプログラム。
    プレビュー画面は表示せず、配信専用に最適化されています。

主な仕様:
    - YOLOv8による高精度人物検出
    - RTSPストリームの別スレッド読み込み（ThreadedVideoCapture）
    - ソースFPS自動取得とフレーム補間機能（30fps未満の場合にフレーム複製）
    - YouTube Live配信（RTMP）
    - プレビュー画面なし（配信専用）

制限事項:
    - YouTube配信専用（stream_keyが必須）
    - フレーム補間はduplicate方式のみ（フレーム複製）

使用方法:
    python face-mosaic-youtube.py <rtsp_url> <stream_key> [options]

例:
    python face-mosaic-youtube.py "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
    python face-mosaic-youtube.py "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6
"""

import cv2
import numpy as np
import subprocess
import sys
import argparse
import threading
from time import perf_counter, sleep, time
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("エラー: ultralyticsパッケージがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install ultralytics")
    sys.exit(1)

def log_ffmpeg_output(process):
    """
    FFmpegのログ出力を別スレッドで処理する関数
    
    Args:
        process: FFmpegのsubprocess.Popenオブジェクト
    """
    while True:
        line = process.stderr.readline()
        if not line:
            break
        line = line.decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        print(f"[FFmpeg] {line}")

class ThreadedVideoCapture:
    """
    RTSPストリームの読み込みを別スレッドで行い、
    常に最新のフレームのみを保持するクラス
    
    属性:
        src: RTSPストリームURL
        cap: OpenCVのVideoCaptureオブジェクト
        fps: ソース映像のFPS（自動取得）
        q: フレームを保持するdeque（最大1フレーム）
        status: スレッドの状態（"running" or "stopped"）
        thread: 読み込みスレッド
    """
    def __init__(self, src, max_queue_size=1):
        """
        初期化
        
        Args:
            src: RTSPストリームURL
            max_queue_size: キューサイズ（デフォルト: 1）
        """
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max_queue_size)
        
        # FPSを取得して保持
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.q = deque(maxlen=max_queue_size)
        self.status = "stopped"
        self.thread = threading.Thread(target=self._update, daemon=True)

    def _update(self):
        """
        別スレッドでフレームを読み込む内部メソッド
        """
        print("[ThreadedVideoCapture] 読み取りスレッドを開始")
        while self.status == "running":
            ret, frame = self.cap.read()
            if not ret:
                print("[ThreadedVideoCapture] フレーム取得失敗。再接続試行...")
                self.cap.release()
                sleep(1)
                
                self.cap = cv2.VideoCapture(self.src)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    print("[ThreadedVideoCapture] 再接続失敗。1秒後にリトライ...")
                    sleep(1)
                continue
            
            self.q.append(frame)
        
        print("[ThreadedVideoCapture] 読み取りスレッドを停止")
        self.cap.release()

    def start(self):
        """
        スレッドを開始する
        
        Returns:
            self: メソッドチェーン用
        """
        if self.status == "stopped":
            self.status = "running"
            self.thread.start()
        return self

    def read(self):
        """
        最新のフレームを取得する
        
        Returns:
            numpy.ndarray: フレーム（BGR形式）、取得できない場合はNone
        """
        try:
            return self.q.pop()
        except IndexError:
            return None

    def stop(self):
        """
        スレッドを停止する
        """
        self.status = "stopped"
        if self.thread.is_alive():
            self.thread.join(timeout=2)

def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """
    画像の指定領域にモザイクを適用する
    
    Args:
        image: 入力画像（BGR形式、numpy.ndarray）
        x: モザイク領域の左上X座標（int）
        y: モザイク領域の左上Y座標（int）
        w: モザイク領域の幅（int）
        h: モザイク領域の高さ（int）
        ratio: モザイクの縮小率（float、デフォルト: 0.05）
    
    Returns:
        numpy.ndarray: モザイク適用後の画像
    """
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return image
    
    face_img = image[y:y+h, x:x+w]
    if face_img.size == 0:
        return image
    
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image

class FrameInterpolator:
    """
    フレームレート補間クラス
    ソースFPSが目標FPSより低い場合、フレームを複製して目標FPSに水増しする
    
    属性:
        sourceFps: ソース映像のFPS（float）
        targetFps: 目標FPS（int）
        needsInterpolation: 補間が必要かどうか（bool）
        framesPerSource: 1つのソースフレームから生成するフレーム数（float）
    """
    def __init__(self, source_fps, target_fps):
        """
        フレーム補間器を初期化
        
        Args:
            source_fps: ソース映像のFPS（float）
            target_fps: 目標FPS（int）
        """
        self.sourceFps = source_fps
        self.targetFps = target_fps
        
        # 補間が必要かどうかを判定
        self.needsInterpolation = source_fps > 0 and source_fps < target_fps
        
        if self.needsInterpolation:
            # 1つのソースフレームから生成するフレーム数
            self.framesPerSource = target_fps / source_fps
            
            print(f"[FrameInterpolator] 補間を有効化: {source_fps}fps -> {target_fps}fps")
            print(f"[FrameInterpolator] 1ソースフレームあたり {self.framesPerSource:.2f} フレームを生成")
        else:
            print(f"[FrameInterpolator] 補間不要: ソースFPS({source_fps}) >= 目標FPS({target_fps})")
    
    def interpolate(self, currentFrame):
        """
        現在のフレームから補間フレームを生成（フレーム複製方式）
        
        Args:
            currentFrame: 現在のフレーム（BGR形式、numpy.ndarray）
        
        Returns:
            list: 補間されたフレームのリスト（空の場合は補間不要）
        """
        if not self.needsInterpolation or currentFrame is None:
            return [currentFrame] if currentFrame is not None else []
        
        interpolatedFrames = []
        
        # フレーム複製方式：各フレームを複数回送信
        numFrames = int(round(self.framesPerSource))
        for _ in range(numFrames):
            interpolatedFrames.append(currentFrame.copy())
        
        return interpolatedFrames

def parse_arguments():
    """
    コマンドライン引数を解析する
    
    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(
        description='監視カメラ映像の顔モザイク処理（YouTube配信専用版）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('rtsp_url', help='監視カメラのRTSPストリームURL')
    parser.add_argument('stream_key', help='YouTubeライブストリーミングキー')
    parser.add_argument('--width', '-W', type=int, default=1280, help='出力映像の幅（デフォルト: 1280）')
    parser.add_argument('--height', '-H', type=int, default=720, help='出力映像の高さ（デフォルト: 720）')
    parser.add_argument('--fps', '-f', type=int, default=30, help='FPS自動取得失敗時のデフォルトFPS（デフォルト: 30）')
    parser.add_argument('--model', '-m', default='yolov8n.pt', 
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLOv8モデル（デフォルト: yolov8n.pt）')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='検出信頼度閾値（デフォルト: 0.5）')
    parser.add_argument('--head-ratio', '-r', type=float, default=0.25, 
                       help='頭部領域の割合（デフォルト: 0.25）')
    parser.add_argument('--interpolate-fps', type=int, default=30, 
                       help='フレーム補間の目標FPS（ソースFPSがこの値より低い場合に補間、デフォルト: 30）')
    
    return parser.parse_args()

def main():
    """
    メイン処理関数
    """
    args = parse_arguments()
    youtube_url = f"rtmp://a.rtmp.youtube.com/live2/{args.stream_key}"
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（YouTube配信専用版）")
    print("=" * 70)
    
    # YOLOv8モデルの読み込み
    print(f"YOLOv8モデル（{args.model}）を読み込んでいます...")
    try:
        model = YOLO(args.model)
        print("モデルの読み込みが完了しました")
    except Exception as e:
        print(f"エラー: YOLOv8モデルの読み込みに失敗しました: {e}")
        print("関数名: main()")
        print(f"引数: model={args.model}")
        print("初回実行時はモデルのダウンロードに時間がかかる場合があります")
        sys.exit(1)
    
    # ビデオキャプチャとFPS判定ロジック
    print("RTSPストリームに接続しています...")
    
    # まずインスタンスを作成（まだスレッドは開始しない）
    cap = ThreadedVideoCapture(args.rtsp_url)
    
    # FPS情報の取得
    source_fps = cap.fps
    print(f"検出されたソースFPS: {source_fps}")
    
    # フレーム補間の判定
    interpolateTargetFps = args.interpolate_fps
    useInterpolation = source_fps > 0 and source_fps < interpolateTargetFps
    
    if useInterpolation:
        # フレーム補間を使用する場合、目標FPSは補間目標FPS
        target_fps = interpolateTargetFps
        print(f"-> 適用FPS: {target_fps} (フレーム補間: {source_fps}fps -> {target_fps}fps)")
    elif source_fps > 0 and source_fps < 120:
        # 補間不要で、ソースFPSが有効な場合
        target_fps = source_fps
        # 整数に近い場合は丸める (29.97 -> 30, 14.9 -> 15)
        if abs(target_fps - round(target_fps)) < 0.1:
            target_fps = round(target_fps)
        print(f"-> 適用FPS: {target_fps} (ソース同期)")
    else:
        # ソースFPSが取得できない、または異常な値の場合
        target_fps = args.fps
        print(f"-> 適用FPS: {target_fps} (デフォルト値)")
    
    # フレーム補間器の初期化
    frameInterpolator = None
    if useInterpolation:
        frameInterpolator = FrameInterpolator(
            source_fps=source_fps,
            target_fps=target_fps
        )
    
    # スレッド開始
    cap.start()
    sleep(2)  # バッファ充填待ち
    
    print(f"入力: {args.rtsp_url}")
    print(f"解像度: {args.width}x{args.height} @ {target_fps}fps")
    print("=" * 70)

    # FFmpeg設定
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{args.width}x{args.height}',
        '-r', str(target_fps),
        '-i', '-',
        '-f', 'lavfi',
        '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        '-fflags', '+genpts',
        '-vsync', 'cfr',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-b:v', '2500k',
        '-maxrate', '2500k',
        '-bufsize', '5000k',
        '-bf', '0',
        '-sc_threshold', '0',
        '-g', str(int(target_fps) * 2),  # GOP長もFPSに合わせて調整（2秒間隔）
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ar', '44100',
        '-flvflags', 'no_duration_filesize',
        '-f', 'flv',
        youtube_url,
    ]
    
    try:
        print("\nFFmpegを起動しています...")
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        log_thread = threading.Thread(target=log_ffmpeg_output, args=(ffmpeg_process,), daemon=True)
        log_thread.start()
        sleep(2)
        
        if ffmpeg_process.poll() is not None:
            print("\n警告: FFmpegプロセスが終了しました。")
            print("関数名: main()")
            print(f"引数: ffmpeg_cmd={ffmpeg_cmd}")
            print("YouTubeストリームキーが正しいか、ネットワーク接続を確認してください")
        else:
            print("\nYouTube Liveへのストリーミングを開始しました")
        
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        print("関数名: main()")
        print("FFmpegをインストールしてPATHに追加してください")
        cap.stop()
        sys.exit(1)
    except Exception as e:
        print(f"エラー: FFmpegの起動に失敗しました: {e}")
        print("関数名: main()")
        print(f"引数: ffmpeg_cmd={ffmpeg_cmd}")
        cap.stop()
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    start_time = time()
    
    try:
        print("処理を開始します（Ctrl+Cで終了）\n")
        
        # フレーム送信間隔の計算（目標FPSに合わせる）
        frameInterval = 1.0 / target_fps if target_fps > 0 else 0.033
        lastFrameTime = perf_counter()
        
        while True:
            frame = cap.read()
            
            if frame is None:
                sleep(0.01)
                continue
            
            frame = cv2.resize(frame, (args.width, args.height))
            
            # YOLO検出
            results = model(frame, classes=[0], conf=args.confidence, verbose=False)
            
            detected_heads = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_w, person_h = x2 - x1, y2 - y1
                    
                    if person_w < 30 or person_h < 50:
                        continue
                    aspect = person_w / person_h if person_h > 0 else 0
                    if aspect < 0.2 or aspect > 3.0:
                        continue
                    
                    head_h = int(person_h * args.head_ratio)
                    head_x = max(0, x1 - int(person_w * 0.1))
                    head_y = max(0, y1 - int(head_h * 0.1))
                    head_w = min(args.width - head_x, person_w + int(person_w * 0.2))
                    head_h = min(args.height - head_y, head_h + int(head_h * 0.2))
                    
                    detected_heads.append((head_x, head_y, head_w, head_h))
            
            # モザイク
            for (x, y, w, h) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h)
            
            total_detections += len(detected_heads)
            
            # フレーム補間の適用
            if frameInterpolator is not None:
                # 補間フレームを生成
                interpolatedFrames = frameInterpolator.interpolate(frame)
                
                # 補間フレームを順次送信
                for interpFrame in interpolatedFrames:
                    try:
                        ffmpeg_process.stdin.write(interpFrame.tobytes())
                        frame_count += 1
                        
                        # 目標FPSに合わせて送信間隔を調整
                        currentTime = perf_counter()
                        elapsed = currentTime - lastFrameTime
                        if elapsed < frameInterval:
                            sleep(frameInterval - elapsed)
                        lastFrameTime = perf_counter()
                    except (BrokenPipeError, IOError) as e:
                        print(f"警告: FFmpegパイプエラー: {e}")
                        print("関数名: main()")
                        print(f"引数: frame_count={frame_count}")
                        raise
            else:
                # 補間なし：フレームをそのまま送信
                try:
                    ffmpeg_process.stdin.write(frame.tobytes())
                    frame_count += 1
                    
                    # 目標FPSに合わせて送信間隔を調整
                    currentTime = perf_counter()
                    elapsed = currentTime - lastFrameTime
                    if elapsed < frameInterval:
                        sleep(frameInterval - elapsed)
                    lastFrameTime = perf_counter()
                except (BrokenPipeError, IOError) as e:
                    print(f"警告: FFmpegパイプエラー: {e}")
                    print("関数名: main()")
                    print(f"引数: frame_count={frame_count}")
                    break
            
            if frame_count % 100 == 0:
                elapsed = time() - start_time
                actual_fps = frame_count / elapsed
                fps_diff = actual_fps - target_fps
                print(f"FPS: {actual_fps:.1f} (目標: {target_fps}, 差: {fps_diff:+.1f}) | 検出数: {len(detected_heads)}")
                
    except KeyboardInterrupt:
        print("\n終了します...")
    except Exception as e:
        print(f"エラー: 予期しないエラーが発生しました: {e}")
        print("関数名: main()")
        import traceback
        traceback.print_exc()
    finally:
        cap.stop()
        if ffmpeg_process:
            try:
                ffmpeg_process.stdin.close()
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=3)
            except Exception as e:
                print(f"警告: FFmpegプロセスの終了処理でエラー: {e}")
        print("完了")

if __name__ == "__main__":
    main()
