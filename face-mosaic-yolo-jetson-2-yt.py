#!/usr/bin/env python3
"""
監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）

NVIDIA Jetson向けに最適化された顔モザイク処理実装（YouTube配信専用版）。
- YOLOv8による高精度人物検出
- TensorRT推論エンジン（自動変換・FP16精度）
- 通常のRTSPデコード（GStreamer不使用）
- ハードウェアエンコード（NVENC）
- プレビューウィンドウなし（配信専用）

技術ブログ用のリファレンス実装です。

使用方法:
    python face-mosaic-yolo-jetson-2-yt.py <rtsp_url> <stream_key> [options]

例:
    python face-mosaic-yolo-jetson-2-yt.py "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
    python face-mosaic-yolo-jetson-2-yt.py "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6

機能:
    - 初回実行時に自動的にTensorRTエンジン（.engine）を生成
    - 2回目以降は高速なTensorRTエンジンを使用
    - 通常のcv2.VideoCapture()でRTSPストリームをデコード（GStreamer不要）
    - NVENCによるハードウェアエンコード（CPU負荷削減）
    - プレビューウィンドウなし（リソース節約、YouTube配信に最適）
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
    """FFmpegの出力をログに記録"""
    while True:
        line = process.stderr.readline()
        if not line:
            break
        line = line.decode('utf-8', errors='ignore').strip()
        if line:
            # 重要なメッセージのみ表示
            if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'connection']):
                print(f"[FFmpeg] {line}")

def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """
    指定された領域にモザイク処理を適用
    
    Args:
        image: 入力画像
        x, y: モザイク領域の左上座標
        w, h: モザイク領域の幅と高さ
        ratio: モザイクの粗さ（小さいほど粗い、推奨: 0.05-0.1）
    
    Returns:
        モザイク処理後の画像
    """
    # 境界チェック
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return image
    
    # モザイク処理する領域を抽出
    face_img = image[y:y+h, x:x+w]
    
    if face_img.size == 0:
        return image
    
    # 縮小してから拡大することでモザイク効果を作成
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, 
                      interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), 
                       interpolation=cv2.INTER_NEAREST)
    
    # 元の画像にモザイクを適用
    image[y:y+h, x:x+w] = mosaic
    
    return image

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
  %(prog)s "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6
  %(prog)s "rtsp://camera/stream" "your-stream-key" --width 1920 --height 1080 --fps 30

配信先: rtmp://a.rtmp.youtube.com/live2 (YouTube Live固定)

モデルの選択:
  yolov8n.pt: Nano (最速、メモリ少)
  yolov8s.pt: Small (バランス)
  yolov8m.pt: Medium (高精度)
  yolov8l.pt: Large (最高精度)
        """
    )
    
    parser.add_argument('rtsp_url', 
                       help='監視カメラのRTSPストリームURL')
    parser.add_argument('stream_key',
                       help='YouTubeライブストリーミングキー')
    parser.add_argument('--width', '-W',
                       type=int,
                       default=1280,
                       help='出力映像の幅 (デフォルト: 1280)')
    parser.add_argument('--height', '-H',
                       type=int,
                       default=720,
                       help='出力映像の高さ (デフォルト: 720)')
    parser.add_argument('--fps', '-f',
                       type=int,
                       default=30,
                       help='フレームレート (デフォルト: 30)')
    parser.add_argument('--model', '-m',
                       default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLOv8モデル (デフォルト: yolov8n.pt)')
    parser.add_argument('--confidence', '-c',
                       type=float,
                       default=0.5,
                       help='検出信頼度閾値 0.0-1.0 (デフォルト: 0.5)')
    parser.add_argument('--head-ratio', '-r',
                       type=float,
                       default=0.25,
                       help='頭部領域の割合 0.1-0.5 (デフォルト: 0.25)')
    parser.add_argument('--no-tensorrt',
                       action='store_true',
                       help='TensorRT変換をスキップしてPyTorchモデルを使用')
    
    return parser.parse_args()

def main():
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # YouTube RTMPストリームURL
    youtube_url = f"rtmp://a.rtmp.youtube.com/live2/{args.stream_key}"
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）")
    print("=" * 70)
    print(f"入力: {args.rtsp_url}")
    print(f"出力: rtmp://a.rtmp.youtube.com/live2/****")
    print(f"解像度: {args.width}x{args.height} @ {args.fps}fps")
    print(f"モデル: {args.model}")
    print(f"検出パラメータ: confidence={args.confidence}, head_ratio={args.head_ratio}")
    print("=" * 70)
    
    # YOLOv8モデルの読み込み（TensorRT最適化対応）
    import os
    import torch
    
    # CUDA利用可否の確認
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("警告: CUDAが利用できません")
        print(f"torch.cuda.is_available(): {cuda_available}")
        print("TensorRT変換をスキップし、PyTorchモデルを使用します")
        print("\nCUDA対応PyTorchのインストール方法:")
        print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 70)
    
    # TensorRTエンジンファイル名を生成（環境依存）
    # 異なるGPU/TensorRTバージョン間では互換性がないため、デバイス名を含める
    device_name = torch.cuda.get_device_name(0).replace(' ', '_') if cuda_available else 'cpu'
    base_name = args.model.replace('.pt', '')
    engine_file = f"{base_name}_{device_name}.engine"
    
    model = None
    model_type = None
    
    # 1. TensorRTエンジンファイルが存在するかチェック
    if os.path.exists(engine_file):
        print(f"TensorRTエンジンファイル（{engine_file}）が見つかりました")
        print("TensorRTエンジンを読み込んでいます...")
        try:
            model = YOLO(engine_file, task='detect')
            model_type = 'TensorRT'
            print("TensorRTエンジンの読み込みが完了しました")
        except Exception as e:
            print(f"警告: TensorRTエンジンの読み込みに失敗しました: {e}")
            print("PyTorchモデルを使用します")
    
    # 2. TensorRTエンジンが読み込めなかった場合、PyTorchモデルから変換を試みる
    if model is None:
        print(f"PyTorchモデル（{args.model}）を読み込んでいます...")
        try:
            model = YOLO(args.model)
            print("PyTorchモデルの読み込みが完了しました")
            
            # CUDAが利用できない、または--no-tensorrtフラグがある場合はTensorRT変換をスキップ
            if not cuda_available:
                print("\nCUDAが利用できないため、TensorRT変換をスキップします")
                print("PyTorchモデル（CPU）をそのまま使用します")
                model_type = 'PyTorch (CPU)'
            elif args.no_tensorrt:
                print("\n--no-tensorrtフラグが指定されたため、TensorRT変換をスキップします")
                print("PyTorchモデルをそのまま使用します")
                model_type = 'PyTorch'
            else:
                # TensorRTへの変換を試みる
                print("\nTensorRTエンジンへの変換を試みています...")
                print("（初回のみ時間がかかります。数分お待ちください）")
                print("※すぐに開始したい場合は Ctrl+C で中断し、--no-tensorrt オプションを使用してください")
                try:
                    # TensorRTへエクスポート
                    # half=True でFP16精度（Jetsonで高速）
                    model.export(format='engine', half=True, device=0)
                    
                    # エクスポート時に生成されるデフォルトのファイル名
                    default_engine = args.model.replace('.pt', '.engine')
                    
                    # デバイス名を含むファイル名にリネーム
                    if os.path.exists(default_engine) and default_engine != engine_file:
                        print(f"\nエンジンファイルを {default_engine} から {engine_file} にリネームしています...")
                        os.rename(default_engine, engine_file)
                    
                    # 変換されたエンジンファイルを再読み込み
                    print(f"\n変換が完了しました。TensorRTエンジン（{engine_file}）を読み込んでいます...")
                    model = YOLO(engine_file, task='detect')
                    model_type = 'TensorRT'
                    print("TensorRTエンジンでの実行準備が完了しました")
                    
                except Exception as e:
                    print(f"\n警告: TensorRTへの変換に失敗しました: {e}")
                    print("PyTorchモデルをそのまま使用します")
                    model_type = 'PyTorch'
                
        except Exception as e:
            print(f"エラー: モデルの読み込みに失敗しました: {e}")
            print("初回実行時はモデルのダウンロードに時間がかかる場合があります")
            sys.exit(1)
    
    print(f"使用するモデル形式: {model_type}")
    print("=" * 70)
    
    # 通常のcv2.VideoCaptureでRTSPストリームを開く（GStreamer不使用）
    print("RTSPストリームに接続しています（通常デコード）...")
    
    cap = cv2.VideoCapture(args.rtsp_url)
    
    if not cap.isOpened():
        print("エラー: RTSPストリームを開けませんでした")
        print("URLが正しいか、カメラが動作しているか確認してください")
        sys.exit(1)
    
    print("接続成功")
    
    # ソースのFPSを取得
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps > 0 and source_fps != args.fps:
        print(f"\n注意: ソースのFPS({source_fps:.1f})と指定されたFPS({args.fps})が異なります")
        print(f"指定されたFPS({args.fps})を使用します")
    
    # フレーム送信の間隔を計算（秒）
    frame_interval = 1.0 / args.fps
    print(f"フレーム送信間隔: {frame_interval:.4f}秒 ({args.fps}fps)")
    
    # 環境に応じたFFmpegエンコーダーの選択
    import platform
    is_jetson = os.path.exists('/etc/nv_tegra_release') or 'tegra' in platform.platform().lower()
    
    if is_jetson:
        # Jetson環境: libx264（ソフトウェアエンコード）を使用
        print("Jetson環境を検出しました。libx264エンコーダーを使用します（CFR/低遅延）")
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{args.width}x{args.height}',
            '-framerate', str(args.fps),  # 入力フレームレート
            '-i', '-',
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            # タイムスタンプ/CFR/低遅延
            '-fflags', '+genpts',
            '-use_wallclock_as_timestamps', '1',
            '-vsync', 'cfr',
            # エンコード
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-b:v', '2500k',
            '-maxrate', '2500k',
            '-bufsize', '5000k',
            '-sc_threshold', '0',
            '-g', str(args.fps * 2),
            '-pix_fmt', 'yuv420p',
            # 音声
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            # RTMP FLV
            '-flvflags', 'no_duration_filesize',
            '-f', 'flv',
            youtube_url,
        ]
    else:
        # 通常のPC環境: NVENCハードウェアエンコーダーを使用
        print("PC環境を検出しました。NVENCエンコーダーを使用します（CFR/低遅延）")
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{args.width}x{args.height}',
            '-framerate', str(args.fps),  # 入力フレームレート
            '-i', '-',
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            # タイムスタンプ/CFR
            '-fflags', '+genpts',
            '-use_wallclock_as_timestamps', '1',
            '-vsync', 'cfr',
            # エンコード（NVENC 低遅延・CBR 固定GOP）
            '-c:v', 'h264_nvenc',
            '-tune', 'll',
            '-rc', 'cbr',
            '-b:v', '2500k',
            '-maxrate', '2500k',
            '-bufsize', '5000k',
            '-bf', '0',
            '-sc_threshold', '0',
            '-g', str(args.fps * 2),
            '-pix_fmt', 'yuv420p',
            # 音声
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            # RTMP FLV
            '-flvflags', 'no_duration_filesize',
            '-f', 'flv',
            youtube_url,
        ]
    
    try:
        print("\nFFmpegを起動しています...")
        print("FFmpegコマンド:")
        print(" ".join(ffmpeg_cmd[:15]) + " ... " + ffmpeg_cmd[-1])
        
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        # FFmpegのログを別スレッドで監視
        log_thread = threading.Thread(target=log_ffmpeg_output, args=(ffmpeg_process,), daemon=True)
        log_thread.start()
        
        # FFmpegが起動するまで少し待つ
        time.sleep(2)
        
        # プロセスが生きているか確認
        if ffmpeg_process.poll() is not None:
            print("\n警告: FFmpegプロセスが予期せず終了しました")
            print("ストリームキーが正しいか確認してください")
        else:
            print("\nYouTube Liveへのストリーミングを開始しました")
            print("YouTube Studio (https://studio.youtube.com) で配信状況を確認してください")
            print("※配信が開始されるまで数秒〜数十秒かかる場合があります\n")
        
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        print("FFmpegをインストールしてPATHに追加してください")
        cap.release()
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    start_time = time()
    skipped_frames = 0
    
    # OpenCVの内部バッファを浅く（効くバックエンドの場合）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 送出タイミング制御用
    next_ts = perf_counter() + frame_interval
    
    try:
        print("処理を開始します（Ctrl+Cで終了）")
        print("CFR安定化モード: 正確なタイミング制御でフレームを送出します\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("警告: フレームの取得に失敗しました。再接続を試みます...")
                cap.release()
                cap = cv2.VideoCapture(args.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # スケジュールをリセット
                next_ts = perf_counter() + frame_interval
                continue
            
            # フレームをリサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # YOLOv8で人物検出
            results = model(
                frame, 
                classes=[0],  # 人物クラス
                conf=args.confidence,
                verbose=False
            )
            
            detected_heads = []
            
            # 検出結果を処理
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 人物の幅と高さ
                    person_w = x2 - x1
                    person_h = y2 - y1
                    
                    # サイズフィルタリング
                    if person_w < 30 or person_h < 50:
                        continue
                    
                    # アスペクト比チェック
                    aspect_ratio = person_w / person_h if person_h > 0 else 0
                    if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                        continue
                    
                    # 頭部領域を計算
                    head_h = int(person_h * args.head_ratio)
                    head_x = max(0, x1 - int(person_w * 0.1))
                    head_y = max(0, y1 - int(head_h * 0.1))
                    head_w = min(args.width - head_x, person_w + int(person_w * 0.2))
                    head_h = min(args.height - head_y, head_h + int(head_h * 0.2))
                    
                    detected_heads.append((head_x, head_y, head_w, head_h))
            
            # モザイク処理
            for (x, y, w, h) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
                total_detections += 1
            
            # 送出タイミング制御：処理後にsleep
            now = perf_counter()
            drift = now - next_ts
            
            if drift < 0:
                # まだ時間がある場合は待機
                sleep(-drift)
            elif drift > frame_interval * 2:
                # 2フレーム以上遅延している場合はスケジュールを調整（バースト防止）
                missed = int(drift // frame_interval)
                next_ts += frame_interval * missed
                skipped_frames += missed
            
            # FFmpegに送信
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("警告: FFmpegプロセスが終了しました")
                break
            except Exception as e:
                print(f"警告: フレーム送信エラー: {e}")
                break
            
            # 送出後に次の予定時刻を進める（重要）
            next_ts += frame_interval
            
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed_time = time() - start_time
                actual_fps = frame_count / elapsed_time
                avg_detections = total_detections / frame_count
                target_fps = args.fps
                fps_diff = actual_fps - target_fps
                print(f"処理済み: {frame_count}フレーム | "
                      f"検出数: {len(detected_heads)} | "
                      f"平均: {avg_detections:.2f} | "
                      f"実FPS: {actual_fps:.1f} (目標: {target_fps}, 差: {fps_diff:+.1f}) | "
                      f"スキップ: {skipped_frames}")
                
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みを検出しました。終了します...")
    
    finally:
        # クリーンアップ
        print("リソースを解放しています...")
        cap.release()
        
        if ffmpeg_process:
            try:
                ffmpeg_process.stdin.close()
            except:
                pass
            
            try:
                ffmpeg_process.terminate()
                try:
                    ffmpeg_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("FFmpegプロセスを強制終了しています...")
                    ffmpeg_process.kill()
                    ffmpeg_process.wait()
            except:
                pass
        
        print("完了しました")

if __name__ == "__main__":
    main()
