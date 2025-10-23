#!/usr/bin/env python3
"""
監視カメラ映像の顔モジイク処理（YOLOv8版）- Jetson最終版

.engineファイルが存在しない場合、自動的に生成する機能を搭載。
初回実行時にTensorRTエンジンをビルドし、2回目以降は高速に起動します。

使用方法:
    # スクリプトを実行するだけ
    python face-mosaic-yolo-jetson.py <rtsp_url> [options]
"""

import cv2
import numpy as np
import subprocess
import sys
import argparse
import os # ファイル存在チェックのために追加

try:
    from ultralytics import YOLO
except ImportError:
    print("エラー: ultralyticsパッケージがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install ultralytics")
    sys.exit(1)

def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """
    指定された領域にモザイク処理を適用
    """
    # 境界チェック
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

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='監視カメラ映像の顔モザイク処理（YOLOv8 Jetson最終版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream"
  %(prog)s "rtsp://camera/stream" --output rtsp://127.0.0.1:8554/mystream
  %(prog)s "rtsp://camera/stream" --model yolov8s.pt --confidence 0.6
        """
    )
    
    parser.add_argument('rtsp_url', help='監視カメラのRTSPストリームURL')
    parser.add_argument('--output', '-o', default='rtsp://127.0.0.1:8554/mystream', help='出力RTSPストリームURL')
    parser.add_argument('--width', '-W', type=int, default=1280, help='処理・出力映像の幅')
    parser.add_argument('--height', '-H', type=int, default=720, help='処理・出力映像の高さ')
    parser.add_argument('--fps', '-f', type=int, default=25, help='フレームレート')
    parser.add_argument('--model', '-m', default='yolov8n.pt', choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'], help='使用するYOLOv8の元モデル名')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='検出信頼度閾値 0.0-1.0')
    parser.add_argument('--head-ratio', '-r', type=float, default=0.25, help='頭部領域の割合 0.1-0.5')
    parser.add_argument('--no-preview', action='store_true', help='プレビューウィンドウを表示しない')
    
    return parser.parse_args()

def create_gstreamer_pipeline(rtsp_url, width, height):
    """ハードウェアデコードを使用するGStreamerパイプライン文字列を生成"""
    return (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1"
    )

def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（YOLOv8 Jetson最終版）")
    print("=" * 70)
    
    # --- 変更点: .engineファイルの存在をチェックし、なければ生成 ---
    engine_file = args.model.replace('.pt', '.engine')
    
    if not os.path.exists(engine_file):
        print(f"TensorRTエンジン '{engine_file}' が見つかりません。")
        print("自動的に生成します。初回は数分かかることがあります...")
        
        command = ['yolo', 'export', f'model={args.model}', 'format=engine', 'device=0']
        
        try:
            # yolo export コマンドを実行
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"'{engine_file}' の生成に成功しました。")
        except FileNotFoundError:
            print("エラー: 'yolo' コマンドが見つかりません。ultralyticsが正しくインストールされているか確認してください。")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"エラー: TensorRTエンジンの生成に失敗しました。")
            print(f"コマンド: {' '.join(command)}")
            print(f"エラー出力:\n{e.stderr}")
            sys.exit(1)
    
    # --- ここから先の処理は同じ ---
    print(f"YOLOv8 TensorRTエンジン（{engine_file}）を読み込んでいます...")
    try:
        model = YOLO(engine_file)
        print("モデルの読み込みが完了しました")
    except Exception as e:
        print(f"エラー: TensorRTエンジンの読み込みに失敗しました: {e}")
        sys.exit(1)
    
    print("GStreamerパイプラインでRTSPストリームに接続しています...")
    gstreamer_pipeline = create_gstreamer_pipeline(args.rtsp_url, args.width, args.height)
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("エラー: GStreamerパイプラインを開けませんでした")
        sys.exit(1)
    
    print("接続成功")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{args.width}x{args.height}',
        '-r', str(args.fps), '-i', '-', '-c:v', 'h264_nvmpi',
        '-b:v', '4M', '-preset', 'p1', '-zerolatency', '1',
        '-f', 'rtsp', args.output,
    ]
    
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ストリーミングを開始しました")
        print(f"\nVLC等で視聴: vlc {args.output}\n")
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        cap.release()
        sys.exit(1)
    
    try:
        print("処理を開始します（'q'キーで終了）\n")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("警告: フレームの取得に失敗しました。")
                break
            
            results = model(frame, classes=[0], conf=args.confidence, verbose=False)
            
            detected_count = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_w = x2 - x1
                    person_h = y2 - y1
                    
                    if person_w < 20 or person_h < 40:
                        continue
                        
                    head_h = int(person_h * args.head_ratio)
                    frame = apply_mosaic(frame, x1, y1, person_w, head_h)
                    detected_count += 1
            
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                print("警告: FFmpegプロセスが終了しました")
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"処理済み: {frame_count}フレーム | 検出数: {detected_count}")
            
            if not args.no_preview:
                cv2.putText(frame, f'Detected: {detected_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('YOLOv8 Face Mosaic (Press Q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n終了します...")
                    break
                    
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みを検出しました。終了します...")
    
    finally:
        print("リソースを解放しています...")
        cap.release()
        cv2.destroyAllWindows()
        
        if 'ffmpeg_process' in locals() and ffmpeg_process.poll() is None:
            try:
                ffmpeg_process.stdin.close()
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=5)
            except Exception as e:
                print(f"FFmpegプロセスの終了中にエラーが発生しました: {e}")
                ffmpeg_process.kill()
        
        print("完了しました")

if __name__ == "__main__":
    main()