#!/usr/bin/env python3
"""
監視カメラ映像の顔モザイク処理（従来手法版）

Haar Cascade + HOGを使用した人物検出とモザイク処理のサンプル実装。
技術ブログ用のリファレンス実装です。

使用方法:
    python face-mosaic.py <rtsp_url> [options]

例:
    python face-mosaic.py "rtsp://admin:password@192.168.1.100:554/stream"
    python face-mosaic.py "rtsp://camera/stream" --output udp://127.0.0.1:9000
"""

import cv2
import numpy as np
import subprocess
import sys
import argparse

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
        description='監視カメラ映像の顔モザイク処理（従来手法版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream"
  %(prog)s "rtsp://camera/stream" --output udp://127.0.0.1:9000
  %(prog)s "rtsp://camera/stream" --width 1920 --height 1080 --fps 30
        """
    )
    
    parser.add_argument('rtsp_url', 
                       help='監視カメラのRTSPストリームURL')
    parser.add_argument('--output', '-o',
                       default='udp://127.0.0.1:8080',
                       help='出力ストリームURL (デフォルト: udp://127.0.0.1:8080)')
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
                       default=25,
                       help='フレームレート (デフォルト: 25)')
    parser.add_argument('--confidence', '-c',
                       type=int,
                       default=8,
                       help='検出信頼度（minNeighbors） (デフォルト: 8)')
    parser.add_argument('--min-size', '-m',
                       type=int,
                       default=50,
                       help='最小検出サイズ (デフォルト: 50)')
    parser.add_argument('--no-preview',
                       action='store_true',
                       help='プレビューウィンドウを表示しない')
    
    return parser.parse_args()

def main():
    # コマンドライン引数を解析
    args = parse_arguments()
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（従来手法版）")
    print("=" * 70)
    print(f"入力: {args.rtsp_url}")
    print(f"出力: {args.output}")
    print(f"解像度: {args.width}x{args.height} @ {args.fps}fps")
    print(f"検出パラメータ: minNeighbors={args.confidence}, minSize={args.min_size}")
    print("=" * 70)
    
    # OpenCVでRTSPストリームを開く
    print("RTSPストリームに接続しています...")
    cap = cv2.VideoCapture(args.rtsp_url)
    
    if not cap.isOpened():
        print("エラー: RTSPストリームを開けませんでした")
        print("URLが正しいか、カメラが動作しているか確認してください")
        sys.exit(1)
    
    print("接続成功")
    
    # 顔検出器の初期化
    print("検出器を初期化しています...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )
    
    if face_cascade.empty() or profile_cascade.empty():
        print("エラー: 顔検出器の読み込みに失敗しました")
        sys.exit(1)
    
    # HOG人物検出器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    print("検出器の初期化が完了しました")
    
    # FFmpegプロセスの設定
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{args.width}x{args.height}',
        '-r', str(args.fps),
        '-i', '-',
        '-f', 'mpegts',
        '-codec:v', 'mpeg1video',
        '-b:v', '2000k',
        '-bf', '0',
        args.output,
    ]
    
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("ストリーミングを開始しました")
        print(f"\nVLCで視聴: vlc {args.output}\n")
        
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        print("FFmpegをインストールしてPATHに追加してください")
        cap.release()
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    
    try:
        print("処理を開始します（'q'キーで終了）\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("警告: フレームの取得に失敗しました。再接続を試みます...")
                cap.release()
                cap = cv2.VideoCapture(args.rtsp_url)
                continue
            
            # フレームをリサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 人物検出（HOG）
            people, _ = hog.detectMultiScale(
                frame, 
                winStride=(16, 16), 
                padding=(4, 4), 
                scale=1.1
            )
            
            detected_heads = []
            
            # 人物の頭部領域を計算
            for (x, y, w, h) in people:
                area_ratio = (w * h) / (args.width * args.height)
                if area_ratio < 0.03 or area_ratio > 0.4:
                    continue
                
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    continue
                
                head_h = int(h * 0.3)
                detected_heads.append((x, y, w, head_h))
            
            # 正面顔検出
            faces_front = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=args.confidence,
                minSize=(args.min_size, args.min_size)
            )
            
            # 横顔検出
            faces_profile = profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=args.confidence,
                minSize=(args.min_size, args.min_size)
            )
            
            # 正面顔を追加
            for (x, y, w, h) in faces_front:
                area_ratio = (w * h) / (args.width * args.height)
                if area_ratio < 0.005 or area_ratio > 0.3:
                    continue
                
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.7 or aspect_ratio > 1.5:
                    continue
                
                margin = int(w * 0.3)
                new_x = max(0, x - margin)
                new_y = max(0, y - margin)
                new_w = min(args.width - new_x, w + margin * 2)
                new_h = min(args.height - new_y, int(h * 1.5))
                detected_heads.append((new_x, new_y, new_w, new_h))
            
            # 横顔を追加
            for (x, y, w, h) in faces_profile:
                area_ratio = (w * h) / (args.width * args.height)
                if area_ratio < 0.005 or area_ratio > 0.3:
                    continue
                
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.6 or aspect_ratio > 2.0:
                    continue
                
                margin = int(w * 0.3)
                new_x = max(0, x - margin)
                new_y = max(0, y - margin)
                new_w = min(args.width - new_x, w + margin * 2)
                new_h = min(args.height - new_y, int(h * 1.5))
                detected_heads.append((new_x, new_y, new_w, new_h))
            
            # 重複を除去
            final_heads = []
            for head in detected_heads:
                x1, y1, w1, h1 = head
                is_duplicate = False
                for existing in final_heads:
                    x2, y2, w2, h2 = existing
                    if (abs(x1 - x2) < max(w1, w2) * 0.5 and 
                        abs(y1 - y2) < max(h1, h2) * 0.5):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    final_heads.append(head)
            
            # モザイク処理
            for (x, y, w, h) in final_heads:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
                total_detections += 1
            
            # FFmpegに送信
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("警告: FFmpegプロセスが終了しました")
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                avg_detections = total_detections / frame_count
                print(f"処理済み: {frame_count}フレーム | "
                      f"検出数: {len(final_heads)} | "
                      f"平均: {avg_detections:.2f}")
            
            # プレビュー表示
            if not args.no_preview:
                cv2.putText(frame, f'Detected: {len(final_heads)}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.imshow('Face Mosaic (Press Q to quit)', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n終了します...")
                    break
                
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みを検出しました。終了します...")
    
    finally:
        # クリーンアップ
        print("リソースを解放しています...")
        cap.release()
        cv2.destroyAllWindows()
        
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
