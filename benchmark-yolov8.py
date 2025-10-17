#!/usr/bin/env python3
"""
YOLOv8パフォーマンスベンチマークツール

RTSPストリームを使用してYOLOv8のパフォーマンスを測定します。
GPU使用状況、FPS、処理時間などの統計情報を収集します。

使用方法:
    python benchmark-yolov8.py <rtsp_url> [options]

例:
    python benchmark-yolov8.py "rtsp://admin:password@192.168.1.100:554/stream"
    python benchmark-yolov8.py "rtsp://camera/stream" --model yolov8s.pt --frames 200
    python benchmark-yolov8.py "rtsp://camera/stream" --device cpu
"""

import cv2
import numpy as np
import sys
import argparse
import time
import subprocess
from collections import deque

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("エラー: 必要なパッケージがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install ultralytics torch")
    sys.exit(1)

def get_gpu_info():
    """NVIDIA GPUの情報を取得"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {
                'name': info[0],
                'memory_total': f"{info[1]} MB",
                'memory_used': f"{info[2]} MB",
                'gpu_util': f"{info[3]}%",
                'temperature': f"{info[4]}°C"
            }
    except:
        pass
    return None

def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """モザイク処理（ベンチマーク用）"""
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return image
    
    face_img = image[y:y+h, x:x+w]
    if face_img.size == 0:
        return image
    
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, 
                      interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), 
                       interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    
    return image

class FPSCounter:
    """FPS計測クラス"""
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
        self.frame_times = []
        
    def update(self):
        """タイムスタンプを更新"""
        self.timestamps.append(time.time())
        
    def get_fps(self):
        """現在のFPSを取得"""
        if len(self.timestamps) < 2:
            return 0.0
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        return (len(self.timestamps) - 1) / time_diff
    
    def add_frame_time(self, frame_time):
        """フレーム処理時間を記録"""
        self.frame_times.append(frame_time)
    
    def get_statistics(self):
        """統計情報を取得"""
        if not self.frame_times:
            return None
        
        frame_times = np.array(self.frame_times)
        fps_values = 1.0 / frame_times
        
        return {
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values),
            'std_fps': np.std(fps_values),
            'avg_frame_time': np.mean(frame_times) * 1000,  # ms
            'total_frames': len(self.frame_times)
        }

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='YOLOv8パフォーマンスベンチマークツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream"
  %(prog)s "rtsp://camera/stream" --model yolov8s.pt --frames 200
  %(prog)s "rtsp://camera/stream" --device cpu --width 1920 --height 1080
        """
    )
    
    parser.add_argument('rtsp_url', 
                       help='監視カメラのRTSPストリームURL')
    parser.add_argument('--model', '-m',
                       default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLOv8モデル (デフォルト: yolov8n.pt)')
    parser.add_argument('--device', '-d',
                       default='cuda',
                       choices=['cuda', 'cpu'],
                       help='使用デバイス (デフォルト: cuda)')
    parser.add_argument('--width', '-W',
                       type=int,
                       default=1280,
                       help='処理する映像の幅 (デフォルト: 1280)')
    parser.add_argument('--height', '-H',
                       type=int,
                       default=720,
                       help='処理する映像の高さ (デフォルト: 720)')
    parser.add_argument('--frames', '-f',
                       type=int,
                       default=100,
                       help='測定するフレーム数 (デフォルト: 100)')
    parser.add_argument('--warmup', '-w',
                       type=int,
                       default=20,
                       help='ウォームアップフレーム数 (デフォルト: 20)')
    parser.add_argument('--confidence', '-c',
                       type=float,
                       default=0.5,
                       help='検出信頼度閾値 (デフォルト: 0.5)')
    parser.add_argument('--head-ratio', '-r',
                       type=float,
                       default=0.25,
                       help='頭部領域の割合 (デフォルト: 0.25)')
    parser.add_argument('--no-display',
                       action='store_true',
                       help='プレビューウィンドウを表示しない')
    parser.add_argument('--csv',
                       help='結果をCSVファイルに保存')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("YOLOv8 パフォーマンスベンチマーク")
    print("=" * 80)
    
    # GPU情報を表示
    if args.device == 'cuda':
        if torch.cuda.is_available():
            gpu_info = get_gpu_info()
            if gpu_info:
                print(f"\nGPU情報:")
                print(f"  名前: {gpu_info['name']}")
                print(f"  メモリ: {gpu_info['memory_used']} / {gpu_info['memory_total']}")
                print(f"  利用率: {gpu_info['gpu_util']}")
                print(f"  温度: {gpu_info['temperature']}")
        else:
            print("\n警告: CUDAが利用できません。CPUモードで実行します。")
            args.device = 'cpu'
    
    print(f"\nベンチマーク設定:")
    print(f"  RTSP URL: {args.rtsp_url}")
    print(f"  モデル: {args.model}")
    print(f"  デバイス: {args.device}")
    print(f"  解像度: {args.width}x{args.height}")
    print(f"  測定フレーム数: {args.frames}")
    print(f"  ウォームアップ: {args.warmup}フレーム")
    print(f"  信頼度閾値: {args.confidence}")
    print("=" * 80)
    
    # YOLOv8モデルの読み込み
    print(f"\nYOLOv8モデル（{args.model}）を読み込んでいます...")
    try:
        model = YOLO(args.model)
        if args.device == 'cuda':
            model.to('cuda')
        print("モデルの読み込みが完了しました")
    except Exception as e:
        print(f"エラー: モデルの読み込みに失敗しました: {e}")
        sys.exit(1)
    
    # RTSPストリームを開く
    print("RTSPストリームに接続しています...")
    cap = cv2.VideoCapture(args.rtsp_url)
    
    if not cap.isOpened():
        print("エラー: RTSPストリームを開けませんでした")
        sys.exit(1)
    
    print("接続成功\n")
    
    # FPSカウンター
    fps_counter = FPSCounter()
    
    frame_count = 0
    warmup_done = False
    detection_count = 0
    
    print("ベンチマークを開始します...\n")
    print(f"{'フレーム':<10} {'FPS':<10} {'検出数':<10} {'処理時間(ms)':<15} {'GPU利用率':<12}")
    print("-" * 80)
    
    try:
        while frame_count < args.warmup + args.frames:
            ret, frame = cap.read()
            
            if not ret:
                print("\n警告: フレームの取得に失敗しました")
                break
            
            # フレームをリサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # 処理時間を測定
            start_time = time.time()
            
            # YOLOv8で人物検出
            results = model(
                frame,
                classes=[0],
                conf=args.confidence,
                verbose=False,
                device=args.device
            )
            
            detected_heads = []
            
            # 検出結果を処理
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    person_w = x2 - x1
                    person_h = y2 - y1
                    
                    if person_w < 30 or person_h < 50:
                        continue
                    
                    aspect_ratio = person_w / person_h if person_h > 0 else 0
                    if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                        continue
                    
                    head_h = int(person_h * args.head_ratio)
                    head_y = y1
                    head_x = x1
                    head_w = person_w
                    
                    margin_w = int(head_w * 0.1)
                    margin_h = int(head_h * 0.1)
                    
                    head_x = max(0, head_x - margin_w)
                    head_y = max(0, head_y - margin_h)
                    head_w = min(args.width - head_x, head_w + margin_w * 2)
                    head_h = min(args.height - head_y, head_h + margin_h * 2)
                    
                    detected_heads.append((head_x, head_y, head_w, head_h))
            
            # モザイク処理
            for (x, y, w, h) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
            
            frame_time = time.time() - start_time
            
            # ウォームアップ完了後に統計を記録
            if frame_count >= args.warmup:
                if not warmup_done:
                    print("\nウォームアップ完了。測定を開始します。\n")
                    print(f"{'フレーム':<10} {'FPS':<10} {'検出数':<10} {'処理時間(ms)':<15} {'GPU利用率':<12}")
                    print("-" * 80)
                    warmup_done = True
                
                fps_counter.add_frame_time(frame_time)
                detection_count += len(detected_heads)
            
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # 10フレームごとに情報を表示
            if warmup_done and (frame_count - args.warmup) % 10 == 0:
                gpu_util = "N/A"
                if args.device == 'cuda':
                    gpu_info = get_gpu_info()
                    if gpu_info:
                        gpu_util = gpu_info['gpu_util']
                
                print(f"{frame_count - args.warmup:<10} {current_fps:<10.2f} {len(detected_heads):<10} "
                      f"{frame_time * 1000:<15.2f} {gpu_util:<12}")
            
            # プレビュー表示
            if not args.no_display:
                cv2.putText(frame, f'FPS: {current_fps:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.putText(frame, f'Detected: {len(detected_heads)}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.imshow('Benchmark Preview (Press Q to quit)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n終了します...")
                    break
            
            frame_count += 1
        
        # 統計情報を表示
        print("\n" + "=" * 80)
        print("ベンチマーク結果")
        print("=" * 80)
        
        stats = fps_counter.get_statistics()
        if stats:
            print(f"\nパフォーマンス統計:")
            print(f"  平均FPS: {stats['avg_fps']:.2f}")
            print(f"  最小FPS: {stats['min_fps']:.2f}")
            print(f"  最大FPS: {stats['max_fps']:.2f}")
            print(f"  標準偏差: {stats['std_fps']:.2f}")
            print(f"  平均処理時間: {stats['avg_frame_time']:.2f} ms")
            print(f"  総フレーム数: {stats['total_frames']}")
            print(f"  平均検出数: {detection_count / stats['total_frames']:.2f} 人/フレーム")
            
            if args.device == 'cuda':
                gpu_info = get_gpu_info()
                if gpu_info:
                    print(f"\n最終GPU状態:")
                    print(f"  利用率: {gpu_info['gpu_util']}")
                    print(f"  メモリ使用: {gpu_info['memory_used']} / {gpu_info['memory_total']}")
                    print(f"  温度: {gpu_info['temperature']}")
            
            # CSVに保存
            if args.csv:
                import csv
                with open(args.csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Model', 'Device', 'Resolution', 'Avg FPS', 'Min FPS', 'Max FPS', 
                                   'Std FPS', 'Avg Frame Time (ms)', 'Frames', 'Avg Detections'])
                    writer.writerow([
                        args.model, args.device, f'{args.width}x{args.height}',
                        f"{stats['avg_fps']:.2f}", f"{stats['min_fps']:.2f}", 
                        f"{stats['max_fps']:.2f}", f"{stats['std_fps']:.2f}",
                        f"{stats['avg_frame_time']:.2f}", stats['total_frames'],
                        f"{detection_count / stats['total_frames']:.2f}"
                    ])
                print(f"\n結果を {args.csv} に保存しました")
        
        print("\n" + "=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nベンチマークが中断されました")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
