# filename: face-mosaic-yolo-jetson-3.py
# python face-mosaic-yolo-jetson-3.py "rtsp://admin:password@192.168.1.100:554/stream"
import sys
import time
import signal
import argparse
from pathlib import Path

import cv2
import numpy as np

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics が見つかりません。 `python3 -m pip install ultralytics` を実行してください。")
    sys.exit(1)


def build_gst_pipeline(rtsp_url: str,
                       use_h265: bool = False,
                       latency: int = 100,
                       drop_on_latency: bool = True,
                       framerate: str = None) -> str:
    """
    Jetson 向けハードウェアデコード（nvv4l2decoder）で RTSP を OpenCV appsink に渡す GStreamer パイプライン
    """
    pay = "rtph265depay" if use_h265 else "rtph264depay"
    parse = "h265parse" if use_h265 else "h264parse"

    # 低遅延寄せのオプション
    latency_opt = f"latency={latency}"
    drop_opt = "drop-on-latency=true" if drop_on_latency else "drop-on-latency=false"
    fr_opt = f"! videorate ! video/x-raw,framerate={framerate}" if framerate else ""

    pipeline = (
        f"rtspsrc location={rtsp_url} {latency_opt} protocols=tcp ! "
        f"{pay} ! {parse} ! "
        f"nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx {fr_opt} ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink emit-signals=false sync=false max-buffers=2 drop=true"
    )
    return pipeline


def pixelate_region(img, x1, y1, x2, y2, pixel=15):
    """
    指定領域をドット（ピクセル化）でモザイク
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return img

    # かなり小さく縮小 → 拡大でピクセル化
    temp = cv2.resize(face, (max(1, w // pixel), max(1, h // pixel)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    return img


def try_load_model():
    """
    顔用モデルの優先順で読み込み。見つからなければ COCO の yolov8n.pt にフォールバック。
    """
    candidates = [
        "yolov8n-face.engine",
        "yolov8n-face.onnx",
        "yolov8n-face.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            print(f"[INFO] Using face model: {c}")
            model = YOLO(c)
            return model, True  # is_face=True

    # フォールバック
    print("[WARN] 顔モデルが見つかりませんでした。COCOの `yolov8n.pt` を使って暫定的に頭部近傍へモザイクを当てます。")
    model = YOLO("yolov8n.pt")
    return model, False  # is_face=False


def main():
    parser = argparse.ArgumentParser(description="Jetson YOLO Face Mosaic (RTSP, GPU)")
    parser.add_argument("rtsp_url", type=str, help="RTSP URL (例: rtsp://user:pass@ip:554/stream)")
    parser.add_argument("--h265", action="store_true", help="入力がH.265のとき指定（既定はH.264）")
    parser.add_argument("--latency", type=int, default=100, help="RTSP jitter buffer latency (ms)")
    parser.add_argument("--conf", type=float, default=0.25, help="検出の信頼度しきい値")
    parser.add_argument("--pixel", type=int, default=15, help="モザイクの粗さ（大きいほど粗）")
    parser.add_argument("--max-fps", type=int, default=0, help="表示側でfps制限（0で無効）")
    parser.add_argument("--show-box", action="store_true", help="デバッグ用に検出枠を表示")
    args = parser.parse_args()

    # モデル読み込み（GPU優先）
    model, is_face = try_load_model()
    try:
        model.to("cuda")
    except Exception as e:
        print(f"[WARN] CUDA 転送に失敗。CPUで実行します: {e}")

    # 半精度試行（TensorRT/ONNX時は無視されることあり）
    try:
        model.fuse()  # 速度最適化（無視されることもあり）
    except Exception:
        pass

    # RTSP → GStreamer pipeline
    gst = build_gst_pipeline(args.rtsp_url, use_h265=args.h265, latency=args.latency)
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] RTSP オープンに失敗。URL/認証/コーデック（--h265）/ネットワークをご確認ください。")
        sys.exit(2)

    win_name = "Face Mosaic Preview (Jetson + YOLO + GPU)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    stop_flag = [False]

    def handle_sigint(sig, frame):
        stop_flag[0] = True
    signal.signal(signal.SIGINT, handle_sigint)

    prev = time.time()
    frame_interval = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0

    while not stop_flag[0]:
        ret, frame = cap.read()
        if not ret or frame is None:
            # 短い再試行
            time.sleep(0.01)
            continue

        # 推論（stream=True で逐次結果を取り出すと低遅延）
        results = model.predict(
            source=frame,
            verbose=False,
            conf=args.conf,
            device=0,          # GPU:0 を明示（CPU時は内部で切替）
            half=True,         # 半精度（環境により無視）
            imgsz=max(frame.shape[0], frame.shape[1]),
            classes=None       # 顔モデルなら全クラス=顔、COCOフォールバック時は後段でpersonのみ使用
        )

        # 結果からモザイク
        r = results[0]  # 1枚入力
        if is_face:
            # 顔モデル想定：検出は顔バウンディングボックス
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    frame = pixelate_region(frame, x1, y1, x2, y2, pixel=args.pixel)
                    if args.show_box:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        else:
            # COCOフォールバック：person クラスのみ拾い、上部（頭部近傍）にモザイク
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                    # COCO で person は 0
                    if cls_id == 0:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        # 上 35% を「頭部領域」として暫定モザイク
                        head_y2 = y1 + int(h * 0.35)
                        frame = pixelate_region(frame, x1, y1, x2, head_y2, pixel=args.pixel)
                        if args.show_box:
                            cv2.rectangle(frame, (x1, y1), (x2, head_y2), (255, 255, 0), 2)

        # FPS 表示
        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2)

        cv2.imshow(win_name, frame)

        # FPS 制限（必要なら）
        if frame_interval > 0:
            # 描画時間を含むため、Key処理前にスリープ
            elapsed = time.time() - now
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # キー処理
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
