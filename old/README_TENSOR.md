了解です。**Jetson Orin Nano（JetPack 6.x 想定）で YOLO を TensorRT 化（.engine）**する最短手順を追記します。まずは **FP16** でのエクスポートを推奨します（安定＆高速）。作成した `yolov8n-face.engine` を、先ほどのスクリプトと同じフォルダに置けば自動で読み込みます。

---

# TensorRT エクスポート手順（YOLOv8 Face → .engine）

## 0) 事前準備

```bash
sudo apt update
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics onnx onnxsim
```

- Jetson 上でビルドしてください（TensorRT は **同一環境でビルドしたエンジン**を使うのが基本）。
- 顔検出モデル（例：`yolov8n-face.pt`）を用意し、作業ディレクトリに置いておきます。

---

## 1) いちばん簡単：Ultralytics から**直接 TensorRT** を出力

```bash
# 出力: yolov8n-face.engine （FP16固定。高速・実運用向け）
yolo export model=yolov8n-face.pt format=engine device=0 half=True imgsz=640 dynamic=False workers=1
```

- `device=0` で GPU 使用、`half=True` で FP16（推奨）。
- `imgsz=640` は一般的な顔検出で十分。必要なら合わせて変更可能（学習サイズと合わせるのが無難）。
- 成功すると `yolov8n-face.engine` が生成されます（同フォルダに配置）。

> **補足**：`dynamic=True` も選べますが、Jetson では **固定解像度（dynamic=False）+ FP16** のほうがビルド・実行とも安定しやすいです。

---

## 2) 代替：ONNX を経由して `trtexec` でビルド

1. ONNX へ書き出し（簡略化あり）

```bash
yolo export model=yolov8n-face.pt format=onnx opset=12 simplify=True imgsz=640
# 出力例: yolov8n-face.onnx
```

2. TensorRT の `trtexec` で .engine 化（FP16 推奨）

```bash
# Jetson に同梱の trtexec を使用（パスは環境により異なる）
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n-face.onnx \
  --saveEngine=yolov8n-face.engine \
  --fp16 \
  --workspace=4096 \
  --verbose \
  --buildOnly
```

- 多解像度を使いたい場合は **動的形状**を指定（Ultralytics ONNX の入力名は多くのケースで `images`）：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n-face.onnx \
  --saveEngine=yolov8n-face.engine \
  --fp16 --workspace=4096 --verbose \
  --minShapes=images:1x3x480x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x1080x1920
```

> ただし、**動的形状はビルドが重くなる・推論時メモリ変動が大きい**などのトレードオフがあります。まずは固定 640 での運用をおすすめします。

---

## 3) 作った `.engine` の動作確認（任意）

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=yolov8n-face.engine \
  --shapes=images:1x3x640x640 \
  --warmUp=200 --duration=300 --streams=1
```

- ここで推定レイテンシ（ms）や処理スループット（FPS）の目安を把握できます。

---

## 4) スクリプトとの連携

- `face-mosaic-yolo-jetson-3.py` と**同じフォルダ**に `yolov8n-face.engine` を置くと自動検出して読み込みます（最優先）。
- もし `.engine` が無い場合は `.onnx` / `.pt` を順に探し、最終的に COCO の `yolov8n.pt` にフォールバックします。

---

## 5) よくある質問・注意点

- **他マシンで作った .engine を持ち込むと落ちる/遅い**
  → TensorRT エンジンは **デバイス依存**です。**Jetson 本体でビルド**してください。JetPack/TensorRT のバージョン差異でも再ビルドが必要になります。
- **INT8 量子化は？**
  → INT8 は**キャリブレーション**が必要でセットアップが複雑です。まずは **FP16** 運用を推奨します（十分高速）。INT8 は実写の代表シーンから校正データを作った上で実施してください（Ultralytics の INT8 エクスポートや、TensorRT のキャリブレータ実装が別途必要）。
- **入力解像度とエンジンの関係**
  → Ultralytics の推論ラッパは内部で画像を `imgsz` にリサイズしてくれるため、カメラ映像が 720p/1080p でも `imgsz=640` のエンジンでそのまま使えます。超高解像で顔が極小なら `imgsz=960` などへ上げると検出率が改善する場合があります（その分コストは上がります）。
- **H.265 カメラ**
  → 受信側はスクリプト実行時に `--h265` を付けるだけで OK（GStreamer のデコーダを切替）。

---

必要であれば、**INT8 キャリブレーションの実施手順（校正データの作り方／キャリブレータ実装）**や、**エンジンの再現性確保（ビルドオプション固定）**, **複数カメラ同時用の最適化（バッチ/スレッド設計）** も追記します！
