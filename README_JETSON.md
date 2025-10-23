# NVIDIA Jetson Orin セットアップガイド

このガイドでは、NVIDIA Jetson Orin Nano Super で顔モザイク処理を実行するための環境構築手順を説明します。

## 目次

- [必要な環境](#必要な環境)
- [インストール手順](#インストール手順)
- [CUDA 対応 PyTorch のインストール](#cuda対応pytorchのインストール)
- [実行方法](#実行方法)
- [パフォーマンス比較](#パフォーマンス比較)
- [トラブルシューティング](#トラブルシューティング)

---

## 必要な環境

### ハードウェア

- NVIDIA Jetson Orin Nano Super
- または Jetson Orin NX / AGX Orin

### ソフトウェア

- JetPack 5.x 以上
- Python 3.8 以上
- CUDA 対応 PyTorch（推奨）

---

## インストール手順

### 1. リポジトリのクローン

```bash
git clone <repository_url>
cd face-mosaic
```

### 2. 基本的な依存関係のインストール

```bash
# システムパッケージの更新
sudo apt update
sudo apt upgrade -y

# 必要なパッケージ
sudo apt install -y python3-pip ffmpeg libopencv-dev

# Python パッケージ
pip3 install opencv-python numpy ultralytics
```

---

## CUDA 対応 PyTorch のインストール

### 重要: パフォーマンスのために CUDA 版が必須

CPU 版 PyTorch では動作しますが、パフォーマンスが大幅に低下します。

### 方法 1: PyTorch 公式（推奨）

```bash
# 既存のPyTorchをアンインストール
pip3 uninstall torch torchvision -y

# CUDA 11.8対応版をインストール
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 方法 2: Jetson 専用ビルド

NVIDIA 公式の Jetson 用 PyTorch を使用:
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

```bash
# JetPack 5.x用 PyTorch 2.x
wget https://nvidia.box.com/shared/static/[最新版のURL]
pip3 install torch-*.whl
```

### インストールの確認

```python
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

期待される出力:

```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8
GPU: Orin
```

---

## 実行方法

### バージョンの選択

#### face-mosaic-yolo-jetson.py（完全最適化版・推奨）

- **デコード**: ハードウェア（GStreamer + NVDEC）
- **推論**: TensorRT（GPU 最適化）
- **エンコード**: libx264（CPU）

```bash
python3 face-mosaic-yolo-jetson.py "rtsp://admin:pass@192.168.1.100:554/stream"
```

**前提条件**: GStreamer とハードウェアデコーダープラグインが必要

```bash
sudo apt install -y gstreamer1.0-tools nvidia-l4t-gstreamer
```

詳細は[README_GST.md](README_GST.md)を参照してください。

#### face-mosaic-yolo-jetson-2.py（シンプル版）

- **デコード**: ソフトウェア（CPU）
- **推論**: TensorRT（GPU 最適化）
- **エンコード**: libx264（CPU）

```bash
python3 face-mosaic-yolo-jetson-2.py "rtsp://admin:pass@192.168.1.100:554/stream"
```

**利点**: GStreamer 不要、セットアップが簡単

### 初回実行（TensorRT 変換）

初回実行時は、PyTorch モデルを TensorRT エンジンに変換します（数分かかります）:

```bash
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream"
```

出力例:

```
TensorRTエンジンへの変換を試みています...
（初回のみ時間がかかります。数分お待ちください）
...
変換が完了しました。TensorRTエンジン（yolov8n_Orin.engine）を読み込んでいます...
```

**生成されるファイル**: `yolov8n_Orin.engine`

### 2 回目以降の実行

TensorRT エンジンが既に存在する場合、即座に起動します:

```bash
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream"
```

### オプション

```bash
# 出力先を指定
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --output udp://192.168.1.200:8080

# 解像度とフレームレートを指定
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --width 1920 --height 1080 --fps 30

# より高精度なモデルを使用
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --model yolov8s.pt

# TensorRT変換をスキップ（テスト用）
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --no-tensorrt

# プレビューなし（ヘッドレス動作）
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --no-preview
```

---

## パフォーマンス比較

### 処理速度の目安（1280x720@25fps）

| 構成             | FPS   | CPU 使用率 | GPU 使用率 | 備考     |
| ---------------- | ----- | ---------- | ---------- | -------- |
| PyTorch (CPU)    | 3-5   | 90-100%    | 0%         | 非推奨   |
| PyTorch (CUDA)   | 10-15 | 30-40%     | 60-70%     |          |
| TensorRT         | 20-25 | 20-30%     | 50-60%     | **推奨** |
| TensorRT + NVDEC | 25-30 | 10-20%     | 60-70%     | **最高** |

### モデル別の推論速度（TensorRT 使用時）

| モデル     | 推論時間 | 精度     | メモリ使用量 |
| ---------- | -------- | -------- | ------------ |
| yolov8n.pt | ~20ms    | 普通     | ~2GB         |
| yolov8s.pt | ~30ms    | 良好     | ~3GB         |
| yolov8m.pt | ~45ms    | 高精度   | ~4GB         |
| yolov8l.pt | ~60ms    | 最高精度 | ~6GB         |

**推奨**: Orin Nano Super では`yolov8n.pt`または`yolov8s.pt`

---

## トラブルシューティング

### 1. CUDA が認識されない

**症状**:

```
警告: CUDAが利用できません
torch.cuda.is_available(): False
```

**原因**: CPU 版 PyTorch がインストールされている

**解決方法**:

```bash
pip3 uninstall torch torchvision -y
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. TensorRT 変換に失敗

**症状**:

```
警告: TensorRTへの変換に失敗しました
```

**解決方法**:

```bash
# TensorRT変換をスキップして実行
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --no-tensorrt
```

### 3. メモリ不足

**症状**: プログラムがクラッシュまたは動作が遅い

**解決方法**:

```bash
# 小さいモデルを使用
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --model yolov8n.pt

# 解像度を下げる
python3 face-mosaic-yolo-jetson-2.py "rtsp://camera/stream" --width 640 --height 480
```

### 4. FFmpeg エンコードエラー

**症状**: FFmpeg がすぐに終了する

**原因**: Jetson 環境が正しく検出されていない

**確認**:

```bash
# Jetson環境の確認
ls /etc/nv_tegra_release
```

### 5. フレームレートが低い

**チェックリスト**:

- [ ] CUDA 対応 PyTorch を使用しているか
- [ ] TensorRT エンジンが生成されているか（`*.engine`ファイル）
- [ ] ネットワーク帯域は十分か
- [ ] 適切なモデルサイズを使用しているか

**パフォーマンス確認**:

```bash
# GPUの使用状況を確認
sudo tegrastats

# プロセスの状態を確認
htop
```

### 6. GStreamer パイプラインエラー（face-mosaic-yolo-jetson.py）

**症状**:

```
エラー: RTSPストリームを開けませんでした
GStreamerとハードウェアデコーダーが正しくインストールされているか確認してください
```

**解決方法**:

```bash
# GStreamerプラグインのインストール
sudo apt install -y nvidia-l4t-gstreamer nvidia-l4t-multimedia

# プラグインの確認
gst-inspect-1.0 nvv4l2decoder
```

詳細は[README_GST.md](README_GST.md)を参照してください。

---

## 環境依存ファイルについて

### TensorRT エンジンファイル

TensorRT エンジン（`*.engine`）は**環境依存**です:

- 開発環境（Windows/RTX）で生成したエンジンは、Jetson では使用不可
- Jetson で生成したエンジンは、他の GPU では使用不可
- 各環境で初回実行時に自動生成されます

**ファイル名の例**:

```
yolov8n_Orin.engine           # Jetson Orin用
yolov8n_NVIDIA_GeForce_RTX_4080_SUPER.engine  # Windows/RTX用
```

これらのファイルは`.gitignore`で管理対象外になっています。

---

## 本番運用のヒント

### 1. 自動起動設定

Systemd サービスとして登録:

```bash
sudo nano /etc/systemd/system/face-mosaic.service
```

```ini
[Unit]
Description=Face Mosaic Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/face-mosaic
ExecStart=/usr/bin/python3 /home/jetson/face-mosaic/face-mosaic-yolo-jetson.py "rtsp://camera/stream" --no-preview
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# サービスを有効化
sudo systemctl enable face-mosaic
sudo systemctl start face-mosaic

# ステータス確認
sudo systemctl status face-mosaic
```

### 2. ログ管理

```bash
# ログをファイルに出力
python3 face-mosaic-yolo-jetson.py "rtsp://camera/stream" --no-preview 2>&1 | tee face-mosaic.log

# または、systemd journalで確認
journalctl -u face-mosaic -f
```

### 3. リモート監視

```bash
# 別のマシンからVLCで確認
vlc udp://@:8080

# または、ffplayで確認
ffplay -fflags nobuffer -flags low_delay udp://192.168.1.100:8080
```

---

## 参考リンク

- [NVIDIA Jetson 公式ドキュメント](https://developer.nvidia.com/embedded/jetson-orin)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [GStreamer セットアップガイド](README_GST.md)

---

## サポート

問題が発生した場合は、以下の情報を含めて Issue を作成してください:

1. Jetson モデル（Orin Nano / NX / AGX）
2. JetPack バージョン: `dpkg -l | grep nvidia-jetpack`
3. PyTorch バージョンと CUDA 状態: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
4. エラーメッセージの全文

---

## まとめ

最高のパフォーマンスを得るための推奨構成:

1. ✅ CUDA 対応 PyTorch をインストール
2. ✅ 初回実行で TensorRT エンジンを生成
3. ✅ face-mosaic-yolo-jetson.py（GStreamer 版）を使用
4. ✅ 適切なモデルサイズを選択（yolov8n or yolov8s）

これにより、リアルタイムで高精度な顔モザイク処理が可能になります。
