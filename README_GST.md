# GStreamer セットアップガイド

このガイドでは、Ubuntu/Jetson 環境に GStreamer と NVIDIA ハードウェアアクセラレーション対応プラグインをインストールする方法を説明します。

## 目次

- [基本的な GStreamer のインストール](#基本的なgstreamerのインストール)
- [Jetson 用ハードウェアデコーダープラグイン](#jetson用ハードウェアデコーダープラグイン)
- [OpenCV の GStreamer サポート確認](#opencvのgstreamerサポート確認)
- [動作確認](#動作確認)
- [トラブルシューティング](#トラブルシューティング)

---

## 基本的な GStreamer のインストール

### Ubuntu/Debian 系

```bash
# システムパッケージの更新
sudo apt update
sudo apt upgrade -y

# GStreamerコアパッケージのインストール
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# 開発用ヘッダーファイル（OpenCV再ビルド用）
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev
```

### RTSP 対応の確認

```bash
# RTSPプラグインの確認
gst-inspect-1.0 rtspsrc

# H.264デコーダーの確認
gst-inspect-1.0 avdec_h264
```

---

## Jetson 用ハードウェアデコーダープラグイン

NVIDIA Jetson デバイスでハードウェアアクセラレーションを使用するには、専用のプラグインが必要です。

### JetPack のインストール

JetPack には必要なプラグインが含まれています。

```bash
# JetPackのバージョン確認
dpkg -l | grep nvidia-jetpack
```

JetPack がインストールされていない場合:

```bash
# NVIDIA SDKマネージャーを使用してインストール
# https://developer.nvidia.com/nvidia-sdk-manager
```

### NVIDIA マルチメディア API のインストール

```bash
# gstreamer-nvv4l2プラグインのインストール（JetPack 4.x/5.x）
sudo apt install -y nvidia-l4t-gstreamer

# 追加のマルチメディアライブラリ
sudo apt install -y \
    nvidia-l4t-multimedia \
    nvidia-l4t-multimedia-utils
```

### ハードウェアデコーダーの確認

```bash
# nvv4l2decoderプラグインの確認
gst-inspect-1.0 nvv4l2decoder

# 利用可能なNVIDIAプラグインの一覧
gst-inspect-1.0 | grep nv
```

期待される出力:

```
nvv4l2decoder:  nvv4l2decoder: NVIDIA v4l2 video decoder
nvvidconv:  nvvidconv: NvVideoConverter Plugin
nvv4l2h264enc:  nvv4l2h264enc: NVIDIA v4l2 H.264 encoder
nvv4l2h265enc:  nvv4l2h265enc: NVIDIA v4l2 H.265 encoder
...
```

---

## OpenCV の GStreamer サポート確認

OpenCV が GStreamer サポート付きでビルドされているか確認します。

### Python で確認

```python
import cv2
print(cv2.getBuildInformation())
```

出力から以下を探します:

```
Video I/O:
  ...
  GStreamer:                   YES (1.16.2)
  ...
```

### GStreamer サポートがない場合

OpenCV を再ビルドする必要があります。

```bash
# 依存関係のインストール
sudo apt install -y \
    build-essential cmake git pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-dev python3-numpy

# OpenCVのソースコード取得
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x  # 最新の安定版を使用

# ビルドディレクトリの作成
mkdir build
cd build

# CMakeでGStreamerサポートを有効化
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_GSTREAMER=ON \
      -D WITH_CUDA=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF ..

# ビルド（時間がかかります）
make -j$(nproc)

# インストール
sudo make install
sudo ldconfig
```

---

## 動作確認

### 1. GStreamer パイプラインのテスト

```bash
# RTSPストリームの受信テスト（ソフトウェアデコード）
gst-launch-1.0 rtspsrc location="rtsp://camera/stream" latency=0 ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

# ハードウェアデコードのテスト（Jetson）
gst-launch-1.0 rtspsrc location="rtsp://camera/stream" latency=0 ! \
  rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! \
  'video/x-raw,format=BGRx' ! videoconvert ! autovideosink
```

### 2. Python でのテスト

```python
import cv2

# GStreamerパイプライン
pipeline = (
    'rtspsrc location="rtsp://camera/stream" latency=0 ! '
    'rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! '
    'video/x-raw,format=BGRx ! videoconvert ! '
    'video/x-raw,format=BGR ! appsink drop=1'
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✓ GStreamerパイプラインが正常に開きました")
    ret, frame = cap.read()
    if ret:
        print(f"✓ フレーム取得成功: {frame.shape}")
    cap.release()
else:
    print("✗ GStreamerパイプラインを開けませんでした")
```

### 3. face-mosaic-yolo-jetson.py のテスト

```bash
# ハードウェアアクセラレーション版の実行
python face-mosaic-yolo-jetson.py "rtsp://camera/stream"
```

正常に動作すれば以下のメッセージが表示されます:

```
RTSPストリームに接続しています（ハードウェアデコード使用）...
接続成功（ハードウェアデコード使用）
```

---

## トラブルシューティング

### エラー: "Could not open resource for reading and writing"

**原因**: GStreamer プラグインが見つからない

**解決方法**:

```bash
# プラグインパスの確認
echo $GST_PLUGIN_PATH

# プラグインの再スキャン
gst-inspect-1.0

# NVIDIAプラグインが見つからない場合
sudo apt install --reinstall nvidia-l4t-gstreamer
```

### エラー: "nvv4l2decoder: No such element"

**原因**: NVIDIA ハードウェアデコーダープラグインがインストールされていない

**解決方法**:

```bash
# JetPackの確認
dpkg -l | grep nvidia-jetpack

# NVIDIAマルチメディアパッケージのインストール
sudo apt install -y nvidia-l4t-gstreamer nvidia-l4t-multimedia

# デバイスノードの確認
ls -l /dev/nvhost-*
```

### OpenCV が"CAP_GSTREAMER"をサポートしていない

**原因**: OpenCV が GStreamer サポートなしでビルドされている

**解決方法**:

1. OpenCV を再ビルド（上記手順参照）
2. または、GStreamer 不使用版を使用:

```bash
python face-mosaic-yolo-jetson-2.py "rtsp://camera/stream"
```

### パフォーマンスが低い

**確認事項**:

```bash
# ハードウェアデコーダーが使用されているか確認
# 実行中に別のターミナルで:
nvidia-smi  # またはjetson_stats (jtop)

# CPU使用率の確認
htop
```

**期待される動作**:

- ハードウェアデコード使用時: GPU 使用率が上がり、CPU 使用率が低い
- ソフトウェアデコード使用時: CPU 使用率が高い

---

## 参考リンク

- [GStreamer 公式ドキュメント](https://gstreamer.freedesktop.org/documentation/)
- [NVIDIA Jetson Linux Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/index.html)
- [OpenCV GStreamer Support](https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html)

---

## バージョン情報

このドキュメントは以下の環境でテスト済みです:

- Ubuntu 18.04/20.04/22.04
- NVIDIA JetPack 4.6/5.0/5.1
- GStreamer 1.14/1.16/1.20
- OpenCV 4.5/4.6/4.8

---

## まとめ

1. **基本インストール**: GStreamer とプラグインをインストール
2. **Jetson 環境**: nvidia-l4t-gstreamer パッケージをインストール
3. **OpenCV 確認**: GStreamer サポートの有無を確認
4. **動作テスト**: gst-launch-1.0 で接続テスト
5. **問題解決**: エラーメッセージに応じて対処

ハードウェアアクセラレーションが正常に動作すれば、CPU 負荷を大幅に削減し、複数のカメラストリームを同時に処理できるようになります。
