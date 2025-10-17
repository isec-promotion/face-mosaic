# 監視カメラ顔モザイク処理システム

監視カメラの映像から顔を検出し、自動的にモザイク処理を行う Python プログラム集です。YOLOv8 を使用した高精度な人物検出と、YouTube Live へのリアルタイム配信機能を提供します。

## 📋 目次

- [特徴](#特徴)
- [プログラム一覧](#プログラム一覧)
- [必要な環境](#必要な環境)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [ツール](#ツール)
- [パフォーマンス](#パフォーマンス)
- [トラブルシューティング](#トラブルシューティング)
- [注意事項](#注意事項)

## ✨ 特徴

- **高精度な顔検出**: YOLOv8 による最新の物体検出技術を採用
- **リアルタイム処理**: 監視カメラの映像をリアルタイムで処理
- **YouTube Live 対応**: 処理した映像を直接 YouTube Live で配信可能
- **柔軟な設定**: 検出精度、モデル選択、解像度などをカスタマイズ可能
- **CUDA 対応**: GPU アクセラレーションで高速処理が可能

## 📦 プログラム一覧

### メインプログラム

#### 1. face-mosaic.py

**プロトタイプ版**

- 基本的な顔検出とモザイク処理
- OpenCV の Haar Cascade を使用
- シンプルだが精度は限定的

#### 2. face-mosaic-yolov8.py ⭐ 推奨

**完成版（YOLOv8 使用）**

- YOLOv8 による高精度な人物検出
- 頭部領域の正確な推定
- ローカルストリーミング対応
- 各種パラメータのカスタマイズが可能

```bash
# 基本的な使用方法
python face-mosaic-yolov8.py "rtsp://admin:password@192.168.1.100:554/stream"

# モデルと解像度を指定
python face-mosaic-yolov8.py "rtsp://camera/stream" --model yolov8s.pt --width 1920 --height 1080
```

**詳細なオプション:**

- `--output`: 出力先 URL（デフォルト: udp://127.0.0.1:8080）
- `--width`, `--height`: 出力解像度（デフォルト: 1280x720）
- `--fps`: フレームレート（デフォルト: 25）
- `--model`: YOLOv8 モデル選択（n/s/m/l）
- `--confidence`: 検出信頼度閾値（0.0-1.0、デフォルト: 0.5）
- `--head-ratio`: 頭部領域の割合（0.1-0.5、デフォルト: 0.25）
- `--no-preview`: プレビュー非表示

#### 3. face-mosaic-youtube.py

**YouTube Live 配信版**

- face-mosaic-yolov8.py の機能を全て継承
- YouTube Live への直接配信機能を追加
- ストリームキーによる簡単配信

```bash
# YouTube Live配信
python face-mosaic-youtube.py "rtsp://camera/stream" --youtube-key YOUR_STREAM_KEY

# フルHD 30fps配信
python face-mosaic-youtube.py "rtsp://camera/stream" \
    --youtube-key YOUR_STREAM_KEY \
    --width 1920 --height 1080 --fps 30
```

詳細は [README_YOUTUBE.md](README_YOUTUBE.md) を参照してください。

### ツール

#### check_cuda.py

PC が CUDA に対応しているか確認するツール

```bash
python check_cuda.py
```

**確認内容:**

- PyTorch のインストール状態
- CUDA の利用可能性
- GPU 情報（名前、メモリ、CUDA Compute Capability）
- 簡易パフォーマンステスト

**出力例:**

```
CUDA available: True
GPU 0:
  Name: NVIDIA GeForce RTX 3060
  Total memory: 12.00 GB
  高速化: 15.2x
```

#### benchmark-yolov8.py

YOLOv8 モデルのパフォーマンス測定ツール

```bash
# CPUでベンチマーク
python benchmark-yolov8.py "rtsp://camera/stream"

# GPUでベンチマーク
python benchmark-yolov8.py "rtsp://camera/stream" --device cuda

# 特定のモデルでテスト
python benchmark-yolov8.py "rtsp://camera/stream" --model yolov8s.pt --frames 300
```

**測定項目:**

- フレーム処理速度（FPS）
- 平均処理時間
- 検出数統計
- デバイス情報

#### example_youtube.bat

YouTube 配信を簡単に開始するための Windows バッチスクリプト

```batch
@echo off
set STREAM_KEY=your-youtube-stream-key-here
set RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream

python face-mosaic-youtube.py "%RTSP_URL%" ^
    --youtube-key %STREAM_KEY% ^
    --width 1280 ^
    --height 720 ^
    --fps 25 ^
    --model yolov8n.pt
```

## 🔧 必要な環境

### システム要件

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 以上
- **メモリ**: 最低 4GB（推奨 8GB 以上）
- **GPU**: NVIDIA GPU（オプション、CUDA 対応で高速化）

### 必須ソフトウェア

1. **Python 3.8+**
2. **FFmpeg**（ストリーミングに必須）
   - Windows: https://ffmpeg.org/download.html
   - インストール確認: `ffmpeg -version`

### Python パッケージ

```bash
pip install ultralytics opencv-python numpy torch torchvision
```

## 📥 インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd face-mosaic
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

または個別にインストール:

```bash
pip install ultralytics opencv-python numpy
```

### 3. FFmpeg のインストール

**Windows:**

1. https://ffmpeg.org/download.html からダウンロード
2. 展開したフォルダの bin ディレクトリを PATH に追加

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

### 4. CUDA 対応（オプション、高速化）

GPU 加速を使用する場合:

```bash
# CUDA対応PyTorchのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 確認
python check_cuda.py
```

## 🚀 使用方法

### クイックスタート

1. **CUDA 環境の確認**（GPU 使用時）

```bash
python check_cuda.py
```

2. **ベンチマークの実行**（オプション）

```bash
python benchmark-yolov8.py "rtsp://your-camera-url"
```

3. **顔モザイク処理の開始**

```bash
# ローカルストリーミング
python face-mosaic-yolov8.py "rtsp://your-camera-url"

# VLCで視聴
vlc udp://@127.0.0.1:8080
```

4. **YouTube Live 配信**

```bash
python face-mosaic-youtube.py "rtsp://your-camera-url" --youtube-key YOUR_STREAM_KEY
```

### モデルの選択

YOLOv8 には複数のモデルサイズがあります:

| モデル     | サイズ | 速度 | 精度     | 用途                            |
| ---------- | ------ | ---- | -------- | ------------------------------- |
| yolov8n.pt | 最小   | 最速 | 標準     | リアルタイム処理、低スペック PC |
| yolov8s.pt | 小     | 高速 | 良好     | バランス型、推奨                |
| yolov8m.pt | 中     | 中速 | 高精度   | 高性能 PC で精度重視            |
| yolov8l.pt | 大     | 低速 | 最高精度 | オフライン処理、最高品質        |

### 推奨設定

#### 低スペック PC / リアルタイム重視

```bash
python face-mosaic-yolov8.py "rtsp://camera/stream" \
    --model yolov8n.pt \
    --width 1280 --height 720 \
    --fps 25 \
    --confidence 0.5
```

#### 高性能 PC / 品質重視

```bash
python face-mosaic-yolov8.py "rtsp://camera/stream" \
    --model yolov8s.pt \
    --width 1920 --height 1080 \
    --fps 30 \
    --confidence 0.6
```

#### GPU 使用時

```bash
# ベンチマークでGPU性能を確認
python benchmark-yolov8.py "rtsp://camera/stream" --device cuda

# 最高品質設定
python face-mosaic-yolov8.py "rtsp://camera/stream" \
    --model yolov8m.pt \
    --width 1920 --height 1080 \
    --fps 30
```

## 🔍 トラブルシューティング

### FFmpeg が見つからない

**エラー:**

```
エラー: FFmpegが見つかりません
```

**解決方法:**

1. FFmpeg をインストール
2. PATH に追加されているか確認: `ffmpeg -version`

### RTSP ストリームに接続できない

**確認項目:**

- RTSP の URL が正しいか
- カメラが稼働しているか
- ネットワーク接続が正常か
- 認証情報（ユーザー名/パスワード）が正しいか

### 映像が遅延する

**対処法:**

1. より軽量なモデルを使用: `--model yolov8n.pt`
2. 解像度を下げる: `--width 1280 --height 720`
3. 信頼度閾値を上げる: `--confidence 0.6`
4. GPU を使用（利用可能な場合）

### CUDA が利用できない

```bash
# 診断ツールを実行
python check_cuda.py
```

表示される指示に従ってください。通常は:

1. NVIDIA ドライバーのインストール/更新
2. CUDA 対応 PyTorch の再インストール

### YouTube 配信できない

**確認事項:**

1. ストリームキーが正しいか
2. YouTube Studio で配信状態を確認
3. 初回配信は 24 時間の待機期間が必要な場合があります
4. インターネット接続が安定しているか（アップロード 5Mbps 以上推奨）

## ⚠️ 注意事項

### プライバシーと法的責任

1. **同意の取得**: 監視カメラ映像を処理・配信する前に、必ず関係者全員の同意を得てください
2. **個人情報保護**: モザイク処理後も個人を特定できる情報が残る可能性があります
3. **法令遵守**: 各国・地域の個人情報保護法、監視カメラ規制を遵守してください
4. **責任**: このソフトウェアの使用による一切の責任は使用者が負います

### セキュリティ

1. **ストリームキーの管理**: YouTube 等のストリームキーは絶対に公開しないでください
2. **認証情報**: RTSP URL に含まれる認証情報の取り扱いに注意してください
3. **アクセス制限**: 配信内容へのアクセスを適切に制限してください

### 技術的制約

1. **処理負荷**: 高解像度・高精度モデルは処理負荷が高くなります
2. **遅延**: リアルタイム処理には若干の遅延が発生します
3. **検出精度**: 照明条件、カメラ角度等により精度が変動する場合があります

## 📚 参考リンク

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 公式リポジトリ
- [YOLOv8 Documentation](https://docs.ultralytics.com/) - 公式ドキュメント
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - FFmpeg ガイド
- [YouTube Live Streaming API](https://developers.google.com/youtube/v3/live/getting-started) - YouTube 配信 API

## 📄 ライセンス

このプログラムは教育・研究目的で提供されています。商用利用の際は、使用する各ライブラリのライセンスを確認してください。

## 🤝 コントリビューション

バグ報告、機能要望、プルリクエストを歓迎します。

---

**開発者向け情報:**

- YOLOv8: Ultralytics ライブラリ使用
- 画像処理: OpenCV
- ストリーミング: FFmpeg
- フレームワーク: PyTorch（CUDA 対応）
