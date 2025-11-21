# YouTube Live 配信機能付き顔モザイクプログラム

YOLOv8 を使用した顔検出とモザイク処理を行いながら、YouTube Live で配信するプログラムです。

## 主な機能

- **YOLOv8 による高精度な人物検出**
- **顔部分への自動モザイク処理**
- **YouTube Live へのリアルタイム配信**
- **ローカルストリーミングにも対応**

## 必要な環境

### 必須ソフトウェア

1. **Python 3.8 以上**
2. **FFmpeg** (YouTube 配信に必須)
   - Windows: https://ffmpeg.org/download.html からダウンロードして PATH に追加
   - インストール確認: `ffmpeg -version`

### 必要な Python パッケージ

```bash
pip install ultralytics opencv-python numpy
```

## YouTube Live 配信の準備

### 1. YouTube でライブ配信を有効化

1. YouTube Studio にアクセス: https://studio.youtube.com/
2. 左メニューから「作成」→「ライブ配信を開始」を選択
3. 初回の場合、ライブ配信の有効化に 24 時間かかることがあります

### 2. ストリームキーの取得

1. YouTube Studio で「作成」→「ライブ配信を開始」
2. 「ウェブカメラ」ではなく「ストリーミング ソフトウェア」を選択
3. 「ストリームキー」をコピー
   - **注意**: ストリームキーは絶対に公開しないでください

## 使用方法

### 基本的な使い方（YouTube 配信）

```bash
python face-mosaic-youtube.py "rtsp://camera_url" --youtube-key YOUR_STREAM_KEY
```

### 解像度とフレームレートの指定

```bash
# フルHD 30fps配信
python face-mosaic-youtube.py "rtsp://camera_url" \
    --youtube-key YOUR_STREAM_KEY \
    --width 1920 \
    --height 1080 \
    --fps 30
```

### モデルの選択

```bash
# より高精度なモデルを使用
python face-mosaic-youtube.py "rtsp://camera_url" \
    --youtube-key YOUR_STREAM_KEY \
    --model yolov8s.pt
```

利用可能なモデル:

- `yolov8n.pt`: Nano（最速、メモリ少）
- `yolov8s.pt`: Small（バランス）
- `yolov8m.pt`: Medium（高精度）
- `yolov8l.pt`: Large（最高精度）

### 検出パラメータの調整

```bash
python face-mosaic-youtube.py "rtsp://camera_url" \
    --youtube-key YOUR_STREAM_KEY \
    --confidence 0.6 \
    --head-ratio 0.3
```

- `--confidence`: 検出信頼度（0.0-1.0、高いほど誤検出が減る）
- `--head-ratio`: 頭部領域の割合（0.1-0.5、大きいほどモザイク範囲が広い）

### プレビューなしで実行

```bash
python face-mosaic-youtube.py "rtsp://camera_url" \
    --youtube-key YOUR_STREAM_KEY \
    --no-preview
```

### ローカルストリーミング（YouTube 以外）

```bash
# UDP出力
python face-mosaic-youtube.py "rtsp://camera_url" \
    --output udp://127.0.0.1:8080

# VLCで視聴
vlc udp://@127.0.0.1:8080
```

## コマンドラインオプション一覧

| オプション      | 短縮形 | デフォルト           | 説明                           |
| --------------- | ------ | -------------------- | ------------------------------ |
| `rtsp_url`      | -      | (必須)               | 監視カメラの RTSP URL          |
| `--youtube-key` | `-y`   | なし                 | YouTube Live ストリームキー    |
| `--output`      | `-o`   | udp://127.0.0.1:8080 | 出力先 URL（YouTube 未指定時） |
| `--width`       | `-W`   | 1280                 | 出力映像の幅                   |
| `--height`      | `-H`   | 720                  | 出力映像の高さ                 |
| `--fps`         | `-f`   | 25                   | フレームレート                 |
| `--model`       | `-m`   | yolov8n.pt           | YOLOv8 モデル                  |
| `--confidence`  | `-c`   | 0.5                  | 検出信頼度閾値                 |
| `--head-ratio`  | `-r`   | 0.25                 | 頭部領域の割合                 |
| `--no-preview`  | -      | False                | プレビュー非表示               |

## トラブルシューティング

### FFmpeg が見つからない

```
エラー: FFmpegが見つかりません
```

**解決方法**: FFmpeg をインストールして PATH に追加してください

### YouTube 配信が開始されない

1. ストリームキーが正しいか確認
2. YouTube Studio で配信状態を確認
3. 初回配信の場合、24 時間の待機期間が必要な場合があります

### 映像が遅延する

**解決方法**:

- より軽量なモデル（yolov8n.pt）を使用
- 解像度を下げる（例: 1280x720）
- `--confidence`を上げて処理を軽減

### RTSP ストリームに接続できない

1. RTSP の URL が正しいか確認
2. カメラが稼働しているか確認
3. ネットワーク接続を確認

## YouTube 配信の推奨設定

| 解像度    | ビットレート | フレームレート | モデル    |
| --------- | ------------ | -------------- | --------- |
| 1280x720  | 2500k        | 25-30 fps      | yolov8n/s |
| 1920x1080 | 4000k        | 25-30 fps      | yolov8s   |

## 注意事項

1. **プライバシー**: 監視カメラ映像を配信する際は、必ず関係者の同意を得てください
2. **ストリームキー**: YouTube ストリームキーは絶対に公開しないでください
3. **処理負荷**: 高解像度・高精度モデルは処理負荷が高くなります
4. **ネットワーク**: 安定したインターネット接続が必要です（アップロード帯域 5Mbps 以上推奨）

## ライセンス

このプログラムは教育・研究目的で提供されています。
商用利用の際は適切なライセンスを確認してください。

## 参考リンク

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [YouTube Live Streaming API](https://developers.google.com/youtube/v3/live/getting-started)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
