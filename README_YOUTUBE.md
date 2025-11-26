# YouTube Live 配信機能付き顔モザイクプログラム

YOLOv8 を使用した顔検出とモザイク処理を行いながら、YouTube Live で配信するプログラムです。

## 主な機能

- **YOLOv8 による高精度な人物検出**
- **顔部分への自動モザイク処理**
- **YouTube Live へのリアルタイム配信**
- **RTSP ストリームの別スレッド読み込み**（安定性向上）
- **FPS 自動取得とフレーム補間**（30fps 未満の場合に自動的にフレーム複製）
- **プレビュー画面なし**（配信専用に最適化）

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
python face-mosaic-youtube.py "rtsp://camera_url" "xxxx-xxxx-xxxx-xxxx"
```

**注意**: `stream_key`は必須引数です（オプションではありません）。`"xxxx-xxxx-xxxx-xxxx"`の部分を実際の YouTube ストリームキーに置き換えてください。

### 解像度とフレームレートの指定

```bash
# フルHD配信（FPSは自動取得、30fps未満の場合は自動補間）
python face-mosaic-youtube.py "rtsp://camera_url" "xxxx-xxxx-xxxx-xxxx" \
    --width 1920 \
    --height 1080
```

**FPS について**:

- ソース FPS を自動取得します
- 30fps 未満の場合は、フレーム複製により自動的に 30fps に補間されます
- 30fps 以上の場合は、ソース FPS をそのまま使用します
- FPS 取得に失敗した場合は、デフォルト値（30fps）を使用します

### モデルの選択

```bash
# より高精度なモデルを使用
python face-mosaic-youtube.py "rtsp://camera_url" "xxxx-xxxx-xxxx-xxxx" \
    --model yolov8s.pt
```

利用可能なモデル:

- `yolov8n.pt`: Nano（最速、メモリ少）
- `yolov8s.pt`: Small（バランス）
- `yolov8m.pt`: Medium（高精度）
- `yolov8l.pt`: Large（最高精度）

### 検出パラメータの調整

```bash
python face-mosaic-youtube.py "rtsp://camera_url" "xxxx-xxxx-xxxx-xxxx" \
    --confidence 0.6 \
    --head-ratio 0.3
```

- `--confidence`: 検出信頼度（0.0-1.0、高いほど誤検出が減る）
- `--head-ratio`: 頭部領域の割合（0.1-0.5、大きいほどモザイク範囲が広い）

### フレーム補間の目標 FPS を変更

```bash
# 補間目標FPSを25fpsに設定（デフォルト: 30fps）
python face-mosaic-youtube.py "rtsp://camera_url" "xxxx-xxxx-xxxx-xxxx" \
    --interpolate-fps 25
```

**フレーム補間について**:

- ソース FPS が`--interpolate-fps`で指定した値より低い場合、フレーム複製により自動的に補間されます
- 例: ソースが 15fps、`--interpolate-fps 30`の場合、各フレームを 2 回送信して 30fps に補間

## コマンドラインオプション一覧

| オプション          | 短縮形 | デフォルト | 説明                                                            |
| ------------------- | ------ | ---------- | --------------------------------------------------------------- |
| `rtsp_url`          | -      | (必須)     | 監視カメラの RTSP URL                                           |
| `stream_key`        | -      | (必須)     | YouTube Live ストリームキー                                     |
| `--width`           | `-W`   | 1280       | 出力映像の幅                                                    |
| `--height`          | `-H`   | 720        | 出力映像の高さ                                                  |
| `--fps`             | `-f`   | 30         | FPS 自動取得失敗時のデフォルト FPS                              |
| `--model`           | `-m`   | yolov8n.pt | YOLOv8 モデル                                                   |
| `--confidence`      | `-c`   | 0.5        | 検出信頼度閾値                                                  |
| `--head-ratio`      | `-r`   | 0.25       | 頭部領域の割合                                                  |
| `--interpolate-fps` | -      | 30         | フレーム補間の目標 FPS（ソース FPS がこの値より低い場合に補間） |

**注意**: このプログラムは YouTube 配信専用です。プレビュー画面は表示されません。

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
- GPU を使用（CUDA 対応の場合）

### FPS が正しく取得できない

**症状**: ソース FPS が 0 や異常な値として表示される

**解決方法**:

- RTSP ストリームの設定を確認
- `--fps`オプションでデフォルト FPS を明示的に指定
- カメラの FPS 設定を確認

### RTSP ストリームに接続できない

1. RTSP の URL が正しいか確認
2. カメラが稼働しているか確認
3. ネットワーク接続を確認

## YouTube 配信の推奨設定

| 解像度    | ビットレート | フレームレート | モデル    | 備考                            |
| --------- | ------------ | -------------- | --------- | ------------------------------- |
| 1280x720  | 2500k        | 30 fps         | yolov8n/s | ソース FPS が低い場合も自動補間 |
| 1920x1080 | 4000k        | 30 fps         | yolov8s   | 高解像度は処理負荷が高い        |

**FPS について**: ソース FPS が 30fps 未満の場合、自動的にフレーム複製により 30fps に補間されます。

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
