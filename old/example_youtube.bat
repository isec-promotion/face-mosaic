@echo off
REM YouTube Live配信の使用例
REM 実行前に以下を編集してください:
REM 1. YOUR_RTSP_URLをカメラのRTSP URLに変更
REM 2. YOUR_STREAM_KEYをYouTubeのストリームキーに変更

echo ======================================
echo YouTube Live配信 - 顔モザイクプログラム
echo ======================================
echo.

REM =========================================
REM 設定（ここを編集してください）
REM =========================================
set RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream
set STREAM_KEY=YOUR_STREAM_KEY_HERE

REM =========================================
REM 基本設定
REM =========================================
REM 解像度: 1280x720 (HD)
REM フレームレート: 25fps
REM モデル: yolov8n.pt (軽量版)

echo 設定確認:
echo RTSP URL: %RTSP_URL%
echo ストリームキー: (表示されません)
echo 解像度: 1280x720
echo フレームレート: 25fps
echo.

echo 配信を開始しますか？
echo Ctrl+Cで中止、Enterキーで続行
pause

python face-mosaic-youtube.py "%RTSP_URL%" --youtube-key "%STREAM_KEY%" --width 1280 --height 720 --fps 25 --model yolov8n.pt

echo.
echo 配信を終了しました
pause
