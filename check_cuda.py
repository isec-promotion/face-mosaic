#!/usr/bin/env python3
"""
CUDA環境診断スクリプト

PyTorchのCUDA対応状況を確認します。
"""

import sys

try:
    import torch
except ImportError:
    print("エラー: PyTorchがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

print("=" * 70)
print("CUDA 環境診断")
print("=" * 70)

# Pythonバージョン
print(f"\nPython version: {sys.version}")

# PyTorchバージョン
print(f"PyTorch version: {torch.__version__}")

# CUDAの利用可能性
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    print("\n✓ CUDA が正常に動作しています")
    print("-" * 70)
    
    # CUDA詳細情報
    print(f"\nCUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # 各GPUの情報
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {'.'.join(map(str, torch.cuda.get_device_capability(i)))}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")
        
        # メモリ使用状況
        if torch.cuda.is_available():
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ GPU を使用してベンチマークを実行できます")
    print("\n推奨コマンド:")
    print('  python benchmark-yolov8.py "rtsp://your-url" --device cuda')
    print("=" * 70)
    
else:
    print("\n✗ CUDA が利用できません")
    print("-" * 70)
    
    # 原因の診断
    print("\n考えられる原因:")
    print("  1. CPU版のPyTorchがインストールされている")
    print("  2. NVIDIAドライバーが正しくインストールされていない")
    print("  3. CUDA Toolkitがインストールされていない")
    
    print("\n解決方法:")
    print("\nステップ1: NVIDIAドライバーの確認")
    print("  コマンドプロンプトで以下を実行:")
    print("    nvidia-smi")
    print("  GPUが認識されていることを確認してください")
    
    print("\nステップ2: PyTorchのCUDA版をインストール")
    print("  以下のコマンドを順番に実行:")
    print("    pip uninstall torch torchvision torchaudio -y")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nステップ3: 再度確認")
    print("    python check_cuda.py")
    
    print("\n詳細な手順は CUDA_SETUP.md を参照してください")
    print("=" * 70)

# 簡易的なパフォーマンステスト
if cuda_available:
    print("\n簡易パフォーマンステスト:")
    print("-" * 70)
    
    try:
        import time
        
        # テスト用のテンソルを作成
        size = 5000
        
        # CPU
        start = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        # GPU
        start = time.time()
        a_gpu = torch.randn(size, size, device='cuda')
        b_gpu = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"行列演算 ({size}x{size}):")
        print(f"  CPU: {cpu_time:.4f} 秒")
        print(f"  GPU: {gpu_time:.4f} 秒")
        print(f"  高速化: {cpu_time / gpu_time:.2f}x")
        print("\n✓ GPU が正常に動作しています")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
    
    print("=" * 70)
