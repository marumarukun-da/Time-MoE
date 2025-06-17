# AWS GPU環境でのTime-MoE構築ガイド

このガイドでは、SSH接続完了後のAWS GPUインスタンス上でTime-MoEの環境構築から動作確認までを詳しく説明します。

**前提条件**: 
- AWS GPU インスタンス（Deep Learning AMI）が起動済み
- SSH接続が完了している

## 目次

1. [環境構築](#1-環境構築)
2. [Time-MoEのセットアップ](#2-time-moeのセットアップ)
3. [動作確認](#3-動作確認)
4. [トラブルシューティング](#4-トラブルシューティング)

## 1. 環境構築

### 1.1 システム更新

```bash
# システムパッケージの更新
sudo apt update && sudo apt upgrade -y
```

### 1.2 CUDA環境の確認

```bash
# CUDAバージョン確認
nvidia-smi

# 期待される出力例
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4               Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   25C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

### 1.3 Python環境の確認とセットアップ

```bash
# Python バージョン確認
python3 --version
# 期待: Python 3.8.x または 3.10.x

# pipの更新
python3 -m pip install --upgrade pip

# 仮想環境作成（推奨）
python3 -m venv time_moe_env
source time_moe_env/bin/activate

# 仮想環境確認
which python
# 期待: /home/ubuntu/time_moe_env/bin/python
```

### 1.3.1 uv（高速パッケージマネージャー）の導入（推奨）

**uvは次世代Pythonパッケージマネージャーで、pipより10-100倍高速に動作します。**

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# uvバージョン確認
uv --version

# uvで仮想環境作成（従来の方法の代替）
uv venv time_moe_env
source time_moe_env/bin/activate

# 仮想環境確認
which python
```

**uvのメリット：**
- ⚡ **高速性**: pipより10-100倍高速なインストール
- 🔧 **pip互換**: 既存のpipコマンドがそのまま使用可能
- 📦 **効率的依存関係解決**: より安定したパッケージ管理

### 1.4 PyTorchの確認

```bash
# PyTorchとCUDAの動作確認
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

**期待される出力例**:
```
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
Device count: 1
Device name: Tesla T4
```

## 2. Time-MoEのセットアップ

### 2.1 リポジトリのクローン

```bash
# Gitのインストール確認
git --version

# Time-MoEリポジトリをクローン
git clone https://github.com/marumarukun-da/Time-MoE.git
cd Time-MoE
```

### 2.2 依存関係のインストール

**方法A: pip使用（従来の方法）**
```bash
# 必要なパッケージのインストール
pip install -r requirements.txt

# 進捗確認（数分かかります）
echo "依存関係インストール完了"
```

**方法B: uv使用（推奨・高速）**
```bash
# uvを使用した高速インストール
uv pip install -r requirements.txt

# 進捗確認（通常1-2分で完了）
echo "依存関係インストール完了"
```

### 2.3 Flash Attentionのインストール（推奨）

**方法A: pip使用**
```bash
# 事前準備
pip install packaging ninja

# Flash Attentionインストール（10-15分かかる場合があります）
echo "Flash Attentionをインストール中... 時間がかかる場合があります"
MAX_JOBS=4 pip install flash-attn==2.6.3 --no-build-isolation
```

**方法B: uv使用（推奨・高速）**
```bash
# 事前準備
uv pip install packaging ninja

# Flash Attentionインストール（uvでも高速化）
echo "Flash Attentionをインストール中..."
MAX_JOBS=4 uv pip install flash-attn==2.6.3 --no-build-isolation
```

**インストール確認（共通）**
```bash
python3 -c "
try:
    import flash_attn
    print('Flash Attention インストール成功')
except ImportError:
    print('Flash Attention インストール失敗 - 続行可能')
"
```

### 2.4 インストール確認

```bash
# Time-MoE固有の依存関係確認
python3 -c "
import torch
from transformers import AutoModelForCausalLM
print('PyTorch:', torch.__version__)
print('Transformers インポート成功')
print('CUDA available:', torch.cuda.is_available())
"
```

## 3. 動作確認

### 3.1 基本的な動作テスト

```bash
# テスト実行ディレクトリ作成
mkdir -p ~/time_moe_test
cd ~/time_moe_test
```

### 3.2 サンプルデータ作成

```bash
# Python スクリプトでテストデータ作成
cat > create_test_data.py << 'EOF'
import json
import numpy as np

# サンプル時系列データ生成
np.random.seed(42)
test_data = []

for i in range(10):
    # 正弦波ベースの時系列データ
    t = np.linspace(0, 4*np.pi, 100)
    signal = np.sin(t) + 0.1*np.random.randn(100) + i*0.1
    test_data.append({"sequence": signal.tolist()})

# JSONL形式で保存
with open('test_data.jsonl', 'w') as f:
    for data in test_data:
        json.dump(data, f)
        f.write('\n')

print(f"テストデータ作成完了: {len(test_data)}個の時系列")
print(f"各系列の長さ: {len(test_data[0]['sequence'])}")
EOF

# テストデータ作成実行
python3 create_test_data.py
```

### 3.3 Time-MoEモデルの動作確認

```bash
# モデル推論テスト
cat > test_inference.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM
import json

print("=== Time-MoE 動作確認テスト ===")

# CUDA確認
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# モデル読み込み
print("\nモデル読み込み中...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map="auto",
        trust_remote_code=True,
    )
    print("✓ モデル読み込み成功")
    print(f"モデルデバイス: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ モデル読み込みエラー: {e}")
    exit(1)

# テストデータ読み込み
print("\nテストデータ読み込み中...")
with open('test_data.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]

# 推論テスト
print("\n推論テスト実行中...")
context_length = 12
test_sequence = test_data[0]['sequence'][:context_length]

# 正規化
import numpy as np
seq_array = np.array(test_sequence)
mean = seq_array.mean()
std = seq_array.std()
normed_seq = (seq_array - mean) / std

# テンソル変換
input_tensor = torch.tensor([normed_seq], dtype=torch.float32)
if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()

try:
    # 推論実行
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=6)
    
    predictions = output[0, -6:].cpu().numpy()
    # 逆正規化
    denorm_predictions = predictions * std + mean
    
    print("✓ 推論実行成功")
    print(f"入力系列: {test_sequence}")
    print(f"予測結果: {denorm_predictions.tolist()}")
    
except Exception as e:
    print(f"✗ 推論エラー: {e}")
    exit(1)

print("\n=== 動作確認完了: すべて正常 ===")
EOF

# 推論テスト実行
cd ~/Time-MoE
python3 ~/time_moe_test/test_inference.py
```

### 3.4 評価機能のテスト

```bash
# 評価スクリプトのテスト
cat > test_evaluation.py << 'EOF'
import pandas as pd
import numpy as np

print("=== 評価機能テスト ===")

# CSV形式のテストデータ作成
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=200, freq='D')
values = np.cumsum(np.random.randn(200)) + 100

test_df = pd.DataFrame({
    'date': dates,
    'value': values
})

test_df.to_csv('test_eval_data.csv', index=False)
print("✓ CSV評価データ作成完了")

# 評価実行
import subprocess
try:
    result = subprocess.run([
        'python3', 'run_eval.py',
        '-d', 'test_eval_data.csv',
        '-p', '10',
        '-c', '50',
        '-b', '1'
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        print("✓ 評価スクリプト実行成功")
        print("出力:", result.stdout[-200:])  # 最後の200文字
    else:
        print(f"評価スクリプトエラー: {result.stderr}")
        
except subprocess.TimeoutExpired:
    print("評価スクリプトがタイムアウトしました")
except Exception as e:
    print(f"評価実行エラー: {e}")

print("=== 評価機能テスト完了 ===")
EOF

python3 test_evaluation.py
```

### 3.5 最終動作確認

```bash
# システム全体の確認
cat > final_check.py << 'EOF'
import torch
import subprocess
import json
import os

print("=== Time-MoE環境 最終確認 ===")

checks = []

# 1. CUDA確認
cuda_available = torch.cuda.is_available()
checks.append(("CUDA", "✓" if cuda_available else "✗"))
if cuda_available:
    checks.append(("GPU", torch.cuda.get_device_name(0)))

# 2. 依存関係確認
try:
    from transformers import AutoModelForCausalLM
    checks.append(("Transformers", "✓"))
except ImportError:
    checks.append(("Transformers", "✗"))

try:
    import flash_attn
    checks.append(("Flash Attention", "✓"))
except ImportError:
    checks.append(("Flash Attention", "✗ (オプション)"))

# 3. ファイル存在確認
files_to_check = [
    'main.py',
    'run_eval.py',
    'requirements.txt',
    'docs/FINE_TUNING_GUIDE.md',
    'docs/AWS_SETUP_GUIDE.md'
]

for file in files_to_check:
    exists = os.path.exists(file)
    checks.append((f"File: {file}", "✓" if exists else "✗"))

# 結果表示
print("\n確認結果:")
for check, status in checks:
    print(f"{check:20} : {status}")

# GPU メモリ使用量
if cuda_available:
    print(f"\nGPU メモリ:")
    print(f"総容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"使用量: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

print("\n=== 環境確認完了 ===")
EOF

python3 final_check.py
```

## 4. トラブルシューティング

### 4.1 よくある問題と解決方法

#### 4.1.1 CUDA認識されない

**症状**: `torch.cuda.is_available()` が `False`

**解決方法**:
```bash
# 1. nvidia-smiコマンド確認
nvidia-smi

# 2. GPUインスタンスタイプ確認
# CPUインスタンス(t2, t3, m5等)ではGPUは使用不可

# 3. ドライバーの再インストール
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

#### 4.1.2 メモリ不足エラー

**症状**: `CUDA out of memory`

**解決方法**:
```bash
# GPU メモリ使用量確認
nvidia-smi

# Python実行時のメモリ制限
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# より小さなバッチサイズを使用
# コード内で batch_size を削減
```

#### 4.1.3 依存関係エラー

**症状**: `ModuleNotFoundError` や `ImportError`

**解決方法**:
```bash
# 仮想環境の確認
which python
source time_moe_env/bin/activate

# 依存関係の再インストール（pip使用）
pip install -r requirements.txt --force-reinstall

# または uv使用（推奨・高速）
uv pip install -r requirements.txt --force-reinstall

# transformersバージョン確認
pip show transformers
# または
uv pip show transformers

# バージョンが4.40.1でない場合（pip）
pip install transformers==4.40.1 --force-reinstall
# または uv使用
uv pip install transformers==4.40.1 --force-reinstall
```

#### 4.1.4 ネットワーク接続エラー

**症状**: モデルダウンロードが失敗

**解決方法**:
```bash
# インターネット接続確認
ping google.com

# Hugging Face Hubの接続確認
python3 -c "from transformers import AutoTokenizer; print('OK')"

# プロキシ設定が必要な場合
export https_proxy=http://proxy-server:port
export http_proxy=http://proxy-server:port
```

## 次のステップ

環境構築が完了したら、以下のドキュメントを参照してファインチューニングを開始してください：

- **ファインチューニングガイド**: `docs/FINE_TUNING_GUIDE.md`
- **CLAUDE.md**: 開発時のガイダンス

---

🎉 **お疲れ様でした！** Time-MoEのAWS GPU環境構築が完了しました。

次は `docs/FINE_TUNING_GUIDE.md` を参照して、独自データでのファインチューニングを開始してください。

このガイドで問題が解決しない場合は、[Time-MoE GitHub Issues](https://github.com/marumarukun-da/Time-MoE/issues) で質問してください。