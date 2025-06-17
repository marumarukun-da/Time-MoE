# Time-MoE ファインチューニング完全ガイド

このガイドでは、Time-MoEを使用して独自の時系列データでファインチューニングを行う方法を詳しく説明します。

## 目次
1. [環境準備](#環境準備)
2. [データ準備](#データ準備)
3. [実行手順](#実行手順)
4. [パラメータ詳細](#パラメータ詳細)
5. [トラブルシューティング](#トラブルシューティング)
6. [実践例](#実践例)

## 1. 環境準備

### 1.1 必要な依存関係のインストール

```bash
# リポジトリをクローン
git clone https://github.com/Time-MoE/Time-MoE.git
cd Time-MoE

# 依存関係をインストール（重要：transformers==4.40.1が必須）
pip install -r requirements.txt

# Flash Attention のインストール（推奨、高速化のため）
pip install flash-attn==2.6.3
# または並列コンパイルで高速化
MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation
```

### 1.2 システム要件確認

- Python 3.10以上
- PyTorch（GPU使用時はCUDA対応版）
- 十分なメモリ（モデルサイズに応じて調整）

## 2. データ準備

### 2.1 対応データ形式

Time-MoEは以下の形式のデータをサポートします：

#### 2.1.1 JSONL形式（推奨）
```jsonl
{"sequence": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
{"sequence": [10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7, 17.8]}
{"sequence": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]}
```

#### 2.1.2 JSON形式
```json
[
  {"sequence": [1.0, 2.0, 3.0, 4.0, 5.0]},
  {"sequence": [10.1, 11.2, 12.3, 13.4, 14.5]},
  {"sequence": [0.5, 0.7, 0.9, 1.1, 1.3]}
]
```

#### 2.1.3 Pickle形式
```python
import pickle

# データ準備
data = [
    {"sequence": [1.0, 2.0, 3.0, 4.0, 5.0]},
    {"sequence": [10.1, 11.2, 12.3, 13.4, 14.5]},
    {"sequence": [0.5, 0.7, 0.9, 1.1, 1.3]}
]

# 保存
with open('my_data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

#### 2.1.4 NumPy形式
```python
import numpy as np

# 各系列は異なる長さでも可
sequences = [
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    np.array([10.1, 11.2, 12.3, 13.4, 14.5, 16.0]),
    np.array([0.5, 0.7, 0.9, 1.1])
]

# 保存
np.save('my_data.npy', sequences, allow_pickle=True)
```

### 2.2 データ前処理のガイドライン

#### 2.2.1 データ品質要件
- **最小系列長**: 各時系列は最低2ポイント以上必要
- **データ型**: 数値データ（float32/float64推奨）
- **欠損値**: 事前に補間または除去
- **異常値**: 必要に応じて事前に処理

#### 2.2.2 データサイズ推奨
- **小規模データセット**: 100-1,000系列 → `--stride 1` を使用
- **中規模データセット**: 1,000-10,000系列 → デフォルト設定
- **大規模データセット**: 10,000系列以上 → デフォルト設定

### 2.3 データディレクトリ構造

```
my_project/
├── data/
│   ├── train_data.jsonl          # メインの訓練データ
│   ├── additional_data.json      # 追加データ（オプション）
│   └── validation_data.jsonl     # 検証データ（オプション）
├── models/                       # 保存されたモデル
└── logs/                         # 訓練ログ
```

## 3. 実行手順

### 3.1 基本的なファインチューニング

#### 3.1.1 CPUでの実行（小規模データ向け）
```bash
python main.py -d /path/to/your/data.jsonl
```

#### 3.1.2 単一GPUでの実行
```bash
python torch_dist_run.py main.py -d /path/to/your/data.jsonl
```

#### 3.1.3 複数GPUでの実行（単一ノード）
```bash
python torch_dist_run.py main.py -d /path/to/your/data.jsonl
```

#### 3.1.4 複数ノードでの分散実行
```bash
# ノード0（マスターノード）
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
python torch_dist_run.py main.py -d /path/to/your/data.jsonl

# ノード1
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=1
python torch_dist_run.py main.py -d /path/to/your/data.jsonl
```

### 3.2 推奨設定例

#### 3.2.1 小規模データセット（<1,000系列）
```bash
python torch_dist_run.py main.py \
  -d /path/to/small_dataset.jsonl \
  --stride 1 \
  --max_length 512 \
  --learning_rate 5e-5 \
  --global_batch_size 32 \
  --num_train_epochs 5
```

#### 3.2.2 中規模データセット（1,000-10,000系列）
```bash
python torch_dist_run.py main.py \
  -d /path/to/medium_dataset.jsonl \
  --max_length 1024 \
  --learning_rate 1e-4 \
  --global_batch_size 64 \
  --num_train_epochs 3
```

#### 3.2.3 大規模データセット（>10,000系列）
```bash
python torch_dist_run.py main.py \
  -d /path/to/large_dataset.jsonl \
  --max_length 1024 \
  --learning_rate 1e-4 \
  --global_batch_size 128 \
  --num_train_epochs 1
```

## 4. パラメータ詳細

### 4.1 データ関連パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--data_path` / `-d` | 必須 | 訓練データのパス |
| `--max_length` | 1024 | 最大系列長（1-4096） |
| `--stride` | max_length | スライディングウィンドウのステップサイズ |
| `--normalization_method` | "zero" | 正規化方法（"none", "zero", "max"） |

### 4.2 モデル関連パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--model_path` / `-m` | "Maple728/TimeMoE-50M" | 事前訓練済みモデル |
| `--from_scratch` | False | ゼロから訓練するかどうか |
| `--attn_implementation` | "auto" | 注意機構実装（"auto", "eager", "flash_attention_2"） |

### 4.3 訓練関連パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--learning_rate` | 1e-4 | 学習率 |
| `--min_learning_rate` | 5e-5 | 最小学習率 |
| `--num_train_epochs` | 1.0 | 訓練エポック数 |
| `--train_steps` | None | 訓練ステップ数（エポックより優先） |
| `--global_batch_size` | 64 | グローバルバッチサイズ |
| `--micro_batch_size` | 16 | デバイスごとのバッチサイズ |

### 4.4 最適化関連パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--lr_scheduler_type` | "cosine" | 学習率スケジューラ |
| `--warmup_ratio` | 0.0 | ウォームアップ比率 |
| `--weight_decay` | 0.1 | 重み減衰 |
| `--adam_beta1` | 0.9 | Adam beta1 |
| `--adam_beta2` | 0.95 | Adam beta2 |

### 4.5 その他の重要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--precision` | "fp32" | 精度（"fp32", "fp16", "bf16"） |
| `--gradient_checkpointing` | False | グラディエントチェックポイント |
| `--save_strategy` | "no" | 保存戦略（"steps", "epoch", "no"） |
| `--logging_steps` | 1 | ログ出力間隔 |

## 5. トラブルシューティング

### 5.1 よくある問題と解決方法

#### 5.1.1 メモリ不足エラー
```
RuntimeError: CUDA out of memory
```

**解決方法:**
```bash
# バッチサイズを減らす
--micro_batch_size 8
--global_batch_size 32

# 系列長を短くする
--max_length 512

# グラディエントチェックポイントを有効化
--gradient_checkpointing

# 精度を下げる
--precision fp16
```

#### 5.1.2 データ読み込みエラー
```
ValueError: Unknown file extension: /path/to/data.txt
```

**解決方法:**
- サポートされている形式（.jsonl, .json, .pkl, .npy）に変換
- データ形式を確認し、`{"sequence": [...]}` 構造にする

#### 5.1.3 transformersバージョンエラー
```
ImportError: cannot import name 'XXX' from 'transformers'
```

**解決方法:**
```bash
pip install transformers==4.40.1 --force-reinstall
```

#### 5.1.4 分散訓練エラー
```
RuntimeError: Address already in use
```

**解決方法:**
```bash
# ポートを変更
--port 29501

# または環境変数で設定
export MASTER_PORT=29501
```

### 5.2 性能最適化のヒント

#### 5.2.1 訓練速度向上
- Flash Attention 2を使用: `--attn_implementation flash_attention_2`
- 適切なバッチサイズを設定: GPU メモリを最大限活用
- 複数GPU使用時: `torch_dist_run.py` を使用

#### 5.2.2 メモリ効率化
- グラディエントチェックポイント: `--gradient_checkpointing`
- 混合精度: `--precision bf16` または `--precision fp16`
- 適切なデータローダー workers: `--dataloader_num_workers 4`

## 6. 実践例

### 6.1 株価データのファインチューニング

#### 6.1.1 データ準備
```python
import pandas as pd
import json

# CSV形式の株価データを読み込み
df = pd.read_csv('stock_prices.csv')

# JSONL形式に変換
with open('stock_data.jsonl', 'w') as f:
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]['close'].tolist()
        if len(symbol_data) >= 10:  # 最低10ポイント以上
            json.dump({"sequence": symbol_data}, f)
            f.write('\n')
```

#### 6.1.2 訓練実行
```bash
python torch_dist_run.py main.py \
  -d stock_data.jsonl \
  --max_length 512 \
  --learning_rate 2e-5 \
  --global_batch_size 64 \
  --num_train_epochs 3 \
  --normalization_method zero \
  --save_strategy epoch \
  --output_path ./models/stock_timemoe
```

### 6.2 センサーデータのファインチューニング

#### 6.2.1 データ準備
```python
import numpy as np
import json

# 複数のセンサー読み値
sensor_data = []
for sensor_id in range(100):  # 100個のセンサー
    # 時系列データを生成（実際のデータに置き換え）
    values = np.random.normal(0, 1, size=1000).cumsum().tolist()
    sensor_data.append({"sequence": values})

# JSONL形式で保存
with open('sensor_data.jsonl', 'w') as f:
    for data in sensor_data:
        json.dump(data, f)
        f.write('\n')
```

#### 6.2.2 訓練実行
```bash
python torch_dist_run.py main.py \
  -d sensor_data.jsonl \
  --max_length 1024 \
  --stride 512 \
  --learning_rate 1e-4 \
  --global_batch_size 128 \
  --num_train_epochs 2 \
  --precision bf16 \
  --gradient_checkpointing \
  --output_path ./models/sensor_timemoe
```

### 6.3 エネルギー消費データのファインチューニング

#### 6.3.1 データ準備
```python
# 時間ごとの電力消費データ
import json
from datetime import datetime, timedelta

# サンプルデータ生成
start_date = datetime(2023, 1, 1)
energy_sequences = []

for building_id in range(50):  # 50の建物
    sequence = []
    for day in range(365):  # 1年分
        # 1日24時間のデータ
        daily_pattern = [20 + 10 * np.sin(2 * np.pi * h / 24) + np.random.normal(0, 2) 
                        for h in range(24)]
        sequence.extend(daily_pattern)
    energy_sequences.append({"sequence": sequence})

# 保存
with open('energy_data.jsonl', 'w') as f:
    for seq in energy_sequences:
        json.dump(seq, f)
        f.write('\n')
```

#### 6.3.2 訓練実行
```bash
python torch_dist_run.py main.py \
  -d energy_data.jsonl \
  --max_length 2048 \
  --learning_rate 5e-5 \
  --global_batch_size 64 \
  --num_train_epochs 1 \
  --save_strategy steps \
  --save_steps 1000 \
  --logging_steps 100 \
  --output_path ./models/energy_timemoe
```

## 7. モデル評価とデプロイ

### 7.1 訓練後のモデル評価

```python
import torch
from transformers import AutoModelForCausalLM

# ファインチューニング済みモデルを読み込み
model = AutoModelForCausalLM.from_pretrained(
    './models/your_timemoe',
    trust_remote_code=True
)

# 予測例
context = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
prediction = model.generate(context, max_new_tokens=10)
print("予測結果:", prediction[0, -10:].tolist())
```

### 7.2 モデルの保存と共有

```bash
# HuggingFace Hubにアップロード（オプション）
huggingface-cli login
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('./models/your_timemoe', trust_remote_code=True)
model.push_to_hub('your-username/your-timemoe-model')
"
```

## 8. 応用例とベストプラクティス

### 8.1 ドメイン特化型のファインチューニング

- **金融データ**: 日次/分単位の価格データで学習率を低く設定
- **IoTセンサー**: 高頻度データでstride値を調整
- **気象データ**: 季節パターンを考慮した長期系列（max_length=2048以上）

### 8.2 継続学習のアプローチ

```bash
# 既存モデルから継続して学習
python torch_dist_run.py main.py \
  -d new_data.jsonl \
  --model_path ./models/previous_timemoe \
  --learning_rate 1e-5 \
  --num_train_epochs 1
```

これで、Time-MoEのファインチューニングを成功させるために必要な全ての情報が網羅されています。不明な点があれば、まずこのガイドを参照し、それでも解決しない場合はGitHubのIssueやDiscussionで質問してください。