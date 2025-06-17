# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Important: This CLAUDE.md file should be updated whenever there are repository updates that introduce new commands, architectural changes, or development workflows that would be helpful for future Claude Code instances.**

## Documentation Guides

### Fine-tuning Guide
A comprehensive fine-tuning guide is available in `docs/FINE_TUNING_GUIDE.md` which provides detailed instructions for:
- Data preparation and format requirements
- Step-by-step fine-tuning procedures
- Parameter optimization recommendations  
- Practical examples with real datasets
- Troubleshooting common issues

### AWS GPU Setup Guide
A complete AWS GPU environment setup guide is available in `docs/AWS_SETUP_GUIDE.md` which covers:
- AWS account setup and GPU instance selection
- SSH connection and security configuration
- Environment setup (CUDA, Python, dependencies)
- Time-MoE installation and verification
- Cost management and troubleshooting
- Designed for non-engineers with detailed step-by-step instructions

Refer to these guides when helping users set up and fine-tune Time-MoE on their custom time series data.

## Project Overview

Time-MoE is a billion-scale time series foundation model family with Mixture of Experts (MoE) architecture. It's the first work to scale time series foundation models up to 2.4 billion parameters, trained from scratch on the Time-300B dataset (300+ billion time points).

Key features:
- Decoder-only transformer architecture with MoE layers
- Auto-regressive time series forecasting
- Context lengths up to 4096 tokens
- Multiple model sizes: 50M, 200M, up to 2.4B parameters
- Built on HuggingFace Transformers with custom implementations

## Dependencies & Installation

**Critical Requirement**: `transformers==4.40.1` (exact version required)

Install dependencies:
```bash
pip install -r requirements.txt
```

Optional but recommended for performance:
```bash
pip install flash-attn==2.6.3
# Or for faster compilation:
MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation
```

Core dependencies: PyTorch, transformers==4.40.1, datasets==2.18.0, accelerate==0.28.0

## Training Commands

### CPU Training
```bash
python main.py -d <data_path>
```

### Single Node (Single/Multi-GPU)
```bash
python torch_dist_run.py main.py -d <data_path>
```

### Multi-Node Distributed Training
```bash
export MASTER_ADDR=<master_addr>
export MASTER_PORT=<master_port>
export WORLD_SIZE=<world_size>
export RANK=<rank>
python torch_dist_run.py main.py -d <data_path>
```

### Training from Scratch
Add `--from_scratch` flag:
```bash
python torch_dist_run.py main.py -d <data_path> --from_scratch
```

### Important Training Parameters
- `--stride 1`: Recommended for small datasets
- `--max_length`: Maximum sequence length (default: 1024, training max: 4096)
- `--normalization_method`: "none", "zero", "max" (default: "zero")
- `--attn_implementation`: "auto", "eager", "flash_attention_2"

## Evaluation Commands

### Benchmark Evaluation
```bash
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 96
```

### Distributed Evaluation
```bash
python torch_dist_run.py run_eval.py -d <data_path> -p <prediction_length>
```

### Common Evaluation Parameters
- `-p, --prediction_length`: Forecast horizon (96, 192, 336, 720)
- `-c, --context_length`: Context window (auto-determined if not specified)
- `-m, --model`: Model path (default: Maple728/TimeMoE-50M)
- `-b, --batch_size`: Batch size for evaluation (default: 32)

## Data Format Requirements

Training data should be in JSONL format with `sequence` field:
```jsonl
{"sequence": [1.0, 2.0, 3.0, ...]}
{"sequence": [11.0, 22.0, 33.0, ...]}
```

Supported formats: `.jsonl`, `.json`, `.pickle`

For Time-300B dataset usage:
```python
from time_moe.datasets.time_moe_dataset import TimeMoEDataset
ds = TimeMoEDataset('Time-300B')
seq = ds[0]  # Get first sequence
```

## Architecture Overview

### Core Components
- **TimeMoeConfig**: Model configuration in `time_moe/models/configuration_time_moe.py`
- **TimeMoeForPrediction**: Main model class in `time_moe/models/modeling_time_moe.py`
- **TimeMoeRunner**: Training orchestrator in `time_moe/runner.py`
- **TimeMoeTrainer**: Custom HuggingFace trainer in `time_moe/trainer/hf_trainer.py`

### Dataset Classes
- **TimeMoEDataset**: Base dataset for Time-300B format
- **TimeMoEWindowDataset**: Sliding window dataset for training
- **BenchmarkEvalDataset**: CSV-based evaluation datasets
- **GeneralEvalDataset**: General format evaluation

### Key Architecture Details
- Decoder-only transformer with MoE layers
- RoPE positional embeddings (rope_theta=10000)
- Configurable number of experts and experts-per-token
- Load balancing loss for expert routing
- Support for both eager and flash attention

## Model Variants

Available pre-trained models on HuggingFace:
- `Maple728/TimeMoE-50M`: Base 50M parameter model
- `Maple728/TimeMoE-200M`: Large 200M parameter model
- Larger variants up to 2.4B parameters

## Important Constraints

- **Maximum sequence length**: 4096 tokens (sum of context + prediction length)
- **Transformers version**: Must use exactly 4.40.1
- **Normalization**: Input sequences should be normalized (zero-mean, unit-variance)
- **Memory**: Large models require significant GPU memory; use gradient checkpointing if needed

## Distributed Training Setup

The `torch_dist_run.py` script automatically handles:
- Environment variable detection (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK)
- GPU availability detection
- Torchrun command construction
- Fallback to CPU training if CUDA unavailable

## Inference Example

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="auto",
    trust_remote_code=True
)

# Normalize input sequence
seqs = torch.randn(2, 12)  # [batch_size, context_length]
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# Generate predictions
output = model.generate(normed_seqs, max_new_tokens=6)
predictions = output[:, -6:] * std + mean  # Denormalize
```