# Machine Translation Inference Guide

This document provides instructions for using the `infer.py` script to perform Chinese-to-English translation using pre-trained Transformer or RNN (GRU) models.

## Prerequisites

Before running inference, ensure you have the following:

1. **Pre-trained model checkpoint** (`.pt` or `.pth` file)
2. **Training data directory** containing the original dataset (required for vocabulary reconstruction)
3. **Cache directory** where vocabulary files are stored
4. **Required dependencies** installed (PyTorch, HanLP, etc.)

## Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--arch` | str | Yes | - | Model architecture: `rnn` or `transformer` |
| `--model_path` | str | Yes | - | Path to the trained model checkpoint |
| `--data_path` | str | No | `./data/` | Directory containing training data |
| `--cache_dir` | str | No | `./cache/` | Directory for cached vocabulary files |
| `--frequency` | int | No | `5` | Minimum word frequency for vocabulary |
| `--max_len` | int | No | `256` | Maximum decoding sequence length |
| `--decode_method` | str | No | `beam` | Decoding strategy: `greedy` or `beam` |
| `--beam_size` | int | No | `5` | Beam size for beam search decoding |

## Usage Examples

### Running with Transformer Model

~~~bash
python infer.py \
    --arch transformer \
    --model_path ./best_trans.pt \
~~~

### Running with RNN (GRU) Model

~~~bash
python infer.py \
    --arch rnn \
    --model_path ./best_rnn.pt \
~~~

## Interactive Commands

Once the inference system is running, you can use the following commands within the interactive shell:

| Command | Action |
|---------|--------|
| `q` or `exit` | Quit the program |
| `greedy` | Switch to greedy search decoding |
| `beam N` | Switch to beam search with beam size N (e.g., `beam 5`) |

## Example Session

~~~text
--- Initializing [TRANSFORMER] Translation System ---
Loading weights: ./checkpoints/transformer_best.pt

============================================================
 🚀 Unified Translation Platform | Current Architecture: TRANSFORMER
 Default Mode: beam (Beam Size: 5)
============================================================
 Commands: 'q' to quit | 'greedy' for greedy search | 'beam N' for beam search

[中文] >>> 今天天气很好
[英文] >>> The weather is very nice today.

[中文] >>> greedy
Switched to Greedy Search

[中文] >>> 机器学习是人工智能的一个分支
[英文] >>> Machine learning is a branch of artificial intelligence.

[中文] >>> beam 10
Switched to Beam Search (Size=10)

[中文] >>> q
~~~

## Model Configuration

The script uses the following default model configurations:

**Transformer Model:**
- Embedding dimension: 256
- Number of attention heads: 8
- Number of layers: 3
- Feed-forward dimension: 2048
- Positional encoding: Absolute
- Normalization: RMSNorm

**RNN Model:**
- Embedding dimension: 300
- Hidden dimension: 512
- RNN type: GRU
- Attention type: Multiplicative

## File Structure

~~~text
project/
├── infer.py           # Inference script
├── transformer.py     # Transformer model definition
├── GRU.py             # RNN/GRU model definition
├── utils.py           # Data loading and tokenization utilities
├── data/              # Training data directory
├── cache/             # Vocabulary cache directory
└── checkpoints/       # Model checkpoint directory
~~~