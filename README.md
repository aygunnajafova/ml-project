# Text-to-SQL Translation

A comprehensive implementation for translating natural language queries to SQL using fine-tuned T5 sequence-to-sequence models.

## 🎯 Project Overview

This project implements Text-to-SQL translation by fine-tuning pre-trained T5 models. The system translates natural language questions into executable SQL queries and evaluates them against a flight database, providing comprehensive metrics and error analysis.

## ✨ Features

- **T5 Fine-tuning**: Fine-tune pre-trained T5 models for Text-to-SQL translation
- **Comprehensive Evaluation**: SQL Exact Match, Record F1, Record EM, and error rate metrics
- **Error Analysis Tools**: Detailed mismatch analysis and debugging utilities
- **Production-Ready**: Configurable early stopping, learning rate scheduling, and checkpointing

## 🚀 Quick Start

### Installation

```bash
conda create -n text-to-sql python=3.10
conda activate text-to-sql
pip install -r requirements.txt
```

### Training

Fine-tune a pre-trained T5 model:
```bash
python train_t5.py --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 20 \
    --patience_epochs 10 \
    --batch_size 32 \
    --test_batch_size 32 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name my_experiment
```

**Train only Encoder** (freeze all decoder layers):
```bash
python train_t5.py --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 20 \
    --patience_epochs 10 \
    --batch_size 32 \
    --test_batch_size 32 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --freeze_decoder_layers "all" \
    --experiment_name encoder_only
```

**Train only Decoder** (freeze all encoder layers):
```bash
python train_t5.py --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 20 \
    --patience_epochs 10 \
    --batch_size 32 \
    --test_batch_size 32 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --freeze_encoder_layers "all" \
    --experiment_name decoder_only
```

**Training with selective layer freezing** (freeze specific layers while training others):
```bash
# Freeze first 3 encoder layers and all decoder layers
python train_t5.py --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 20 \
    --freeze_encoder_layers "0,1,2" \
    --freeze_decoder_layers "all" \
    --experiment_name frozen_layers_experiment

# Freeze embeddings and specific decoder layers
python train_t5.py --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 20 \
    --freeze_decoder_layers "0,1" \
    --freeze_embeddings \
    --experiment_name partial_freeze
```

### Evaluation

Evaluate model predictions on the development set:
```bash
python evaluate.py \
    --predicted_sql results/t5_ft_my_experiment_dev.sql \
    --predicted_records records/t5_ft_my_experiment_dev.pkl \
    --development_sql data/dev.sql \
    --development_records records/ground_truth_dev.pkl
```

### Error Analysis

Analyze prediction errors and mismatches:
```bash
python find_mismatches.py \
    --experiment_name my_experiment \
    --finetune
```

## 📊 Evaluation Metrics

The system computes three primary metrics:

- **SQL Exact Match (EM)**: Exact string match between predicted and ground truth SQL
- **Record Exact Match (EM)**: Exact match of database query results
- **Record F1 Score**: Token-level F1 score on query result sets
- **Error Rate**: Percentage of queries that produce SQL execution errors

## 📁 Project Structure

```
ml-project/
├── data/              # Dataset files (train/dev/test NL and SQL queries)
├── results/           # Generated SQL predictions (.sql files)
├── records/           # Database query results (.pkl files)
├── checkpoints/       # Model checkpoints (.pt files)
├── mismatches/       # Error analysis reports
├── train_t5.py       # Main training script
├── evaluate.py       # Evaluation script
└── find_mismatches.py # Error analysis tool
```

## 🛠️ Technical Details

### Model Architecture

- **Base Model**: Google T5-small (60M parameters)
- **Tokenizer**: T5TokenizerFast
- **Input Format**: Natural language queries with "translate to SQL: " prefix
- **Output Format**: SQL queries compatible with SQLite

### Training Features

- **Optimizer**: AdamW with configurable weight decay
- **Scheduler**: Cosine or linear warmup scheduling
- **Early Stopping**: Patience-based stopping on validation metrics
- **Generation**: Beam search (default: 2 beams, max length: 512)
- **Layer Freezing**: Freeze specific encoder/decoder layers, embeddings, or LM head for selective fine-tuning

### Dataset

- **Training Set**: 4,225 natural language → SQL pairs
- **Development Set**: 466 pairs for validation
- **Test Set**: 431 natural language queries (no ground truth SQL)
- **Database**: Flight database with multiple tables and relationships

## 📦 Dependencies

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- CUDA (recommended for GPU training)

See `requirements.txt` for complete dependency list.

## 🔬 Experiment Tracking

The project supports Weights & Biases integration for experiment tracking:
```bash
python train_t5.py --finetune --use_wandb --experiment_name tracked_experiment
```

## 📝 License

MIT License

## 🤝 Contributing

This is a personal research project. Feel free to fork and adapt for your own use cases.
