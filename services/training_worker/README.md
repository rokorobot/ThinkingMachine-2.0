# Training Worker - LoRA Fine-Tuning

This directory contains the training worker service for fine-tuning models using LoRA (Low-Rank Adaptation).

## Overview

The training worker is responsible for:
1. Fetching training run configurations from the database
2. Loading base models and applying LoRA adapters
3. Fine-tuning on distilled datasets
4. Saving adapter weights
5. Registering new model versions
6. Updating training run status

## Usage

### Via Admin API
```bash
curl -X POST http://localhost:8080/admin/distill-and-train \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "gpt2",
    "target_name": "tm-v2",
    "min_reward": 0.7,
    "limit": 1000
  }'
```

### Manual Execution
```bash
python train_tm_model.py --run-id <training_run_uuid>
```

## Configuration

Training configuration is stored in `training_runs.config` (JSONB):

```json
{
  "epochs": 1,
  "batch_size": 2,
  "grad_accum_steps": 8,
  "learning_rate": 1e-4,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "lora_target_modules": ["q_proj", "v_proj"],
  "max_length": 1024,
  "output_dir": "models/tm-v2"
}
```

## LoRA Parameters

- **r**: LoRA rank (default: 8) - Lower = fewer parameters
- **alpha**: LoRA alpha (default: 16) - Scaling factor
- **dropout**: Dropout rate (default: 0.05)
- **target_modules**: Which layers to adapt (default: ["q_proj", "v_proj"])

## Output

Trained models are saved to `models/<target_name>/` with:
- LoRA adapter weights
- Tokenizer configuration
- Training logs

## Integration

The training worker integrates with:
- `services/distillation/trainer_launcher.py` - Launches training jobs
- `services/distillation/dataset_builder.py` - Prepares training data
- Database tables: `training_runs`, `model_versions`
- Orchestrator for candidate evaluation
