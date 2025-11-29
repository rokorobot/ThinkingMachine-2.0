# train_tm_model.py
"""
Training Worker - LoRA Fine-Tuning for Thinking Machine 2.0

This script handles one training_run at a time:
1. Fetches training_run configuration from database
2. Loads base model and applies LoRA adapters
3. Fine-tunes on distilled dataset
4. Saves adapter weights
5. Registers new model_version in database
6. Updates training_run with results

Usage:
    python train_tm_model.py --run-id <uuid>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import psycopg2
import psycopg2.extras


# --------- DB helpers ---------

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tm_user:tm_pass@localhost:5432/thinking_machine")


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def update_training_run_status(run_id: str, status: str, metrics: Dict[str, Any] | None = None, logs_path: str | None = None):
    """Update training_run status and metrics."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_runs
                SET status = %s,
                    metrics = COALESCE(metrics, '{}'::jsonb) || %s::jsonb,
                    logs_path = COALESCE(%s, logs_path),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (status, json.dumps(metrics or {}), logs_path, run_id),
            )
        conn.commit()


def fetch_training_run(run_id: str) -> Dict[str, Any]:
    """Fetch training_run configuration from database."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, base_model, target_name, dataset_path, config
                FROM training_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
    if not row:
        raise RuntimeError(f"training_run {run_id} not found")
    return dict(row)


def register_model_version(
    name: str,
    base_model: str,
    location: str,
    status: str,
    performance_score: float | None,
    metadata: Dict[str, Any],
) -> str:
    """Register a new model version in the database."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_versions (name, base_model, location, status, performance_score, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (name, base_model, location, status, performance_score, json.dumps(metadata)),
            )
            model_id = cur.fetchone()[0]
        conn.commit()
    return model_id


def attach_model_version_to_run(run_id: str, model_version_id: str):
    """Link model_version to training_run."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_runs
                SET model_version_id = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (model_version_id, run_id),
            )
        conn.commit()


# --------- Dataset & Prompting ---------

def build_prompt(example: Dict[str, Any]) -> str:
    """
    Convert dataset record to training prompt.
    Assumes JSONL with fields: prompt, ideal_response, domain (optional).
    """
    user = example["prompt"]
    answer = example["ideal_response"]
    return f"<s>[USER]\n{user}\n\n[ASSISTANT]\n{answer}</s>"


def load_training_dataset(dataset_path: str):
    """Load dataset from JSONL using Hugging Face datasets."""
    ds = load_dataset("json", data_files=dataset_path)
    return ds["train"]


# --------- LoRA Training Logic ---------

def train_lora(run_id: str):
    """Main LoRA fine-tuning function."""
    # 1) Fetch training_run config
    tr = fetch_training_run(run_id)
    base_model_name = tr["base_model"]
    target_name = tr["target_name"]
    dataset_path = tr["dataset_path"]
    config = tr["config"] or {}

    output_dir = config.get("output_dir", f"models/{target_name}")
    os.makedirs(output_dir, exist_ok=True)

    # 2) Mark status = running
    update_training_run_status(run_id, "running", metrics={"phase": "starting"})

    # 3) Load tokenizer & base model
    print(f"[training] Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # 4) Apply LoRA
    lora_r = config.get("lora_r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.05)
    lora_target_modules = config.get("lora_target_modules", ["q_proj", "v_proj"])

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5) Load dataset
    print(f"[training] Loading dataset from {dataset_path}")
    raw_ds = load_training_dataset(dataset_path)

    def tokenize_fn(example):
        full_prompt = build_prompt(example)
        tokens = tokenizer(
            full_prompt,
            truncation=True,
            max_length=config.get("max_length", 1024),
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_ds = raw_ds.map(tokenize_fn, batched=False, remove_columns=raw_ds.column_names)

    # 6) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7) TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("grad_accum_steps", 8),
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 100),
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 2),
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        evaluation_strategy="no",
        report_to=[],
    )

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # 9) Train
    print("[training] Starting LoRA fine-tuning...")
    train_result = trainer.train()
    train_metrics = train_result.metrics or {}
    print("[training] Training complete:", train_metrics)

    # 10) Save adapter
    print(f"[training] Saving LoRA adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 11) Compute performance score (negative train loss)
    train_loss = float(train_metrics.get("train_loss", 0.0))
    performance_score = -train_loss

    # 12) Register model_version in DB
    location = f"local:{output_dir}"
    model_metadata = {
        "trained_from_run": run_id,
        "train_metrics": train_metrics,
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": lora_target_modules,
        },
    }
    model_id = register_model_version(
        name=target_name,
        base_model=base_model_name,
        location=location,
        status="candidate",
        performance_score=performance_score,
        metadata=model_metadata,
    )

    # 13) Attach model_version to training_run and mark completed
    attach_model_version_to_run(run_id, model_id)
    update_training_run_status(
        run_id,
        "completed",
        metrics={"train_loss": train_loss, "performance_score": performance_score},
        logs_path=str(Path(output_dir) / "training_log.json"),
    )

    print(f"[training] Finished run {run_id}, model_version={model_id}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Training Worker for Thinking Machine 2.0")
    parser.add_argument("--run-id", required=True, help="training_runs.id to execute")
    args = parser.parse_args()

    run_id = args.run_id
    try:
        train_lora(run_id)
    except Exception as e:
        print(f"[training] ERROR in run {run_id}: {e}")
        update_training_run_status(
            run_id,
            "failed",
            metrics={"error": str(e)},
        )
        raise


if __name__ == "__main__":
    main()
