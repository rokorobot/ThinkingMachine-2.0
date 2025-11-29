from __future__ import annotations

from typing import Any, Dict
import subprocess
import json
from pathlib import Path

from libs import db
from .model_registry import register_model_version


def create_training_run(
    base_model: str,
    target_name: str,
    dataset_path: str,
    config: Dict[str, Any],
) -> str:
    """
    Insert a training_runs row with status 'pending'.
    """
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (
                    base_model, target_name, status, config, dataset_path
                ) VALUES (%s, %s, 'pending', %s::jsonb, %s)
                RETURNING id
                """,
                (base_model, target_name, json.dumps(config), dataset_path),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
    return str(run_id)


def launch_local_training(run_id: str, script_path: str = "train_tm_model.py") -> None:
    """
    Simple version: run local training script that:
      - reads training_runs row
      - trains a new model
      - updates training_runs + model_versions
    Here we just spawn a subprocess; you can replace with SLURM/k8s submission.
    """
    subprocess.Popen(["python", script_path, "--run-id", run_id])
