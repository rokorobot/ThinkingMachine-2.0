from __future__ import annotations
import argparse, json
from libs import db
from services.distillation.model_registry import register_model_version, set_model_status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_id = args.run_id

    # 1) Load run config
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT base_model, target_name, dataset_path, config FROM training_runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
    if not row:
        print(f"Run ID {run_id} not found.")
        return

    base_model, target_name, dataset_path, config = row[0], row[1], row[2], row[3]

    print(f"Starting training for {target_name} based on {base_model}...")
    print(f"Dataset: {dataset_path}")
    print(f"Config: {config}")

    # 2) Set status=running
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE training_runs SET status = 'running' WHERE id = %s",
                (run_id,),
            )
        conn.commit()

    # 3) Run actual training (HF/DeepSpeed/etc.) - pseudo:
    # train_model(base_model, dataset_path, config, output_dir)
    output_dir = f"/models/{target_name}"  # or NFS / S3
    # ... training logic here ...
    import time
    time.sleep(2) # Simulate training

    # 4) Save model & register
    model_id = register_model_version(
        name=target_name,
        base_model=base_model,
        location=f"local:{output_dir}",
        status="candidate",
        performance_score=0.95, # Mock score
        metadata={"trained_from_run": run_id},
    )

    # 5) Update training_runs with metrics, link model_version_id, status
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_runs 
                SET status = 'completed', 
                    model_version_id = %s,
                    metrics = '{"train_loss": 0.1, "eval_loss": 0.2}'::jsonb
                WHERE id = %s
                """,
                (model_id, run_id),
            )
        conn.commit()
    
    print(f"Training completed. Model registered as {model_id}")

if __name__ == "__main__":
    main()
