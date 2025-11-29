
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.append(os.getcwd())

# Mock libs.db
sys.modules["libs.db"] = MagicMock()
# Mock connection and cursor
mock_conn = MagicMock()
mock_cur = MagicMock()
sys.modules["libs.db"].get_conn.return_value.__enter__.return_value = mock_conn
mock_conn.cursor.return_value.__enter__.return_value = mock_cur

# Mock fetchall for dataset builder
mock_cur.fetchall.return_value = [
    ("trace1", "prompt1", "response1", {"reward_score": 0.9}, "general", "policy1"),
    ("trace2", "prompt2", "response2", {"reward_score": 0.95}, "coding", "policy1"),
]
# Mock fetchone for model registry and trainer launcher
mock_cur.fetchone.side_effect = [
    ["model_id_123"], # register_model_version
    ["run_id_456"],   # create_training_run
    {"id": "model_id_123", "name": "tm-v2", "status": "active"}, # get_active_model
]

# Mock libs.llm.client
sys.modules["libs.llm.client"] = MagicMock()

# Import services after mocking
from services.distillation.dataset_builder import export_distillation_dataset
from services.distillation.trainer_launcher import create_training_run
from services.distillation.model_registry import get_active_model
from libs.llm.client import LLMClient, LLMConfig

def test_distillation_pipeline():
    print("Running Distillation Pipeline Verification...")
    
    # 1. Test Dataset Export
    print("\nTesting export_distillation_dataset...")
    output_path = "data/distill/test_dataset.jsonl"
    result_path = export_distillation_dataset(output_path)
    print(f"Exported to: {result_path}")
    assert result_path == str(os.path.abspath(output_path))
    
    # 2. Test Training Run Creation
    print("\nTesting create_training_run...")
    run_id = create_training_run(
        base_model="test-model",
        target_name="tm-v2-test",
        dataset_path=output_path,
        config={"epochs": 1}
    )
    print(f"Created run ID: {run_id}")
    assert run_id == "run_id_456"
    
    # 3. Test Model Registry
    print("\nTesting get_active_model...")
    active_model = get_active_model()
    print(f"Active model: {active_model}")
    assert active_model["name"] == "tm-v2"

    # 4. Test LLMClient Routing
    print("\nTesting LLMClient routing...")
    client = LLMClient(LLMConfig(backend="openai", model="default-model"))
    # Mock _chat_openai
    client._chat_openai = MagicMock(return_value="response")
    
    client.chat([], override_model="overridden-model")
    client._chat_openai.assert_called_with([], "overridden-model")
    print("LLMClient routing verified.")

    print("\nAll verification steps passed!")

if __name__ == "__main__":
    test_distillation_pipeline()
