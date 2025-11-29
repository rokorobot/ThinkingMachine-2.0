
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.append(os.getcwd())

# Mock libs.db
sys.modules["libs.db"] = MagicMock()
sys.modules["libs.db"].get_active_policy_version.return_value = {"id": "test_policy", "routing": {}, "tool_use": {}, "safety_overrides": {}}
sys.modules["libs.db"].get_active_self_prompt.return_value = {"id": "test_prompt", "merged": "Test system prompt", "editable": {}}

# Mock libs.user_memory
sys.modules["libs.user_memory"] = MagicMock()
sys.modules["libs.user_memory"].get_or_create_user.return_value = {"id": "test_user", "profile": {"preferences": {"detail_level": "concise"}}}
sys.modules["libs.user_memory"].search_user_memories.return_value = []
sys.modules["libs.user_memory"].get_top_recent_memories.return_value = []

# Mock libs.llm.client
sys.modules["libs.llm.client"] = MagicMock()
mock_llm_client = MagicMock()
mock_llm_client.chat.return_value = "This is a test response."
sys.modules["libs.llm.client"].LLMClient.from_env.return_value = mock_llm_client

# Import the pipeline after mocking
from thinking_core.pipeline import run_thinking_pipeline

def test_pipeline():
    print("Running Thinking Machine 2.0 Pipeline Verification...")
    
    task = {
        "input_text": "Write a plan to build a rocket.",
        "user_external_id": "user123",
        "session_id": "sess1",
        "task_id": "task1"
    }
    
    try:
        result = run_thinking_pipeline(task)
        
        print("\nPipeline execution successful!")
        print(f"Final Output: {result['final_output']}")
        print(f"Draft Output: {result['draft']}")
        print(f"World Model Context: {result['world_model_ctx']}")
        print(f"Retrieval Context: {result['retrieval_ctx']}")
        print(f"Reflection Notes: {result['reflection_notes']}")
        
        assert result["final_output"] == "This is a test response."
        assert result["user_id"] == "test_user"
        assert result["policy_id"] == "test_policy"
        
        print("\nAll assertions passed.")
        
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
