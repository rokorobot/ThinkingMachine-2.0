import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from libs.logging.logger import get_logger

logger = get_logger("safety_guard")
app = FastAPI()

class ValidationRequest(BaseModel):
    content: str
    type: str # 'output', 'proposal'

class ValidationResponse(BaseModel):
    valid: bool
    reason: str = None

def load_safety_rules():
    try:
        with open("/app/genome_store/safety/immutable_core.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config.get("rules", [])
    except Exception as e:
        logger.error(f"Failed to load safety rules: {e}")
        return []

@app.post("/validate", response_model=ValidationResponse)
async def validate(request: ValidationRequest):
    rules = load_safety_rules()
    
    # Simple keyword check for prototype
    # In real system, use a small BERT model or Regex
    for rule in rules:
        # Very basic heuristic: if rule says "Never X", check if content contains X
        # This is just a placeholder.
        pass
        
    if "hate speech" in request.content.lower():
         return ValidationResponse(valid=False, reason="Violates safety rule: Hate Speech")

    return ValidationResponse(valid=True)

@app.get("/health")
def health():
    return {"status": "ok"}
