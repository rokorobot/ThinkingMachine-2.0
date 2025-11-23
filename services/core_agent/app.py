import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.core_agent.core_agent import handle_task
from libs.logging.logger import get_logger

logger = get_logger("core_agent")
app = FastAPI()

class TaskRequest(BaseModel):
    input: str
    session_id: str = "default"
    domain: str = "general"

class TaskResponse(BaseModel):
    output: str

@app.post("/agent/act", response_model=TaskResponse)
async def act(request: TaskRequest):
    logger.info(f"Received task: {request.input}")
    
    task = {
        "input_text": request.input,
        "session_id": request.session_id,
        "domain": request.domain
    }
    
    try:
        output = handle_task(task)
        return TaskResponse(output=output)
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
