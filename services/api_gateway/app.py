from __future__ import annotations

from fastapi import FastAPI

from services.api_gateway.routers import admin
from services.core_agent.core_agent import handle_task  # existing

from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="Thinking Machine API Gateway")


class TaskRequest(BaseModel):
    input_text: str
    domain: Optional[str] = "general"
    task_type: Optional[str] = "chat"
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    user_external_id: Optional[str] = None
    remember: Optional[bool] = True
    memory_note: Optional[str] = None


class TaskResponse(BaseModel):
    output_text: str


@app.post("/task", response_model=TaskResponse)
def submit_task(req: TaskRequest):
    task_dict = {
        "input_text": req.input_text,
        "domain": req.domain,
        "task_type": req.task_type,
        "session_id": req.session_id,
        "task_id": req.task_id,
        "user_external_id": req.user_external_id,
        "remember": req.remember,
        "memory_note": req.memory_note,
    }
    output_text = handle_task(task_dict)
    return TaskResponse(output_text=output_text)


@app.get("/health")
def health():
    return {"status": "ok"}

# Mount admin/game-theory endpoints
app.include_router(admin.router)
