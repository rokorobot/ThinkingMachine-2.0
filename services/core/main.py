from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from services.core.database import get_db
from services.core.llm_client import get_llm_client
from services.common.models import SelfPrompt, PolicyVersion, Trace
import json

app = FastAPI()
llm_client = get_llm_client()

class TaskRequest(BaseModel):
    input: str

class TaskResponse(BaseModel):
    output: str
    trace_id: str

@app.post("/task", response_model=TaskResponse)
async def handle_task(request: TaskRequest, db: Session = Depends(get_db)):
    # 1. Load Active Context (Self-Prompt & Policy)
    active_prompt = db.query(SelfPrompt).filter(SelfPrompt.is_active == True).first()
    active_policy = db.query(PolicyVersion).filter(PolicyVersion.is_active == True).first()

    system_prompt = "You are a helpful AI."
    prompt_id = None
    if active_prompt:
        system_prompt = active_prompt.content
        prompt_id = active_prompt.id
    
    policy_id = None
    if active_policy:
        # Append policy rules to system prompt or handle logic
        system_prompt += f"\n\nFollow these policies: {json.dumps(active_policy.rules)}"
        policy_id = active_policy.id

    # 2. Execute Reasoning
    try:
        output = await llm_client.generate(request.input, system_prompt=system_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Log Trace
    trace = Trace(
        task_input=request.input,
        result_output=output,
        policy_version_id=policy_id,
        self_prompt_id=prompt_id,
        metadata_={"model": llm_client.model}
    )
    db.add(trace)
    db.commit()
    db.refresh(trace)

    return TaskResponse(output=output, trace_id=str(trace.id))

@app.get("/health")
def health():
    return {"status": "ok"}
