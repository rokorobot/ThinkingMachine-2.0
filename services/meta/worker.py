import time
import json
import os
from sqlalchemy.orm import Session
from services.core.database import SessionLocal
from services.common.models import Trace, Proposal
from services.core.llm_client import get_llm_client
from services.meta.prompts import GAME_THEORY_STRATEGIST_PROMPT

llm_client = get_llm_client()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def run_reflection_loop():
    print("Starting Reflection Loop...")
    while True:
        db = SessionLocal()
        try:
            # 1. Find failing traces (e.g., negative feedback or no feedback but high uncertainty - simplified here)
            # For demo, we look for traces with user_feedback containing 'thumbs_down'
            # In reality, we'd query for traces created in the last X minutes that haven't been analyzed.
            
            # This is a simplified query. In prod, we need a 'analyzed' flag on Trace.
            failing_traces = db.query(Trace).filter(
                Trace.user_feedback['thumbs_down'].astext == 'true'
            ).limit(5).all()

            if failing_traces:
                print(f"Found {len(failing_traces)} failing traces. Analyzing...")
                
                # Aggregate context
                context = "\n".join([f"Input: {t.task_input}\nOutput: {t.result_output}\nFeedback: {t.user_feedback}" for t in failing_traces])
                
                # 2. Game Theory Analysis
                prompt = f"{GAME_THEORY_STRATEGIST_PROMPT}\n\nFAILURES:\n{context}"
                
                try:
                    response = await llm_client.generate(prompt, model="gpt-4o") # Use smart model for meta-cognition
                    # Clean response to ensure JSON
                    response = response.replace("```json", "").replace("```", "")
                    data = json.loads(response)
                    
                    # 3. Create Proposal
                    proposal = Proposal(
                        type=data['proposal_type'],
                        payload=data['payload'],
                        reasoning=data['reasoning'],
                        status='pending'
                    )
                    db.add(proposal)
                    db.commit()
                    print(f"Created proposal: {proposal.id}")
                    
                    # Mark traces as analyzed (TODO: Add column or separate table)
                    
                except Exception as e:
                    print(f"Error in reflection: {e}")

            else:
                print("No failing traces found. Sleeping...")
        
        finally:
            db.close()
        
        time.sleep(60) # Run every minute

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_reflection_loop())
