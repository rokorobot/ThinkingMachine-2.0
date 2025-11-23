import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
from services.core.database import SessionLocal
from services.common.models import Trace, Proposal, Experiment, PolicyVersion

st.set_page_config(page_title="Thinking Machine Dashboard", layout="wide")

def get_db():
    return SessionLocal()

st.title("Thinking Machine: Governance Dashboard")

tab1, tab2, tab3 = st.tabs(["Traces", "Proposals", "Experiments"])

with tab1:
    st.header("Recent Traces")
    db = get_db()
    traces = db.query(Trace).order_by(Trace.created_at.desc()).limit(50).all()
    db.close()
    
    data = []
    for t in traces:
        data.append({
            "ID": str(t.id),
            "Input": t.task_input,
            "Output": t.result_output,
            "Feedback": t.user_feedback,
            "Created": t.created_at
        })
    st.dataframe(pd.DataFrame(data))

with tab2:
    st.header("Proposals")
    db = get_db()
    proposals = db.query(Proposal).order_by(Proposal.created_at.desc()).all()
    
    for p in proposals:
        with st.expander(f"{p.type}: {p.status} ({p.created_at})"):
            st.write(f"**Reasoning:** {p.reasoning}")
            st.json(p.payload)
            
            if p.status == 'pending':
                col1, col2 = st.columns(2)
                if col1.button("Approve", key=f"approve_{p.id}"):
                    p.status = 'approved' # Orchestrator should pick this up (or we set to pending if orchestrator waits for approval)
                    # Actually, orchestrator picks up 'pending'. 
                    # Let's say Orchestrator picks up 'approved' to start experiment? 
                    # Or Orchestrator picks up 'pending', creates experiment, then waits for 'approved' to deploy?
                    # For simplicity: Orchestrator picks up 'pending'. 
                    # So 'Approve' here might mean "Force Deploy" or "Retry".
                    # Let's assume proposals start as 'draft' and need approval to become 'pending'?
                    # Or proposals are 'pending' and Orchestrator runs them automatically?
                    # Let's say we want to Approve deployment.
                    pass
                if col2.button("Reject", key=f"reject_{p.id}"):
                    p.status = 'rejected'
                    db.commit()
                    st.experimental_rerun()
    db.close()

with tab3:
    st.header("Experiments")
    db = get_db()
    experiments = db.query(Experiment).order_by(Experiment.id.desc()).all()
    db.close()
    
    data = []
    for e in experiments:
        data.append({
            "ID": str(e.id),
            "Status": e.status,
            "Baseline": str(e.baseline_policy_id),
            "Candidate": str(e.candidate_policy_id),
            "Results": e.result_summary
        })
    st.dataframe(pd.DataFrame(data))
