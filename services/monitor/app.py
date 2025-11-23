import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import json
import yaml
from sqlalchemy import create_engine, text

# Config
st.set_page_config(page_title="Thinking Machine: Mission Control", layout="wide", page_icon="🧠")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@postgres:5432/thinking_machine")
API_URL = os.environ.get("API_URL", "http://api_gateway:8080")

# DB Connection
@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

engine = get_engine()

def get_data(query, params=None):
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def load_safety_core():
    try:
        with open("/workspace/genome_store/safety/immutable_core.yaml", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "# Safety Core file not found."

# --- Sidebar Controls ---
st.sidebar.title("🧠 Mission Control")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("### 🎮 Operator Actions")
if st.sidebar.button("⚡ Trigger Game Theory Opt"):
    try:
        res = requests.post(f"{API_URL}/admin/game-theory/optimize", json={"domain": "medical", "hours": 24, "commit": True})
        if res.status_code == 200:
            st.sidebar.success(f"Optimization Triggered! Proposal: {res.json().get('proposal_id')}")
        else:
            st.sidebar.error(f"Failed: {res.text}")
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")

# --- Main Dashboard Structure ---
# Mapping to User's 6 Categories
tabs = st.tabs([
    "🚀 Ops & KPIs",           # 6. Operational
    "🧠 Cognitive Engine",     # 1. Core Cognitive (Memory, Knowledge)
    "🧬 Self-Reprogramming",   # 2. Self-Reprogramming (Genome, Evolution, Strategy)
    "🛡️ Safety & Governance",  # 4. Safety (Immutable Core, Audit)
    "💬 Interaction & Traces", # 3. Interaction & 5. Meta-Cognition (Traces, Analysis)
])

# --- Tab 1: Operations & KPIs ---
with tabs[0]:
    st.header("Operational Status")
    
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Health Check
    try:
        health = requests.get(f"{API_URL}/health", timeout=1)
        status = "Online 🟢" if health.status_code == 200 else "Error 🔴"
    except:
        status = "Offline ⚫"
    col1.metric("System Health", status)
    
    # Success Rate (Proxy: Avg Reward > 0.5)
    success_df = get_data("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN (metadata->>'reward_score')::float > 0.5 THEN 1 ELSE 0 END) as success
        FROM traces WHERE created_at > NOW() - INTERVAL '24 hours'
    """)
    if not success_df.empty and success_df.iloc[0]['total'] > 0:
        rate = (success_df.iloc[0]['success'] / success_df.iloc[0]['total']) * 100
        col2.metric("Success Rate (24h)", f"{rate:.1f}%")
    else:
        col2.metric("Success Rate (24h)", "N/A")

    # Latency
    latency_df = get_data("SELECT AVG((metadata->>'latency_ms')::float) as lat FROM traces WHERE created_at > NOW() - INTERVAL '24 hours'")
    lat = latency_df.iloc[0]['lat']
    col3.metric("Avg Latency", f"{lat:.0f} ms" if pd.notnull(lat) else "N/A")
    
    # Active Users
    users_df = get_data("SELECT COUNT(DISTINCT user_id) as c FROM traces WHERE created_at > NOW() - INTERVAL '24 hours'")
    col4.metric("Active Users (24h)", users_df.iloc[0]['c'])

    st.markdown("### 📉 Telemetry")
    activity_df = get_data("""
        SELECT date_trunc('hour', created_at) as hour, count(*) as count 
        FROM traces 
        WHERE created_at > NOW() - INTERVAL '24 hours' 
        GROUP BY 1 ORDER BY 1
    """)
    if not activity_df.empty:
        st.line_chart(activity_df.set_index('hour'))

# --- Tab 2: Cognitive Engine ---
with tabs[1]:
    st.header("Core Cognitive Engine")
    
    col_mem, col_know = st.columns(2)
    
    with col_mem:
        st.subheader("💾 Long-Term Memory")
        users_count = get_data("SELECT COUNT(*) as c FROM users").iloc[0]['c']
        mem_count = get_data("SELECT COUNT(*) as c FROM user_memories").iloc[0]['c']
        st.metric("Total Users", users_count)
        st.metric("Total Memories", mem_count)
        
        st.markdown("**Recent User Memories**")
        recent_mems = get_data("""
            SELECT u.external_id, m.kind, m.text, m.importance 
            FROM user_memories m 
            JOIN users u ON m.user_id = u.id 
            ORDER BY m.created_at DESC LIMIT 5
        """)
        st.dataframe(recent_mems, use_container_width=True)

    with col_know:
        st.subheader("🌍 World Model & Knowledge")
        st.info("Vector Database Status: Connected (Qdrant)")
        # Placeholder for actual vector DB stats
        st.markdown("""
        - **Static Knowledge**: Loaded from `genome_store/knowledge`
        - **Dynamic Knowledge**: Web Search (Enabled), APIs (Enabled)
        - **Uncertainty Handling**: Active
        """)
        
    st.subheader("👤 User Modeling")
    user_input = st.text_input("Inspect User Profile (External ID)", "")
    if user_input:
        u_profile = get_data("SELECT * FROM users WHERE external_id = %s", params={'id': user_input})
        if not u_profile.empty:
            st.json(u_profile.iloc[0].to_dict())
        else:
            st.warning("User not found.")

# --- Tab 3: Self-Reprogramming ---
with tabs[2]:
    st.header("Self-Reprogramming Loop")
    
    st.subheader("🧬 Active Genome")
    col_pol, col_prm = st.columns(2)
    with col_pol:
        st.markdown("**Active Policy**")
        active_policy = get_data("SELECT label, created_at, routing, tool_use FROM policy_versions WHERE is_active = TRUE")
        if not active_policy.empty:
            st.json(active_policy.iloc[0].to_dict())
        else:
            st.warning("No active policy.")
            
    with col_prm:
        st.markdown("**Active Self-Prompt**")
        active_prompt = get_data("SELECT created_at, merged FROM self_prompts WHERE is_active = TRUE")
        if not active_prompt.empty:
            st.text_area("Self-Prompt Content", active_prompt.iloc[0]['merged'], height=200)
        else:
            st.warning("No active self-prompt.")

    st.divider()
    
    st.subheader("🧠 Game Theory Strategy")
    # Fetch preview
    try:
        res = requests.get(f"{API_URL}/admin/game-theory/preview?domain=medical&hours=24")
        if res.status_code == 200:
            data = res.json()
            mixes = data.get("mixes", [])
            mix_data = []
            for m in mixes:
                for i, strat in enumerate(m['strategies']):
                    mix_data.append({"Player": m['player'], "Strategy": strat, "Probability": m['mix'][i]})
            
            df_mix = pd.DataFrame(mix_data)
            fig = px.bar(df_mix, x="Player", y="Probability", color="Strategy", title="Current Strategic Equilibrium", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Recommended Strategy: **{data.get('chosen_strategy')}**")
        else:
            st.warning("Game Theory API unavailable.")
    except:
        st.error("Connection to Game Theory API failed.")

    st.divider()
    
    col_prop, col_exp = st.columns(2)
    with col_prop:
        st.subheader("Proposals (Evolution)")
        props = get_data("SELECT id, type, status, reason, created_at FROM proposals ORDER BY created_at DESC LIMIT 10")
        st.dataframe(props, use_container_width=True)
        
    with col_exp:
        st.subheader("Experiments (Validation)")
        exps = get_data("SELECT id, status, result_summary, created_at FROM experiments ORDER BY created_at DESC LIMIT 10")
        st.dataframe(exps, use_container_width=True)

# --- Tab 4: Safety & Governance ---
with tabs[3]:
    st.header("Safety, Alignment & Governance")
    
    col_safe, col_audit = st.columns([1, 2])
    
    with col_safe:
        st.subheader("🛡️ Immutable Safety Core")
        safety_yaml = load_safety_core()
        st.code(safety_yaml, language="yaml")
        st.caption("Read-only view of `genome_store/safety/immutable_core.yaml`")
        
    with col_audit:
        st.subheader("📜 Change Audit Log")
        audit_df = get_data("""
            SELECT created_at, type, author, reason, status 
            FROM proposals 
            WHERE status IN ('accepted', 'rejected') 
            ORDER BY created_at DESC
        """)
        st.dataframe(audit_df, use_container_width=True)
        
    st.subheader("Human-in-the-Loop Review")
    pending_reviews = get_data("SELECT id, type, reason, payload FROM proposals WHERE status = 'pending'")
    if not pending_reviews.empty:
        for _, row in pending_reviews.iterrows():
            with st.expander(f"Review Proposal: {row['id']} ({row['type']})"):
                st.write(f"**Reason:** {row['reason']}")
                st.json(row['payload'])
                c1, c2 = st.columns(2)
                with c1:
                    st.button("Approve ✅", key=f"app_{row['id']}")
                with c2:
                    st.button("Reject ❌", key=f"rej_{row['id']}")
    else:
        st.info("No proposals pending human review.")

# --- Tab 5: Interaction & Traces ---
with tabs[4]:
    st.header("Interaction & Meta-Cognition")
    
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        domain_filter = st.text_input("Filter Domain", "")
    with c2:
        show_errors = st.checkbox("Show Low Confidence / Errors Only")
        
    query = """
        SELECT id, created_at, domain, input_text, output_text, metadata, user_id 
        FROM traces WHERE 1=1
    """
    params = {}
    if domain_filter:
        query += " AND domain = %(domain)s"
        params['domain'] = domain_filter
    if show_errors:
        query += " AND ((metadata->>'reward_score')::float < 0.6 OR (metadata->>'hallucination_flag')::bool = TRUE)"
        
    query += " ORDER BY created_at DESC LIMIT 50"
    
    traces = get_data(query, params)
    
    for _, row in traces.iterrows():
        meta = row['metadata']
        score = float(meta.get('reward_score', 0))
        color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
        
        with st.expander(f"[{row['created_at']}] {row['domain']} (Score: {score:.2f})"):
            st.markdown(f"**User:** {row['input_text']}")
            st.markdown(f"**Agent:** {row['output_text']}")
            
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reward Score", f"{score:.2f}")
            m2.metric("Latency", f"{meta.get('latency_ms')} ms")
            m3.metric("Hallucination", str(meta.get('hallucination_flag')))
            
            st.json(meta)
            if row['user_id']:
                st.caption(f"User ID: {row['user_id']}")
