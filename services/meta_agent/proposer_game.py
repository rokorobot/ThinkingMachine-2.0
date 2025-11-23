from libs import db
from genome_store.skills.code.game_strategy import recommend_policy_patch
from statistics import mean

def propose_from_game_theory(metric_window_hours=24):
    # aggregate windowed metrics from traces
    traces = db.get_problematic_traces(hours=metric_window_hours, limit=500)
    
    if not traces:
        return
        
    # compute rollups you care about
    # Handle missing keys gracefully
    accuracies = [t["metadata"].get("reward_score", 0) for t in traces]
    safeties = [int(not t["metadata"].get("hallucination_flag", False)) for t in traces]
    latencies = [t["metadata"].get("latency_ms", 0) for t in traces]
    user_sats = [t.get("user_feedback", {}).get("thumbs_up", 0) for t in traces]

    rollup = {
        "accuracy":  mean(accuracies) if accuracies else 0,
        "safety":    mean(safeties) if safeties else 0,
        "latency":   mean(latencies) if latencies else 0,
        "user_sat":  mean(user_sats) if user_sats else 0,
    }
    
    rec = recommend_policy_patch(rollup)
    payload = {"domain":"medical","change": rec["patch"], "game_strategy": rec["chosen_strategy"], "mix": rec["mix"]}
    
    active_policy = db.get_active_policy_version()
    active_prompt = db.get_active_self_prompt()
    
    if not active_policy or not active_prompt:
        return

    db.insert_proposal(
        created_by="meta_agent",
        proposal_type="policy_update",
        payload=payload,
        current_policy_version=active_policy["id"],
        current_self_prompt_id=active_prompt["id"],
        reason=f"Game-theory equilibrium favored {rec['chosen_strategy']}"
    )
    print(f"meta_agent: created game-theoretic proposal ({rec['chosen_strategy']})")
