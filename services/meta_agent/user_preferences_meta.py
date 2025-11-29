from __future__ import annotations

from typing import Dict, List
from collections import Counter

from libs import db


def get_active_users_with_recent_traces(hours: int = 72, min_traces: int = 10) -> List[str]:
    """
    Find users with enough recent interaction to infer preferences.
    Returns list of user_ids.
    """
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                FROM traces
                WHERE user_id IS NOT NULL
                  AND created_at > NOW() - (%s || ' hours')::interval
                GROUP BY user_id
                HAVING COUNT(*) >= %s
                """,
                (hours, min_traces),
            )
            rows = cur.fetchall()
    return [r[0] for r in rows]


def fetch_user_traces(user_id: str, hours: int = 72) -> List[Dict]:
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT input_text, output_text, metadata, user_feedback
                FROM traces
                WHERE user_id = %s
                  AND created_at > NOW() - (%s || ' hours')::interval
                ORDER BY created_at DESC
                """,
                (user_id, hours),
            )
            rows = cur.fetchall()

    traces = []
    for input_text, output_text, metadata, user_feedback in rows:
        traces.append(
            {
                "input_text": input_text,
                "output_text": output_text,
                "metadata": metadata or {},
                "user_feedback": user_feedback or {},
            }
        )
    return traces


def infer_preferences_from_traces(traces: List[Dict]) -> Dict[str, str]:
    """
    Super simple heuristics; you can replace with an LLM-based classifier later.
    We'll infer:
      - tone: 'direct' vs 'gentle'
      - detail_level: 'concise' vs 'detailed'
      - safety_bias: 'strict' vs 'relaxed'
    Based on:
      - user_feedback tags: 'like_direct', 'like_concise', etc.
      - metadata: reward_score, hallucination_flag, etc.
    """
    tone_votes = Counter()
    detail_votes = Counter()
    safety_votes = Counter()

    for t in traces:
        fb = t["user_feedback"]
        # Example: your UI or integration might send structured feedback tags
        tag = fb.get("tag")  # e.g., 'too_blunt', 'too_soft', 'too_long', 'too_short'
        thumbs_up = fb.get("thumbs_up", False)
        thumbs_down = fb.get("thumbs_down", False)

        # Tone
        if tag == "too_blunt" and thumbs_down:
            tone_votes["gentle"] += 2
        if tag == "too_soft" and thumbs_down:
            tone_votes["direct"] += 2
        if thumbs_up and tag == "direct_helpful":
            tone_votes["direct"] += 3
        if thumbs_up and tag == "kind_helpful":
            tone_votes["gentle"] += 3

        # Detail
        if tag == "too_long" and thumbs_down:
            detail_votes["concise"] += 3
        if tag == "too_short" and thumbs_down:
            detail_votes["detailed"] += 3
        if thumbs_up and tag == "just_right_detail":
            detail_votes["balanced"] += 2

        # Safety bias (here we just check hallucination feedback / corrections)
        if fb.get("flag_unsafe_output"):
            safety_votes["strict"] += 3
        if fb.get("complained_too_cautious"):
            safety_votes["relaxed"] += 2

    prefs: Dict[str, str] = {}

    if tone_votes:
        tone = tone_votes.most_common(1)[0][0]
        prefs["tone"] = tone

    if detail_votes:
        detail = detail_votes.most_common(1)[0][0]
        # normalize 'balanced' otherwise fallback
        if detail == "balanced":
            detail = "medium"
        prefs["detail_level"] = detail

    if safety_votes:
        safety = safety_votes.most_common(1)[0][0]
        prefs["safety_bias"] = safety

    return prefs


def build_user_policy_overlay_from_prefs(prefs: Dict[str, str]) -> Dict:
    """
    Map user preferences to routing overrides.
    This is just an example; adjust to your policy schema.
    """
    routing_override: Dict = {}

    tone = prefs.get("tone")
    if tone == "direct":
        routing_override.setdefault("style", {})["directness"] = "high"
    elif tone == "gentle":
        routing_override.setdefault("style", {})["directness"] = "low"

    detail = prefs.get("detail_level")
    if detail == "concise":
        routing_override.setdefault("style", {})["max_tokens_per_reply"] = 256
    elif detail == "detailed":
        routing_override.setdefault("style", {})["max_tokens_per_reply"] = 1024

    safety_bias = prefs.get("safety_bias")
    if safety_bias == "strict":
        routing_override.setdefault("safety", {})["extra_checks"] = True
        routing_override["safety"]["min_sources"] = 3
    elif safety_bias == "relaxed":
        routing_override.setdefault("safety", {})["extra_checks"] = False

    return routing_override


def run_user_preference_meta_cycle(hours: int = 72, min_traces: int = 10) -> None:
    """
    Meta-agent routine:
      - find users with sufficient data
      - infer preferences
      - update users.profile['preferences']
      - update or create user_policies overlay
    """
    user_ids = get_active_users_with_recent_traces(hours=hours, min_traces=min_traces)
    if not user_ids:
        print("user_pref_meta: no users with enough traces")
        return

    active_policy = db.get_active_policy_version()
    if not active_policy:
        print("user_pref_meta: no active global policy")
        return

    for user_id in user_ids:
        traces = fetch_user_traces(user_id, hours=hours)
        prefs = infer_preferences_from_traces(traces)
        if not prefs:
            continue

        # 1) update profile
        db.update_user_profile_preferences(user_id, prefs)
        print(f"user_pref_meta: updated preferences for user {user_id}: {prefs}")

        # 2) build and upsert overlay
        routing_override = build_user_policy_overlay_from_prefs(prefs)
        if routing_override:
            db.upsert_user_policy_overlay(
                user_id=user_id,
                base_policy_id=active_policy["id"],
                routing_override=routing_override,
                tool_use_override={},  # optionally add per-user tool settings
            )
            print(f"user_pref_meta: updated policy overlay for user {user_id}")
