from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Player:
    name: str
    strategies: List[str]


@dataclass
class GameModel:
    """
    General n-player normal-form game.

    payoff_tensors[i] has shape:
      (n_strats_player0, n_strats_player1, ..., n_strats_player{n-1})
    and gives payoffs for player i.
    """
    players: List[Player]
    payoff_tensors: List[np.ndarray]


def expected_payoff_for_player(player_idx: int,
                               payoff_tensor: np.ndarray,
                               mixes: List[np.ndarray]) -> np.ndarray:
    """
    Compute expected payoff for each pure strategy of player_idx given
    mixed strategies of all players (including player_idx).

    Returns a 1D array of shape (n_strats_player_idx,)
    giving expected payoff of each pure strategy.
    """
    # Start with full payoff tensor
    tensor = payoff_tensor.copy()
    n_players = len(mixes)

    # Contract over other players' strategy distributions
    for i in range(n_players):
        if i == player_idx:
            continue
        # tensordot along axis i with mixes[i]
        tensor = np.tensordot(tensor, mixes[i], axes=([i], [0]))

    # After contracting all other players, remaining axis 0 corresponds to player_idx
    # but axis ordering may have changed depending on contraction order; to be safe,
    # we flatten and then reshape to n_strats for player_idx.
    n_strats = tensor.shape[0] if isinstance(tensor, np.ndarray) else 1
    return np.array(tensor).reshape(n_strats)


def nash_via_fictitious_play_n(game: GameModel, iters: int = 200, lr: float = 0.05) -> List[np.ndarray]:
    """
    Very rough n-player fictitious play:
      - initialize all players with uniform mixes
      - repeatedly compute best response for each player vs others' mixes
      - update their mix toward best response with learning rate lr
    """
    n_players = len(game.players)
    mixes: List[np.ndarray] = []
    for p in game.players:
        m = len(p.strategies)
        mixes.append(np.ones(m) / m)

    for _ in range(iters):
        for i in range(n_players):
            payoff_tensor = game.payoff_tensors[i]
            exp_payoffs = expected_payoff_for_player(i, payoff_tensor, mixes)
            br = np.zeros_like(exp_payoffs)
            br[np.argmax(exp_payoffs)] = 1.0
            mixes[i] = (1.0 - lr) * mixes[i] + lr * br
            mixes[i] = mixes[i] / mixes[i].sum()

    return mixes


def build_adaptability_game_n_players(metrics: Dict[str, float],
                                      weights: Dict[str, float] | None = None) -> GameModel:
    """
    Example: 3 players for adaptability:
      - AgentPolicy: how aggressive/conservative is reasoning
      - SafetyRegulator: how strict safety filters/thresholds are
      - UserTrust: how sensitive user satisfaction is to errors vs latency

    metrics keys (example): accuracy, safety, latency, user_sat
    weights keys: same, representing relative importance in payoff.
    """
    if weights is None:
        weights = {"accuracy": 0.4, "safety": 0.3, "latency": 0.1, "user_sat": 0.2}

    # Normalize weights
    total_w = sum(weights.values()) or 1.0
    weights = {k: v / total_w for k, v in weights.items()}

    # Players
    agent = Player("AgentPolicy", ["Conservative", "Balanced", "Aggressive"])
    regulator = Player("SafetyRegulator", ["Strict", "Moderate", "Lenient"])
    user = Player("UserTrust", ["RiskAverse", "Neutral", "RiskSeeking"])
    players = [agent, regulator, user]

    # Strategy counts
    nA, nR, nU = len(agent.strategies), len(regulator.strategies), len(user.strategies)

    # Toy payoff surfaces derived from metrics (you’ll refine this)
    # Idea: reward high accuracy & safety, penalize latency, model user_sat differently.
    base_acc = metrics.get("accuracy", 0.7)
    base_safe = metrics.get("safety", 0.9)
    base_lat = metrics.get("latency", 200.0)  # ms
    base_sat = metrics.get("user_sat", 0.6)

    # Agent payoff: wants high accuracy & user_sat, balanced safety, moderate latency
    payoff_agent = np.zeros((nA, nR, nU))
    for i, a_strat in enumerate(agent.strategies):
        for j, r_strat in enumerate(regulator.strategies):
            for k, u_strat in enumerate(user.strategies):
                # Example shaping: more aggressive may increase accuracy but drop safety
                acc = base_acc + (0.03 if a_strat == "Aggressive" else -0.01 if a_strat == "Conservative" else 0.0)
                safety = base_safe + (0.04 if r_strat == "Strict" else -0.03 if r_strat == "Lenient" else 0.0)
                latency = base_lat + (30 if r_strat == "Strict" else -20 if a_strat == "Aggressive" else 0.0)
                user_sat = base_sat

                # UserTrust preference
                if u_strat == "RiskAverse":
                    user_sat += 0.03 * safety - 0.02 * (1 - acc)
                elif u_strat == "RiskSeeking":
                    user_sat += 0.03 * acc - 0.01 * safety

                # Scale into payoff: higher is better
                pay = (
                    weights["accuracy"] * acc +
                    weights["safety"] * safety +
                    weights["user_sat"] * user_sat -
                    weights["latency"] * (latency / 1000.0)  # penalty
                )
                payoff_agent[i, j, k] = pay

    # Regulator payoff: loves safety, dislikes low safety, mild penalty for latency
    payoff_reg = np.zeros_like(payoff_agent)
    for i in range(nA):
        for j, r_strat in enumerate(regulator.strategies):
            for k, u_strat in enumerate(user.strategies):
                safety = base_safe + (0.04 if r_strat == "Strict" else -0.03 if r_strat == "Lenient" else 0.0)
                latency = base_lat + (30 if r_strat == "Strict" else -10 if r_strat == "Lenient" else 0.0)
                payoff_reg[i, j, k] = 0.7 * safety - 0.3 * (latency / 1000.0)

    # User payoff: cares about user_sat and how well their risk profile is matched
    payoff_user = np.zeros_like(payoff_agent)
    for i, a_strat in enumerate(agent.strategies):
        for j, r_strat in enumerate(regulator.strategies):
            for k, u_strat in enumerate(user.strategies):
                acc = base_acc + (0.03 if a_strat == "Aggressive" else -0.01 if a_strat == "Conservative" else 0.0)
                safety = base_safe + (0.04 if r_strat == "Strict" else -0.03 if r_strat == "Lenient" else 0.0)
                user_sat = base_sat
                # Match risk preferences
                risk_mismatch_penalty = 0.0
                if u_strat == "RiskAverse" and a_strat == "Aggressive":
                    risk_mismatch_penalty = 0.05
                if u_strat == "RiskSeeking" and a_strat == "Conservative":
                    risk_mismatch_penalty = 0.03

                user_sat += 0.02 * acc + 0.02 * safety - risk_mismatch_penalty
                payoff_user[i, j, k] = user_sat

    return GameModel(players=players,
                     payoff_tensors=[payoff_agent, payoff_reg, payoff_user])


def recommend_policy_patch_from_metrics(metrics: Dict[str, float],
                                        weights: Dict[str, float] | None = None) -> Dict[str, object]:
    """
    High-level: build n-player game, compute equilibrium mixes, map AgentPolicy
    equilibrium to a concrete policy patch.
    """
    gm = build_adaptability_game_n_players(metrics, weights)
    mixes = nash_via_fictitious_play_n(gm, iters=300, lr=0.05)
    agent_mix = mixes[0]  # AgentPolicy
    choice_idx = int(np.argmax(agent_mix))
    choice = gm.players[0].strategies[choice_idx]

    # Map agent strategy to concrete patch (you’ll extend this)
    if choice == "Conservative":
        patch = {
            "require_multi_source_check": True,
            "min_sources": 3,
            "ask_clarifying_if_ambiguous": True,
            "allow_single_source_if_high_conf": False,
        }
    elif choice == "Balanced":
        patch = {
            "require_multi_source_check": True,
            "min_sources": 2,
            "ask_clarifying_if_ambiguous": True,
        }
    else:  # Aggressive
        patch = {
            "require_multi_source_check": True,
            "min_sources": 1,
            "allow_single_source_if_high_conf": True,
        }

    return {
        "chosen_strategy": choice,
        "patch": patch,
        "mixes": [
            {
                "player": p.name,
                "strategies": p.strategies,
                "mix": mixes[i].tolist(),
            }
            for i, p in enumerate(gm.players)
        ],
    }
