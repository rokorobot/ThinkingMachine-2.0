def tournament_config(domain:str, rounds=5, batch_per_round=64):
    return {
      "type":"repeated_game",
      "domain":domain,
      "rounds":rounds,
      "batch_per_round":batch_per_round,
      "environment_states":[ "StrictSafety","LenientSafety" ],
      "scoring": { "accuracy":0.4,"safety":0.3,"user_sat":0.2,"latency":0.1 }
    }
