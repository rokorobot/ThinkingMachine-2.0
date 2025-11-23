import time
import asyncio
from libs.logging.logger import get_logger
from services.meta_agent.meta_agent import run_reflection_cycle
from services.meta_agent.proposer_game import propose_from_game_theory

logger = get_logger("meta_agent")

async def run_worker_loop():
    logger.info("Starting Meta Agent Reflection Loop...")
    while True:
        try:
            # Run heuristic reflection
            run_reflection_cycle()
            
            # Run Game Theory reflection
            propose_from_game_theory()
            
        except Exception as e:
            logger.error(f"Error in reflection loop: {e}")
        
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run_worker_loop())
