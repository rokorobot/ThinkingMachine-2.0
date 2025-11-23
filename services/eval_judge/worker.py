import asyncio
from libs.logging.logger import get_logger
from services.eval_judge.eval_judge import run_eval_loop

logger = get_logger("eval_judge")

async def run_worker_loop():
    logger.info("Starting Eval Judge Loop...")
    run_eval_loop()

if __name__ == "__main__":
    asyncio.run(run_worker_loop())
