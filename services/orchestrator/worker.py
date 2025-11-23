import asyncio
from libs.logging.logger import get_logger
from services.orchestrator.orchestrator import run_orchestrator_loop

logger = get_logger("orchestrator")

async def run_worker_loop():
    logger.info("Starting Orchestrator Loop...")
    # run_orchestrator_loop is blocking (has its own while True), so we run it in executor or just call it if we don't need async
    # For simplicity, just call it.
    run_orchestrator_loop()

if __name__ == "__main__":
    asyncio.run(run_worker_loop())
