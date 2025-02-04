from tenacity import retry, stop_after_attempt, before_sleep_log
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), before_sleep=before_sleep_log(logger, logging.INFO))
def demo_function():
    print("Attempting...")
    raise RuntimeError("Failure!")


try:
    demo_function()
except RuntimeError as e:
    print(f"All retries failed: {e}")
