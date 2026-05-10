"""
utils.py — Shared utilities for logging and execution retry handling.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging() -> logging.Logger:
    """Configures and returns the application logger."""
    _logger = logging.getLogger("VisionAuto")
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        _logger.addHandler(handler)
    return _logger


logger = setup_logging()


def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    Decorator to retry a function execution upon failure.

    Args:
        max_attempts: Maximum number of execution attempts.
        delay: Seconds to wait between attempts.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        f"[{func.__name__}] attempt {attempt}/{max_attempts} "
                        f"failed: {exc}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
