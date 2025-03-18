import os
import logging
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str): The logging level (DEBUG, INFO, WARNING, ERROR).
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Set up logging
    log_file = os.path.join(
        "logs", f'rag_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    # Get the numeric logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"
