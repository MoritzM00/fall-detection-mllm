"""
Logging utilities for setting up Rich-based logging with file output.

This module provides a clean interface for configuring logging across the project.
"""

import logging

import torch.distributed as dist
from rich.console import Console
from rich.logging import RichHandler

# Global console instance (shared across the application)
console = Console()


def get_rank_prefix():
    """Get the current process rank as a string prefix."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        return f"[Rank {rank}] "
    return ""


class RankAndModuleFilter(logging.Filter):
    """Filter that adds rank and module name to log records."""

    def filter(self, record):
        # Only add if not already added (check for marker attribute)
        if not hasattr(record, "_rank_module_added"):
            # Get the formatted message first
            original_message = record.getMessage()

            # Add rank prefix and module name
            rank_prefix = get_rank_prefix()

            # Format: [Rank X/Y] [module.name] message
            if rank_prefix:
                formatted = f"{rank_prefix} {original_message}"
            else:
                formatted = f"{original_message}"

            # Replace the message - this modifies the record in place
            record.msg = formatted
            record.args = ()  # Clear args to prevent re-formatting
            record._rank_module_added = True  # Mark as processed

            # IMPORTANT: Also update the message cache if it exists
            if hasattr(record, "message"):
                record.message = formatted

        return True


class RankFormatter(logging.Formatter):
    """Custom formatter for file output - rank and module are already in message from filter."""

    def format(self, record):
        # The message already has rank and module from RankAndModuleFilter
        return super().format(record)


def setup_logging(
    log_file: str = "local_logs.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    suppress_libraries: bool = True,
) -> tuple[Console, RichHandler, logging.FileHandler]:
    """
    Configure logging with Rich console output and file logging.

    This function should be called BEFORE importing heavy libraries (transformers, torch, etc.)
    to ensure logging configuration takes precedence.

    Args:
        log_file: Path to the log file
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        suppress_libraries: Whether to suppress noisy library loggers (default: True)

    Returns:
        Tuple of (console, rich_handler, file_handler) for potential reconfiguration later
    """
    # Configure file handler with rank information
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(RankFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    # Add filter to file handler
    file_handler.addFilter(RankAndModuleFilter())

    # Configure rich console handler with module name and rank
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=False,  # No timestamps in console
        show_path=False,
        markup=True,
        omit_repeated_times=False,
        show_level=True,  # Show log level
    )
    rich_handler.setLevel(console_level)
    # Add filter to console handler
    rich_handler.addFilter(RankAndModuleFilter())

    # Force configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Remove all existing handlers

    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy library loggers
    if suppress_libraries:
        suppress_noisy_loggers()

    return console, rich_handler, file_handler


def suppress_noisy_loggers():
    """
    Suppress INFO-level logging from noisy libraries.

    This sets commonly chatty libraries (transformers, torch, PIL, etc.) to WARNING level
    to reduce log clutter.
    """
    noisy_libraries = [
        "transformers",
        "transformers.utils",
        "transformers.utils.import_utils",
        "torch",
        "PIL",
        "matplotlib",
        "urllib3",
        "filelock",
        "huggingface_hub",
        "datasets",  # Suppress "PyTorch version X available" messages
        "datasets.config",
    ]

    for lib in noisy_libraries:
        logging.getLogger(lib).setLevel(logging.WARNING)


def reconfigure_logging_after_wandb(
    rich_handler: RichHandler,
    file_handler: logging.FileHandler,
):
    """
    Reconfigure logging after WandB initialization.

    WandB modifies the logging configuration, so we need to reclaim control.
    Call this immediately after wandb.init().

    Args:
        rich_handler: The Rich handler to restore
        file_handler: The file handler to restore
    """
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    # Re-suppress noisy libraries (WandB might have changed their levels)
    suppress_noisy_loggers()

    # Extra aggressive: Also set propagate=False for datasets to prevent any propagation
    logging.getLogger("datasets").propagate = False


def disable_logging_for_non_main_process(local_rank: int):
    """
    Disable logging on non-main processes in distributed training.

    Args:
        local_rank: The local rank of the process (0 = main process)
    """
    if local_rank != 0:
        import os
        import sys

        # Disable all logging
        logging.getLogger().handlers = []
        logging.getLogger().addHandler(logging.NullHandler())

        # Redirect stdout and stderr to /dev/null to suppress print statements
        # This prevents the Trainer from printing metrics on non-main processes
        devnull = open(os.devnull, "w")  # noqa: SIM115
        sys.stdout = devnull
        sys.stderr = devnull
