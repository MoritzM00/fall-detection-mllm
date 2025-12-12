"""
Rich-based formatting utilities for displaying metrics and results.

This module provides functions for formatting and displaying evaluation metrics
using Rich tables and other Rich components.
"""

import math
from typing import Any

from rich.table import Table
from utils.logging_utils import console


def format_metric_value(value: Any) -> str:
    """
    Format a metric value for display.

    Args:
        value: The value to format (can be int, float, bool, str, None, etc.)

    Returns:
        Formatted string representation of the value
    """
    if value is None:
        return "None"
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        elif math.isinf(value):
            return "∞" if value > 0 else "-∞"
        else:
            # Limit to 5 decimal places, strip trailing zeros
            return f"{value:.5f}".rstrip("0").rstrip(".")
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def print_metrics_table(metrics: dict[str, Any], title: str = "Evaluation Metrics"):
    """
    Print metrics in a nicely formatted Rich table.

    Args:
        metrics: Dictionary of metric names to values
        title: Title for the table
    """
    # Create table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=False)
    table.add_column("Value", justify="right", style="green")

    # Sort metrics for consistent display
    sorted_metrics = sorted(metrics.items())

    # Add rows
    for key, value in sorted_metrics:
        formatted_value = format_metric_value(value)
        table.add_row(key, formatted_value)

    # Print table
    console.print(table)
