"""Evaluation utilities for video model evaluation (vLLM compatible).

This module provides evaluation functions that work with vLLM inference outputs,
without requiring HuggingFace Trainer or Accelerator dependencies.
"""

import json
import logging
import os
import time
from typing import Any

import numpy as np
from torch.utils.data import Dataset

import wandb
from falldet.evaluation.subgroup import perform_subgroup_evaluation
from falldet.evaluation.visual import visualize_evaluation_results
from falldet.metrics.base import compute_metrics
from falldet.utils.latex import format_subgroup_latex_table
from falldet.utils.wandb import log_confusion_matrix, log_videos_with_predictions

logger = logging.getLogger(__name__)

# TODO: implement combined evaluation function that handles multiple datasets


# adapted from fall-da/evaluation_utils::evaluate_individual_dataset
def evaluate_predictions(
    dataset: Dataset,
    predictions: list[str] | list[int] | np.ndarray,
    references: list[str] | list[int] | np.ndarray,
    dataset_name: str,
    output_dir: str = "outputs",
    save_results: bool = True,
    run: wandb.Run | None = None,
    log_videos: int = 1,
) -> dict[str, Any]:
    """
    Evaluate predictions against ground truth labels.

    Args:
        predictions: Predicted labels (strings or indices)
        references: Ground truth labels (strings or indices)
        dataset_name: Name of the dataset for logging
        output_dir: Directory to save results (default: "outputs")
        save_results: Whether to save results to JSON file

    Returns:
        Dictionary of computed metrics
    """
    logger.info(f"Evaluating {len(predictions)} predictions on {dataset_name}")
    is_wanfall = dataset_name.lower().startswith("wanfall") or dataset_name.lower().startswith(
        "wan_fall"
    )

    # Compute comprehensive metrics
    metrics = compute_metrics(y_pred=predictions, y_true=references)

    visualize_evaluation_results(metrics, title=f"Test Results: {dataset_name}")

    if run and log_videos > 0:
        log_videos_with_predictions(
            dataset=dataset,
            predictions=predictions,
            references=references,
            dataset_name=dataset_name,
            n_videos=log_videos,
        )

    # Perform subgroup evaluation if applicable
    subgroup_results = None
    if is_wanfall and predictions is not None:
        subgroup_results = perform_subgroup_evaluation(
            dataset=dataset,
            predictions=predictions,
            references=references,
            dataset_name=dataset_name,
            run=run,
        )

    if run:
        for k, v in metrics.items():
            wandb.log({f"{dataset_name}_{k}": v})

        log_confusion_matrix(
            predictions=predictions,
            references=references,
            dataset_name=dataset_name,
        )

    if save_results:
        save_evaluation_results(metrics, subgroup_results, output_dir)

    return metrics


def save_evaluation_results(
    all_results: dict[str, Any],
    subgroup_results: dict[str, Any],
    output_dir: str,
) -> None:
    """
    Save evaluation results to files (YAML and LaTeX).

    Args:
        all_results: All evaluation results
        subgroup_results: Subgroup evaluation results
        output_dir: Directory to save results
    """
    results_dir = os.path.join(output_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON results
    results_file = os.path.join(results_dir, f"test_results_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved evaluation results to {results_file}")

    # Save subgroup LaTeX tables if available
    if subgroup_results:
        latex_file = os.path.join(
            results_dir, f"subgroup_tables_{time.strftime('%Y%m%d-%H%M%S')}.tex"
        )
        logger.info(f"Generating subgroup LaTeX tables for {len(subgroup_results)} dataset(s)")

        with open(latex_file, "w") as f:
            f.write("% Subgroup Evaluation LaTeX Tables\n")
            f.write("% Generated automatically from WanFall evaluation\n\n")

            for dataset_name, subgroup_metrics in subgroup_results.items():
                f.write(f"\n% Dataset: {dataset_name}\n")
                logger.info(
                    f"  Processing dataset: {dataset_name} with subgroups: {list(subgroup_metrics.keys())}"
                )

                for subgroup_key in ["age_group", "ethnicity", "gender", "bmi_band"]:
                    if subgroup_key in subgroup_metrics:
                        logger.info(f"    Generating LaTeX table for {subgroup_key}")
                        table_latex = format_subgroup_latex_table(
                            subgroup_metrics,
                            subgroup_key,
                            metric_keys=[
                                "accuracy",
                                "balanced_accuracy",
                                "macro_f1",
                                "fall_sensitivity",
                                "fall_specificity",
                                "fall_f1",
                            ],
                        )
                        f.write("\n" + table_latex + "\n")
                    else:
                        logger.info(f"    No data for {subgroup_key}")

        logger.info(f"Saved subgroup LaTeX tables to {latex_file}")

        # Also print LaTeX tables to console for easy copy-paste
        print("\n" + "=" * 80)
        print("SUBGROUP LATEX TABLES (copy-paste ready)".center(80))
        print("=" * 80)

        for dataset_name, subgroup_metrics in subgroup_results.items():
            print(f"\n% Dataset: {dataset_name.upper()}")
            print("-" * 80)

            for subgroup_key in ["age_group", "ethnicity", "gender", "bmi_band"]:
                if subgroup_key in subgroup_metrics:
                    table_latex = format_subgroup_latex_table(
                        subgroup_metrics,
                        subgroup_key,
                        metric_keys=[
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                            "fall_sensitivity",
                            "fall_specificity",
                            "fall_f1",
                        ],
                    )
                    print(f"\n{table_latex}\n")

        print("=" * 80)
