"""Embedding I/O and retrieval utilities.

Centralises embedding persistence (save / load), filename conventions, and
top-k cosine retrieval so that both the inference script and the few-shot
samplers share the same logic.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_embeddings(path: str | Path) -> tuple[torch.Tensor, list[dict]]:
    """Load a .pt embeddings file.

    Args:
        path: Path to the .pt file with 'embeddings' and 'samples' keys.

    Returns:
        Tuple of (embeddings tensor [n, dim], list of sample metadata dicts).

    Raises:
        FileNotFoundError: If the embedding file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}. Run the embed task first to generate embeddings."
        )
    data = torch.load(path, map_location="cpu", weights_only=True)
    embeddings: torch.Tensor = data["embeddings"]
    samples: list[dict] = data["samples"]
    logger.info(f"Loaded embeddings from {path}: shape={embeddings.shape}")
    return embeddings, samples


def get_embedding_filename(
    dataset_name: str,
    mode: str,
    num_frames: int,
    model_fps: float | int,
    model_name: str,
    data_size: int | None,
) -> str:
    """Derive the embedding filename from config parameters.

    ``dataset_name`` should already include the split suffix (e.g.
    ``"OOPS_cs"``).

    Returns:
        Filename like ``"OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt"``.
    """
    fps_str = str(model_fps).replace(".", "_") if isinstance(model_fps, float) else str(model_fps)
    size_str = str(data_size) if data_size is not None else "none"
    return f"{dataset_name}_{mode}_{num_frames}@{fps_str}_{model_name}_{size_str}.pt"


def retrieve_topk(
    query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, k: int = 10
) -> list[dict]:
    """Retrieve top-k results based on cosine similarity.

    Args:
        query_embeddings: Tensor of shape (num_queries, embedding_dim)
        corpus_embeddings: Tensor of shape (num_corpus, embedding_dim)
        k: Number of top results to retrieve per query
    Returns:
        List of dicts with 'ranked_indices' and 'ranked_scores' for each query.
    """
    similarity_scores = query_embeddings @ corpus_embeddings.T
    results = []
    for i in range(len(query_embeddings)):
        scores = similarity_scores[i].cpu().float().numpy()
        ranked_indices = np.argsort(scores)[::-1][:k]
        ranked_scores = scores[ranked_indices]
        results.append(
            {"ranked_indices": ranked_indices.tolist(), "ranked_scores": ranked_scores.tolist()}
        )
    return results


def save_embeddings(
    embeddings: torch.Tensor,
    samples: list[dict],
    output_dir: str | Path,
    dataset_name: str,
    mode: str,
    num_frames: int,
    model_fps: float | int,
    model_name: str,
    data_size: int | None,
) -> Path:
    """Save embeddings and sample metadata to a .pt file.

    Args:
        embeddings: Tensor of shape (n, dim).
        samples: List of per-sample metadata dicts.
        output_dir: Directory in which to write the file.
        dataset_name: Dataset name (including split suffix, e.g. ``"OOPS_cs"``).
        mode: Data mode (e.g. ``"train"``, ``"val"``).
        num_frames: Number of frames per clip.
        model_fps: Sampling FPS used by the model.
        model_name: Embedding model name (e.g. ``"Qwen3-VL-Embedding-2B"``).
        data_size: Input resolution (e.g. ``448``), or None.

    Returns:
        Path to the saved .pt file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = get_embedding_filename(
        dataset_name, mode, num_frames, model_fps, model_name, data_size
    )
    output_path = output_dir / filename

    if output_path.exists():
        logger.warning(f"Embedding file already exists and will be overwritten: {output_path}")

    torch.save(
        {
            "embeddings": embeddings,
            "samples": samples,
        },
        output_path,
    )
    logger.info(f"Saved embeddings to {output_path} (shape: {embeddings.shape})")
    return output_path
