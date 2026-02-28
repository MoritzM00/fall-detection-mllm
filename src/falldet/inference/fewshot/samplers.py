"""Per-query exemplar sampling for few-shot inference.

All samplers implement the same interface: sample(query_index) -> list[int],
returning corpus indices for the given query. The inference loop loads the
actual exemplar dicts from the train dataset.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from falldet.schemas import InferenceConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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
    data = torch.load(path, map_location="cpu", weights_only=False)
    embeddings: torch.Tensor = data["embeddings"]
    samples: list[dict] = data["samples"]
    logger.info(f"Loaded embeddings from {path}: shape={embeddings.shape}")
    return embeddings, samples


def get_embedding_filename(
    dataset_name: str, mode: str, num_frames: int, model_fps: float | int
) -> str:
    """Derive the embedding filename from config parameters.

    Matches the naming convention in ``_save_embeddings()`` in
    ``vllm_inference.py``.  ``dataset_name`` should already include the
    split suffix (e.g. ``"OOPS_cs"``).

    Returns:
        Filename like ``"OOPS_cs_train_16@7_5.pt"``.
    """
    fps_str = str(model_fps).replace(".", "_") if isinstance(model_fps, float) else str(model_fps)
    return f"{dataset_name}_{mode}_{num_frames}@{fps_str}.pt"


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


# ---------------------------------------------------------------------------
# Sampler ABC
# ---------------------------------------------------------------------------


class ExemplarSampler(ABC):
    """Base class for per-query exemplar samplers.

    Every sampler holds a reference to the corpus (train) dataset and
    returns **indices** into it.  The inference loop is responsible for
    loading the actual exemplar dicts via ``corpus[idx]``.
    """

    def __init__(self, corpus: Dataset, num_shots: int = 5, seed: int = 0):
        self.corpus = corpus

        if num_shots <= 0:
            raise ValueError("num_shots must be positive.")
        self.num_shots = num_shots

    @abstractmethod
    def sample(self, query_index: int) -> list[int]:
        """Return corpus indices for the given query."""


# ---------------------------------------------------------------------------
# Concrete samplers
# ---------------------------------------------------------------------------


class RandomSampler(ExemplarSampler):
    """Uniformly random sampling – resamples on every call."""

    def __init__(self, corpus: Dataset, num_shots: int = 5, seed: int = 0):
        super().__init__(corpus, num_shots)
        self.rng = np.random.default_rng(seed)

    def sample(self, query_index: int) -> list[int]:
        """Return freshly sampled random indices (``query_index`` ignored)."""
        n = min(self.num_shots, len(self.corpus))  # type: ignore[arg-type]
        return self.rng.choice(len(self.corpus), size=n, replace=False).tolist()  # type: ignore[arg-type]


class BalancedRandomSampler(ExemplarSampler):
    """Balanced sampling across classes – resamples on every call.

    Distributes shots roughly equally across classes, using the
    ``video_segments`` attribute of the corpus dataset to read labels.
    """

    def __init__(self, corpus: Dataset, num_shots: int = 5, seed: int = 0):
        super().__init__(corpus, num_shots)
        self.rng = np.random.default_rng(seed)
        self._class_to_indices: dict[str, list[int]] | None = None

    def _build_class_index(self) -> dict[str, list[int]]:
        """Build and cache a mapping from class labels to corpus indices."""
        if self._class_to_indices is not None:
            return self._class_to_indices

        class_to_indices: dict[str, list[int]] = {}
        for idx in range(len(self.corpus)):  # type: ignore[arg-type]
            segment = self.corpus.video_segments[idx]  # type: ignore[union-attr]
            label: str = segment["label_str"]
            class_to_indices.setdefault(label, []).append(idx)

        self._class_to_indices = class_to_indices
        logger.info(f"Built class index: {len(class_to_indices)} classes")
        return class_to_indices

    def sample(self, query_index: int) -> list[int]:
        """Return freshly sampled balanced indices (``query_index`` ignored)."""
        class_to_indices = self._build_class_index()
        classes = sorted(class_to_indices.keys())
        num_classes = len(classes)
        if not classes:
            return []

        # Distribute shots as evenly as possible across classes
        shots_per_class = {cls: self.num_shots // num_classes for cls in classes}
        remainder = self.num_shots % num_classes
        for cls in self.rng.choice(classes, remainder, replace=False):
            shots_per_class[cls] += 1

        indices: list[int] = []
        for cls, num_to_sample in shots_per_class.items():
            available = class_to_indices.get(cls, [])
            n = min(num_to_sample, len(available))
            if n > 0:
                sampled = self.rng.choice(available, n, replace=False).tolist()
                indices.extend(sampled)

        self.rng.shuffle(indices)
        return indices


class SimilaritySampler(ExemplarSampler):
    """Cosine-similarity retrieval – pre-computes all retrievals at init.

    Performs a single batched matrix multiply to compute cosine similarity
    between all queries and all corpus items, then stores the top-k indices
    per query for O(1) lookup during inference.
    """

    def __init__(
        self,
        corpus: Dataset,
        num_shots: int = 5,
        query_embeddings: torch.Tensor | None = None,
        corpus_embeddings: torch.Tensor | None = None,
    ):
        super().__init__(corpus, num_shots)

        if query_embeddings is None or corpus_embeddings is None:
            raise ValueError(
                "SimilaritySampler requires both query_embeddings and corpus_embeddings."
            )

        # L2-normalise for cosine similarity
        query_norm = F.normalize(query_embeddings.float(), dim=1)
        corpus_norm = F.normalize(corpus_embeddings.float(), dim=1)

        # Batched cosine similarity: [num_queries, num_corpus]
        similarity = query_norm @ corpus_norm.T

        k = min(num_shots, corpus_embeddings.shape[0])
        topk_scores, topk_indices = torch.topk(similarity, k=k, dim=1)

        self._retrievals: list[list[int]] = topk_indices.tolist()
        self._scores: list[list[float]] = topk_scores.tolist()

        logger.info(
            f"SimilaritySampler: {len(query_embeddings)} queries, "
            f"{len(corpus_embeddings)} corpus, top-{k}"
        )

    def sample(self, query_index: int) -> list[int]:
        """Return pre-computed top-k corpus indices for ``query_index``."""
        return self._retrievals[query_index]

    def get_scores(self, query_index: int) -> list[float]:
        """Return cosine similarity scores for the retrieved exemplars."""
        return self._scores[query_index]

    def __len__(self) -> int:
        """Number of queries with pre-computed retrievals."""
        return len(self._retrievals)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SAMPLER_REGISTRY: dict[str, type[ExemplarSampler]] = {
    "random": RandomSampler,
    "balanced": BalancedRandomSampler,
}


def create_sampler(
    config: InferenceConfig,
    corpus: Dataset,
    query_embeddings: torch.Tensor | None = None,
    corpus_embeddings: torch.Tensor | None = None,
) -> ExemplarSampler:
    """Create the appropriate sampler from config.

    Args:
        config: Inference configuration.
        corpus: Training dataset.
        query_embeddings: Required for similarity mode.
        corpus_embeddings: Required for similarity mode.

    Returns:
        An ``ExemplarSampler`` instance.
    """
    strategy = config.prompt.shot_selection
    num_shots = config.prompt.num_shots
    seed = config.prompt.exemplar_seed

    if strategy == "similarity":
        return SimilaritySampler(
            corpus=corpus,
            num_shots=num_shots,
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
        )

    sampler_cls = _SAMPLER_REGISTRY.get(strategy)
    if sampler_cls is None:
        raise ValueError(f"Unknown shot_selection strategy: {strategy!r}")

    return sampler_cls(corpus=corpus, num_shots=num_shots, seed=seed)
