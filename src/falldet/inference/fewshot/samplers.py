"""Per-query exemplar sampling for few-shot inference.

All samplers implement the same interface: sample(query_index) -> list[int],
returning corpus indices for the given query. The inference loop loads the
actual exemplar dicts from the train dataset.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from falldet.schemas import InferenceConfig

logger = logging.getLogger(__name__)


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
        assert len(corpus) == corpus_embeddings.shape[0], (  # type: ignore[arg-type]
            "Corpus embeddings must match corpus size."
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
    "similarity": SimilaritySampler,
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

    sampler_cls = _SAMPLER_REGISTRY.get(strategy)
    if sampler_cls is None:
        raise ValueError(f"Unknown shot_selection strategy: {strategy!r}")

    if sampler_cls is SimilaritySampler:
        return sampler_cls(
            corpus=corpus,
            num_shots=num_shots,
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
        )

    return sampler_cls(corpus=corpus, num_shots=num_shots, seed=seed)
