"""Per-query exemplar sampling for few-shot inference.

All samplers implement the same interface: sample(query_index) -> list[int],
returning corpus indices for the given query. The convenience method
get_exemplars(query_index) loads the actual exemplar dicts from the corpus.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

from falldet.embeddings import compute_similarity_scores
from falldet.schemas import ExemplarOrdering, InferenceConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler ABC
# ---------------------------------------------------------------------------


class ExemplarSampler(ABC):
    """Base class for per-query exemplar samplers.

    Every sampler holds a reference to the corpus (train) dataset and
    returns **indices** into it.  Use ``get_exemplars(query_index)`` to
    get the actual exemplar dicts directly.
    """

    def __init__(self, corpus: Dataset, num_shots: int = 5, seed: int = 0):
        self.corpus = corpus

        if num_shots <= 0:
            raise ValueError("num_shots must be positive.")
        self.num_shots = num_shots

    @abstractmethod
    def sample(self, query_index: int) -> list[int]:
        """Return corpus indices for the given query."""

    def get_exemplars(self, query_index: int) -> list[dict]:
        """Sample indices and return the corresponding corpus items."""
        indices = self.sample(query_index)
        with ThreadPoolExecutor(max_workers=len(indices)) as executor:
            return list(executor.map(self.corpus.__getitem__, indices))

    def log_cache_stats(self) -> None:
        """Log cache stats from the exemplar corpus, if available."""
        if hasattr(self.corpus, "log_cache_stats"):
            self.corpus.log_cache_stats()  # type: ignore[union-attr]

    def get_batch_exemplars(self, query_indices: list[int]) -> list[list[dict]]:
        """Load exemplars for multiple queries in a single thread pool.

        Instead of calling ``get_exemplars`` sequentially for each query,
        this flattens all corpus loads into one pool so I/O overlaps across
        the entire batch.
        """
        all_indices = [self.sample(q) for q in query_indices]

        # Flatten into (corpus_index, batch_pos, shot_pos) for parallel loading
        flat_keys: list[tuple[int, int, int]] = []
        for batch_pos, indices in enumerate(all_indices):
            for shot_pos, idx in enumerate(indices):
                flat_keys.append((idx, batch_pos, shot_pos))

        with ThreadPoolExecutor(max_workers=min(len(flat_keys), 16)) as executor:
            flat_items = list(executor.map(self.corpus.__getitem__, [k[0] for k in flat_keys]))

        # Reassemble into list[list[dict]] preserving order
        result: list[list[dict]] = [[] for _ in query_indices]
        for (_, batch_pos, _), item in zip(flat_keys, flat_items):
            result[batch_pos].append(item)
        return result


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
        n = len(self.corpus)  # type: ignore[arg-type]
        k = min(self.num_shots, n)
        return self.rng.choice(n, size=k, replace=False).tolist()


class BalancedRandomSampler(ExemplarSampler):
    """Balanced sampling across classes – resamples on every call.

    Distributes shots roughly equally across classes, using the
    ``video_segments`` attribute of the corpus dataset to read labels.
    """

    def __init__(
        self,
        corpus: Dataset,
        num_shots: int = 5,
        seed: int = 0,
        exemplar_ordering: ExemplarOrdering = ExemplarOrdering.RANDOM,
    ):
        super().__init__(corpus, num_shots)
        self.rng = np.random.default_rng(seed)
        self.exemplar_ordering = exemplar_ordering
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

        if self.exemplar_ordering == ExemplarOrdering.RANDOM:
            self.rng.shuffle(indices)
        elif self.exemplar_ordering in (ExemplarOrdering.ASCENDING, ExemplarOrdering.DESCENDING):
            freq = {cls: len(idxs) for cls, idxs in class_to_indices.items()}
            label_of = {
                idx: self.corpus.video_segments[idx]["label_str"]  # type: ignore[union-attr]
                for idx in indices
            }
            indices.sort(
                key=lambda i: freq[label_of[i]],
                reverse=(self.exemplar_ordering == ExemplarOrdering.DESCENDING),
            )

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
        exemplar_ordering: ExemplarOrdering = ExemplarOrdering.DESCENDING,
        seed: int = 0,
    ):
        super().__init__(corpus, num_shots)

        if query_embeddings is None or corpus_embeddings is None:
            raise ValueError(
                "SimilaritySampler requires both query_embeddings and corpus_embeddings."
            )
        assert len(corpus) == corpus_embeddings.shape[0], (  # type: ignore[arg-type]
            "Corpus embeddings must match corpus size."
        )
        assert query_embeddings.shape[1] == corpus_embeddings.shape[1], (
            f"Query and corpus embeddings must have the same dimension, but got "
            f"query dim {query_embeddings.shape[1]} and corpus dim {corpus_embeddings.shape[1]}."
        )

        similarity = compute_similarity_scores(query_embeddings, corpus_embeddings)

        k = min(num_shots, corpus_embeddings.shape[0])
        topk_scores, topk_indices = torch.topk(similarity, k=k, dim=1)

        self._retrievals: list[list[int]] = topk_indices.tolist()
        self._scores: list[list[float]] = topk_scores.tolist()

        if exemplar_ordering == ExemplarOrdering.ASCENDING:
            self._retrievals = [r[::-1] for r in self._retrievals]
            self._scores = [s[::-1] for s in self._scores]
        elif exemplar_ordering == ExemplarOrdering.RANDOM:
            rng = np.random.default_rng(seed)
            for i in range(len(self._retrievals)):
                perm = rng.permutation(len(self._retrievals[i]))
                self._retrievals[i] = [self._retrievals[i][j] for j in perm]
                self._scores[i] = [self._scores[i][j] for j in perm]

        logger.info(
            f"SimilaritySampler: {len(query_embeddings)} queries, "
            f"{len(corpus_embeddings)} corpus, top-{k}, ordering={exemplar_ordering}"
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
            exemplar_ordering=config.prompt.exemplar_ordering,
            seed=seed,
        )

    if sampler_cls is BalancedRandomSampler:
        return sampler_cls(
            corpus=corpus,
            num_shots=num_shots,
            seed=seed,
            exemplar_ordering=config.prompt.exemplar_ordering,
        )

    return sampler_cls(corpus=corpus, num_shots=num_shots, seed=seed)


# ---------------------------------------------------------------------------
# High-level setup helper
# ---------------------------------------------------------------------------


def setup_fewshot_sampler(
    config: InferenceConfig,
    dataset_name: str,
) -> ExemplarSampler:
    """Load train data, embeddings, and create the exemplar sampler.

    Encapsulates the full few-shot setup so callers only need a single call.

    Args:
        config: Inference configuration (must have ``num_shots > 0``).
        dataset_name: Name of the dataset (including split suffix).

    Returns:
        A ready-to-use ``ExemplarSampler`` whose ``corpus`` is the train dataset.
    """
    from typing import cast

    from falldet.data.video_dataset_factory import get_video_datasets

    # Load train dataset for exemplar video access
    train_datasets = get_video_datasets(
        config=config,
        mode="train",
        split=config.data.split,
        size=config.data.size,
        seed=config.data.seed,
        return_individual=True,
    )
    train_datasets = cast(dict[str, object], train_datasets)
    train_dataset = list(train_datasets["individual"].values())[0]  # type: ignore[union-attr]
    logger.info(f"Train dataset loaded: {len(train_dataset)} samples for exemplar access")

    # Enable in-memory caching for the exemplar corpus only (not the test dataset).
    # ThreadPoolExecutor in get_batch_exemplars shares memory across threads, so
    # repeated corpus accesses within a run hit O(1) dict lookup after the first load.
    if config.data.cache_in_memory:
        train_dataset.enable_memory_cache()
        logger.info("In-memory cache enabled for exemplar corpus")

    # Load embeddings if needed for similarity-based retrieval
    query_embeddings: torch.Tensor | None = None
    corpus_embeddings: torch.Tensor | None = None
    if config.prompt.shot_selection == "similarity":
        from pathlib import Path

        from falldet.embeddings import get_embedding_filename, load_embeddings

        emb_dir = Path(config.embeddings_dir)
        emb_model_name = config.embedding_model_name or config.model.name
        train_emb_file = get_embedding_filename(
            dataset_name,
            "train",
            config.num_frames,
            config.model_fps,
            model_name=emb_model_name,
            data_size=config.data.size,
        )
        query_emb_file = get_embedding_filename(
            dataset_name,
            config.data.mode,
            config.num_frames,
            config.model_fps,
            model_name=emb_model_name,
            data_size=config.data.size,
        )
        corpus_embeddings, _ = load_embeddings(emb_dir / train_emb_file)
        query_embeddings, _ = load_embeddings(emb_dir / query_emb_file)

        # Slice query embeddings to match num_samples Subset
        if config.num_samples is not None:
            n = min(config.num_samples, len(query_embeddings))
            query_embeddings = query_embeddings[:n]
            logger.info(f"Sliced query embeddings to {n} (num_samples={config.num_samples})")

    return create_sampler(
        config,
        train_dataset,
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
    )
