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
# Private helpers
# ---------------------------------------------------------------------------


def _build_class_to_indices(corpus: Dataset) -> dict[str, list[int]]:
    """Build a mapping from class label strings to corpus indices."""
    class_to_indices: dict[str, list[int]] = {}
    for idx in range(len(corpus)):  # type: ignore[arg-type]
        label: str = corpus.video_segments[idx]["label_str"]  # type: ignore[union-attr]
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def _apply_exemplar_ordering(
    retrievals: list[list[int]],
    scores: list[list[float]],
    ordering: ExemplarOrdering,
    seed: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Reorder pre-computed retrievals and scores in place."""
    if ordering == ExemplarOrdering.ASCENDING:
        retrievals = [r[::-1] for r in retrievals]
        scores = [s[::-1] for s in scores]
    elif ordering == ExemplarOrdering.RANDOM:
        rng = np.random.default_rng(seed)
        for i in range(len(retrievals)):
            perm = rng.permutation(len(retrievals[i]))
            retrievals[i] = [retrievals[i][j] for j in perm]
            scores[i] = [scores[i][j] for j in perm]
    return retrievals, scores


class _PrecomputedSampler(ExemplarSampler):
    """Base for samplers that pre-compute all retrievals at init for O(1) lookup."""

    def __init__(self, corpus: Dataset, num_shots: int = 5, seed: int = 0):
        super().__init__(corpus, num_shots)
        self._retrievals: list[list[int]] = []
        self._scores: list[list[float]] = []

    def sample(self, query_index: int) -> list[int]:
        return self._retrievals[query_index]

    def get_scores(self, query_index: int) -> list[float]:
        return self._scores[query_index]

    def __len__(self) -> int:
        return len(self._retrievals)


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

        self._class_to_indices = _build_class_to_indices(self.corpus)
        dist_str = ", ".join(
            f"{cls}: {len(idxs)}"
            for cls, idxs in sorted(
                self._class_to_indices.items(), key=lambda x: len(x[1]), reverse=True
            )
        )
        logger.info(
            f"Built class index: {len(self._class_to_indices)} classes — distribution: {dist_str}"
        )
        return self._class_to_indices

    def sample(self, query_index: int) -> list[int]:
        """Return freshly sampled balanced indices (``query_index`` ignored)."""
        class_to_indices = self._build_class_index()
        classes = sorted(class_to_indices.keys())
        num_classes = len(classes)
        if not classes:
            return []

        # Distribute shots across classes, giving priority to most frequent classes
        classes_by_freq = sorted(classes, key=lambda c: len(class_to_indices[c]), reverse=True)
        base = self.num_shots // num_classes
        remainder = self.num_shots % num_classes
        shots_per_class = {
            cls: base + (1 if i < remainder else 0) for i, cls in enumerate(classes_by_freq)
        }

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


class SimilaritySampler(_PrecomputedSampler):
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

        self._retrievals = topk_indices.tolist()
        self._scores = topk_scores.tolist()
        self._retrievals, self._scores = _apply_exemplar_ordering(
            self._retrievals, self._scores, exemplar_ordering, seed
        )

        logger.info(
            f"SimilaritySampler: {len(query_embeddings)} queries, "
            f"{len(corpus_embeddings)} corpus, top-{k}, ordering={exemplar_ordering}"
        )


class PerClassSimilaritySampler(_PrecomputedSampler):
    """Per-class cosine-similarity retrieval – one exemplar per selected class.

    For each query, computes the best-matching (highest cosine similarity)
    corpus item within every class, then selects the top-k classes ranked by
    that per-class maximum similarity.  This guarantees class diversity: each
    selected class contributes exactly one exemplar.

    ``num_shots`` must be <= the number of distinct classes in the corpus.
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
                "PerClassSimilaritySampler requires both query_embeddings and corpus_embeddings."
            )
        assert len(corpus) == corpus_embeddings.shape[0], (  # type: ignore[arg-type]
            "Corpus embeddings must match corpus size."
        )
        assert query_embeddings.shape[1] == corpus_embeddings.shape[1], (
            f"Query and corpus embeddings must have the same dimension, but got "
            f"query dim {query_embeddings.shape[1]} and corpus dim {corpus_embeddings.shape[1]}."
        )

        class_to_indices = _build_class_to_indices(corpus)
        num_classes = len(class_to_indices)
        if num_shots > num_classes:
            raise ValueError(
                f"num_shots ({num_shots}) must be <= number of classes ({num_classes}) "
                "for PerClassSimilaritySampler."
            )

        similarity = compute_similarity_scores(query_embeddings, corpus_embeddings)
        num_queries = query_embeddings.shape[0]

        # Pre-compute per-class index tensors once to avoid repeated allocation in the query loop
        class_index_tensors = [torch.tensor(idxs) for idxs in class_to_indices.values()]
        class_index_lists = list(class_to_indices.values())

        for qi in range(num_queries):
            sim_row = similarity[qi]
            class_best: list[tuple[float, int]] = []
            for idx_tensor, indices in zip(class_index_tensors, class_index_lists):
                local_best = sim_row[idx_tensor].argmax().item()
                best_idx = indices[int(local_best)]
                class_best.append((sim_row[best_idx].item(), best_idx))

            class_best.sort(key=lambda x: x[0], reverse=True)
            top_k = class_best[:num_shots]
            self._scores.append([s for s, _ in top_k])
            self._retrievals.append([i for _, i in top_k])

        self._retrievals, self._scores = _apply_exemplar_ordering(
            self._retrievals, self._scores, exemplar_ordering, seed
        )

        logger.info(
            f"PerClassSimilaritySampler: {num_queries} queries, "
            f"{len(corpus_embeddings)} corpus, {num_classes} classes, "
            f"top-{num_shots} classes, ordering={exemplar_ordering}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SAMPLER_REGISTRY: dict[str, type[ExemplarSampler]] = {
    "random": RandomSampler,
    "balanced": BalancedRandomSampler,
    "similarity": SimilaritySampler,
    "per_class_similarity": PerClassSimilaritySampler,
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
        return SimilaritySampler(
            corpus=corpus,
            num_shots=num_shots,
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
            exemplar_ordering=config.prompt.exemplar_ordering,
            seed=seed,
        )

    if sampler_cls is PerClassSimilaritySampler:
        return PerClassSimilaritySampler(
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
    if config.prompt.shot_selection in ("similarity", "per_class_similarity"):
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
