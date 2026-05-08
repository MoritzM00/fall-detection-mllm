"""Tests for per-query exemplar samplers."""

import pytest
import torch

from falldet.embeddings import get_embedding_filename, load_embeddings
from falldet.inference.fewshot.samplers import (
    BalancedRandomSampler,
    PerClassSimilaritySampler,
    RandomSampler,
    SimilaritySampler,
)
from falldet.schemas import ExemplarOrdering

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class MockDataset:
    """Minimal mock dataset with ``video_segments`` for balanced sampling."""

    def __init__(self, num_samples: int = 20, num_classes: int = 4):
        self.num_samples = num_samples
        self.labels = [f"action_{i % num_classes}" for i in range(num_samples)]
        self.video_segments = [{"label_str": self.labels[i]} for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "video": torch.randn(4, 3, 8, 8),
            "label_str": self.labels[idx],
            "label": idx % 4,
            "idx": idx,
        }


# ---------------------------------------------------------------------------
# ExemplarSampler.get_exemplars
# ---------------------------------------------------------------------------


class TestGetExemplars:
    def test_returns_correct_count_and_keys(self):
        ds = MockDataset(num_samples=20)
        sampler = RandomSampler(ds, num_shots=5, seed=0)
        exemplars = sampler.get_exemplars(query_index=0)

        assert len(exemplars) == 5
        for ex in exemplars:
            assert isinstance(ex, dict)
            assert "video" in ex
            assert "label_str" in ex

    def test_matches_sample_indices(self):
        ds = MockDataset(num_samples=20)
        sampler = RandomSampler(ds, num_shots=3, seed=42)

        # get_exemplars should return the same items as manual indexing
        indices = sampler.sample(query_index=0)
        sampler2 = RandomSampler(ds, num_shots=3, seed=42)
        exemplars = sampler2.get_exemplars(query_index=0)

        for idx, ex in zip(indices, exemplars):
            assert ex["idx"] == ds[idx]["idx"]


# ---------------------------------------------------------------------------
# RandomSampler
# ---------------------------------------------------------------------------


class TestRandomSampler:
    def test_returns_correct_count(self):
        ds = MockDataset(num_samples=20)
        sampler = RandomSampler(ds, num_shots=5, seed=0)
        indices = sampler.sample(query_index=0)

        assert len(indices) == 5
        assert all(0 <= i < 20 for i in indices)

    def test_resamples_each_call(self):
        ds = MockDataset(num_samples=50)
        sampler = RandomSampler(ds, num_shots=5, seed=0)

        results = [sampler.sample(query_index=i) for i in range(10)]
        # With high probability, not all calls return the same set
        assert len(set(tuple(r) for r in results)) > 1

    def test_reproducibility_with_same_seed(self):
        ds = MockDataset(num_samples=50)
        s1 = RandomSampler(ds, num_shots=5, seed=42)
        s2 = RandomSampler(ds, num_shots=5, seed=42)

        # Same seed produces same sequence of calls
        assert s1.sample(0) == s2.sample(0)
        assert s1.sample(1) == s2.sample(1)

    def test_different_seeds_differ(self):
        ds = MockDataset(num_samples=50)
        s1 = RandomSampler(ds, num_shots=5, seed=42)
        s2 = RandomSampler(ds, num_shots=5, seed=99)

        assert s1.sample(0) != s2.sample(0)

    def test_handles_k_larger_than_corpus(self):
        ds = MockDataset(num_samples=3)
        sampler = RandomSampler(ds, num_shots=10, seed=0)
        indices = sampler.sample(0)

        assert len(indices) == 3

    def test_no_duplicates_in_single_call(self):
        ds = MockDataset(num_samples=50)
        sampler = RandomSampler(ds, num_shots=10, seed=0)
        indices = sampler.sample(0)

        assert len(set(indices)) == len(indices)


# ---------------------------------------------------------------------------
# BalancedRandomSampler
# ---------------------------------------------------------------------------


class TestBalancedRandomSampler:
    def test_returns_correct_count(self):
        ds = MockDataset(num_samples=20, num_classes=4)
        sampler = BalancedRandomSampler(ds, num_shots=8, seed=0)
        indices = sampler.sample(query_index=0)

        assert len(indices) == 8

    def test_covers_multiple_classes(self):
        """With equal class sizes and enough shots, all classes should appear."""
        ds = MockDataset(num_samples=40, num_classes=4)
        # Run many draws to ensure all classes are covered at least once
        sampler = BalancedRandomSampler(ds, num_shots=20, seed=0)
        indices = sampler.sample(query_index=0)

        classes = {ds.video_segments[i]["label_str"] for i in indices}
        assert len(classes) == 4

    def test_even_distribution(self):
        """Shots should be distributed roughly equally across classes."""
        # Imbalanced dataset: action_0 has 80 samples, action_1 has 20
        num_samples = 100
        labels = ["action_0"] * 80 + ["action_1"] * 20
        ds = MockDataset(num_samples=num_samples, num_classes=2)
        ds.labels = labels
        ds.video_segments = [{"label_str": labels[i]} for i in range(num_samples)]

        # Average over many draws
        counts = {"action_0": 0, "action_1": 0}
        num_trials = 200
        sampler = BalancedRandomSampler(ds, num_shots=10, seed=42)
        for trial in range(num_trials):
            indices = sampler.sample(query_index=trial)
            for i in indices:
                counts[ds.video_segments[i]["label_str"]] += 1

        # With even distribution, each class should get ~5 shots per trial
        ratio = counts["action_0"] / counts["action_1"]
        assert 0.8 < ratio < 1.2, f"Expected ratio ~1.0, got {ratio:.2f}"

    def test_resamples_each_call(self):
        ds = MockDataset(num_samples=100, num_classes=4)
        sampler = BalancedRandomSampler(ds, num_shots=4, seed=0)

        results = [tuple(sorted(sampler.sample(i))) for i in range(10)]
        assert len(set(results)) > 1

    def test_class_index_cached(self):
        ds = MockDataset(num_samples=20, num_classes=4)
        sampler = BalancedRandomSampler(ds, num_shots=4, seed=0)

        sampler.sample(0)
        idx1 = sampler._class_to_indices

        sampler.sample(1)
        idx2 = sampler._class_to_indices

        assert idx1 is idx2  # Same object (cached)

    def test_ascending_ordering_by_class_frequency(self):
        """ascending: rarest class first, most frequent class last."""
        num_samples = 60
        # action_0: 10 samples (rare), action_1: 20, action_2: 30 (frequent)
        labels = ["action_0"] * 10 + ["action_1"] * 20 + ["action_2"] * 30
        ds = MockDataset(num_samples=num_samples, num_classes=3)
        ds.labels = labels
        ds.video_segments = [{"label_str": labels[i]} for i in range(num_samples)]

        sampler = BalancedRandomSampler(
            ds, num_shots=3, seed=0, exemplar_ordering=ExemplarOrdering.ASCENDING
        )
        indices = sampler.sample(query_index=0)
        assert len(indices) == 3

        freq = {"action_0": 10, "action_1": 20, "action_2": 30}
        freqs = [freq[ds.video_segments[i]["label_str"]] for i in indices]
        assert freqs == sorted(freqs)

    def test_descending_ordering_by_class_frequency(self):
        """descending: most frequent class first, rarest class last."""
        num_samples = 60
        labels = ["action_0"] * 10 + ["action_1"] * 20 + ["action_2"] * 30
        ds = MockDataset(num_samples=num_samples, num_classes=3)
        ds.labels = labels
        ds.video_segments = [{"label_str": labels[i]} for i in range(num_samples)]

        sampler = BalancedRandomSampler(
            ds, num_shots=3, seed=0, exemplar_ordering=ExemplarOrdering.DESCENDING
        )
        indices = sampler.sample(query_index=0)
        assert len(indices) == 3

        freq = {"action_0": 10, "action_1": 20, "action_2": 30}
        freqs = [freq[ds.video_segments[i]["label_str"]] for i in indices]
        assert freqs == sorted(freqs, reverse=True)


# ---------------------------------------------------------------------------
# SimilaritySampler
# ---------------------------------------------------------------------------


class TestSimilaritySampler:
    @pytest.fixture()
    def embeddings(self):
        """Create deterministic embeddings for testing."""
        torch.manual_seed(42)
        corpus = torch.randn(20, 64)
        query = torch.randn(10, 64)
        return query, corpus

    def test_returns_correct_count(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds, num_shots=5, query_embeddings=query, corpus_embeddings=corpus
        )

        indices = sampler.sample(query_index=0)
        assert len(indices) == 5
        assert all(0 <= i < 20 for i in indices)

    def test_self_retrieval_identity(self):
        """When query == corpus, top-1 should be the sample itself."""
        torch.manual_seed(0)
        embs = torch.randn(10, 64)
        ds = MockDataset(num_samples=10)
        sampler = SimilaritySampler(ds, num_shots=3, query_embeddings=embs, corpus_embeddings=embs)

        for i in range(10):
            top1 = sampler.sample(i)[0]
            assert top1 == i, f"query {i} top-1 was {top1}"

    def test_scores_descending(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds, num_shots=5, query_embeddings=query, corpus_embeddings=corpus
        )

        for qi in range(len(query)):
            scores = sampler.get_scores(qi)
            assert scores == sorted(scores, reverse=True)

    def test_scores_are_valid_cosine(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds, num_shots=5, query_embeddings=query, corpus_embeddings=corpus
        )

        for qi in range(len(query)):
            for s in sampler.get_scores(qi):
                assert -1.0 - 1e-5 <= s <= 1.0 + 1e-5

    def test_handles_k_larger_than_corpus(self):
        torch.manual_seed(0)
        ds = MockDataset(num_samples=3)
        query = torch.randn(5, 32)
        corpus = torch.randn(3, 32)
        sampler = SimilaritySampler(
            ds, num_shots=10, query_embeddings=query, corpus_embeddings=corpus
        )

        indices = sampler.sample(0)
        assert len(indices) == 3

    def test_different_queries_get_different_results(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds, num_shots=5, query_embeddings=query, corpus_embeddings=corpus
        )

        results = {tuple(sampler.sample(i)) for i in range(len(query))}
        assert len(results) > 1

    def test_len(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds, num_shots=5, query_embeddings=query, corpus_embeddings=corpus
        )

        assert len(sampler) == len(query)

    def test_missing_embeddings_raises(self):
        ds = MockDataset(num_samples=10)
        with pytest.raises(ValueError, match="requires both"):
            SimilaritySampler(ds, num_shots=3, query_embeddings=None, corpus_embeddings=None)

    def test_descending_ordering_is_descending(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds,
            num_shots=5,
            query_embeddings=query,
            corpus_embeddings=corpus,
            exemplar_ordering=ExemplarOrdering.DESCENDING,
        )

        for qi in range(len(query)):
            scores = sampler.get_scores(qi)
            assert scores == sorted(scores, reverse=True)

    def test_ascending_ordering_is_ascending(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler = SimilaritySampler(
            ds,
            num_shots=5,
            query_embeddings=query,
            corpus_embeddings=corpus,
            exemplar_ordering=ExemplarOrdering.ASCENDING,
        )

        for qi in range(len(query)):
            scores = sampler.get_scores(qi)
            assert scores == sorted(scores)

    def test_ascending_reverses_descending_indices(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler_desc = SimilaritySampler(
            ds,
            num_shots=5,
            query_embeddings=query,
            corpus_embeddings=corpus,
            exemplar_ordering=ExemplarOrdering.DESCENDING,
        )
        sampler_asc = SimilaritySampler(
            ds,
            num_shots=5,
            query_embeddings=query,
            corpus_embeddings=corpus,
            exemplar_ordering=ExemplarOrdering.ASCENDING,
        )

        for qi in range(len(query)):
            assert sampler_desc.sample(qi) == sampler_asc.sample(qi)[::-1]
            assert sampler_desc.get_scores(qi) == sampler_asc.get_scores(qi)[::-1]

    def test_random_ordering_same_set_different_order(self, embeddings):
        query, corpus = embeddings
        ds = MockDataset(num_samples=20)
        sampler_desc = SimilaritySampler(
            ds,
            num_shots=5,
            query_embeddings=query,
            corpus_embeddings=corpus,
            exemplar_ordering=ExemplarOrdering.DESCENDING,
        )
        sampler_rand = SimilaritySampler(
            ds,
            num_shots=5,
            query_embeddings=query,
            corpus_embeddings=corpus,
            exemplar_ordering=ExemplarOrdering.RANDOM,
            seed=7,
        )

        for qi in range(len(query)):
            assert sorted(sampler_desc.sample(qi)) == sorted(sampler_rand.sample(qi))
        # At least one query should have a different order
        assert any(sampler_desc.sample(qi) != sampler_rand.sample(qi) for qi in range(len(query)))


# ---------------------------------------------------------------------------
# PerClassSimilaritySampler
# ---------------------------------------------------------------------------


class TestPerClassSimilaritySampler:
    @pytest.fixture()
    def class_corpus(self):
        """Corpus with 4 classes × 5 samples = 20 items, plus matched embeddings."""
        torch.manual_seed(0)
        num_classes = 4
        samples_per_class = 5
        num_corpus = num_classes * samples_per_class
        labels = [f"action_{i // samples_per_class}" for i in range(num_corpus)]
        ds = MockDataset(num_samples=num_corpus, num_classes=num_classes)
        ds.labels = labels
        ds.video_segments = [{"label_str": labels[i]} for i in range(num_corpus)]
        corpus_embs = torch.randn(num_corpus, 64)
        query_embs = torch.randn(10, 64)
        return ds, query_embs, corpus_embs

    def test_returns_k_exemplars(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        sampler = PerClassSimilaritySampler(
            ds, num_shots=3, query_embeddings=query_embs, corpus_embeddings=corpus_embs
        )
        indices = sampler.sample(0)
        assert len(indices) == 3

    def test_each_exemplar_from_distinct_class(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        sampler = PerClassSimilaritySampler(
            ds, num_shots=3, query_embeddings=query_embs, corpus_embeddings=corpus_embs
        )
        for qi in range(len(query_embs)):
            indices = sampler.sample(qi)
            classes = [ds.video_segments[i]["label_str"] for i in indices]
            assert len(classes) == len(set(classes)), f"Duplicate class in query {qi}: {classes}"

    def test_selects_best_within_class(self, class_corpus):
        """Each returned index must be the highest-similarity item within its class."""
        import torch.nn.functional as F

        ds, query_embs, corpus_embs = class_corpus
        sampler = PerClassSimilaritySampler(
            ds, num_shots=4, query_embeddings=query_embs, corpus_embeddings=corpus_embs
        )
        q_norm = F.normalize(query_embs.float(), dim=1)
        c_norm = F.normalize(corpus_embs.float(), dim=1)
        sim = q_norm @ c_norm.T  # (num_queries, num_corpus)

        class_to_indices: dict[str, list[int]] = {}
        for idx in range(len(ds)):
            lbl = ds.video_segments[idx]["label_str"]
            class_to_indices.setdefault(lbl, []).append(idx)

        for qi in range(len(query_embs)):
            retrieved = sampler.sample(qi)
            for idx in retrieved:
                lbl = ds.video_segments[idx]["label_str"]
                class_indices = class_to_indices[lbl]
                best_in_class = max(class_indices, key=lambda i: sim[qi, i].item())
                assert idx == best_in_class, (
                    f"query {qi}, class {lbl}: expected corpus idx {best_in_class}, got {idx}"
                )

    def test_descending_scores(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        sampler = PerClassSimilaritySampler(
            ds,
            num_shots=3,
            query_embeddings=query_embs,
            corpus_embeddings=corpus_embs,
            exemplar_ordering=ExemplarOrdering.DESCENDING,
        )
        for qi in range(len(query_embs)):
            scores = sampler.get_scores(qi)
            assert scores == sorted(scores, reverse=True)

    def test_ascending_scores(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        sampler = PerClassSimilaritySampler(
            ds,
            num_shots=3,
            query_embeddings=query_embs,
            corpus_embeddings=corpus_embs,
            exemplar_ordering=ExemplarOrdering.ASCENDING,
        )
        for qi in range(len(query_embs)):
            scores = sampler.get_scores(qi)
            assert scores == sorted(scores)

    def test_random_ordering_same_set(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        sampler_desc = PerClassSimilaritySampler(
            ds,
            num_shots=3,
            query_embeddings=query_embs,
            corpus_embeddings=corpus_embs,
            exemplar_ordering=ExemplarOrdering.DESCENDING,
        )
        sampler_rand = PerClassSimilaritySampler(
            ds,
            num_shots=3,
            query_embeddings=query_embs,
            corpus_embeddings=corpus_embs,
            exemplar_ordering=ExemplarOrdering.RANDOM,
            seed=7,
        )
        for qi in range(len(query_embs)):
            assert sorted(sampler_desc.sample(qi)) == sorted(sampler_rand.sample(qi))

    def test_k_exceeds_classes_raises(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        with pytest.raises(ValueError, match="num_shots"):
            PerClassSimilaritySampler(
                ds, num_shots=5, query_embeddings=query_embs, corpus_embeddings=corpus_embs
            )

    def test_len(self, class_corpus):
        ds, query_embs, corpus_embs = class_corpus
        sampler = PerClassSimilaritySampler(
            ds, num_shots=2, query_embeddings=query_embs, corpus_embeddings=corpus_embs
        )
        assert len(sampler) == len(query_embs)

    def test_missing_embeddings_raises(self):
        ds = MockDataset(num_samples=10)
        with pytest.raises(ValueError, match="requires both"):
            PerClassSimilaritySampler(
                ds, num_shots=2, query_embeddings=None, corpus_embeddings=None
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestLoadEmbeddings:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Embedding file not found"):
            load_embeddings(tmp_path / "nonexistent.pt")

    def test_loads_correctly(self, tmp_path):
        embs = torch.randn(10, 64)
        samples = [{"label_str": f"cls_{i}"} for i in range(10)]
        path = tmp_path / "test.pt"
        torch.save({"embeddings": embs, "samples": samples}, path)

        loaded_embs, loaded_samples = load_embeddings(path)
        assert torch.allclose(loaded_embs, embs)
        assert loaded_samples == samples


class TestGetEmbeddingFilename:
    def test_float_fps(self):
        name = get_embedding_filename("OOPS_cs", "train", 16, 7.5, "Qwen3-VL-Embedding-2B", 448)
        assert name == "OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt"

    def test_int_fps(self):
        name = get_embedding_filename("OOPS_cs", "val", 16, 8, "Qwen3-VL-Embedding-2B", 448)
        assert name == "OOPS_cs_val_16@8_Qwen3-VL-Embedding-2B_448.pt"

    def test_different_mode(self):
        name = get_embedding_filename("OOPS_cs", "test", 9, 7.5, "InternVL3-2B", 448)
        assert name == "OOPS_cs_test_9@7_5_InternVL3-2B_448.pt"

    def test_data_size_none(self):
        name = get_embedding_filename("OOPS_cs", "train", 16, 7.5, "Qwen3-VL-Embedding-2B", None)
        assert name == "OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_none.pt"
