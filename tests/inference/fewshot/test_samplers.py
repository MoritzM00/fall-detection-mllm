"""Tests for per-query exemplar samplers."""

import pytest
import torch

from falldet.inference.fewshot.samplers import (
    BalancedRandomSampler,
    RandomSampler,
    SimilaritySampler,
    get_embedding_filename,
    load_embeddings,
)

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
        ds = MockDataset(num_samples=40, num_classes=4)
        sampler = BalancedRandomSampler(ds, num_shots=8, seed=0)
        indices = sampler.sample(query_index=0)

        classes = {ds.video_segments[i]["label_str"] for i in indices}
        assert len(classes) == 4

    def test_roughly_equal_distribution(self):
        ds = MockDataset(num_samples=100, num_classes=4)
        sampler = BalancedRandomSampler(ds, num_shots=8, seed=0)
        indices = sampler.sample(query_index=0)

        counts: dict[str, int] = {}
        for i in indices:
            label = ds.video_segments[i]["label_str"]
            counts[label] = counts.get(label, 0) + 1

        for count in counts.values():
            assert count == 2  # 8 / 4 classes

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
        name = get_embedding_filename("OOPS_cs", "train", 16, 7.5)
        assert name == "OOPS_cs_train_16@7_5.pt"

    def test_int_fps(self):
        name = get_embedding_filename("OOPS_cs", "val", 16, 8)
        assert name == "OOPS_cs_val_16@8.pt"

    def test_different_mode(self):
        name = get_embedding_filename("OOPS_cs", "test", 9, 7.5)
        assert name == "OOPS_cs_test_9@7_5.pt"
