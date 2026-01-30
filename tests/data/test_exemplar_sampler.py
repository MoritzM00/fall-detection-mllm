"""Tests for the ExemplarSampler class."""

import pytest
import torch

from infreqact.data.exemplar_sampler import ExemplarSampler


class MockDataset:
    """Mock dataset for testing ExemplarSampler."""

    def __init__(self, num_samples: int = 20, num_classes: int = 4):
        """Initialize mock dataset.

        Args:
            num_samples: Total number of samples
            num_classes: Number of action classes
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.labels = [f"action_{i % num_classes}" for i in range(num_samples)]

        # Create video_segments attribute to mimic real dataset
        self.video_segments = [{"label_str": self.labels[i]} for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Return a mock sample."""
        return {
            "video": torch.randn(16, 3, 224, 224),  # Mock video tensor
            "label_str": self.labels[idx],
            "label": idx % self.num_classes,
            "idx": idx,
        }


class TestExemplarSampler:
    """Tests for ExemplarSampler."""

    def test_sample_zero_shots(self):
        """Test that zero shots returns empty list."""
        dataset = MockDataset()
        sampler = ExemplarSampler(dataset, num_shots=0)
        exemplars = sampler.sample()

        assert exemplars == []
        assert sampler._cache == []

    def test_sample_random_returns_correct_count(self):
        """Test random sampling returns requested number of exemplars."""
        dataset = MockDataset(num_samples=20)
        num_shots = 4
        sampler = ExemplarSampler(dataset, num_shots=num_shots, strategy="random")
        exemplars = sampler.sample()

        assert len(exemplars) == num_shots
        for ex in exemplars:
            assert "video" in ex
            assert "label_str" in ex

    def test_sample_balanced_returns_correct_count(self):
        """Test balanced sampling returns requested number of exemplars."""
        dataset = MockDataset(num_samples=20, num_classes=4)
        num_shots = 4
        sampler = ExemplarSampler(dataset, num_shots=num_shots, strategy="balanced")
        exemplars = sampler.sample()

        assert len(exemplars) == num_shots

    def test_sample_balanced_covers_multiple_classes(self):
        """Test balanced sampling covers multiple action classes."""
        dataset = MockDataset(num_samples=40, num_classes=4)
        num_shots = 8
        sampler = ExemplarSampler(dataset, num_shots=num_shots, strategy="balanced")
        exemplars = sampler.sample()

        # Collect unique classes from exemplars
        classes = set(ex["label_str"] for ex in exemplars)

        # With 8 shots across 4 classes, should have samples from multiple classes
        assert len(classes) >= 2
        # Ideally all 4 classes should be represented
        assert len(classes) == 4

    def test_sample_balanced_roughly_equal_distribution(self):
        """Test balanced sampling produces roughly equal samples per class."""
        dataset = MockDataset(num_samples=100, num_classes=4)
        num_shots = 8
        sampler = ExemplarSampler(dataset, num_shots=num_shots, strategy="balanced")
        exemplars = sampler.sample()

        # Count samples per class
        class_counts = {}
        for ex in exemplars:
            label = ex["label_str"]
            class_counts[label] = class_counts.get(label, 0) + 1

        # Each class should have exactly 2 samples (8 shots / 4 classes)
        for count in class_counts.values():
            assert count == 2

    def test_sample_caching(self):
        """Test that second call returns cached exemplars."""
        dataset = MockDataset()
        sampler = ExemplarSampler(dataset, num_shots=4, strategy="random")

        exemplars1 = sampler.sample()
        exemplars2 = sampler.sample()

        # Should be the exact same list object (cached)
        assert exemplars1 is exemplars2
        assert sampler._cache is exemplars1

    def test_sample_reproducibility(self):
        """Test that same seed produces same exemplars."""
        dataset = MockDataset(num_samples=50)
        seed = 42

        sampler1 = ExemplarSampler(dataset, num_shots=4, strategy="random", seed=seed)
        exemplars1 = sampler1.sample()

        sampler2 = ExemplarSampler(dataset, num_shots=4, strategy="random", seed=seed)
        exemplars2 = sampler2.sample()

        # Should produce the same indices
        indices1 = [ex["idx"] for ex in exemplars1]
        indices2 = [ex["idx"] for ex in exemplars2]
        assert indices1 == indices2

    def test_sample_different_seeds_produce_different_results(self):
        """Test that different seeds produce different exemplars."""
        dataset = MockDataset(num_samples=50)

        sampler1 = ExemplarSampler(dataset, num_shots=4, strategy="random", seed=42)
        exemplars1 = sampler1.sample()

        sampler2 = ExemplarSampler(dataset, num_shots=4, strategy="random", seed=123)
        exemplars2 = sampler2.sample()

        # Should produce different indices (with high probability)
        indices1 = [ex["idx"] for ex in exemplars1]
        indices2 = [ex["idx"] for ex in exemplars2]
        assert indices1 != indices2

    def test_sample_exceeds_dataset_size(self):
        """Test handling when num_shots exceeds dataset size."""
        dataset = MockDataset(num_samples=5)
        sampler = ExemplarSampler(dataset, num_shots=10, strategy="random")
        exemplars = sampler.sample()

        # Should return at most len(dataset) exemplars
        assert len(exemplars) <= len(dataset)
        assert len(exemplars) == 5

    def test_sample_exceeds_per_class_availability(self):
        """Test handling when num_shots exceeds samples per class."""
        # 8 samples, 4 classes = 2 samples per class
        dataset = MockDataset(num_samples=8, num_classes=4)
        # Request 8 shots with balanced - 2 per class
        sampler = ExemplarSampler(dataset, num_shots=8, strategy="balanced")
        exemplars = sampler.sample()

        # Should handle gracefully
        assert len(exemplars) == 8

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        dataset = MockDataset()
        sampler = ExemplarSampler(dataset, num_shots=4, strategy="invalid")  # type: ignore

        with pytest.raises(ValueError, match="Unknown strategy"):
            sampler.sample()

    def test_class_index_is_cached(self):
        """Test that class index is built once and cached."""
        dataset = MockDataset(num_samples=20, num_classes=4)
        sampler = ExemplarSampler(dataset, num_shots=4, strategy="balanced")

        # First call builds the index
        sampler.sample()
        index1 = sampler._class_to_indices

        # Clear sample cache but not class index
        sampler._cache = None
        sampler.sample()
        index2 = sampler._class_to_indices

        # Should be the same object (cached)
        assert index1 is index2

    def test_class_index_has_correct_structure(self):
        """Test that class index maps labels to correct indices."""
        dataset = MockDataset(num_samples=12, num_classes=3)
        sampler = ExemplarSampler(dataset, num_shots=3, strategy="balanced")
        sampler.sample()

        class_index = sampler._class_to_indices
        assert class_index is not None

        # Should have 3 classes
        assert len(class_index) == 3

        # Each class should have 4 indices (12 samples / 3 classes)
        for label, indices in class_index.items():
            assert len(indices) == 4
            # All indices should map back to this label
            for idx in indices:
                assert dataset.video_segments[idx]["label_str"] == label
