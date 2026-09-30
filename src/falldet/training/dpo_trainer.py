"""Minimal TRL trainer adaptation for lazy video preference datasets."""

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from trl import DPOConfig, DPOTrainer


class VideoDPOTrainer(DPOTrainer):
    """Keep the lazy dataset intact; the custom collator performs processing."""

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: DPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        del processing_class, args, dataset_name
        return dataset

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Reduce eval logits to teacher-forced argmax tokens of the chosen answers.

        The collator stacks all chosen rows before all rejected rows. For the chosen
        half, return (argmax token ids, gold ids masked to -100 outside the answer),
        aligned so ``preds[:, t]`` is the prediction for ``labels[:, t]``. The answer
        is correct under greedy decoding iff every unmasked position matches.
        """

        loss, logits, _ = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
        if prediction_loss_only or logits is None:
            return loss, None, None

        num_chosen = logits.shape[0] // 2
        completion_mask = inputs["completion_mask"][:num_chosen, 1:].to(logits.device).bool()
        preds = logits[:num_chosen, :-1].argmax(dim=-1)
        labels = inputs["input_ids"][:num_chosen, 1:].to(logits.device)
        return loss, preds, labels.masked_fill(~completion_mask, -100)
