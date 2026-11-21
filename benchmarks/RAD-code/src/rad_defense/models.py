from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ContrieverEncoder:
    """Masked-mean Contriever encoder used for dense retrieval."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        # The pinned Contriever revision is distributed as a PyTorch checkpoint rather
        # than safetensors. Let Transformers select the available weight format.
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model.to(self.device).eval()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                output = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0)


class TransformersLanguageModel:
    """Deterministic Hugging Face backend for generation and likelihood scoring."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        max_new_tokens: int = 1024,
        device_map: str = "auto",
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map=device_map,
            torch_dtype="auto",
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model.eval()
        self._torch = torch

    @property
    def input_device(self):
        return self.model.get_input_embeddings().weight.device

    def generate(self, prompts: Sequence[str], *, system_prompt: str | None = None) -> list[str]:
        messages = [
            [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in messages
        ]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.input_device)
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prefix_length = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(outputs[:, prefix_length:], skip_special_tokens=True)

    def conditional_log_likelihood(
        self, context: str, continuation: str, *, length_normalize: bool = False
    ) -> float:
        context_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=True).input_ids
        continuation_ids = self.tokenizer(
            continuation, return_tensors="pt", add_special_tokens=False
        ).input_ids
        input_ids = self._torch.cat([context_ids, continuation_ids], dim=1).to(self.input_device)
        with self._torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[:, :-1]
        targets = input_ids[:, 1:]
        token_log_probs = self._torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        start = context_ids.shape[1] - 1
        selected = token_log_probs[:, start : start + continuation_ids.shape[1]]
        score = selected.sum()
        if length_normalize and selected.numel():
            score = score / selected.numel()
        return float(score.item())

    def harmful_probabilities(self, prompts: Sequence[str]) -> list[float]:
        import torch.nn.functional as functional

        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in messages
        ]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.input_device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[:, -1]

        token_groups: dict[int, int] = {}
        for group, variants in enumerate((("Bad", "bad"), ("Good", "good"))):
            for value in variants:
                ids = self.tokenizer.encode(value, add_special_tokens=False)
                if len(ids) == 1:
                    token_groups.setdefault(ids[0], group)
        if set(token_groups.values()) != {0, 1}:
            raise ValueError("model tokenizer must represent at least one Bad and Good label as one token")
        ids = list(token_groups)
        probabilities = functional.softmax(logits[:, ids], dim=-1)
        return [
            float(sum(row[i].item() for i, token_id in enumerate(ids) if token_groups[token_id] == 0))
            for row in probabilities
        ]
