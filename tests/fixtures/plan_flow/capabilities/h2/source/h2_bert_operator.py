from __future__ import annotations

import hashlib
import json
import math

from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("demo_bert_feature_mapper")
class DemoBertFeatureMapper(Mapper):
    """Produce a compact, deterministic CPU feature receipt from a local BERT artifact."""

    def __init__(
        self,
        model_path: str = "",
        output_key: str = "bert_feature_receipt",
        max_length: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_path:
            raise ValueError("model_path must refer to a published ModelArtifact directory")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.output_key = output_key
        self.max_length = max_length
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        ).to("cpu")
        self.model.eval()

    def process_single(self, sample):
        encoded = self.tokenizer(
            str(sample[self.text_key]),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        values = [round(float(value), 6) for value in pooled[0].tolist()]
        norm = math.sqrt(sum(value * value for value in values))
        checksum = hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        sample[self.output_key] = {
            "checksum_sha256": checksum,
            "hidden_size": len(values),
            "l2_norm": round(norm, 6),
            "token_count": int(encoded["attention_mask"].sum().item()),
        }
        return sample
