"""Teacher-forcing token-level prediction error estimation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class UnitInput:
    unit_id: str
    unit_answer: str
    question: str = ""
    context_before: str = ""
    route: str = "direct"
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenPEMetrics:
    n_tokens: int
    mean_nll: Optional[float]
    mean_entropy: Optional[float]
    mean_confidence: Optional[float]
    low_conf_ratio: Optional[float]
    pe_token: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnitTokenPE:
    unit_id: str
    unit_answer: str
    route: str
    metrics: TokenPEMetrics
    token_logprobs: Optional[List[float]] = None
    token_confidences: Optional[List[float]] = None
    token_entropies: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pe_token(self) -> Optional[float]:
        return self.metrics.pe_token

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_answer": self.unit_answer,
            "route": self.route,
            "metrics": self.metrics.to_dict(),
            "token_logprobs": self.token_logprobs,
            "token_confidences": self.token_confidences,
            "token_entropies": self.token_entropies,
            "metadata": self.metadata,
        }


@dataclass
class TokenPEReport:
    units: List[UnitTokenPE]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_pe_token(self) -> Optional[float]:
        values = [unit.pe_token for unit in self.units if unit.pe_token is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [unit.to_dict() for unit in self.units],
            "metadata": self.metadata,
            "mean_pe_token": self.mean_pe_token,
        }


class TokenPEEstimator:
    """Estimate token-level prediction error for intermediate answer units."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        low_conf_threshold: float = 0.2,
        compute_entropy: bool = True,
        keep_token_details: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.low_conf_threshold = low_conf_threshold
        self.compute_entropy = compute_entropy
        self.keep_token_details = keep_token_details

        self.model.to(self.device)
        self.model.eval()

    def build_context(self, unit: UnitInput) -> str:
        sections: List[str] = []
        if unit.question:
            sections.append(f"Question:\n{unit.question}")
        if unit.context_before:
            sections.append(f"Previous reasoning:\n{unit.context_before}")
        if unit.evidence:
            evidence_text = "\n".join(
                f"[{index}] {item}" for index, item in enumerate(unit.evidence, start=1)
            )
            sections.append(f"Evidence:\n{evidence_text}")
        sections.append("Now generate the current step:\n ")
        return "\n\n".join(sections)

    def evaluate_unit(self, unit: UnitInput) -> UnitTokenPE:
        answer_text = unit.unit_answer.strip()
        if not answer_text:
            return self._empty_result(unit)

        context_text = self.build_context(unit)
        full_text = context_text + answer_text
        input_ids, attention_mask, answer_indices = self._tokenize(
            context_text, full_text
        )

        valid_indices = [
            index
            for index in answer_indices
            if 0 < index < input_ids.shape[1]
        ]
        if not valid_indices:
            return self._empty_result(unit)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        token_logprobs: List[float] = []
        token_confidences: List[float] = []
        token_entropies: List[float] = []

        for index in valid_indices:
            prev_logits = logits[0, index - 1]
            log_probs = torch.log_softmax(prev_logits, dim=-1)
            probs = torch.softmax(prev_logits, dim=-1)
            token_id = input_ids[0, index]

            token_logprobs.append(float(log_probs[token_id].item()))
            token_confidences.append(float(probs[token_id].item()))
            if self.compute_entropy:
                entropy = -(probs * log_probs).sum().item()
                token_entropies.append(float(entropy))

        n_tokens = len(token_logprobs)
        if n_tokens == 0:
            return self._empty_result(unit)

        mean_nll = float(-sum(token_logprobs) / n_tokens)
        mean_confidence = float(sum(token_confidences) / n_tokens)
        low_conf_ratio = float(
            sum(value < self.low_conf_threshold for value in token_confidences)
            / n_tokens
        )
        mean_entropy = (
            float(sum(token_entropies) / n_tokens)
            if self.compute_entropy
            else None
        )
        metrics = TokenPEMetrics(
            n_tokens=n_tokens,
            mean_nll=mean_nll,
            mean_entropy=mean_entropy,
            mean_confidence=mean_confidence,
            low_conf_ratio=low_conf_ratio,
            pe_token=float(1.0 - math.exp(-mean_nll)),
        )
        return UnitTokenPE(
            unit_id=unit.unit_id,
            unit_answer=unit.unit_answer,
            route=unit.route,
            metrics=metrics,
            token_logprobs=token_logprobs if self.keep_token_details else None,
            token_confidences=token_confidences if self.keep_token_details else None,
            token_entropies=(
                token_entropies
                if self.keep_token_details and self.compute_entropy
                else None
            ),
            metadata=unit.metadata,
        )

    def evaluate_units(self, units: List[UnitInput]) -> TokenPEReport:
        return TokenPEReport(units=[self.evaluate_unit(unit) for unit in units])

    def save_report(self, report: TokenPEReport, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)

    def _tokenize(
        self, context_text: str, full_text: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int]]:
        try:
            encoded = self.tokenizer(
                full_text,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            answer_start_char = len(context_text)
            answer_indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if (start, end) != (0, 0) and end > answer_start_char
            ]
            return (
                encoded["input_ids"],
                encoded.get("attention_mask"),
                answer_indices,
            )
        except (KeyError, NotImplementedError, TypeError, ValueError):
            return self._tokenize_fallback(context_text, full_text)

    def _tokenize_fallback(
        self, context_text: str, full_text: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int]]:
        context_ids = self.tokenizer.encode(
            context_text, add_special_tokens=False
        )
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        start = min(len(context_ids), input_ids.shape[1])
        answer_indices = list(range(start, input_ids.shape[1]))
        return input_ids, encoded.get("attention_mask"), answer_indices

    @staticmethod
    def _empty_result(unit: UnitInput) -> UnitTokenPE:
        return UnitTokenPE(
            unit_id=unit.unit_id,
            unit_answer=unit.unit_answer,
            route=unit.route,
            metrics=TokenPEMetrics(
                n_tokens=0,
                mean_nll=None,
                mean_entropy=None,
                mean_confidence=None,
                low_conf_ratio=None,
                pe_token=None,
            ),
            metadata=unit.metadata,
        )
