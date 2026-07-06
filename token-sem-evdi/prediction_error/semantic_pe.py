"""Semantic-level prediction error estimation for intermediate answer units."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .token_pe import UnitInput

try:
    import torch
except ImportError:  # pragma: no cover - numpy/list embedders do not need torch
    torch = None  # type: ignore[assignment]


def _json_safe(value: Any) -> Any:
    """Convert common numeric containers to values accepted by json.dump."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


@dataclass
class SemanticPEMetrics:
    num_samples: int
    num_valid_samples: int
    mean_pairwise_similarity: Optional[float]
    semantic_dispersion: Optional[float]
    num_clusters: int
    cluster_distribution: List[int]
    cluster_entropy: Optional[float]
    original_consistency: Optional[float]
    original_semantic_error: Optional[float]
    pe_sem: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnitSemanticPE:
    unit_id: str
    unit_answer: str
    route: str
    samples: List[str]
    metrics: SemanticPEMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pe_sem(self) -> Optional[float]:
        return self.metrics.pe_sem

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_answer": self.unit_answer,
            "route": self.route,
            "samples": self.samples,
            "metrics": self.metrics.to_dict(),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class SemanticPEReport:
    units: List[UnitSemanticPE]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_pe_sem(self) -> Optional[float]:
        values = [unit.pe_sem for unit in self.units if unit.pe_sem is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [unit.to_dict() for unit in self.units],
            "metadata": _json_safe(self.metadata),
            "mean_pe_sem": self.mean_pe_sem,
        }


class SemanticPEEstimator:
    """Estimate semantic uncertainty from repeated answer-unit samples."""

    def __init__(
        self,
        generator: Any,
        embedder: Any,
        num_samples: int = 10,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        similarity_threshold: float = 0.85,
        entropy_weight: float = 0.6,
        dispersion_weight: float = 0.4,
        original_error_weight: float = 0.0,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative")
        if dispersion_weight < 0:
            raise ValueError("dispersion_weight must be non-negative")
        if original_error_weight < 0:
            raise ValueError("original_error_weight must be non-negative")

        self.generator = generator
        self.embedder = embedder
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.similarity_threshold = similarity_threshold
        self.entropy_weight = entropy_weight
        self.dispersion_weight = dispersion_weight
        self.original_error_weight = original_error_weight

    def build_context(self, unit: UnitInput) -> str:
        if unit.context_text is not None:
            if unit.context_text.endswith((" ", "\n")):
                return unit.context_text
            return unit.context_text + " "

        sections: List[str] = []
        if unit.question:
            sections.append(f"Question:\n{unit.question}")
        if unit.context_before:
            sections.append(f"Previous reasoning:\n{unit.context_before}")
        if unit.evidence:
            evidence_text = "\n".join(
                f"[{index}] {item}"
                for index, item in enumerate(unit.evidence, start=1)
            )
            sections.append(f"Evidence:\n{evidence_text}")
        sections.append("Now generate the current step:\n ")
        return "\n\n".join(sections)

    def evaluate_unit(self, unit: UnitInput) -> UnitSemanticPE:
        context_text = self.build_context(unit)
        samples = self._sample_texts(context_text, self.num_samples)
        metadata = self._unit_metadata(unit)

        if len(samples) < 2:
            metadata["insufficient_semantic_samples"] = True
            original_consistency, original_error = (
                self._original_metrics(unit.unit_answer, samples)
                if samples
                else (None, None)
            )
            num_valid_samples = len(samples)
            return UnitSemanticPE(
                unit_id=unit.unit_id,
                unit_answer=unit.unit_answer,
                route=unit.route,
                samples=samples,
                metrics=SemanticPEMetrics(
                    num_samples=int(self.num_samples),
                    num_valid_samples=int(num_valid_samples),
                    mean_pairwise_similarity=None,
                    semantic_dispersion=None,
                    num_clusters=int(num_valid_samples),
                    cluster_distribution=(
                        [int(num_valid_samples)] if samples else []
                    ),
                    cluster_entropy=None,
                    original_consistency=original_consistency,
                    original_semantic_error=original_error,
                    pe_sem=None,
                ),
                metadata=metadata,
            )

        metadata["insufficient_semantic_samples"] = False
        sample_embeddings = self._embed_texts(samples)
        if sample_embeddings.shape[0] != len(samples):
            raise ValueError(
                "Embedder returned a different number of embeddings than texts."
            )

        mean_similarity, dispersion = self._pairwise_metrics(sample_embeddings)
        cluster_distribution = self._cluster_embeddings(sample_embeddings)
        num_clusters = len(cluster_distribution)
        cluster_entropy = self._cluster_entropy(
            cluster_distribution, len(samples)
        )
        original_consistency, original_error = self._original_metrics(
            unit.unit_answer, samples
        )
        pe_sem = self._combine_metrics(
            cluster_entropy, dispersion, original_error
        )

        metrics = SemanticPEMetrics(
            num_samples=int(self.num_samples),
            num_valid_samples=int(len(samples)),
            mean_pairwise_similarity=mean_similarity,
            semantic_dispersion=dispersion,
            num_clusters=int(num_clusters),
            cluster_distribution=[int(size) for size in cluster_distribution],
            cluster_entropy=cluster_entropy,
            original_consistency=original_consistency,
            original_semantic_error=original_error,
            pe_sem=pe_sem,
        )
        return UnitSemanticPE(
            unit_id=unit.unit_id,
            unit_answer=unit.unit_answer,
            route=unit.route,
            samples=samples,
            metrics=metrics,
            metadata=metadata,
        )

    def evaluate_units(
        self,
        units: List[UnitInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticPEReport:
        return SemanticPEReport(
            units=[self.evaluate_unit(unit) for unit in units],
            metadata=metadata or {},
        )

    def save_report(self, report: SemanticPEReport, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)

    def _sample_texts(self, prompt: str, num_samples: int) -> List[str]:
        raw_outputs: List[Any] = []

        if hasattr(self.generator, "sample"):
            try:
                result = self.generator.sample(
                    prompt=prompt,
                    num_samples=num_samples,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            except TypeError:
                try:
                    result = self.generator.sample(prompt, num_samples)
                except TypeError:
                    result = self.generator.sample(prompt)
            raw_outputs.append(result)
        elif hasattr(self.generator, "generate"):
            for _ in range(num_samples):
                raw_outputs.append(self.generator.generate(prompt))
        elif callable(self.generator):
            for _ in range(num_samples):
                raw_outputs.append(self.generator(prompt))
        else:
            raise TypeError(
                "generator must provide sample(), generate(), or be callable"
            )

        samples: List[str] = []
        for output in raw_outputs:
            if isinstance(output, tuple):
                output = output[0] if output else None
            candidates = output if isinstance(output, list) else [output]
            for candidate in candidates:
                if isinstance(candidate, str):
                    text = candidate.strip()
                    if text:
                        samples.append(text)
        return samples[:num_samples]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float64)

        if hasattr(self.embedder, "encode"):
            try:
                embeddings = self.embedder.encode(
                    texts, normalize_embeddings=True
                )
            except TypeError:
                embeddings = self.embedder.encode(texts)
        elif callable(self.embedder):
            embeddings = self.embedder(texts)
        else:
            raise TypeError("embedder must provide encode() or be callable")

        if torch is not None and isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        array = np.asarray(embeddings, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Embeddings must be a one- or two-dimensional array.")

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return np.divide(
            array,
            norms,
            out=np.zeros_like(array, dtype=np.float64),
            where=norms > 0,
        )

    @staticmethod
    def _pairwise_metrics(
        embeddings: np.ndarray,
    ) -> tuple[Optional[float], Optional[float]]:
        if embeddings.shape[0] < 2:
            return None, None
        similarity_matrix = embeddings @ embeddings.T
        row_indices, column_indices = np.triu_indices(
            embeddings.shape[0], k=1
        )
        similarities = np.clip(
            similarity_matrix[row_indices, column_indices], 0.0, 1.0
        )
        mean_similarity = float(np.mean(similarities))
        return mean_similarity, float(1.0 - mean_similarity)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> List[int]:
        centroids: List[np.ndarray] = []
        cluster_sums: List[np.ndarray] = []
        cluster_sizes: List[int] = []

        for embedding in embeddings:
            cluster_index: Optional[int] = None
            for index, centroid in enumerate(centroids):
                similarity = float(np.dot(embedding, centroid))
                if similarity >= self.similarity_threshold:
                    cluster_index = index
                    break

            if cluster_index is None:
                centroids.append(embedding.copy())
                cluster_sums.append(embedding.copy())
                cluster_sizes.append(1)
                continue

            cluster_sums[cluster_index] += embedding
            cluster_sizes[cluster_index] += 1
            centroid = cluster_sums[cluster_index]
            norm = float(np.linalg.norm(centroid))
            centroids[cluster_index] = centroid / norm if norm > 0 else centroid

        return cluster_sizes

    @staticmethod
    def _cluster_entropy(
        cluster_distribution: List[int], num_valid_samples: int
    ) -> float:
        num_clusters = len(cluster_distribution)
        if num_clusters <= 1:
            return 0.0
        probabilities = np.asarray(cluster_distribution, dtype=np.float64)
        probabilities /= num_valid_samples
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        return float(entropy / math.log(num_clusters))

    def _original_metrics(
        self, unit_answer: str, samples: List[str]
    ) -> tuple[Optional[float], Optional[float]]:
        original = unit_answer.strip()
        if not original:
            return None, None
        embeddings = self._embed_texts([original, *samples])
        if embeddings.shape[0] != len(samples) + 1:
            raise ValueError(
                "Embedder returned a different number of embeddings than texts."
            )
        similarities = np.clip(embeddings[1:] @ embeddings[0], 0.0, 1.0)
        consistency = float(np.mean(similarities))
        return consistency, float(1.0 - consistency)

    def _combine_metrics(
        self,
        cluster_entropy: Optional[float],
        semantic_dispersion: Optional[float],
        original_semantic_error: Optional[float],
    ) -> Optional[float]:
        weighted_values: List[tuple[float, float]] = []
        if cluster_entropy is not None and self.entropy_weight != 0:
            weighted_values.append((self.entropy_weight, cluster_entropy))
        if semantic_dispersion is not None and self.dispersion_weight != 0:
            weighted_values.append((self.dispersion_weight, semantic_dispersion))
        if (
            original_semantic_error is not None
            and self.original_error_weight > 0
        ):
            weighted_values.append(
                (self.original_error_weight, original_semantic_error)
            )
        weight_sum = sum(weight for weight, _ in weighted_values)
        if not weighted_values or weight_sum == 0:
            return None
        value = sum(weight * metric for weight, metric in weighted_values)
        return float(np.clip(value / weight_sum, 0.0, 1.0))

    def _unit_metadata(self, unit: UnitInput) -> Dict[str, Any]:
        metadata = dict(unit.metadata)
        metadata.update(
            {
                "requested_num_samples": int(self.num_samples),
                "max_new_tokens": int(self.max_new_tokens),
                "temperature": float(self.temperature),
                "top_p": float(self.top_p),
                "similarity_threshold": float(self.similarity_threshold),
                "entropy_weight": float(self.entropy_weight),
                "dispersion_weight": float(self.dispersion_weight),
                "original_error_weight": float(self.original_error_weight),
            }
        )
        return metadata
