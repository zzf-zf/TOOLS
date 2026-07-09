"""Alignment and reporting for unit-level prediction error estimates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .token_pe import UnitInput

try:
    import torch
except ImportError:  # pragma: no cover - tensor conversion is optional
    torch = None  # type: ignore[assignment]


def _json_safe(value: Any) -> Any:
    """Convert common containers and numeric objects for JSON serialization."""
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
class UnitPERecord:
    unit_id: str
    route: str
    unit_answer: str
    token_pe: Optional[Dict[str, Any]] = None
    semantic_pe: Optional[Dict[str, Any]] = None
    evidence_pe: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pe_token(self) -> Optional[float]:
        if self.token_pe is None:
            return None
        return self.token_pe.get("pe_token")

    @property
    def pe_sem(self) -> Optional[float]:
        if self.semantic_pe is None:
            return None
        return self.semantic_pe.get("pe_sem")

    @property
    def pe_evid(self) -> Optional[float]:
        if self.evidence_pe is None:
            return None
        return self.evidence_pe.get("pe_evid")

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(
            {
                "unit_id": self.unit_id,
                "route": self.route,
                "unit_answer": self.unit_answer,
                "token_pe": self.token_pe,
                "semantic_pe": self.semantic_pe,
                "evidence_pe": self.evidence_pe,
                "metadata": self.metadata,
            }
        )


@dataclass
class PEAlignmentSummary:
    num_units: int
    num_token_available: int
    num_semantic_available: int
    num_evidence_available: int
    mean_pe_token: Optional[float]
    mean_pe_sem: Optional[float]
    mean_pe_evid: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PEAlignmentReport:
    units: List[UnitPERecord]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> PEAlignmentSummary:
        token_values = [
            float(unit.pe_token)
            for unit in self.units
            if unit.pe_token is not None
        ]
        semantic_values = [
            float(unit.pe_sem)
            for unit in self.units
            if unit.pe_sem is not None
        ]
        evidence_values = [
            float(unit.pe_evid)
            for unit in self.units
            if unit.pe_evid is not None
        ]
        return PEAlignmentSummary(
            num_units=int(len(self.units)),
            num_token_available=int(len(token_values)),
            num_semantic_available=int(len(semantic_values)),
            num_evidence_available=int(len(evidence_values)),
            mean_pe_token=(
                float(sum(token_values) / len(token_values))
                if token_values
                else None
            ),
            mean_pe_sem=(
                float(sum(semantic_values) / len(semantic_values))
                if semantic_values
                else None
            ),
            mean_pe_evid=(
                float(sum(evidence_values) / len(evidence_values))
                if evidence_values
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [unit.to_dict() for unit in self.units],
            "summary": self.summary.to_dict(),
            "metadata": _json_safe(self.metadata),
        }


class UnitPEAligner:
    """Align token, semantic, and evidence PE for the same answer unit."""

    def __init__(
        self,
        token_estimator: Optional[Any] = None,
        semantic_estimator: Optional[Any] = None,
        evidence_estimator: Optional[Any] = None,
        include_semantic_samples: bool = True,
        include_token_details: bool = False,
        include_evidence_details: bool = True,
    ) -> None:
        self.token_estimator = token_estimator
        self.semantic_estimator = semantic_estimator
        self.evidence_estimator = evidence_estimator
        self.include_semantic_samples = include_semantic_samples
        self.include_token_details = include_token_details
        self.include_evidence_details = include_evidence_details

    def evaluate_unit(self, unit: UnitInput) -> UnitPERecord:
        metadata = dict(unit.metadata)
        token_pe = self._evaluate_token(unit)
        semantic_pe = self._evaluate_semantic(unit)
        evidence_pe = self._evaluate_evidence(unit)
        return UnitPERecord(
            unit_id=unit.unit_id,
            route=unit.route,
            unit_answer=unit.unit_answer,
            token_pe=token_pe,
            semantic_pe=semantic_pe,
            evidence_pe=evidence_pe,
            metadata=metadata,
        )

    def evaluate_units(
        self,
        units: List[UnitInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PEAlignmentReport:
        return PEAlignmentReport(
            units=[self.evaluate_unit(unit) for unit in units],
            metadata=metadata or {},
        )

    def save_report(self, report: PEAlignmentReport, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)

    def _evaluate_token(self, unit: UnitInput) -> Dict[str, Any]:
        unavailable: Dict[str, Any] = {
            "available": False,
            "pe_token": None,
            "metrics": None,
            "metadata": {},
        }
        if self.token_estimator is None:
            return unavailable

        try:
            result = self.token_estimator.evaluate_unit(unit)
            token_pe: Dict[str, Any] = {
                "available": result.pe_token is not None,
                "pe_token": result.pe_token,
                "metrics": result.metrics.to_dict(),
                "metadata": result.metadata,
            }
            if self.include_token_details:
                token_pe.update(
                    {
                        "token_logprobs": result.token_logprobs,
                        "token_confidences": result.token_confidences,
                        "token_entropies": result.token_entropies,
                    }
                )
            return token_pe
        except Exception as error:
            unavailable["error"] = str(error)
            return unavailable

    def _evaluate_semantic(self, unit: UnitInput) -> Dict[str, Any]:
        unavailable: Dict[str, Any] = {
            "available": False,
            "pe_sem": None,
            "metrics": None,
            "metadata": {},
        }
        if self.include_semantic_samples:
            unavailable["samples"] = []
        if self.semantic_estimator is None:
            return unavailable

        try:
            result = self.semantic_estimator.evaluate_unit(unit)
            semantic_pe: Dict[str, Any] = {
                "available": result.pe_sem is not None,
                "pe_sem": result.pe_sem,
                "metrics": result.metrics.to_dict(),
                "metadata": result.metadata,
            }
            if self.include_semantic_samples:
                semantic_pe["samples"] = result.samples
            return semantic_pe
        except Exception as error:
            unavailable["samples"] = []
            unavailable["error"] = str(error)
            return unavailable

    def _evaluate_evidence(self, unit: UnitInput) -> Dict[str, Any]:
        unavailable: Dict[str, Any] = {
            "available": False,
            "pe_evid": None,
            "evidence_support_score": None,
            "evidence_status": "UNAVAILABLE",
            "num_atomic_facts": 0,
            "num_verifiable_facts": 0,
            "num_supported_facts": 0,
            "num_unsupported_facts": 0,
            "num_non_verifiable_facts": 0,
            "metadata": {},
        }
        if self.include_evidence_details:
            unavailable.update(
                {
                    "atomic_facts": [],
                    "retrieved_evidence": [],
                    "afv_results": [],
                }
            )
        if self.evidence_estimator is None:
            return unavailable

        try:
            report = self.evidence_estimator.evaluate(
                answer=unit.unit_answer,
                route=unit.route,
                metadata=unit.metadata,
                question=unit.question or None,
            )
            report_dict = report.to_dict() if hasattr(report, "to_dict") else {}
            if not isinstance(report_dict, dict):
                report_dict = {}

            pe_evid = report_dict.get("PE_evid")
            if pe_evid is None:
                pe_evid = getattr(report, "pe_evid", None)
            evidence_support_score = report_dict.get("evidence_support_score")
            if evidence_support_score is None:
                evidence_support_score = getattr(
                    report, "evidence_support_score", None
                )

            atomic_facts = report_dict.get("atomic_facts", [])
            afv_results = report_dict.get("afv_results", [])
            num_atomic_facts = int(
                report_dict.get("num_atomic_facts", len(atomic_facts))
            )
            num_verifiable_facts = int(
                report_dict.get(
                    "num_verifiable_facts",
                    sum(
                        str(item.get("validity", "VERIFIABLE")).upper()
                        == "VERIFIABLE"
                        for item in atomic_facts
                        if isinstance(item, dict)
                    ),
                )
            )
            num_supported_facts = int(
                report_dict.get(
                    "num_supported_facts",
                    sum(
                        str(item.get("label", "")).upper() == "SUPPORTED"
                        for item in afv_results
                        if isinstance(item, dict)
                    ),
                )
            )
            num_unsupported_facts = int(
                report_dict.get(
                    "num_unsupported_facts",
                    sum(
                        str(item.get("label", "")).upper()
                        in {"UNSUPPORTED", "NO_EVIDENCE"}
                        for item in afv_results
                        if isinstance(item, dict)
                    ),
                )
            )
            num_non_verifiable_facts = int(
                report_dict.get(
                    "num_non_verifiable_facts",
                    max(num_atomic_facts - num_verifiable_facts, 0),
                )
            )

            if "afv_results" not in report_dict:
                validations = getattr(report, "validations", [])
                num_atomic_facts = len(validations)
                num_supported_facts = sum(
                    bool(getattr(item, "supported", False))
                    for item in validations
                )
                num_verifiable_facts = sum(
                    bool(getattr(getattr(item, "fact", None), "verifiable", True))
                    for item in validations
                )
                num_unsupported_facts = sum(
                    not bool(getattr(item, "supported", False))
                    for item in validations
                    if bool(getattr(getattr(item, "fact", None), "verifiable", True))
                )
                num_non_verifiable_facts = max(
                    num_atomic_facts - num_verifiable_facts, 0
                )

            evidence_pe: Dict[str, Any] = {
                "available": pe_evid is not None,
                "pe_evid": pe_evid,
                "evidence_support_score": evidence_support_score,
                "evidence_status": report_dict.get(
                    "evidence_status",
                    getattr(report, "evidence_status", None),
                ),
                "num_atomic_facts": int(num_atomic_facts),
                "num_verifiable_facts": int(num_verifiable_facts),
                "num_supported_facts": int(num_supported_facts),
                "num_unsupported_facts": int(num_unsupported_facts),
                "num_non_verifiable_facts": int(num_non_verifiable_facts),
                "metadata": report_dict.get(
                    "metadata", getattr(report, "metadata", {})
                ),
            }
            if self.include_evidence_details:
                evidence_pe.update(
                    {
                        "atomic_facts": atomic_facts,
                        "retrieved_evidence": report_dict.get(
                            "retrieved_evidence", []
                        ),
                        "afv_results": afv_results,
                    }
                )
            return _json_safe(evidence_pe)
        except Exception as error:
            unavailable["error"] = str(error)
            return unavailable
