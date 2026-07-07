#!/usr/bin/env python3
"""Run the unit-level TriPE alignment pipeline on the debug dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
EXAMPLES_DIR = SCRIPT_PATH.parent
PROJECT_DIR = EXAMPLES_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from prediction_error import (  # noqa: E402
    SemanticPEEstimator,
    TokenPEEstimator,
    UnitInput,
    UnitPEAligner,
)


REQUIRED_UNIT_FIELDS = {
    "unit_id",
    "question",
    "context_text",
    "unit_answer",
    "route",
    "expected_pattern",
    "primary_test",
    "note",
}


def load_debug_units(path: Path) -> List[Dict[str, Any]]:
    """Load and validate unit-level JSONL records."""
    units: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        missing = REQUIRED_UNIT_FIELDS.difference(item)
        if missing:
            raise ValueError(
                f"{path}:{line_number} missing fields: {sorted(missing)}"
            )
        if item["route"] not in {"direct", "retrieve"}:
            raise ValueError(
                f"{path}:{line_number} has invalid route: {item['route']!r}"
            )
        units.append(item)
    return units


def to_unit_input(item: Dict[str, Any]) -> UnitInput:
    """Convert one debug record to the shared UnitInput structure."""
    return UnitInput(
        unit_id=item["unit_id"],
        question=item["question"],
        context_text=item["context_text"],
        unit_answer=item["unit_answer"],
        route=item["route"],
        metadata={
            "primary_test": item["primary_test"],
            "expected_pattern": item["expected_pattern"],
            "note": item["note"],
        },
    )


def load_evidence_corpus(path: Path) -> List[Dict[str, Any]]:
    """Load and validate the small JSON evidence corpus."""
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(corpus, list):
        raise ValueError(f"{path} must contain a JSON list")
    for index, passage in enumerate(corpus):
        if not isinstance(passage, dict):
            raise ValueError(f"{path}: passage {index} must be an object")
        missing = {"title", "text", "metadata"}.difference(passage)
        if missing:
            raise ValueError(
                f"{path}: passage {index} missing fields: {sorted(missing)}"
            )
    return corpus


@dataclass
class _DictMetrics:
    values: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.values)


class FakeTokenEstimator:
    """Deterministic token PE fixture for alignment debugging."""

    VALUES = {
        "stable_correct": 0.15,
        "confident_hallucination": 0.18,
        "hedged_but_correct": 0.55,
        "semantic_instability": 0.35,
        "weakly_verifiable_explanation": 0.40,
        "retrieval_supported_fact": 0.20,
        "retrieval_contradicted_or_unsupported_fact": 0.22,
        "non_atomic_or_abstract_claim": 0.45,
    }

    def evaluate_unit(self, unit: UnitInput) -> Any:
        primary_test = unit.metadata.get("primary_test", "")
        pe_token = float(self.VALUES.get(primary_test, 0.3))
        mean_nll = float(-math.log(max(1.0 - pe_token, 1e-12)))
        metrics = _DictMetrics(
            {
                "n_tokens": 10,
                "mean_nll": mean_nll,
                "mean_entropy": float(0.5 + pe_token),
                "mean_confidence": float(1.0 - pe_token),
                "low_conf_ratio": float(pe_token / 2.0),
                "pe_token": pe_token,
            }
        )
        return SimpleNamespace(
            pe_token=pe_token,
            metrics=metrics,
            metadata={"tokenization_method": "mock"},
            token_logprobs=None,
            token_confidences=None,
            token_entropies=None,
        )


class FakeSemanticEstimator:
    """Deterministic semantic PE fixture for alignment debugging."""

    VALUES = {
        "stable_correct": 0.05,
        "confident_hallucination": 0.08,
        "hedged_but_correct": 0.10,
        "semantic_instability": 0.85,
        "weakly_verifiable_explanation": 0.45,
        "retrieval_supported_fact": 0.10,
        "retrieval_contradicted_or_unsupported_fact": 0.12,
        "non_atomic_or_abstract_claim": 0.50,
    }

    def evaluate_unit(self, unit: UnitInput) -> Any:
        primary_test = unit.metadata.get("primary_test", "")
        pe_sem = float(self.VALUES.get(primary_test, 0.3))
        has_multiple_clusters = pe_sem >= 0.4
        distribution = [2, 2] if has_multiple_clusters else [4]
        metrics = _DictMetrics(
            {
                "num_samples": 4,
                "num_valid_samples": 4,
                "mean_pairwise_similarity": float(1.0 - pe_sem),
                "semantic_dispersion": pe_sem,
                "num_clusters": len(distribution),
                "cluster_distribution": distribution,
                "cluster_entropy": float(pe_sem if has_multiple_clusters else 0.0),
                "original_consistency": float(1.0 - pe_sem),
                "original_semantic_error": pe_sem,
                "pe_sem": pe_sem,
            }
        )
        return SimpleNamespace(
            pe_sem=pe_sem,
            metrics=metrics,
            samples=[
                unit.unit_answer,
                unit.unit_answer,
                "mock semantic sample",
            ],
            metadata={
                "requested_num_samples": 4,
                "insufficient_semantic_samples": False,
            },
        )


class _FakeEvidenceReport:
    def __init__(
        self,
        answer: str,
        question: Optional[str],
        route: Optional[str],
        metadata: Dict[str, Any],
        support_score: Optional[float],
        pe_evid: Optional[float],
        labels: List[str],
    ) -> None:
        self.answer = answer
        self.question = question
        self.route = route
        self.metadata = metadata
        self.support_score = support_score
        self.pe_evid = pe_evid
        self.labels = labels

    def to_dict(self) -> Dict[str, Any]:
        atomic_facts = [
            {
                "fact_id": f"fact-{index:04d}",
                "text": self.answer,
                "source_sentence": None,
            }
            for index, _ in enumerate(self.labels, start=1)
        ]
        afv_results = [
            {
                "fact_id": f"fact-{index:04d}",
                "label": label,
                "confidence": 1.0,
                "raw_output": label,
            }
            for index, label in enumerate(self.labels, start=1)
        ]
        return {
            "question": self.question,
            "answer": self.answer,
            "atomic_facts": atomic_facts,
            "retrieved_evidence": [],
            "afv_results": afv_results,
            "evidence_support_score": self.support_score,
            "PE_evid": self.pe_evid,
            "route": self.route,
            "metadata": self.metadata,
        }


class FakeEvidenceEstimator:
    """Deterministic evidence PE fixture for alignment debugging."""

    VALUES = {
        "stable_correct": (1.0, 0.0, ["SUPPORTED"]),
        "confident_hallucination": (0.0, 1.0, ["UNSUPPORTED"]),
        "hedged_but_correct": (1.0, 0.0, ["SUPPORTED"]),
        "semantic_instability": (None, None, []),
        "weakly_verifiable_explanation": (
            0.5,
            0.5,
            ["SUPPORTED", "UNSUPPORTED"],
        ),
        "retrieval_supported_fact": (1.0, 0.0, ["SUPPORTED"]),
        "retrieval_contradicted_or_unsupported_fact": (
            0.0,
            1.0,
            ["UNSUPPORTED"],
        ),
        "non_atomic_or_abstract_claim": (None, None, []),
    }

    def evaluate(
        self,
        answer: str,
        route: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> _FakeEvidenceReport:
        report_metadata = metadata or {}
        primary_test = report_metadata.get("primary_test", "")
        support_score, pe_evid, labels = self.VALUES.get(
            primary_test, (None, None, [])
        )
        return _FakeEvidenceReport(
            answer=answer,
            question=question,
            route=route,
            metadata=report_metadata,
            support_score=support_score,
            pe_evid=pe_evid,
            labels=labels,
        )


class HuggingFaceGeneratorWrapper:
    """Expose one causal LM through the AFG/AFV and semantic APIs."""

    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def _encode(self, prompt: str) -> Tuple[Dict[str, Any], int]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_length = int(encoded["input_ids"].shape[1])
        try:
            device = next(self.model.parameters()).device
            if device.type != "meta":
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
        except (AttributeError, StopIteration, RuntimeError, TypeError):
            pass
        return encoded, input_length

    def _pad_token_id(self) -> Optional[int]:
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return None

    def _postprocess_generated_text(self, text: str) -> str:
        """Keep a generated continuation close to one current-step answer."""
        original = text.strip()
        cleaned = original
        for marker in ("Answer:", "答案："):
            if marker in cleaned:
                cleaned = cleaned.rsplit(marker, 1)[-1].strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0].strip()

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            cleaned = lines[0]

        cleaned = re.sub(
            r"^(?:[-*]\s+|\d+[.)、]\s*|Step\s*\d+\s*[:：]\s*|步骤\s*\d+\s*[:：]\s*)+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        if len(cleaned) > 80:
            match = re.search(r"[。？！.]", cleaned)
            if match is not None:
                cleaned = cleaned[: match.end()].strip()

        return cleaned or original

    def generate(self, prompt: str) -> str:
        # Raw deterministic generation for AFG/AFV; keep bullets and numbering.
        encoded, input_length = self._encode(prompt)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._pad_token_id(),
        )
        decoded = self.tokenizer.decode(
            output_ids[0, input_length:], skip_special_tokens=True
        )
        return decoded.strip()

    def sample(
        self,
        prompt: str,
        num_samples: int,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        # Constrained and postprocessed stochastic generation for semantic PE.
        constrained_prompt = (
            "请只生成当前步骤的一句话答案。"
            "不要编号，不要解释，不要写代码，不要输出 Answer，不要扩展背景。"
            "只输出答案本身。\n\n"
            + prompt
        )
        encoded, input_length = self._encode(constrained_prompt)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            pad_token_id=self._pad_token_id(),
        )
        return [
            self._postprocess_generated_text(
                self.tokenizer.decode(
                    sequence[input_length:], skip_special_tokens=True
                )
            ).strip()
            for sequence in output_ids
        ]


def build_real_estimators(
    args: argparse.Namespace,
    evidence_corpus: List[Dict[str, Any]],
) -> Tuple[Any, Any, Any]:
    """Load real model-backed estimators, only when explicitly requested."""
    if not args.model_path:
        raise ValueError("--model-path is required in real mode")
    if not args.embedder_path:
        raise ValueError("--embedder-path is required in real mode")

    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "real mode requires transformers and sentence-transformers"
        ) from error

    from fact_verification.afg import AtomicFactGenerator
    from fact_verification.afv import AtomicFactValidator
    from fact_verification.pipeline import FactVerificationPipeline
    from fact_verification.retriever import InMemoryRetriever

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    generator = HuggingFaceGeneratorWrapper(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    token_estimator = TokenPEEstimator(
        model=model,
        tokenizer=tokenizer,
        keep_token_details=args.include_token_details,
    )
    embedder = SentenceTransformer(args.embedder_path)
    semantic_estimator = SemanticPEEstimator(
        generator=generator,
        embedder=embedder,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        similarity_threshold=args.similarity_threshold,
    )
    evidence_estimator = FactVerificationPipeline(
        afg=AtomicFactGenerator(generator),
        afv=AtomicFactValidator(generator),
        retriever=InMemoryRetriever(evidence_corpus),
        top_k=args.top_k,
    )
    return token_estimator, semantic_estimator, evidence_estimator


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary_table(report: Any) -> None:
    """Print one compact line per aligned unit."""
    headers = [
        "unit_id",
        "primary_test",
        "pe_token",
        "pe_sem",
        "pe_evid",
        "expected",
        "evid_support",
        "facts",
    ]
    rows: List[List[str]] = []
    for record in report.units:
        expected_pattern = record.metadata.get("expected_pattern", {})
        expected = (
            f"T:{expected_pattern.get('pe_token', '-')}, "
            f"S:{expected_pattern.get('pe_sem', '-')}, "
            f"E:{expected_pattern.get('pe_evid', '-')}"
        )
        evidence_pe = record.evidence_pe or {}
        rows.append(
            [
                record.unit_id,
                str(record.metadata.get("primary_test", "-")),
                _format_value(record.pe_token),
                _format_value(record.pe_sem),
                _format_value(record.pe_evid),
                expected,
                _format_value(evidence_pe.get("evidence_support_score")),
                (
                    f"{evidence_pe.get('num_supported_facts', 0)}/"
                    f"{evidence_pe.get('num_atomic_facts', 0)}"
                ),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(
            " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )
    print("\nSummary:", json.dumps(report.summary.to_dict(), ensure_ascii=False))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unit-level TriPE alignment on the debug dataset."
    )
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument(
        "--units", default=str(EXAMPLES_DIR / "debug_units.jsonl")
    )
    parser.add_argument(
        "--evidence-corpus",
        default=str(EXAMPLES_DIR / "debug_evidence_corpus.json"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--embedder-path", default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens; for real-mode debug sampling, 48 or 64 is usually cleaner.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-token-details", action="store_true")
    parser.add_argument("--no-semantic-samples", action="store_true")
    parser.add_argument("--no-evidence-details", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    units_path = Path(args.units)
    corpus_path = Path(args.evidence_corpus)
    raw_units = load_debug_units(units_path)
    units = [to_unit_input(item) for item in raw_units]
    evidence_corpus = load_evidence_corpus(corpus_path)

    if args.mode == "mock":
        estimators = (
            FakeTokenEstimator(),
            FakeSemanticEstimator(),
            FakeEvidenceEstimator(),
        )
    else:
        estimators = build_real_estimators(args, evidence_corpus)

    aligner = UnitPEAligner(
        token_estimator=estimators[0],
        semantic_estimator=estimators[1],
        evidence_estimator=estimators[2],
        include_semantic_samples=not args.no_semantic_samples,
        include_token_details=args.include_token_details,
        include_evidence_details=not args.no_evidence_details,
    )
    report = aligner.evaluate_units(
        units,
        metadata={
            "mode": args.mode,
            "units_path": str(units_path.resolve()),
            "evidence_corpus_path": str(corpus_path.resolve()),
            "model_path": args.model_path,
            "embedder_path": args.embedder_path,
        },
    )
    default_name = f"debug_unit_pe_alignment_{args.mode}.json"
    output_path = Path(args.output) if args.output else RESULTS_DIR / default_name
    aligner.save_report(report, str(output_path))

    print_summary_table(report)
    print(f"\nSaved report: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
