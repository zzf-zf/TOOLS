from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fact_verification.afg import AtomicFactGenerator  # noqa: E402
from fact_verification.schemas import AtomicFact, FactValidation, VerificationReport  # noqa: E402


class FakeGenerator:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, prompt: str) -> str:
        return self.output


def _extract_texts(output: str, answer: str = "debug answer") -> list[str]:
    generator = AtomicFactGenerator(FakeGenerator(output))
    return [fact.text for fact in generator.extract(answer)]


def _extract(output: str, answer: str = "debug answer"):
    generator = AtomicFactGenerator(FakeGenerator(output))
    return generator.extract(answer)


def test_extracts_two_atomic_facts_from_compound_answer() -> None:
    texts = _extract_texts(
        "- 北京交通大学位于北京。\n"
        "- 北京交通大学是教育部直属高校。",
        answer="北京交通大学位于北京，是教育部直属高校。",
    )

    assert texts == [
        "北京交通大学位于北京。",
        "北京交通大学是教育部直属高校。",
    ]


def test_removes_duplicate_bullets() -> None:
    texts = _extract_texts(
        "- 北京交通大学位于北京市海淀区。\n"
        "* 北京交通大学位于北京市海淀区。\n"
        "1. 北京交通大学位于北京市海淀区"
    )

    assert texts == ["北京交通大学位于北京市海淀区。"]


def test_splits_multiple_bullets_on_one_line() -> None:
    texts = _extract_texts(
        "北京交通大学位于北京。 - 北京交通大学是教育部直属高校。 "
        "2. 北京交通大学设有多个学院。"
    )

    assert texts == [
        "北京交通大学位于北京。",
        "北京交通大学是教育部直属高校。",
        "北京交通大学设有多个学院。",
    ]


def test_filters_output_and_atomic_fact_headers() -> None:
    texts = _extract_texts(
        "###\n"
        "Output:\n"
        "Atomic Facts:\n"
        "Answer:\n"
        "- 北京交通大学位于北京市海淀区。\n"
        "-\n"
        "*\n"
        "•\n"
    )

    assert texts == ["北京交通大学位于北京市海淀区。"]


def test_normalizes_hedged_location_fact() -> None:
    facts = _extract("- 北京交通大学应该是位于北京市海淀区。")

    assert [fact.text for fact in facts] == ["北京交通大学位于北京市海淀区。"]
    assert facts[0].validity == "VERIFIABLE"


def test_splits_merged_chinese_claims() -> None:
    texts = _extract_texts("- 北京交通大学位于北京市海淀区，是教育部直属高校。")

    assert texts == [
        "北京交通大学位于北京市海淀区。",
        "北京交通大学是教育部直属高校。",
    ]


def test_marks_abstract_opinion_as_not_verifiable() -> None:
    facts = _extract("- 跨学科视角能够帮助研究者从多个层面理解人工智能系统。")

    assert len(facts) == 1
    assert facts[0].validity == "NOT_VERIFIABLE"
    assert facts[0].drop_reason == "not_verifiable_or_opinion"


def test_strips_not_verifiable_label_from_verifiable_fact() -> None:
    facts = _extract("- NOT_VERIFIABLE: 北京交通大学位于北京市海淀区。")

    assert [fact.text for fact in facts] == ["北京交通大学位于北京市海淀区。"]
    assert facts[0].validity == "VERIFIABLE"


def test_drops_model_explanation_lines() -> None:
    texts = _extract_texts(
        "- 北京交通大学位于北京市海淀区。\n"
        "- The provided answer contains a single statement that can be directly extracted as an atomic fact."
    )

    assert texts == ["北京交通大学位于北京市海淀区。"]


def test_restores_spacing_between_latin_and_chinese_tokens() -> None:
    texts = _extract_texts("- Transformer架构的核心机制包括自注意力机制。")

    assert texts == ["Transformer 架构的核心机制包括自注意力机制。"]


def test_drops_noun_phrase_fragments_without_verifiable_relation() -> None:
    texts = _extract_texts(
        "- Transformer架构的核心机制是循环神经网络结构。\n"
        "- Transformer架构的核心机制。"
    )

    assert texts == ["Transformer 架构的核心机制是循环神经网络结构。"]


def test_verification_report_excludes_not_verifiable_facts_from_score() -> None:
    supported = AtomicFact(
        text="北京交通大学位于北京市海淀区。",
        fact_id="fact-0001",
        validity="VERIFIABLE",
    )
    abstract = AtomicFact(
        text="跨学科视角能够帮助研究者理解人工智能系统。",
        fact_id="fact-0002",
        validity="NOT_VERIFIABLE",
        drop_reason="not_verifiable_or_opinion",
    )
    report = VerificationReport(
        answer="debug answer",
        validations=[
            FactValidation(fact=supported, label="SUPPORTED", evidence=[]),
            FactValidation(fact=abstract, label="NOT_VERIFIABLE", evidence=[]),
        ],
    )

    assert report.evidence_support_score == 1.0
    assert report.pe_evid == 0.0
    assert report.to_dict()["num_non_verifiable_facts"] == 1


def test_verification_report_returns_none_when_no_verifiable_facts() -> None:
    abstract = AtomicFact(
        text="跨学科视角能够帮助研究者理解人工智能系统。",
        fact_id="fact-0001",
        validity="NOT_VERIFIABLE",
        drop_reason="not_verifiable_or_opinion",
    )
    report = VerificationReport(
        answer="debug answer",
        validations=[
            FactValidation(fact=abstract, label="NOT_VERIFIABLE", evidence=[]),
        ],
    )

    assert report.evidence_status == "NO_VERIFIABLE_FACTS"
    assert report.evidence_support_score is None
    assert report.pe_evid is None
