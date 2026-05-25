"""
Automated evaluation harness for the F1 Strategy Intelligence pipeline.
Run from the project root:  python eval/auto_eval.py
"""
import sys
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

from dataclasses import dataclass  # noqa: E402
from typing import List  # noqa: E402
from graph import run_pipeline  # noqa: E402

CONFIDENCE_THRESHOLD = 0.6
PASS_KEYWORD_THRESHOLD = 0.5   # at least 50% of expected keywords must match


@dataclass
class TestCase:
    question: str
    session: str                        # FP1 | Sprint | Race
    expected_keywords: List[str]


@dataclass
class EvalResult:
    question: str
    session: str
    expected_keywords: List[str]
    keyword_match: float                # 0.0–1.0
    matched_keywords: List[str]
    missing_keywords: List[str]
    has_evidence: bool
    has_citations: bool
    confidence_above_threshold: bool
    confidence_score: float
    response_time: float                # seconds
    passed: bool
    error: str = ""


# NOTE: Driver-specific keywords (Antonelli, Russell etc) and incident
# keywords (red flag, Colapinto) are not testable until Jolpica
# populates post-session results, typically 24-48hrs after each session.
# These queries test what the system is designed for: regulation
# reasoning and strategy synthesis.
TEST_CASES: List[TestCase] = [
    TestCase(
        question="What does the 2026 technical regulation say about tyre compounds?",
        session="Regulation",
        expected_keywords=["compound", "tyre", "regulation"],
    ),
    TestCase(
        question="What alternative strategies exist for a one-stop race at Montreal?",
        session="Strategy",
        expected_keywords=["medium", "hard", "pit stop"],
    ),
    TestCase(
        question="What tyre strategy does FP1 data suggest for the race?",
        session="FP1",
        expected_keywords=["medium", "compound", "stint"],
    ),
    TestCase(
        question="What does Leclerc need to do strategically at Montreal to close the championship gap?",
        session="Strategy",
        expected_keywords=["Ferrari", "strategy", "compound"],
    ),
    TestCase(
        question="How do the 2026 regulations affect pit stop strategy at Montreal?",
        session="Regulation",
        expected_keywords=["regulation", "tyre", "pit"],
    ),
]


def _keyword_hit(keyword: str, text: str) -> bool:
    return keyword.lower() in text.lower()


def evaluate(tc: TestCase) -> EvalResult:
    t0 = time.time()
    error = ""
    try:
        result = run_pipeline(tc.question)
    except Exception as exc:
        elapsed = time.time() - t0
        return EvalResult(
            question=tc.question,
            session=tc.session,
            expected_keywords=tc.expected_keywords,
            keyword_match=0.0,
            matched_keywords=[],
            missing_keywords=tc.expected_keywords,
            has_evidence=False,
            has_citations=False,
            confidence_above_threshold=False,
            confidence_score=0.0,
            response_time=elapsed,
            passed=False,
            error=str(exc),
        )
    elapsed = time.time() - t0

    if result is None:
        return EvalResult(
            question=tc.question,
            session=tc.session,
            expected_keywords=tc.expected_keywords,
            keyword_match=0.0,
            matched_keywords=[],
            missing_keywords=tc.expected_keywords,
            has_evidence=False,
            has_citations=False,
            confidence_above_threshold=False,
            confidence_score=0.0,
            response_time=elapsed,
            passed=False,
            error="run_pipeline returned None",
        )

    # Build the full text blob to search for keywords
    full_text = " ".join([
        result.recommendation,
        result.reasoning,
        " ".join(result.supporting_evidence),
        " ".join(result.alternative_strategies),
    ])

    matched = [kw for kw in tc.expected_keywords if _keyword_hit(kw, full_text)]
    missing = [kw for kw in tc.expected_keywords if not _keyword_hit(kw, full_text)]
    keyword_match = len(matched) / len(tc.expected_keywords) if tc.expected_keywords else 1.0

    has_evidence = len(result.supporting_evidence) > 0
    has_citations = len(result.citations) > 0
    confidence_ok = result.confidence_score > CONFIDENCE_THRESHOLD

    passed = (
        keyword_match >= PASS_KEYWORD_THRESHOLD
        and has_evidence
        and confidence_ok
    )

    return EvalResult(
        question=tc.question,
        session=tc.session,
        expected_keywords=tc.expected_keywords,
        keyword_match=keyword_match,
        matched_keywords=matched,
        missing_keywords=missing,
        has_evidence=has_evidence,
        has_citations=has_citations,
        confidence_above_threshold=confidence_ok,
        confidence_score=result.confidence_score,
        response_time=elapsed,
        passed=passed,
        error=error,
    )


def _tick(value: bool) -> str:
    return "✅" if value else "❌"


def write_results_md(results: List[EvalResult], path: str) -> None:
    lines = [
        "# F1 Strategy Intelligence — Eval Results",
        "",
        "| Query | Session | Keyword Match | Evidence | Citations | Confidence | Response Time | Pass/Fail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        short_q = r.question[:55] + "…" if len(r.question) > 55 else r.question
        km = f"{r.keyword_match:.0%} ({', '.join(r.matched_keywords) or '—'})"
        lines.append(
            f"| {short_q} "
            f"| {r.session} "
            f"| {km} "
            f"| {_tick(r.has_evidence)} "
            f"| {_tick(r.has_citations)} "
            f"| {r.confidence_score:.2f} {_tick(r.confidence_above_threshold)} "
            f"| {r.response_time:.1f}s "
            f"| {_tick(r.passed)} |"
        )

    lines += ["", "## Missing Keywords Detail", ""]
    for r in results:
        if r.missing_keywords:
            lines.append(f"- **{r.question[:70]}**: missing `{'`, `'.join(r.missing_keywords)}`")
        if r.error:
            lines.append(f"- **{r.question[:70]}**: ERROR — {r.error}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    print(f"Running {len(TEST_CASES)} eval queries…\n")

    results: List[EvalResult] = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {tc.session} — {tc.question[:70]}")
        r = evaluate(tc)
        status = "PASS" if r.passed else "FAIL"
        print(f"  → {status}  keyword_match={r.keyword_match:.0%}  "
              f"confidence={r.confidence_score:.2f}  time={r.response_time:.1f}s")
        if r.missing_keywords:
            print(f"  missing keywords: {r.missing_keywords}")
        if r.error:
            print(f"  error: {r.error}")
        results.append(r)

    out_path = "eval/results.md"
    write_results_md(results, out_path)
    print(f"\nResults written to {out_path}")

    passed = sum(1 for r in results if r.passed)
    avg_confidence = sum(r.confidence_score for r in results) / len(results)
    avg_time = sum(r.response_time for r in results) / len(results)

    print(f"\n{'='*40}")
    print(f"  {passed}/{len(results)} queries passed")
    print(f"  Average confidence : {avg_confidence:.2f}")
    print(f"  Average response   : {avg_time:.1f}s")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
