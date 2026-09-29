"""
Extracts a structured analysis (DocumentAnalysis) from feedback documents
— via the Claude API, or, while there's no key yet, via mock mode, which
assembles a plausible response from ground_truth.json.

Usage:
    python extractor.py                 # process all documents
    python extractor.py --force         # ignore the cache, recompute everything
    python extractor.py --doc C001_survey_2026-09-01.txt   # a single file only
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from config import EXTRACTED_DIR, GROUND_TRUTH_JSON, MODEL, USE_MOCK
from loader import load_documents
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import DocumentAnalysis

# ground_truth.json is read once at module import time — it's small and
# doesn't change while the script runs.
_GROUND_TRUTH: dict = json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Mock mode: a plausible DocumentAnalysis without calling the Claude API.
# The logic below is deliberately simple (regexes and keywords, not real
# text understanding) — the goal isn't accuracy, it's letting the rest of
# the pipeline (schema, cache, CLI) be built and tested without an API
# key. Quotes are a rough approximation, not guaranteed to match what a
# real analysis would pick.
# ---------------------------------------------------------------------------

_CSAT_PATTERN = re.compile(r"CSAT \(1-5\):\s*(\d)")

_SENTIMENT_BASE_SCORE = {
    "positive": 0.7,
    "neutral": 0.0,
    "mixed": -0.05,
    "negative": -0.7,
}

_CATEGORY_KEYWORDS = {
    "onboarding": ["onboarding", "setup", "new staff", "new team", "training"],
    "bugs_performance": [
        "bug", "slow", "load", "outage", "crash", "logs out", "log out",
        "drops", "downtime", "session",
    ],
    "usability": ["confusing", "unclear", "not obvious", "hard to", "opaque"],
    "missing_feature": ["schedule", "export", "dark mode", "audit log", "summary"],
    "integration": [
        "integration", "sync", "sso", "okta", "azure ad", "google workspace",
        "wms", "api",
    ],
    "pricing": ["pricing", "price", "invoice", "billing", "cost"],
    "support": ["support", "response time", "ticket", "helpdesk"],
}

_PRAISE_KEYWORDS = ["thank", "appreciate", "great", "smooth", "helpful", "happy", "love"]
_CHURN_KEYWORDS = ["competitor", "evaluat", "cancel", "renew", "leav", "alternative"]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitting — good enough for mock quotes. First
    split on lines (each dialogue turn / email line already starts on
    its own line anyway), then split each line on punctuation too —
    otherwise multi-line paragraphs would glue together into one "quote"
    with line breaks baked into it."""
    fragments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for piece in re.split(r"(?<=[.!?])\s+", line):
            piece = piece.strip()
            if piece:
                fragments.append(piece)
    return fragments


def _find_quote(text: str, keywords: list[str]) -> str:
    """Looks for a fragment containing one of the keywords; if nothing
    matches, falls back to the first sufficiently substantial fragment
    (skipping short boilerplate lines like "Date: 2026-09-10")."""
    sentences = _split_sentences(text)
    substantial = [s for s in sentences if len(s) >= 20]
    pool = substantial or sentences
    keywords_lower = [k.lower() for k in keywords]

    for sentence in pool:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords_lower):
            return sentence[:160]

    return pool[0][:160] if pool else text[:160]


def _normalize_sentiment(raw: str) -> str:
    """ground_truth.json stores sentiment as free text (e.g. 'superficially
    positive / underlying negative (silent decline)') — normalize it down
    to one of the 4 values our schema understands."""
    lowered = raw.lower()
    if "negative" in lowered and "positive" not in lowered:
        return "negative"
    if "mixed" in lowered or ("positive" in lowered and "negative" in lowered):
        return "mixed"
    if "positive" in lowered:
        return "positive"
    return "neutral"


# The reference date "how late is this document" is measured against.
# The later a document is relative to this date, the stronger the trend
# shift applied to it — so for a client marked "declining" in
# ground_truth, early documents end up slightly less negative than late
# ones (and the reverse for "improving"). This exists specifically for
# stage 3: aggregator.py compares the first and second half of a
# client's documents by sentiment_score, and without this shift every
# document for a given client in mock mode would get the exact same
# score (it used to be computed from client_id alone, ignoring each
# document's own date) — and the trend would always come out "stable".
_TREND_ANCHOR_DATE = date(2026, 7, 1)


def _trend_adjustment(doc_date: str, trend: str) -> float:
    trend_lower = trend.lower()
    if "declin" not in trend_lower and "improv" not in trend_lower:
        return 0.0

    days_since_anchor = max((date.fromisoformat(doc_date) - _TREND_ANCHOR_DATE).days, 0)
    magnitude = min(days_since_anchor / 30 * 0.10, 0.3)
    return -magnitude if "declin" in trend_lower else magnitude


def _sentiment_score(sentiment: str, gt: dict, doc_date: str) -> float:
    score = _SENTIMENT_BASE_SCORE[sentiment]
    score += _trend_adjustment(doc_date, gt.get("sentiment_trend", ""))
    return max(-1.0, min(1.0, round(score, 2)))


def _extract_csat(doc: dict) -> int | None:
    if doc["doc_type"] != "survey":
        return None
    match = _CSAT_PATTERN.search(doc["text"])
    return int(match.group(1)) if match else None


def _guess_category(text: str) -> str:
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def _guess_severity(gt: dict) -> str:
    churn_risk = gt.get("churn_risk", "").lower()
    if "high" in churn_risk:
        return "high"
    if "medium" in churn_risk:
        return "medium"
    return "low"


def _keywords_from(phrase: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{4,}", phrase)[:5]


def _build_pain_points(doc: dict, gt: dict) -> list[dict]:
    return [
        {
            "issue": issue,
            "category": _guess_category(issue),
            "severity": _guess_severity(gt),
            "quote": _find_quote(doc["text"], _keywords_from(issue)),
        }
        for issue in gt.get("pain_points", [])[:3]
    ]


def _build_feature_requests(doc: dict, gt: dict) -> list[dict]:
    return [
        {
            "request": request,
            "quote": _find_quote(doc["text"], _keywords_from(request)),
        }
        for request in gt.get("feature_requests", [])[:3]
    ]


def _build_praise(doc: dict, sentiment: str) -> list[dict]:
    if sentiment not in ("positive", "mixed"):
        return []
    return [
        {
            "what": "Client expresses satisfaction with the product or support",
            "quote": _find_quote(doc["text"], _PRAISE_KEYWORDS),
        }
    ]


def _build_churn_signals(doc: dict, gt: dict) -> list[dict]:
    if "low" in gt.get("churn_risk", "").lower():
        return []
    return [
        {
            "signal": "Possible churn risk (see ground_truth churn_risk)",
            "quote": _find_quote(doc["text"], _CHURN_KEYWORDS),
        }
    ]


def _hidden_dissatisfaction(sentiment: str, gt: dict) -> bool:
    # If the tone is already openly negative, there's nothing "hidden."
    if sentiment == "negative":
        return False
    overall = gt.get("overall_sentiment", "").lower()
    trend = gt.get("sentiment_trend", "").lower()
    return "silent" in overall or "declining" in trend


def _summary(gt: dict) -> str:
    sentences = _split_sentences(gt.get("notes", ""))
    return " ".join(sentences[:2]) if sentences else gt.get("notes", "")


def _mock_analysis(doc: dict) -> DocumentAnalysis:
    gt = _GROUND_TRUTH[doc["client_id"]]
    sentiment = _normalize_sentiment(gt["overall_sentiment"])

    data = {
        "client_id": doc["client_id"],
        "doc_type": doc["doc_type"],
        "date": doc["date"],
        "sentiment": sentiment,
        "sentiment_score": _sentiment_score(sentiment, gt, doc["date"]),
        "csat_score": _extract_csat(doc),
        "pain_points": _build_pain_points(doc, gt),
        "feature_requests": _build_feature_requests(doc, gt),
        "praise": _build_praise(doc, sentiment),
        "churn_signals": _build_churn_signals(doc, gt),
        "hidden_dissatisfaction": _hidden_dissatisfaction(sentiment, gt),
        "summary": _summary(gt),
    }
    return DocumentAnalysis.model_validate(data)


# ---------------------------------------------------------------------------
# Live mode: an actual call to the Claude API.
# ---------------------------------------------------------------------------


def _call_claude(doc: dict) -> str:
    from anthropic import Anthropic

    from config import ANTHROPIC_API_KEY

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(doc)}],
    )
    return response.content[0].text


def _parse_and_validate(raw_text: str) -> DocumentAnalysis:
    # json.loads raises json.JSONDecodeError if the text isn't JSON at all.
    # model_validate raises pydantic.ValidationError if it is JSON but the
    # wrong shape (missing field, wrong type, number out of range...).
    data = json.loads(raw_text)
    return DocumentAnalysis.model_validate(data)


def _analyze_via_api(doc: dict) -> DocumentAnalysis:
    raw_text = _call_claude(doc)
    try:
        return _parse_and_validate(raw_text)
    except (json.JSONDecodeError, ValidationError) as first_error:
        print(f"    Invalid response ({first_error.__class__.__name__}), retrying...")
        raw_text = _call_claude(doc)
        return _parse_and_validate(raw_text)  # if it fails again, the exception propagates to the caller


def analyze_document(doc: dict) -> DocumentAnalysis | None:
    """Analyzes one document. Returns None if analysis still fails after
    the retry (the error is logged right here)."""
    if USE_MOCK:
        return _mock_analysis(doc)

    try:
        return _analyze_via_api(doc)
    except (json.JSONDecodeError, ValidationError) as error:
        print(f"    Error: couldn't get a valid JSON response after retrying ({error}). Skipping.")
        return None


# ---------------------------------------------------------------------------
# Disk cache and CLI.
# ---------------------------------------------------------------------------


def _cache_path(doc: dict) -> Path:
    stem = Path(doc["source_file"]).stem
    return EXTRACTED_DIR / f"{stem}.json"


def process_documents(documents: list[dict], force: bool) -> tuple[int, int, int]:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    processed = skipped = errors = 0

    for doc in documents:
        cache_path = _cache_path(doc)
        if cache_path.exists() and not force:
            print(f"  {doc['source_file']}: already cached, skipping")
            skipped += 1
            continue

        print(f"  {doc['source_file']}: analyzing...")
        result = analyze_document(doc)
        if result is None:
            errors += 1
            continue

        cache_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        processed += 1

    return processed, skipped, errors


def run_extraction(force: bool = False, documents: list[dict] | None = None) -> tuple[int, int, int]:
    """Entry point for calling stage 2 from outside the command line —
    e.g. from app.py (Streamlit). documents=None means "take all
    documents from data/documents/"; process_documents already did all
    the real work (caching, calling analyze_document) and was already
    self-contained — this wrapper only exists so callers don't also have
    to import load_documents themselves."""
    if documents is None:
        documents = load_documents()
    return process_documents(documents, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract analysis from feedback documents")
    parser.add_argument("--force", action="store_true", help="ignore the cache and recompute everything")
    parser.add_argument("--doc", help="process a single document only, e.g. C001_survey_2026-09-01.txt")
    args = parser.parse_args()

    documents = load_documents()

    if args.doc:
        documents = [d for d in documents if d["source_file"] == args.doc]
        if not documents:
            print(f"Document not found: {args.doc}")
            sys.exit(1)

    mode = "MOCK (no API calls)" if USE_MOCK else "LIVE (Claude API)"
    print(f"Mode: {mode}, model: {MODEL}")
    print(f"Documents to process: {len(documents)}\n")

    processed, skipped, errors = run_extraction(force=args.force, documents=documents)

    print("\nTotals:")
    print(f"  processed: {processed}")
    print(f"  skipped (cached): {skipped}")
    print(f"  errors: {errors}")


if __name__ == "__main__":
    main()
