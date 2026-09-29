"""
Aggregates analyze_document results (data/extracted/*.json) into one
per-client summary — ClientMetrics.

Important: everything in this file is plain Python arithmetic (sums,
averages, Counter tallies). There's no call to Claude here, and there
shouldn't be — counting "how many documents," "average CSAT," or "which
pain point category shows up most" doesn't need a language model.
Interpreting those numbers (risk, strategy) is Claude's job in the next
step (synthesizer.py) — precisely because that requires reasoning, not
counting.
"""

import json
from collections import Counter
from statistics import mean

from config import EXTRACTED_DIR, SENTIMENT_TREND_THRESHOLD
from loader import load_clients
from schemas import (
    ChurnSignalRecord,
    ClientMetrics,
    DocumentAnalysis,
    FeatureRequestRecord,
)


def _load_client_documents(client_id: str) -> list[DocumentAnalysis]:
    """Reads every cached DocumentAnalysis for one client. Filenames in
    data/extracted/ start with client_id_, e.g. C001_survey_2026-09-01.json
    — that's enough to pick out the right ones without re-parsing the
    original .txt files."""
    documents = []
    for path in sorted(EXTRACTED_DIR.glob(f"{client_id}_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        documents.append(DocumentAnalysis.model_validate(data))
    return documents


def _compute_sentiment_trend(documents: list[DocumentAnalysis]) -> str:
    """Sorts documents by date, splits them in half, and compares the
    average sentiment_score of the second half against the first. A
    difference smaller than the threshold (config.SENTIMENT_TREND_THRESHOLD)
    in absolute value counts as "stable" — a guard against noise from
    one or two documents looking like a real trend."""
    if len(documents) < 2:
        return "stable"

    by_date = sorted(documents, key=lambda d: d.date)
    scores = [d.sentiment_score for d in by_date]

    mid = len(scores) // 2
    first_half, second_half = scores[:mid], scores[mid:]
    diff = mean(second_half) - mean(first_half)

    if diff > SENTIMENT_TREND_THRESHOLD:
        return "improving"
    if diff < -SENTIMENT_TREND_THRESHOLD:
        return "declining"
    return "stable"


def compute_metrics(client_id: str) -> ClientMetrics:
    documents = _load_client_documents(client_id)
    if not documents:
        raise ValueError(
            f"No analyzed documents found for {client_id} in {EXTRACTED_DIR}. "
            "Run extractor.py first."
        )

    clients = load_clients()
    matching_rows = clients.loc[clients["client_id"] == client_id]
    if matching_rows.empty:
        raise ValueError(f"Client {client_id} not found in clients.csv")
    client_row = matching_rows.iloc[0]

    doc_types = Counter(doc.doc_type for doc in documents)

    csat_scores = [doc.csat_score for doc in documents if doc.csat_score is not None]
    avg_csat = round(mean(csat_scores), 2) if csat_scores else None

    avg_sentiment_score = round(mean(doc.sentiment_score for doc in documents), 2)

    pain_counts: Counter = Counter()
    high_severity_count = 0
    for doc in documents:
        for pain in doc.pain_points:
            pain_counts[pain.category] += 1
            if pain.severity == "high":
                high_severity_count += 1

    all_feature_requests = [
        FeatureRequestRecord(
            request=fr.request, quote=fr.quote, date=doc.date, doc_type=doc.doc_type
        )
        for doc in documents
        for fr in doc.feature_requests
    ]
    all_churn_signals = [
        ChurnSignalRecord(
            signal=cs.signal, quote=cs.quote, date=doc.date, doc_type=doc.doc_type
        )
        for doc in documents
        for cs in doc.churn_signals
    ]

    return ClientMetrics(
        client_id=client_id,
        company_name=client_row["company_name"],
        plan=client_row["plan"],
        mrr_usd=int(client_row["mrr_usd"]),
        doc_count=len(documents),
        doc_types=dict(doc_types),
        avg_csat=avg_csat,
        avg_sentiment_score=avg_sentiment_score,
        sentiment_trend=_compute_sentiment_trend(documents),
        pain_counts_by_category=dict(pain_counts),
        high_severity_count=high_severity_count,
        all_feature_requests=all_feature_requests,
        all_churn_signals=all_churn_signals,
        hidden_dissatisfaction=any(doc.hidden_dissatisfaction for doc in documents),
    )


def _print_all_metrics() -> None:
    for client_id in load_clients()["client_id"]:
        print(compute_metrics(client_id).model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    _print_all_metrics()
