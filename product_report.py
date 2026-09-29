"""
Stage 4: a cross-client report for the product team.

Three steps, a different tool for each:

  STEP A — group feature requests by meaning.       Claude (mock is a weak approximation)
  STEP B — metrics per cluster and pain points.      plain Python, no LLM
  STEP C — conclusions and recommendations.          Claude (mock is a template)

Usage:
    python product_report.py

Requires client reports to already exist (stage 3, data/reports/clients/)
— if they don't, the script stops with a clear message instead of a hard-
to-read traceback.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from config import (
    CLIENT_REPORTS_DIR,
    EXTRACTED_DIR,
    MODEL,
    PRIORITY_WEIGHTS,
    REPORTS_DIR,
    USE_MOCK,
)
from loader import load_clients
from prompts import (
    CLUSTERING_SYSTEM_PROMPT,
    PRODUCT_SYSTEM_PROMPT,
    build_clustering_prompt,
    build_product_prompt,
)
from schemas import (
    ClientReport,
    ClusterMetrics,
    DocumentAnalysis,
    FeatureClusterResult,
    FeatureRequestItem,
    PainPointSummary,
    ProductInsights,
    ProductReport,
    RequestCluster,
)


class ClusteringValidationError(ValueError):
    """The clustering result is syntactically valid (passed pydantic) but
    breaks the "every id in exactly one cluster" rule."""


# ---------------------------------------------------------------------------
# Loading stage 4's inputs + clear errors if the earlier stages haven't
# run yet.
# ---------------------------------------------------------------------------


def _require_documents() -> list[DocumentAnalysis]:
    if not EXTRACTED_DIR.exists() or not any(EXTRACTED_DIR.glob("*.json")):
        raise SystemExit(
            f"No analyzed documents found in {EXTRACTED_DIR}.\n"
            "Run stage 2 first: python extractor.py"
        )
    return [
        DocumentAnalysis.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(EXTRACTED_DIR.glob("*.json"))
    ]


def _require_client_reports() -> dict[str, ClientReport]:
    if not CLIENT_REPORTS_DIR.exists() or not any(CLIENT_REPORTS_DIR.glob("*.json")):
        raise SystemExit(
            f"No client reports found in {CLIENT_REPORTS_DIR}.\n"
            "Run stage 3 first: python synthesizer.py"
        )
    return {
        report.client_id: report
        for report in (
            ClientReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(CLIENT_REPORTS_DIR.glob("*.json"))
        )
    }


# ---------------------------------------------------------------------------
# STEP A — group feature requests by meaning.
# ---------------------------------------------------------------------------


def collect_feature_requests(documents: list[DocumentAnalysis]) -> list[FeatureRequestItem]:
    """Collects every feature_request from every document into one flat
    list with short ids (FR001, FR002, ...) — these ids are only used for
    references inside the clustering step, not as a permanent identifier."""
    items = []
    counter = 0
    for doc in sorted(documents, key=lambda d: (d.client_id, d.date)):
        for fr in doc.feature_requests:
            counter += 1
            items.append(
                FeatureRequestItem(
                    id=f"FR{counter:03d}",
                    client_id=doc.client_id,
                    date=doc.date,
                    request=fr.request,
                    quote=fr.quote,
                )
            )
    return items


def _mock_clusters(requests: list[FeatureRequestItem]) -> list[RequestCluster]:
    """Mock mode's honest limitation: this only groups by exact
    (normalized) request text. It will NOT merge different phrasings of
    the same meaning (e.g. three different SSO phrasings from three
    different clients stay three separate clusters) — that needs real
    text understanding, i.e. Claude. The mock gives a structurally valid
    result here so steps B and C can be built and tested without waiting
    on a key."""
    groups: dict[str, list[str]] = defaultdict(list)
    display_name: dict[str, str] = {}

    for item in requests:
        key = " ".join(item.request.lower().split())
        groups[key].append(item.id)
        display_name.setdefault(key, item.request)

    return [
        RequestCluster(
            cluster_name=display_name[key],
            description=(
                f"Mock cluster: {len(ids)} mention(s) with the exact same wording. "
                "Real semantic clustering (merging different phrasings of the same "
                "meaning) requires Claude — see the code comment above."
            ),
            request_ids=ids,
        )
        for key, ids in groups.items()
    ]


def _call_claude_for_clusters(requests: list[FeatureRequestItem]) -> str:
    from anthropic import Anthropic

    from config import ANTHROPIC_API_KEY

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    requests_json = json.dumps([r.model_dump() for r in requests], indent=2)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=CLUSTERING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_clustering_prompt(requests_json)}],
    )
    return response.content[0].text


def _parse_clusters(raw_text: str) -> list[RequestCluster]:
    data = json.loads(raw_text)
    return FeatureClusterResult.model_validate(data).clusters


def _validate_clusters(clusters: list[RequestCluster], valid_ids: set[str]) -> None:
    """Three checks, all mandatory: every id is assigned, no duplicates,
    no invented ids. Runs for both mock and live Claude output — the
    same guarantee regardless of mode."""
    seen: set[str] = set()
    unknown: set[str] = set()
    duplicates: set[str] = set()

    for cluster in clusters:
        for request_id in cluster.request_ids:
            if request_id not in valid_ids:
                unknown.add(request_id)
            elif request_id in seen:
                duplicates.add(request_id)
            seen.add(request_id)

    missing = valid_ids - seen

    problems = []
    if missing:
        problems.append(f"not assigned to any cluster: {sorted(missing)}")
    if duplicates:
        problems.append(f"assigned to more than one cluster: {sorted(duplicates)}")
    if unknown:
        problems.append(f"reference ids that don't exist: {sorted(unknown)}")

    if problems:
        raise ClusteringValidationError("; ".join(problems))


def _trivial_clusters(requests: list[FeatureRequestItem]) -> list[RequestCluster]:
    """Fallback for when even a retried live API call fails validation:
    one request per cluster. Worse than real clustering, but the report
    still gets built instead of failing outright."""
    return [
        RequestCluster(
            cluster_name=r.request,
            description="Fallback: clustering failed validation, one request = one cluster.",
            request_ids=[r.id],
        )
        for r in requests
    ]


def get_clusters(requests: list[FeatureRequestItem]) -> list[RequestCluster]:
    valid_ids = {r.id for r in requests}

    if USE_MOCK:
        clusters = _mock_clusters(requests)
        _validate_clusters(clusters, valid_ids)
        return clusters

    try:
        clusters = _parse_clusters(_call_claude_for_clusters(requests))
        _validate_clusters(clusters, valid_ids)
        return clusters
    except (json.JSONDecodeError, ValidationError, ClusteringValidationError) as first_error:
        print(f"    Clustering failed validation ({first_error}), retrying...")
        try:
            clusters = _parse_clusters(_call_claude_for_clusters(requests))
            _validate_clusters(clusters, valid_ids)
            return clusters
        except (json.JSONDecodeError, ValidationError, ClusteringValidationError) as second_error:
            print(f"    Retry also failed ({second_error}). Falling back.")
            return _trivial_clusters(requests)


# ---------------------------------------------------------------------------
# STEP B — metrics per cluster and pain points. Plain Python, no LLM:
# MRR sums, client/plan counts, min-max normalization for priority_score.
# None of these numbers can "get it wrong" the way a language model can —
# which is exactly why this isn't Claude.
# ---------------------------------------------------------------------------


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalization into the [0, 1] range. If every cluster has
    the same value for a metric, that metric can't distinguish between
    them — treat it as maximal for all of them (1.0) rather than 0, so we
    don't artificially deflate priority_score where there's just too
    little data to compare."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def compute_cluster_metrics(
    clusters: list[RequestCluster],
    requests_by_id: dict[str, FeatureRequestItem],
    clients_df,
    client_reports: dict[str, ClientReport],
) -> list[ClusterMetrics]:
    raw: list[ClusterMetrics] = []

    for cluster in clusters:
        items = [requests_by_id[rid] for rid in cluster.request_ids]
        client_ids = sorted({item.client_id for item in items})
        client_rows = clients_df[clients_df["client_id"].isin(client_ids)]

        total_mrr_usd = int(client_rows["mrr_usd"].sum())
        plans = dict(Counter(client_rows["plan"]))

        at_risk_mrr_usd = int(
            sum(
                int(client_rows.loc[client_rows["client_id"] == cid, "mrr_usd"].iloc[0])
                for cid in client_ids
                if client_reports.get(cid) is not None and client_reports[cid].risk_level == "high"
            )
        )

        dates = sorted(item.date for item in items)

        seen_quotes: list[str] = []
        for item in items:
            if item.quote not in seen_quotes:
                seen_quotes.append(item.quote)
            if len(seen_quotes) == 3:
                break

        raw.append(
            ClusterMetrics(
                cluster_name=cluster.cluster_name,
                description=cluster.description,
                client_count=len(client_ids),
                clients=client_ids,
                total_mrr_usd=total_mrr_usd,
                plans=plans,
                at_risk_mrr_usd=at_risk_mrr_usd,
                mention_count=len(items),
                first_mentioned=dates[0],
                last_mentioned=dates[-1],
                top_quotes=seen_quotes,
                priority_score=0.0,  # placeholder — computed below across all clusters at once
            )
        )

    mrr_norm = _normalize([c.total_mrr_usd for c in raw])
    count_norm = _normalize([c.client_count for c in raw])
    risk_norm = _normalize([c.at_risk_mrr_usd for c in raw])

    scored = []
    for cluster, mrr_n, count_n, risk_n in zip(raw, mrr_norm, count_norm, risk_norm):
        score = (
            PRIORITY_WEIGHTS["mrr"] * mrr_n
            + PRIORITY_WEIGHTS["client_count"] * count_n
            + PRIORITY_WEIGHTS["at_risk"] * risk_n
        )
        scored.append(cluster.model_copy(update={"priority_score": round(score, 3)}))

    return sorted(scored, key=lambda c: c.priority_score, reverse=True)


def compute_pain_point_summary(documents: list[DocumentAnalysis], clients_df) -> list[PainPointSummary]:
    by_category: dict[str, dict] = defaultdict(lambda: {"count": 0, "clients": set(), "high_severity": 0})

    for doc in documents:
        for pain in doc.pain_points:
            entry = by_category[pain.category]
            entry["count"] += 1
            entry["clients"].add(doc.client_id)
            if pain.severity == "high":
                entry["high_severity"] += 1

    summaries = [
        PainPointSummary(
            category=category,
            count=data["count"],
            affected_mrr_usd=int(clients_df[clients_df["client_id"].isin(data["clients"])]["mrr_usd"].sum()),
            high_severity_count=data["high_severity"],
        )
        for category, data in by_category.items()
    ]

    return sorted(summaries, key=lambda s: s.count, reverse=True)


# ---------------------------------------------------------------------------
# STEP C — conclusions for the product team. Claude again (or a mock
# template): the input is step B's already-computed metrics, not raw
# documents.
# ---------------------------------------------------------------------------


def _mock_product_insights(
    clusters: list[ClusterMetrics],
    pain_summary: list[PainPointSummary],
    risk_by_client: dict[str, str],
) -> ProductInsights:
    top_clusters = clusters[:3]  # already sorted by priority_score

    summary_parts = [
        f"{len(clusters)} distinct feature request theme(s) identified across "
        f"{sum(c.mention_count for c in clusters)} mention(s)."
    ]
    if top_clusters:
        top = top_clusters[0]
        at_risk_note = f", including ${top.at_risk_mrr_usd} at high churn risk" if top.at_risk_mrr_usd else ""
        summary_parts.append(
            f"Top priority: '{top.cluster_name}' — requested by {top.client_count} "
            f"client(s) representing ${top.total_mrr_usd} MRR{at_risk_note}."
        )
    if pain_summary:
        worst = pain_summary[0]
        summary_parts.append(
            f"'{worst.category}' is the most common pain point category "
            f"({worst.count} occurrence(s), ${worst.affected_mrr_usd} MRR affected)."
        )

    top_recommendations = [
        {
            "cluster_name": c.cluster_name,
            "why_now": (
                f"Requested by {c.client_count} client(s) (${c.total_mrr_usd} MRR total"
                + (f", ${c.at_risk_mrr_usd} at high churn risk" if c.at_risk_mrr_usd else "")
                + f"), mentioned {c.mention_count} time(s) between {c.first_mentioned} and {c.last_mentioned}."
            ),
            "business_impact": f"priority_score={c.priority_score:.2f} (weighted by MRR, client count, at-risk MRR).",
        }
        for c in top_clusters
    ]

    pain_point_insights = [
        {
            "issue": p.category.replace("_", " ").title(),
            "explanation": (
                f"{p.count} occurrence(s) across clients representing ${p.affected_mrr_usd} MRR"
                + (f", {p.high_severity_count} marked high severity" if p.high_severity_count else "")
                + "."
            ),
        }
        for p in pain_summary[:3]
    ]

    quick_wins = [
        f"'{c.cluster_name}': narrow ask, {c.mention_count} mention(s) from a single client — "
        "worth checking with engineering whether scope is smaller than priority_score suggests."
        for c in clusters
        if c.client_count == 1 and c.mention_count >= 2
    ][:3] or ["No obvious quick wins from current metrics; see top_recommendations for larger bets."]

    risks_if_ignored = [
        {
            "description": f"Clients requesting '{c.cluster_name}' include high-churn-risk accounts.",
            "clients_at_risk": [cid for cid in c.clients if risk_by_client.get(cid) == "high"],
            "mrr_at_risk_usd": c.at_risk_mrr_usd,
        }
        for c in clusters
        if c.at_risk_mrr_usd > 0
    ]

    return ProductInsights.model_validate(
        {
            "executive_summary": " ".join(summary_parts),
            "top_recommendations": top_recommendations,
            "pain_point_insights": pain_point_insights,
            "quick_wins": quick_wins,
            "risks_if_ignored": risks_if_ignored,
        }
    )


def _call_claude_for_insights(
    clusters: list[ClusterMetrics], pain_summary: list[PainPointSummary], risk_by_client: dict[str, str]
) -> str:
    from anthropic import Anthropic

    from config import ANTHROPIC_API_KEY

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_product_prompt(
        clusters_json=json.dumps([c.model_dump() for c in clusters], indent=2),
        pain_summary_json=json.dumps([p.model_dump() for p in pain_summary], indent=2),
        risk_by_client_json=json.dumps(risk_by_client, indent=2),
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=PRODUCT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_and_validate_insights(raw_text: str) -> ProductInsights:
    return ProductInsights.model_validate(json.loads(raw_text))


def generate_product_insights(
    clusters: list[ClusterMetrics], pain_summary: list[PainPointSummary], risk_by_client: dict[str, str]
) -> ProductInsights:
    if USE_MOCK:
        return _mock_product_insights(clusters, pain_summary, risk_by_client)

    raw_text = _call_claude_for_insights(clusters, pain_summary, risk_by_client)
    try:
        return _parse_and_validate_insights(raw_text)
    except (json.JSONDecodeError, ValidationError) as first_error:
        print(f"    Invalid step C response ({first_error.__class__.__name__}), retrying...")
        raw_text = _call_claude_for_insights(clusters, pain_summary, risk_by_client)
        return _parse_and_validate_insights(raw_text)


# ---------------------------------------------------------------------------
# Rendering and saving (JSON, Markdown, CSV).
# ---------------------------------------------------------------------------


def _render_markdown(report: ProductReport) -> str:
    lines = ["# Product Feedback Report", ""]

    lines.append("## Executive Summary")
    lines.append(report.insights.executive_summary)
    lines.append("")

    lines.append("## Feature Request Priorities")
    lines.append("| Rank | Cluster | Clients | Total MRR | At-Risk MRR | Mentions | Priority |")
    lines.append("|---|---|---|---|---|---|---|")
    for rank, c in enumerate(report.clusters, start=1):
        lines.append(
            f"| {rank} | {c.cluster_name} | {c.client_count} | ${c.total_mrr_usd} | "
            f"${c.at_risk_mrr_usd} | {c.mention_count} | {c.priority_score:.2f} |"
        )
    lines.append("")

    lines.append("## Top Recommendations")
    for rec in report.insights.top_recommendations:
        lines.append(f"### {rec.cluster_name}")
        lines.append(f"**Why now:** {rec.why_now}")
        lines.append(f"**Business impact:** {rec.business_impact}")
        lines.append("")

    lines.append("## Pain Point Insights")
    for insight in report.insights.pain_point_insights:
        lines.append(f"### {insight.issue}")
        lines.append(insight.explanation)
        lines.append("")

    lines.append("## Quick Wins")
    for win in report.insights.quick_wins:
        lines.append(f"- {win}")
    lines.append("")

    lines.append("## Risks If Ignored")
    for risk in report.insights.risks_if_ignored:
        lines.append(f"### {risk.description}")
        clients_str = ", ".join(risk.clients_at_risk) or "n/a"
        lines.append(f"Clients at risk: {clients_str} (MRR: ${risk.mrr_at_risk_usd})")
        lines.append("")

    lines.append("## Pain Point Summary by Category")
    lines.append("| Category | Count | Affected MRR | High Severity |")
    lines.append("|---|---|---|---|")
    for p in report.pain_point_summary:
        lines.append(f"| {p.category} | {p.count} | ${p.affected_mrr_usd} | {p.high_severity_count} |")
    lines.append("")

    return "\n".join(lines)


def _write_csv(clusters: list[ClusterMetrics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "cluster_name",
                "description",
                "client_count",
                "clients",
                "total_mrr_usd",
                "plans",
                "at_risk_mrr_usd",
                "mention_count",
                "first_mentioned",
                "last_mentioned",
                "priority_score",
                "top_quotes",
            ]
        )
        for rank, c in enumerate(clusters, start=1):
            writer.writerow(
                [
                    rank,
                    c.cluster_name,
                    c.description,
                    c.client_count,
                    "; ".join(c.clients),
                    c.total_mrr_usd,
                    "; ".join(f"{plan}:{count}" for plan, count in c.plans.items()),
                    c.at_risk_mrr_usd,
                    c.mention_count,
                    c.first_mentioned,
                    c.last_mentioned,
                    c.priority_score,
                    " | ".join(c.top_quotes),
                ]
            )


def build_product_report() -> ProductReport:
    documents = _require_documents()
    client_reports = _require_client_reports()
    clients_df = load_clients()
    risk_by_client = {cid: report.risk_level for cid, report in client_reports.items()}

    print("Step A: grouping feature requests...")
    requests = collect_feature_requests(documents)
    requests_by_id = {r.id: r for r in requests}
    clusters_raw = get_clusters(requests)
    print(f"  {len(requests)} request(s) -> {len(clusters_raw)} cluster(s)")

    print("Step B: cluster and pain point metrics...")
    cluster_metrics = compute_cluster_metrics(clusters_raw, requests_by_id, clients_df, client_reports)
    pain_summary = compute_pain_point_summary(documents, clients_df)

    print("Step C: conclusions for the product team...")
    insights = generate_product_insights(cluster_metrics, pain_summary, risk_by_client)

    return ProductReport(clusters=cluster_metrics, pain_point_summary=pain_summary, insights=insights)


def save_product_report(report: ProductReport) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    (REPORTS_DIR / "product_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (REPORTS_DIR / "product_report.md").write_text(_render_markdown(report), encoding="utf-8")
    _write_csv(report.clusters, REPORTS_DIR / "feature_requests.csv")


def main() -> None:
    mode = "MOCK (no API calls)" if USE_MOCK else "LIVE (Claude API)"
    print(f"Mode: {mode}, model: {MODEL}\n")

    report = build_product_report()
    save_product_report(report)

    print(f"\nSaved to {REPORTS_DIR}:")
    print("  product_report.json")
    print("  product_report.md")
    print("  feature_requests.csv")

    print("\nCluster priorities:")
    header = f"{'rank':<6}{'cluster':<45}{'clients':>8}{'mrr':>9}{'at_risk':>9}{'score':>8}"
    print(header)
    print("-" * len(header))
    for rank, c in enumerate(report.clusters, start=1):
        name = c.cluster_name if len(c.cluster_name) <= 43 else c.cluster_name[:40] + "..."
        print(f"{rank:<6}{name:<45}{c.client_count:>8}{c.total_mrr_usd:>9}{c.at_risk_mrr_usd:>9}{c.priority_score:>8.2f}")


if __name__ == "__main__":
    main()
