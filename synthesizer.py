"""
Synthesizes a per-client report: takes ClientMetrics (computed by
aggregator.py) and adds ClientInsights — the interpretation, which in
mock mode is assembled by simple rules, and in live mode is written by
Claude.

Usage:
    python synthesizer.py                # all clients
    python synthesizer.py --client C003  # a single client only
"""

import argparse
import json

from pydantic import ValidationError

from aggregator import compute_metrics
from config import CLIENT_REPORTS_DIR, MODEL, USE_MOCK
from loader import load_clients
from prompts import SYNTHESIS_SYSTEM_PROMPT, build_synthesis_prompt
from schemas import ClientInsights, ClientMetrics, ClientReport

# ---------------------------------------------------------------------------
# Mock mode: ClientInsights is assembled by rules over ClientMetrics,
# without calling Claude. As in extractor.py, the goal isn't perfect
# accuracy but a plausible, schema-valid result, so everything else
# (aggregator, reports, CLI) can already be built and tested. Since the
# reasoning here is simulated with if/else rules rather than real text
# understanding, it's necessarily cruder than what live Claude would
# produce on the same data.
# ---------------------------------------------------------------------------


def _mock_risk(metrics: ClientMetrics) -> tuple[str, str]:
    has_churn_signals = len(metrics.all_churn_signals) > 0
    declining = metrics.sentiment_trend == "declining"
    high_value = metrics.plan == "Enterprise" or metrics.mrr_usd >= 5000

    if has_churn_signals and declining:
        level = "high"
    elif has_churn_signals or (metrics.hidden_dissatisfaction and declining):
        level = "high" if high_value else "medium"
    elif metrics.hidden_dissatisfaction:
        level = "medium" if metrics.high_severity_count > 0 else "low"
    else:
        level = "low"

    reasons = []
    if has_churn_signals:
        reasons.append(f"{len(metrics.all_churn_signals)} explicit churn signal(s) on record")
    if declining:
        reasons.append("sentiment trend over the period is declining")
    if metrics.hidden_dissatisfaction:
        reasons.append(
            "hidden dissatisfaction detected (polite tone / decent CSAT alongside "
            "repeated unresolved complaints)"
        )
    if high_value:
        reasons.append(f"high-value account (${metrics.mrr_usd} MRR, {metrics.plan} plan) — losing it costs more")
    if not reasons:
        reasons.append("no churn signals, sentiment is stable or improving")

    return level, "; ".join(reasons) + "."


def _mock_key_themes(metrics: ClientMetrics) -> list[dict]:
    themes = []

    top_categories = sorted(
        metrics.pain_counts_by_category.items(), key=lambda kv: kv[1], reverse=True
    )[:2]
    for category, count in top_categories:
        themes.append(
            {
                "theme": category.replace("_", " ").title(),
                "description": (
                    f"'{category}' is the most frequently tagged pain point category, "
                    f"appearing {count} time(s) across this client's documents."
                ),
                # There's no quote tied specifically to the category
                # (ClientMetrics only stores a count) — leave evidence
                # honestly empty rather than fabricate a quote.
                "evidence": [],
            }
        )

    if metrics.all_feature_requests:
        unique_requests = sorted({fr.request for fr in metrics.all_feature_requests})
        themes.append(
            {
                "theme": "Outstanding feature requests",
                "description": (
                    f"{len(unique_requests)} distinct feature request(s) logged, "
                    f"{len(metrics.all_feature_requests)} mention(s) total across documents."
                ),
                "evidence": [fr.quote for fr in metrics.all_feature_requests[:2]],
            }
        )

    if metrics.all_churn_signals:
        themes.append(
            {
                "theme": "Churn risk signals",
                "description": f"{len(metrics.all_churn_signals)} explicit churn signal(s) found.",
                "evidence": [cs.quote for cs in metrics.all_churn_signals[:2]],
            }
        )

    if metrics.hidden_dissatisfaction:
        themes.append(
            {
                "theme": "Hidden dissatisfaction",
                "description": (
                    "Tone reads as polite or positive, but the same issues keep "
                    "resurfacing without resolution — a purely tone-based read "
                    "would miss this."
                ),
                "evidence": [fr.quote for fr in metrics.all_feature_requests[:1]],
            }
        )

    return themes or [
        {
            "theme": "No major issues detected",
            "description": "No recurring pain points, feature requests, or churn signals stood out.",
            "evidence": [],
        }
    ]


def _mock_response_strategy(metrics: ClientMetrics, risk_level: str) -> dict:
    top_request = metrics.all_feature_requests[0].request if metrics.all_feature_requests else None
    has_churn_signals = len(metrics.all_churn_signals) > 0

    immediate_actions = []
    if risk_level == "high":
        immediate_actions.append("Schedule a call with the client within 2 business days.")
        immediate_actions.append("Loop in a senior stakeholder (CSM lead or VP) before the call.")
    elif risk_level == "medium":
        immediate_actions.append("Send a personal check-in email within 5 business days.")
    else:
        immediate_actions.append("No urgent action needed; continue regular quarterly check-ins.")
    if top_request:
        immediate_actions.append(f"Get a concrete status or timeline on: {top_request}.")

    talking_points = ["Thank the client for their continued feedback and patience."]
    if top_request:
        talking_points.append(f"Acknowledge the outstanding request directly: {top_request}.")
    if metrics.hidden_dissatisfaction:
        talking_points.append(
            "Ask directly how things are really going — don't rely only on the polite "
            "tone or a decent CSAT score."
        )
    if has_churn_signals:
        talking_points.append(
            "Address the relationship/reliability concerns head-on rather than waiting "
            "for the client to raise them again."
        )

    what_to_avoid = ["Don't promise a specific ship date you can't confirm with product/engineering."]
    if risk_level == "high":
        what_to_avoid.append(
            "Don't let this go another cycle without a concrete update — repeated "
            "'still unscheduled' answers are part of why risk is elevated."
        )
    if metrics.hidden_dissatisfaction:
        what_to_avoid.append("Don't take a polite tone or a decent CSAT score at face value as 'everything is fine.'")

    return {
        "immediate_actions": immediate_actions,
        "talking_points": talking_points,
        "what_to_avoid": what_to_avoid,
    }


def _mock_follow_up_draft(metrics: ClientMetrics) -> str:
    top_request = metrics.all_feature_requests[0].request if metrics.all_feature_requests else None

    lines = [f"Hi {metrics.company_name} team,", ""]
    lines.append("Thanks for all the feedback you've shared with us recently.")
    if top_request:
        lines.append(
            f"I wanted to follow up specifically on {top_request} — I know this has "
            "come up more than once, and I want to get you a real status rather than "
            "another 'in progress.'"
        )
    lines.append("Could we grab 20 minutes this week to talk through where things stand and what you need from us next?")
    lines.append("")
    lines.append("Best,")
    lines.append("Your Customer Success Team")
    return "\n".join(lines)


def _mock_executive_summary(metrics: ClientMetrics, risk_level: str) -> str:
    parts = [
        f"{metrics.company_name} ({metrics.plan}, ${metrics.mrr_usd} MRR) is currently "
        f"assessed as {risk_level} risk."
    ]
    if metrics.avg_csat is not None:
        parts.append(
            f"Average CSAT across {metrics.doc_count} document(s) is {metrics.avg_csat}/5, "
            f"sentiment trend is {metrics.sentiment_trend}."
        )
    else:
        parts.append(
            f"No CSAT survey data yet; sentiment trend across {metrics.doc_count} "
            f"document(s) is {metrics.sentiment_trend}."
        )
    if metrics.hidden_dissatisfaction:
        parts.append(
            "Note: tone reads as polite or positive, but underlying signals suggest "
            "real, understated dissatisfaction."
        )
    if metrics.all_churn_signals:
        parts.append(
            f"There are {len(metrics.all_churn_signals)} explicit churn signal(s) on "
            "record that need direct follow-up."
        )
    return " ".join(parts)


def _mock_insights(metrics: ClientMetrics) -> ClientInsights:
    risk_level, risk_reasoning = _mock_risk(metrics)
    data = {
        "executive_summary": _mock_executive_summary(metrics, risk_level),
        "risk_level": risk_level,
        "risk_reasoning": risk_reasoning,
        "key_themes": _mock_key_themes(metrics),
        "response_strategy": _mock_response_strategy(metrics, risk_level),
        "follow_up_draft": _mock_follow_up_draft(metrics),
    }
    return ClientInsights.model_validate(data)


# ---------------------------------------------------------------------------
# Live mode: an actual call to the Claude API, following the same pattern
# (call → parse JSON → validate with pydantic → one retry) as extractor.py.
# ---------------------------------------------------------------------------


def _call_claude_for_insights(metrics: ClientMetrics) -> str:
    from anthropic import Anthropic

    from config import ANTHROPIC_API_KEY

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_synthesis_prompt(metrics.model_dump_json(indent=2))}],
    )
    return response.content[0].text


def _parse_and_validate_insights(raw_text: str) -> ClientInsights:
    data = json.loads(raw_text)
    return ClientInsights.model_validate(data)


def generate_insights(metrics: ClientMetrics) -> ClientInsights:
    if USE_MOCK:
        return _mock_insights(metrics)

    raw_text = _call_claude_for_insights(metrics)
    try:
        return _parse_and_validate_insights(raw_text)
    except (json.JSONDecodeError, ValidationError) as first_error:
        print(f"    Invalid synthesis response ({first_error.__class__.__name__}), retrying...")
        raw_text = _call_claude_for_insights(metrics)
        return _parse_and_validate_insights(raw_text)


def build_client_report(client_id: str) -> ClientReport:
    metrics = compute_metrics(client_id)
    insights = generate_insights(metrics)
    return ClientReport.model_validate({**metrics.model_dump(), **insights.model_dump()})


# ---------------------------------------------------------------------------
# Saving the report and the CLI.
# ---------------------------------------------------------------------------


def _render_markdown(report: ClientReport) -> str:
    lines = [f"# {report.company_name} ({report.client_id})", ""]

    lines.append(f"**Plan:** {report.plan} | **MRR:** ${report.mrr_usd} | **Risk:** {report.risk_level.upper()}")
    avg_csat = report.avg_csat if report.avg_csat is not None else "n/a"
    lines.append(
        f"**Docs analyzed:** {report.doc_count} | **Avg CSAT:** {avg_csat} | "
        f"**Sentiment trend:** {report.sentiment_trend} | "
        f"**Hidden dissatisfaction:** {report.hidden_dissatisfaction}"
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(report.executive_summary)
    lines.append("")

    lines.append("## Risk Assessment")
    lines.append(f"**Level:** {report.risk_level}")
    lines.append(f"**Reasoning:** {report.risk_reasoning}")
    lines.append("")

    lines.append("## Key Themes")
    for theme in report.key_themes:
        lines.append(f"### {theme.theme}")
        lines.append(theme.description)
        for quote in theme.evidence:
            lines.append(f"> {quote}")
        lines.append("")

    lines.append("## Response Strategy")
    lines.append("**Immediate actions:**")
    for action in report.response_strategy.immediate_actions:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("**Talking points:**")
    for point in report.response_strategy.talking_points:
        lines.append(f"- {point}")
    lines.append("")
    lines.append("**What to avoid:**")
    for item in report.response_strategy.what_to_avoid:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Follow-up Draft")
    lines.append("```")
    lines.append(report.follow_up_draft)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _save_report(report: ClientReport) -> None:
    CLIENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    (CLIENT_REPORTS_DIR / f"{report.client_id}.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (CLIENT_REPORTS_DIR / f"{report.client_id}.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )


def _print_summary_table(reports: list[ClientReport]) -> None:
    header = f"{'client':<8}{'plan':<12}{'mrr':>8}{'avg_csat':>10}{'trend':>12}{'risk':>8}"
    print(header)
    print("-" * len(header))
    for r in reports:
        avg_csat = f"{r.avg_csat:.1f}" if r.avg_csat is not None else "n/a"
        print(f"{r.client_id:<8}{r.plan:<12}{r.mrr_usd:>8}{avg_csat:>10}{r.sentiment_trend:>12}{r.risk_level:>8}")


def run_synthesis(client_ids: list[str] | None = None) -> list[ClientReport]:
    """Entry point for calling stage 3 from outside the command line —
    e.g. from app.py (Streamlit). Builds and saves a report for every
    client, skipping (with a console log) any client synthesis fails
    for, instead of aborting the whole run over one client."""
    if client_ids is None:
        client_ids = list(load_clients()["client_id"])

    reports = []
    for client_id in client_ids:
        print(f"{client_id}: building report...")
        try:
            report = build_client_report(client_id)
        except (json.JSONDecodeError, ValidationError) as error:
            print(f"  Synthesis error for {client_id}: {error}. Skipping.")
            continue

        _save_report(report)
        reports.append(report)

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize per-client reports")
    parser.add_argument("--client", help="process a single client only, e.g. C001")
    args = parser.parse_args()

    client_ids = [args.client] if args.client else None

    mode = "MOCK (no API calls)" if USE_MOCK else "LIVE (Claude API)"
    print(f"Mode: {mode}, model: {MODEL}\n")

    reports = run_synthesis(client_ids)

    print(f"\nReports saved: {len(reports)} (in {CLIENT_REPORTS_DIR})\n")
    _print_summary_table(reports)


if __name__ == "__main__":
    main()
