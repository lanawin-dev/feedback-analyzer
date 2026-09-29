"""
Streamlit interface for Feedback Analyzer.

Usage:
    streamlit run app.py

None of the business logic is reimplemented here — app.py only calls the
functions that already exist (extractor.run_extraction,
synthesizer.run_synthesis, product_report.build_product_report /
save_product_report) and displays whatever has already been computed
into data/extracted/ and data/reports/. If those files don't exist yet,
every tab explains what to click instead of failing.
"""

import json
from datetime import date

import pandas as pd
import streamlit as st

import extractor
import product_report
import synthesizer
from config import (
    CLIENT_REPORTS_DIR,
    DOCUMENTS_DIR,
    EXTRACTED_DIR,
    MODEL,
    REPORTS_DIR,
    USE_MOCK,
)
from loader import load_clients, load_documents
from schemas import ClientReport, DocumentAnalysis, ProductReport

st.set_page_config(page_title="Feedback Analyzer", page_icon="📊", layout="wide")

# One sequential blue for single-series bar/line charts, and a status
# traffic-light triad for risk_level — colors come from the project's
# design system (the dataviz skill's references/palette.md): a "status"
# color is never reused as a regular series color, and risk always keeps
# a text label (low/medium/high) next to the color so the value never
# reads through color alone.
BAR_COLOR = "#2a78d6"
RISK_COLORS = {"low": "#0ca30c", "medium": "#fab219", "high": "#d03b3b"}
RISK_LABELS = {"low": "🟢 low", "medium": "🟡 medium", "high": "🔴 high"}


# ---------------------------------------------------------------------------
# Data loading with caching.
#
# Streamlit reruns the entire app.py file top to bottom on ABSOLUTELY any
# interaction — a button click, a selectbox choice, even typing a
# character into a text_area. Without caching, that would mean re-reading
# every JSON file in data/extracted/ and data/reports/ on every click.
# @st.cache_data remembers a function's result (keyed by its arguments —
# there are none here, so each function caches a single result) and
# returns it again until the cache is explicitly cleared. Files on disk
# aren't part of that calculation — which is why the cache has to be
# cleared by hand after running the pipeline (see _run_pipeline).
# ---------------------------------------------------------------------------


@st.cache_data
def _load_clients_df() -> pd.DataFrame:
    return load_clients()


@st.cache_data
def _load_raw_documents() -> list[dict]:
    return load_documents()


@st.cache_data
def _load_analyzed_documents() -> list[DocumentAnalysis]:
    if not EXTRACTED_DIR.exists():
        return []
    return [
        DocumentAnalysis.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(EXTRACTED_DIR.glob("*.json"))
    ]


@st.cache_data
def _load_client_reports() -> dict[str, ClientReport]:
    if not CLIENT_REPORTS_DIR.exists():
        return {}
    return {
        report.client_id: report
        for report in (
            ClientReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(CLIENT_REPORTS_DIR.glob("*.json"))
        )
    }


@st.cache_data
def _load_product_report() -> ProductReport | None:
    path = REPORTS_DIR / "product_report.json"
    if not path.exists():
        return None
    return ProductReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _empty_state(message: str) -> None:
    st.info(f"{message}\n\nClick **▶️ Run analysis** in the sidebar on the left.")


# ---------------------------------------------------------------------------
# Sidebar: mode indicator + running the whole pipeline.
# ---------------------------------------------------------------------------


def _run_pipeline(force: bool) -> None:
    progress = st.sidebar.progress(0, text="Running analysis...")

    # st.status is a collapsible block with a step history and a spinner;
    # expanded=True keeps it open while it runs, and status.update() at
    # the end switches the title/icon to "done." It plays the same role
    # as the argparse-driven console progress in extractor.py /
    # synthesizer.py / product_report.py, just for the browser.
    try:
        with st.status("Running the pipeline: extractor → synthesizer → product_report", expanded=True) as status:
            st.write("**Step 1/3** — analyzing documents (`extractor.py`)")
            processed, skipped, errors = extractor.run_extraction(force=force)
            st.write(f"Processed: {processed}, cached: {skipped}, errors: {errors}")
            progress.progress(33, text="Step 1/3 done")

            st.write("**Step 2/3** — building client reports (`synthesizer.py`)")
            client_reports = synthesizer.run_synthesis()
            st.write(f"Reports built: {len(client_reports)}")
            progress.progress(66, text="Step 2/3 done")

            st.write("**Step 3/3** — building the product report (`product_report.py`)")
            product = product_report.build_product_report()
            product_report.save_product_report(product)
            st.write("Product report saved")
            progress.progress(100, text="Done")

            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as error:  # surface the error in the UI instead of crashing the app
        st.sidebar.error(f"Analysis failed: {error}")
        progress.empty()
        return

    progress.empty()

    # Files on disk changed, but @st.cache_data doesn't know that —
    # without clearing it explicitly, every tab would keep showing data
    # cached from before this run.
    st.cache_data.clear()

    # st.rerun() immediately restarts the script top to bottom — without
    # it, the user would only see the updated data after their next click
    # somewhere else.
    st.rerun()


def render_sidebar() -> None:
    st.sidebar.title("⚙️ Controls")

    if USE_MOCK:
        st.sidebar.info("🧪 Mock mode")
    else:
        st.sidebar.success("🟢 Claude API")
    st.sidebar.caption(f"Model: `{MODEL}`")

    st.sidebar.divider()

    force = st.sidebar.checkbox(
        "Recalculate all",
        value=False,
        help="Ignore the cache in data/extracted/ and recompute every document's analysis from scratch (same as --force in the CLI).",
    )

    if st.sidebar.button("▶️ Run analysis", width="stretch", type="primary"):
        _run_pipeline(force=force)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Documents analyzed: {len(_load_analyzed_documents())}\n\n"
        f"Client reports: {len(_load_client_reports())}"
    )


# ---------------------------------------------------------------------------
# "Overview" tab.
# ---------------------------------------------------------------------------


def render_overview_tab() -> None:
    clients_df = _load_clients_df()
    client_reports = _load_client_reports()
    documents = _load_analyzed_documents()

    if not client_reports:
        _empty_state("No client reports yet.")
        return

    total_mrr_at_risk = sum(r.mrr_usd for r in client_reports.values() if r.risk_level == "high")
    csat_values = [r.avg_csat for r in client_reports.values() if r.avg_csat is not None]
    avg_csat_overall = round(sum(csat_values) / len(csat_values), 2) if csat_values else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients", len(clients_df))
    col2.metric("Documents analyzed", len(documents))
    col3.metric("Average CSAT", avg_csat_overall if avg_csat_overall is not None else "—")
    col4.metric("MRR at risk", f"${total_mrr_at_risk:,}")

    st.divider()
    st.subheader("Clients")

    rows = []
    for _, client_row in clients_df.iterrows():
        report = client_reports.get(client_row["client_id"])
        rows.append(
            {
                "client_id": client_row["client_id"],
                "company": client_row["company_name"],
                "plan": client_row["plan"],
                "mrr_usd": client_row["mrr_usd"],
                "avg_csat": report.avg_csat if report else None,
                "trend": report.sentiment_trend if report else "—",
                "risk": RISK_LABELS.get(report.risk_level, "—") if report else "—",
            }
        )
    table_df = pd.DataFrame(rows)

    def _highlight_risk(value: str) -> str:
        for level, color in RISK_COLORS.items():
            if level in value:
                return f"background-color: {color}33"  # hex 33 ≈ 20% opacity — a tint, not a solid fill
        return ""

    st.dataframe(
        table_df.style.map(_highlight_risk, subset=["risk"]),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Average CSAT by client")
        csat_df = table_df.dropna(subset=["avg_csat"])[["company", "avg_csat"]].set_index("company")
        if not csat_df.empty:
            st.bar_chart(csat_df, color=BAR_COLOR)
        else:
            st.caption("No CSAT data yet.")

    with chart_col2:
        st.subheader("Pain points by category")
        category_counts: dict[str, int] = {}
        for doc in documents:
            for pain in doc.pain_points:
                category_counts[pain.category] = category_counts.get(pain.category, 0) + 1
        if category_counts:
            pain_df = pd.DataFrame(
                sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True),
                columns=["category", "count"],
            ).set_index("category")
            st.bar_chart(pain_df, color=BAR_COLOR)
        else:
            st.caption("No pain points yet.")


# ---------------------------------------------------------------------------
# "Client" tab.
# ---------------------------------------------------------------------------


def render_client_tab() -> None:
    clients_df = _load_clients_df()
    client_reports = _load_client_reports()
    raw_documents = _load_raw_documents()
    analyzed_documents = _load_analyzed_documents()

    if not client_reports:
        _empty_state("No client reports yet.")
        return

    company_by_id = dict(zip(clients_df["client_id"], clients_df["company_name"]))
    selected_id = st.selectbox(
        "Client",
        options=list(clients_df["client_id"]),
        format_func=lambda cid: f"{cid} — {company_by_id.get(cid, '?')}",
    )

    report = client_reports.get(selected_id)
    if report is None:
        st.warning(f"No report for {selected_id} yet. Run the analysis in the sidebar.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Plan", report.plan)
    col2.metric("MRR", f"${report.mrr_usd:,}")
    col3.metric("Avg CSAT", report.avg_csat if report.avg_csat is not None else "—")
    col4.metric("Risk", RISK_LABELS.get(report.risk_level, report.risk_level))

    client_docs = sorted((d for d in analyzed_documents if d.client_id == selected_id), key=lambda d: d.date)
    if len(client_docs) >= 2:
        st.subheader("Sentiment over time")
        sentiment_df = pd.DataFrame(
            {
                "date": [d.date for d in client_docs],
                "sentiment_score": [d.sentiment_score for d in client_docs],
            }
        ).set_index("date")
        st.line_chart(sentiment_df, color=BAR_COLOR)

    st.divider()
    st.subheader("Executive Summary")
    st.write(report.executive_summary)
    st.caption(f"**Risk reasoning:** {report.risk_reasoning}")

    st.divider()
    st.subheader("Themes")
    for theme in report.key_themes:
        # st.expander collapses the block by default — a client with
        # several themes and quotes doesn't turn the page into one long
        # wall of text until the user chooses to open a theme.
        with st.expander(theme.theme):
            st.write(theme.description)
            for quote in theme.evidence:
                st.markdown(f"> {quote}")

    st.divider()
    st.subheader("Response Strategy")
    strat_col1, strat_col2, strat_col3 = st.columns(3)
    with strat_col1:
        st.markdown("**Immediate actions**")
        for action in report.response_strategy.immediate_actions:
            st.markdown(f"- {action}")
    with strat_col2:
        st.markdown("**Talking points**")
        for point in report.response_strategy.talking_points:
            st.markdown(f"- {point}")
    with strat_col3:
        st.markdown("**What to avoid**")
        for item in report.response_strategy.what_to_avoid:
            st.markdown(f"- {item}")

    st.divider()
    st.subheader("Follow-up Draft")
    # key=f"draft_{selected_id}" gives each client its own "slot" in
    # st.session_state: edits to the draft for C001 won't be lost or mixed
    # up with C002 if you switch between them in the selectbox and come
    # back — Streamlit itself keeps the current text_area content under
    # this key across script reruns.
    st.text_area(
        "Feel free to edit before sending",
        value=report.follow_up_draft,
        height=160,
        key=f"draft_{selected_id}",
    )

    st.divider()
    st.subheader("Client Documents")

    analyzed_lookup = {(d.client_id, d.doc_type, d.date): d for d in analyzed_documents}
    docs_for_client = sorted((d for d in raw_documents if d["client_id"] == selected_id), key=lambda d: d["date"])

    for raw_doc in docs_for_client:
        key = (raw_doc["client_id"], raw_doc["doc_type"], raw_doc["date"])
        analyzed = analyzed_lookup.get(key)

        with st.expander(raw_doc["source_file"]):
            left, right = st.columns(2)
            with left:
                st.caption("Original text")
                st.text(raw_doc["text"])
            with right:
                st.caption("Extracted analysis")
                if analyzed is None:
                    st.caption("Not analyzed yet.")
                else:
                    st.write(f"**Sentiment:** {analyzed.sentiment} ({analyzed.sentiment_score:+.2f})")
                    if analyzed.csat_score is not None:
                        st.write(f"**CSAT:** {analyzed.csat_score}/5")
                    st.write(f"**Hidden dissatisfaction:** {analyzed.hidden_dissatisfaction}")
                    st.write(f"**Summary:** {analyzed.summary}")
                    if analyzed.pain_points:
                        st.write("**Pain points:**")
                        for p in analyzed.pain_points:
                            st.markdown(f"- _{p.category}/{p.severity}_ — {p.issue}")
                    if analyzed.feature_requests:
                        st.write("**Feature requests:**")
                        for fr in analyzed.feature_requests:
                            st.markdown(f"- {fr.request}")

    st.divider()

    md_path = CLIENT_REPORTS_DIR / f"{selected_id}.md"
    if md_path.exists():
        # Unlike a regular button, st.download_button doesn't run any code
        # in app.py — it just hands the browser bytes that are already
        # ready, under the given file name and MIME type.
        st.download_button(
            "⬇️ Download client report (.md)",
            data=md_path.read_text(encoding="utf-8"),
            file_name=md_path.name,
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# "Product Report" tab.
# ---------------------------------------------------------------------------


def render_product_tab() -> None:
    product = _load_product_report()
    if product is None:
        _empty_state("No product report yet.")
        return

    st.subheader("Executive Summary")
    st.write(product.insights.executive_summary)

    st.divider()
    st.subheader("Feature Request Cluster Priorities")
    clusters_df = pd.DataFrame(
        [
            {
                "cluster": c.cluster_name,
                "clients": c.client_count,
                "total_mrr_usd": c.total_mrr_usd,
                "at_risk_mrr_usd": c.at_risk_mrr_usd,
                "mentions": c.mention_count,
                "priority_score": c.priority_score,
            }
            # product_report.py already sorts clusters by priority_score when saving
            for c in product.clusters
        ]
    )
    st.dataframe(clusters_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Clusters in Detail")
    for cluster in product.clusters:
        with st.expander(f"{cluster.cluster_name} — priority {cluster.priority_score:.2f}"):
            st.write(cluster.description)
            st.write(f"**Clients:** {', '.join(cluster.clients)}")
            st.write(f"**Plans:** {cluster.plans}")
            if cluster.top_quotes:
                st.write("**Quotes:**")
                for quote in cluster.top_quotes:
                    st.markdown(f"> {quote}")

    st.divider()
    st.subheader("Recommendations")
    for rec in product.insights.top_recommendations:
        st.markdown(f"**{rec.cluster_name}**")
        st.markdown(f"- Why now: {rec.why_now}")
        st.markdown(f"- Business impact: {rec.business_impact}")

    st.subheader("Quick Wins")
    for win in product.insights.quick_wins:
        st.markdown(f"- {win}")

    st.subheader("Risks If Ignored")
    for risk in product.insights.risks_if_ignored:
        clients_str = ", ".join(risk.clients_at_risk) or "n/a"
        st.markdown(f"- {risk.description} — clients: {clients_str}, MRR at risk: ${risk.mrr_at_risk_usd:,}")

    st.divider()
    dl_col1, dl_col2 = st.columns(2)
    md_path = REPORTS_DIR / "product_report.md"
    csv_path = REPORTS_DIR / "feature_requests.csv"
    with dl_col1:
        if md_path.exists():
            st.download_button(
                "⬇️ Download product_report.md",
                data=md_path.read_text(encoding="utf-8"),
                file_name="product_report.md",
                mime="text/markdown",
            )
    with dl_col2:
        if csv_path.exists():
            st.download_button(
                "⬇️ Download feature_requests.csv",
                data=csv_path.read_bytes(),
                file_name="feature_requests.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# "Upload" tab.
# ---------------------------------------------------------------------------


def _save_document(client_id: str, doc_type: str, doc_date: date, text: str) -> bool:
    text = text.strip()
    if not text:
        st.error("Empty document — nothing to save.")
        return False

    filename = f"{client_id}_{doc_type}_{doc_date.isoformat()}.txt"
    target_path = DOCUMENTS_DIR / filename

    if target_path.exists():
        st.error(f"File {filename} already exists. Pick a different date or document type.")
        return False

    target_path.write_text(text, encoding="utf-8")
    st.success(f"Saved: {filename}")

    # We have a new .txt in data/documents/, but no analysis for it yet —
    # clear only the raw-documents cache (not everything at once via
    # st.cache_data.clear()), so the client's document list shows it
    # right away without discarding already-computed reports.
    _load_raw_documents.clear()
    return True


def render_upload_tab() -> None:
    clients_df = _load_clients_df()
    company_by_id = dict(zip(clients_df["client_id"], clients_df["company_name"]))

    st.write(
        "A new document is saved to `data/documents/` with the name "
        "`{client_id}_{doc_type}_{YYYY-MM-DD}.txt` — that's exactly the "
        "format `loader.py` parses. After saving, click **▶️ Run analysis** "
        "in the sidebar so the document gets picked up in the reports."
    )

    client_id = st.selectbox(
        "Client",
        options=list(clients_df["client_id"]),
        format_func=lambda cid: f"{cid} — {company_by_id.get(cid, '?')}",
        key="upload_client_id",
    )
    doc_type = st.selectbox(
        "Document type",
        options=["chat_transcript", "email_thread", "survey"],
        key="upload_doc_type",
    )
    default_date = st.date_input("Document date", value=date.today(), key="upload_default_date")

    st.divider()
    upload_mode = st.radio(
        "Text source", ["Upload file(s)", "Paste text manually"], horizontal=True
    )

    if upload_mode == "Upload file(s)":
        # st.file_uploader returns a list of UploadedFile — file-like
        # objects that live only in this browser session's memory;
        # nothing hits disk until we call write_text ourselves.
        uploaded_files = st.file_uploader(
            ".txt files", type=["txt"], accept_multiple_files=True
        )

        if uploaded_files:
            entries = [(uploaded_files[0], default_date)]
            if len(uploaded_files) > 1:
                st.caption(
                    "Multiple files uploaded — each needs its own date, "
                    "otherwise the filenames would collide."
                )
                entries = [
                    (f, st.date_input(f"Date for {f.name}", value=default_date, key=f"upload_date_{i}"))
                    for i, f in enumerate(uploaded_files)
                ]

            if st.button("💾 Save"):
                saved = 0
                for uploaded_file, file_date in entries:
                    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                    if _save_document(client_id, doc_type, file_date, text):
                        saved += 1
                if saved:
                    st.info(f"Files saved: {saved}")
    else:
        manual_text = st.text_area("Document text", height=250, key="upload_manual_text")
        if st.button("💾 Save"):
            _save_document(client_id, doc_type, default_date, manual_text)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("📊 Smart Customer Feedback Analyzer")
    st.caption("A learning project: customer feedback analysis for B2B SaaS.")

    render_sidebar()

    # st.tabs — navigation without a page reload: switching tabs is also
    # a full script rerun, Streamlit just remembers which tab was active
    # and only renders the content inside its own "with" block.
    tab_overview, tab_client, tab_product, tab_upload = st.tabs(
        ["📈 Overview", "🧑‍💼 Client", "🏭 Product Report", "⬆️ Upload"]
    )

    with tab_overview:
        render_overview_tab()
    with tab_client:
        render_client_tab()
    with tab_product:
        render_product_tab()
    with tab_upload:
        render_upload_tab()


if __name__ == "__main__":
    main()
