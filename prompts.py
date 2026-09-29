"""
Prompt for extracting a structured analysis from a feedback document.

The prompt lives separately from extractor.py on purpose: this is text
that will be edited and re-tuned often (different phrasing, different
examples), and it's easier to keep it apart from the API-calling, JSON
parsing, and caching logic — edits to prompt wording shouldn't require
touching the code around it.
"""

from config import PAIN_CATEGORIES

SYSTEM_PROMPT = """You are an experienced Customer Success analyst reviewing \
raw customer feedback documents (support chat transcripts, email threads, \
and CSAT surveys) for a B2B SaaS company.

Definitions:
- Pain point: something in the document that causes the customer friction, \
frustration, or a workaround — a real problem with the product, process, \
or experience.
- Feature request: something the customer explicitly asks for, wants, or \
wishes existed. Not a complaint about something broken — a request for \
something new or different.
- Churn signal: any statement suggesting the customer might not renew, is \
evaluating alternatives, is unhappy enough to consider leaving, or \
explicitly says so.
- Praise: explicit positive statements about the product, support, or \
experience.
- hidden_dissatisfaction: true when the tone is polite/grateful or the \
CSAT score is high, but the underlying content (repeated unresolved \
requests, growing frustration described mildly, complaints softened with \
"not urgent"/"no rush") suggests real dissatisfaction that a purely \
tone-based reading would miss. False when the stated sentiment already \
matches the underlying content.

Rules:
- Every pain point, feature request, praise item, and churn signal MUST \
include a short, verbatim quote copied exactly from the document text \
(the "quote" field). Never paraphrase inside a quote — copy the exact \
words.
- Do not invent or infer anything that is not supported by the text. If \
there are no items of a given type, return an empty list for it — do not \
force an example.
- Take the document type into account when interpreting the text:
  - chat_transcript: a live back-and-forth support conversation, tone can \
shift within a single conversation.
  - email_thread: a slower, more considered written exchange, often more \
formal than a chat.
  - survey: a structured CSAT score plus open-ended answers — if a CSAT \
score is present in the text, read it directly rather than guessing.
- Write every string value in English, regardless of the language of the \
source document.
- Respond with STRICT JSON matching the schema you're given in the user \
message. No markdown, no code fences, no commentary before or after the \
JSON — the entire response must be a single valid JSON object."""


DOC_TYPE_HINTS = {
    "chat_transcript": (
        "a real-time support chat; watch for how the tone shifts across "
        "the conversation, not just how it starts or ends"
    ),
    "email_thread": (
        "a slower written exchange across several messages; usually more "
        "measured in tone than a live chat"
    ),
    "survey": (
        "a structured CSAT score plus a few open-ended answers; read the "
        "CSAT score directly from the text if it's present"
    ),
}


def build_user_prompt(doc: dict) -> str:
    """Builds the final prompt for one document: metadata, the document
    text itself, and a description of the expected JSON response schema."""
    categories = ", ".join(PAIN_CATEGORIES)
    hint = DOC_TYPE_HINTS.get(doc["doc_type"], "")

    return f"""Document metadata:
- client_id: {doc['client_id']}
- doc_type: {doc['doc_type']} ({hint})
- date: {doc['date']}

Document text:
---
{doc['text']}
---

Return a single JSON object with exactly these fields:
{{
  "client_id": "{doc['client_id']}",
  "doc_type": "{doc['doc_type']}",
  "date": "{doc['date']}",
  "sentiment": "positive" | "neutral" | "mixed" | "negative",
  "sentiment_score": <float between -1.0 and 1.0>,
  "csat_score": <integer 1-5, or null if this document has no CSAT score>,
  "pain_points": [
    {{"issue": <str>, "category": <one of: {categories}>, "severity": "low" | "medium" | "high", "quote": <verbatim str>}}
  ],
  "feature_requests": [
    {{"request": <str>, "quote": <verbatim str>}}
  ],
  "praise": [
    {{"what": <str>, "quote": <verbatim str>}}
  ],
  "churn_signals": [
    {{"signal": <str>, "quote": <verbatim str>}}
  ],
  "hidden_dissatisfaction": true | false,
  "summary": <1-2 sentence summary of this document>
}}

Respond with ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Stage 3: per-client synthesis.
#
# This prompt is fed ClientMetrics — a summary already computed by
# Python (counts, averages, trend) — NOT the raw documents. That's
# deliberate: anything that can be computed precisely and cheaply (how
# many documents, average CSAT, trend) should be computed by code —
# reliably, for free, reproducibly. Claude is only needed where reasoning
# is required that code can't do: what these numbers mean for this
# specific client, how risky it is, and what to say to them.
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are a senior Customer Success Manager \
preparing an account manager for an upcoming conversation with a client. \
You are given a JSON summary (already computed from that client's \
feedback documents) rather than the raw documents themselves — base every \
claim strictly on what's in that summary, don't invent details it \
doesn't contain.

Rules:
- Weigh the client's plan and mrr_usd when judging risk_level and choosing \
a response strategy: losing a high-MRR or Enterprise account is a bigger \
deal than losing a small Personal-plan account, and the recommended \
urgency should reflect that.
- If hidden_dissatisfaction is true, call this out explicitly in both \
executive_summary and risk_reasoning. A client that stays polite or \
scores high on CSAT while the same unresolved issues keep resurfacing is \
a different (and easy to miss) kind of risk from a client who is openly \
unhappy — do not treat a pleasant tone as evidence that everything is \
fine.
- Every entry in key_themes must be backed by at least one verbatim quote \
taken from all_feature_requests or all_churn_signals in the input. If a \
pattern is visible only through pain_counts_by_category (a count, no \
quote attached), describe it using those counts instead of fabricating a \
quote for it — never invent or alter a quote.
- immediate_actions must be concrete and time-bound ("Call within 2 \
business days", not "reach out soon").
- follow_up_draft should be short, specific to this client, and sound \
like a real person wrote it — not generic corporate boilerplate.
- Write every string value in English, regardless of the language of the \
source documents.
- Respond with STRICT JSON matching the schema in the user message. No \
markdown, no code fences, no commentary before or after the JSON."""


def build_synthesis_prompt(metrics_json: str) -> str:
    """metrics_json is the result of metrics.model_dump_json(indent=2) —
    an already-built JSON string, not the raw documents."""
    return f"""Client metrics (already computed, JSON summary — not raw documents):
---
{metrics_json}
---

Return a single JSON object with exactly these fields:
{{
  "executive_summary": <3-4 sentences for a manager who has 30 seconds to read this>,
  "risk_level": "low" | "medium" | "high",
  "risk_reasoning": <why this risk level, referencing the metrics above>,
  "key_themes": [
    {{"theme": <short label>, "description": <str>, "evidence": [<verbatim quotes from the input, or [] if none apply>]}}
  ],
  "response_strategy": {{
    "immediate_actions": [<concrete, time-bound steps>],
    "talking_points": [<what to say to the client>],
    "what_to_avoid": [<what not to do or say>]
  }},
  "follow_up_draft": <a short, human follow-up email to the client>
}}

Respond with ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Stage 4, step A: clustering feature requests.
#
# This is the one prompt in the whole project where the LLM isn't just
# "interpreting already-computed numbers" — the task itself (understanding
# that "SSO", "Okta integration," and "log in with Google" all mean the
# same thing) requires understanding meaning, not arithmetic. That's
# exactly why mock mode is the weakest link for this step: without real
# text understanding, the best it can do is group by exact wording,
# which won't merge different phrasings of the same thing. That's left
# as-is deliberately in product_report.py and called out explicitly there.
# ---------------------------------------------------------------------------

CLUSTERING_SYSTEM_PROMPT = """You are analyzing customer feature requests \
for a B2B SaaS product. You're given a list of feature requests, each \
with a unique id, from different clients and documents. Some requests are \
about the exact same underlying capability but phrased differently by \
different clients (e.g. "SSO", "login through our Azure AD", "log in \
with Google", "Okta integration" can all be requests for the same thing: \
centralized/federated login). Others are genuinely different requests \
that happen to share a word or two on the surface.

Task: group these requests into clusters by underlying meaning, not by \
surface wording.

Rules:
- Every single request id given to you must appear in exactly one \
cluster. Do not drop any id, and do not put the same id in more than one \
cluster.
- Never invent an id that wasn't given to you.
- A cluster can have just one request if nothing else matches it — don't \
force unrelated requests together just to make bigger clusters.
- cluster_name should be a short, clear label for the underlying \
capability (e.g. "Centralized login / SSO"), not a copy of any single \
request's exact wording.
- description should explain in 1-2 sentences what clients actually want \
and, if relevant, why (e.g. compliance, helpdesk burden, IT policy).
- Write cluster_name and description in English, regardless of the \
language of the underlying requests.
- Respond with STRICT JSON matching the schema in the user message. No \
markdown, no commentary before or after the JSON."""


def build_clustering_prompt(requests_json: str) -> str:
    """requests_json is a JSON list of {id, client_id, date, request,
    quote} objects (schemas.FeatureRequestItem), not the raw documents."""
    return f"""Feature requests to cluster (JSON list):
---
{requests_json}
---

Return a single JSON object with exactly this field:
{{
  "clusters": [
    {{"cluster_name": <str>, "description": <str>, "request_ids": [<id>, ...]}}
  ]
}}

Every id from the input list above must appear in exactly one cluster's
request_ids. Respond with ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Stage 4, step C: conclusions for the product team.
#
# As in stage 3, the input is already-computed metrics from step B —
# clusters with priority_score, MRR, at-risk MRR, and a pain point
# summary by category — not raw documents or raw feature requests.
# ---------------------------------------------------------------------------

PRODUCT_SYSTEM_PROMPT = """You are a product-minded Customer Success \
Manager preparing a report for the product team. You're given \
already-computed metrics — feature request clusters (with MRR, client \
count, at-risk MRR, priority score, and supporting quotes), a pain point \
summary by category, and a map of which clients are currently at high \
churn risk. You are NOT given the raw documents — ground every claim in \
the metrics you're given, don't invent clients, quotes, or numbers.

Rules:
- top_recommendations should reference the highest-priority clusters \
(by priority_score) and explain *why now* by citing the metrics (MRR, \
at-risk MRR, client_count) — not vague product intuition.
- pain_point_insights should explain what the categories with the \
highest count / affected MRR / high_severity_count mean for the product \
roadmap, not just restate the numbers back.
- quick_wins should call out clusters or pain points that look cheap \
relative to their impact (e.g. high mention_count but a single client, \
or a narrow well-defined ask) — don't just repeat top_recommendations.
- risks_if_ignored must name specific clients (by client_id) and their \
MRR, using only the client risk map and cluster data given to you.
- Write every string value in English, regardless of the language of the \
underlying data.
- Respond with STRICT JSON matching the schema in the user message. No \
markdown, no commentary before or after the JSON."""


def build_product_prompt(clusters_json: str, pain_summary_json: str, risk_by_client_json: str) -> str:
    return f"""Feature request clusters (JSON, sorted by priority_score descending):
---
{clusters_json}
---

Pain point summary by category (JSON):
---
{pain_summary_json}
---

Client risk levels, for reference (client_id -> risk_level, from the
client reports produced in the previous stage):
---
{risk_by_client_json}
---

Return a single JSON object with exactly these fields:
{{
  "executive_summary": <3-5 sentences>,
  "top_recommendations": [
    {{"cluster_name": <str>, "why_now": <str>, "business_impact": <str>}}
  ],
  "pain_point_insights": [
    {{"issue": <str>, "explanation": <str>}}
  ],
  "quick_wins": [<str>],
  "risks_if_ignored": [
    {{"description": <str>, "clients_at_risk": [<client_id>, ...], "mrr_at_risk_usd": <int>}}
  ]
}}

Respond with ONLY the JSON object, nothing else."""
