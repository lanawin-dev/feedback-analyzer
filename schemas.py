"""
Pydantic models for the result of analyzing a single document.

What pydantic is: a library that describes the "shape" of data through a
plain Python class with type annotations (BaseModel), then validates that
the data it's given (say, JSON returned by Claude) actually matches that
shape — correct types, required fields present, numbers within range, and
so on. If something's off, pydantic raises a ValidationError with a clear
explanation of exactly what didn't match. That's especially useful
wherever data arrives from "outside" (an API, a file) and can't be
trusted in advance.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from config import PAIN_CATEGORIES


class PainPoint(BaseModel):
    issue: str
    category: str
    severity: Literal["low", "medium", "high"]
    quote: str

    @field_validator("category")
    @classmethod
    def category_must_be_known(cls, value: str) -> str:
        # Categories live in config.PAIN_CATEGORIES, not hardcoded again
        # here — otherwise the two lists would drift out of sync over time.
        if value not in PAIN_CATEGORIES:
            raise ValueError(
                f"Unknown pain point category: {value!r}. "
                f"Expected one of: {PAIN_CATEGORIES}"
            )
        return value


class FeatureRequest(BaseModel):
    request: str
    quote: str


class Praise(BaseModel):
    what: str
    quote: str


class ChurnSignal(BaseModel):
    signal: str
    quote: str


class DocumentAnalysis(BaseModel):
    """The result of analyzing a single feedback document."""

    client_id: str
    doc_type: Literal["chat_transcript", "email_thread", "survey"]
    date: str

    sentiment: Literal["positive", "neutral", "mixed", "negative"]
    # Field(ge=..., le=...) is a pydantic constraint: "must be >= -1.0 and
    # <= 1.0." If the model (or the mock) returns, say, 1.4, pydantic
    # refuses to build the object and explains why.
    sentiment_score: float = Field(ge=-1.0, le=1.0)

    # CSAT only exists on survey documents, None everywhere else.
    # Optional[int] = can be an int, or can be None.
    csat_score: Optional[int] = Field(default=None, ge=1, le=5)

    # default_factory=list gives each new object its own fresh empty
    # list. Using default=[] directly would share one mutable list across
    # every instance — a classic Python gotcha — so pydantic (like Python
    # in general) asks for a factory instead.
    pain_points: list[PainPoint] = Field(default_factory=list)
    feature_requests: list[FeatureRequest] = Field(default_factory=list)
    praise: list[Praise] = Field(default_factory=list)
    churn_signals: list[ChurnSignal] = Field(default_factory=list)

    # Polite tone / decent CSAT, but the client is unhappy in substance.
    hidden_dissatisfaction: bool

    summary: str


# ---------------------------------------------------------------------------
# Stage 3: per-client aggregation.
#
# Two different sources of truth, so two different classes:
# - ClientMetrics is computed by aggregator.py with plain Python (sums,
#   averages, counts) from already-saved DocumentAnalysis. No LLM.
# - ClientInsights is written by Claude (or the mock in synthesizer.py) —
#   this is interpretation: conclusions, strategy, a follow-up draft,
#   the kind of thing that needs reasoning, not arithmetic.
# ClientReport just merges both sets of fields into one flat object, so a
# client report has a single JSON representation.
# ---------------------------------------------------------------------------


class FeatureRequestRecord(BaseModel):
    """A feature request with provenance attached — needed for the report,
    and so Claude can reference the date/document type during synthesis,
    not just the request text."""

    request: str
    quote: str
    date: str
    doc_type: Literal["chat_transcript", "email_thread", "survey"]


class ChurnSignalRecord(BaseModel):
    signal: str
    quote: str
    date: str
    doc_type: Literal["chat_transcript", "email_thread", "survey"]


class ClientMetrics(BaseModel):
    """A purely computed per-client summary — not one field here is
    decided by an LLM, everything comes from arithmetic over
    DocumentAnalysis + clients.csv."""

    client_id: str
    company_name: str
    plan: str
    mrr_usd: int

    doc_count: int
    doc_types: dict[str, int]

    avg_csat: Optional[float] = None
    avg_sentiment_score: float

    sentiment_trend: Literal["improving", "stable", "declining"]

    pain_counts_by_category: dict[str, int]
    high_severity_count: int

    all_feature_requests: list[FeatureRequestRecord] = Field(default_factory=list)
    all_churn_signals: list[ChurnSignalRecord] = Field(default_factory=list)

    hidden_dissatisfaction: bool


class ThemeEvidence(BaseModel):
    theme: str
    description: str
    evidence: list[str] = Field(default_factory=list)


class ResponseStrategy(BaseModel):
    immediate_actions: list[str] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    what_to_avoid: list[str] = Field(default_factory=list)


class ClientInsights(BaseModel):
    """Interpretation of the metrics — the part that genuinely needs
    reasoning rather than counting, which is why this (and only this) is
    what Claude does."""

    executive_summary: str
    risk_level: Literal["low", "medium", "high"]
    risk_reasoning: str
    key_themes: list[ThemeEvidence] = Field(default_factory=list)
    response_strategy: ResponseStrategy
    follow_up_draft: str


class ClientReport(ClientMetrics, ClientInsights):
    """ClientMetrics + ClientInsights as one flat object — easier to save
    and render into a single JSON/Markdown file per client."""


# ---------------------------------------------------------------------------
# Stage 4: cross-client report for the product team.
#
# Three steps — three different kinds of task — so three groups of models:
# - FeatureRequestItem / RequestCluster: step A, grouping by meaning —
#   this is a semantic task (understanding that "SSO" and "log in with
#   Google" are the same thing), and code fundamentally can't do this as
#   well as Claude, which is why this is the one step where mock mode is
#   a genuinely weak approximation rather than just "the same thing, but free."
# - ClusterMetrics / PainPointSummary: step B, pure aggregation — Python.
# - ProductInsights: step C, interpreting step B's metrics for the
#   product team — Claude again, on the same principle as ClientInsights
#   in stage 3.
# ---------------------------------------------------------------------------


class FeatureRequestItem(BaseModel):
    """A single feature request with its own id — the unit of grouping in
    step A. The id is only used for references inside the clustering step
    (short and unambiguous, unlike the full request text)."""

    id: str
    client_id: str
    date: str
    request: str
    quote: str


class RequestCluster(BaseModel):
    """The result of clustering — what Claude (or the mock) returned for step A."""

    cluster_name: str
    description: str
    request_ids: list[str] = Field(default_factory=list)


class FeatureClusterResult(BaseModel):
    """Wrapper for validating step A's raw JSON response (a plain list of
    RequestCluster underneath — the wrapper exists because pydantic
    validates JSON objects, not bare lists)."""

    clusters: list[RequestCluster]


class ClusterMetrics(BaseModel):
    """Metrics for a single cluster — computed entirely by product_report.py
    with plain Python; not one field in ClusterMetrics is decided by an LLM."""

    cluster_name: str
    description: str

    client_count: int
    clients: list[str] = Field(default_factory=list)

    total_mrr_usd: int
    plans: dict[str, int] = Field(default_factory=dict)
    at_risk_mrr_usd: int

    mention_count: int
    first_mentioned: str
    last_mentioned: str

    top_quotes: list[str] = Field(default_factory=list)

    priority_score: float = Field(ge=0.0, le=1.0)


class PainPointSummary(BaseModel):
    """Summary for a single pain point category, across all clients at once."""

    category: str
    count: int
    affected_mrr_usd: int
    high_severity_count: int


class TopRecommendation(BaseModel):
    cluster_name: str
    why_now: str
    business_impact: str


class PainPointInsight(BaseModel):
    issue: str
    explanation: str


class RiskIfIgnored(BaseModel):
    description: str
    clients_at_risk: list[str] = Field(default_factory=list)
    mrr_at_risk_usd: int


class ProductInsights(BaseModel):
    """Interpretation of step B's metrics for the product team — written
    by Claude (or the mock template in product_report.py)."""

    executive_summary: str
    top_recommendations: list[TopRecommendation] = Field(default_factory=list)
    pain_point_insights: list[PainPointInsight] = Field(default_factory=list)
    quick_wins: list[str] = Field(default_factory=list)
    risks_if_ignored: list[RiskIfIgnored] = Field(default_factory=list)


class ProductReport(BaseModel):
    """The final stage-4 report: step B's metrics (unchanged, as computed
    by Python) + step C's interpretation (as written by Claude/the mock)."""

    clusters: list[ClusterMetrics]
    pain_point_summary: list[PainPointSummary]
    insights: ProductInsights
