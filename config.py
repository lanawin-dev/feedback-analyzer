"""
Central place for the project's paths and settings.

Every other module imports paths from here instead of assembling them by
hand — if the folder layout ever changes, this is the only file that
needs to change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv() looks for a .env file in the current directory and copies
# KEY=value pairs from it into environment variables (os.environ), as if
# you'd run `export ANTHROPIC_API_KEY=...` in the terminal before
# starting the script. If there's no .env file, it silently does nothing —
# no error.
load_dotenv()

# The folder this config.py lives in — every other path is built from it.
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CLIENTS_CSV = DATA_DIR / "clients.csv"
GROUND_TRUTH_JSON = DATA_DIR / "ground_truth.json"
EXTRACTED_DIR = DATA_DIR / "extracted"
REPORTS_DIR = DATA_DIR / "reports"
CLIENT_REPORTS_DIR = REPORTS_DIR / "clients"

# Claude model used for text analysis.
MODEL = "claude-sonnet-5"

# The key is read from the ANTHROPIC_API_KEY environment variable (put
# there by load_dotenv() from .env). os.getenv returns None if the
# variable isn't set — that's fine while we're running in mock mode.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def _read_use_mock() -> bool:
    """USE_MOCK is controlled from the outside rather than hardcoded,
    specifically for deployment: on Streamlit Cloud (or any other host)
    the mode needs to be switchable from the hosting settings, without
    touching code or redeploying. Lookup order:

    1. The USE_MOCK environment variable — works locally (via .env,
       already read by load_dotenv() above) and on almost any server.
    2. st.secrets["USE_MOCK"] — on Streamlit Cloud, secrets are NOT
       copied into os.environ by default, only into st.secrets, so
       without this step secrets.toml simply wouldn't work there.
    3. If it's set nowhere — True. A safe default: without explicitly
       opting in, the app shouldn't try to call a paid API.
    """
    raw = os.getenv("USE_MOCK")

    if raw is None:
        try:
            import streamlit as st

            raw = st.secrets.get("USE_MOCK")
        except Exception:
            # No streamlit, no secrets.toml, or the script wasn't started
            # via `streamlit run` — any of these just means "no secrets
            # configured," not "something broke."
            raw = None

    if raw is None:
        return True

    return str(raw).strip().lower() not in {"false", "0", "no", "off"}


USE_MOCK = _read_use_mock()

# Pain point categories that the model (or the mock) must pick from for
# schemas.PainPoint.category. Kept in one place so prompts.py and
# schemas.py always reference the same set of values.
PAIN_CATEGORIES = [
    "onboarding",
    "bugs_performance",
    "usability",
    "missing_feature",
    "integration",
    "pricing",
    "support",
    "other",
]

# aggregator.py compares the average sentiment_score of the first half of
# a client's documents (by date) against the second half. If the
# difference is smaller than this threshold in absolute value, the trend
# is considered "stable" rather than improving/declining. Kept here so it
# can be tuned without digging into aggregator.py's logic.
SENTIMENT_TREND_THRESHOLD = 0.15

# Weights for the feature-request cluster priority_score in
# product_report.py. Must sum to 1.0. Each metric is min-max normalized
# to 0..1 across all clusters, then combined with this weight:
#   mrr          — how much total MRR stands behind the request
#   client_count — how many distinct clients are asking for it (not one voice)
#   at_risk      — how much of that MRR is already at high churn risk,
#                  i.e. "may be lost right now if nothing is done"
PRIORITY_WEIGHTS = {
    "mrr": 0.5,
    "client_count": 0.3,
    "at_risk": 0.2,
}
