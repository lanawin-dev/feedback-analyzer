"""
Loads raw data: clients from CSV and documents from data/documents/.

Documents aren't parsed "for meaning" here — this is just reading files
and splitting the filename into its parts. Actual analysis (sentiment,
pain points) happens in stage 2, once the Claude API is wired in.
"""

import re
from pathlib import Path

import pandas as pd

from config import CLIENTS_CSV, DOCUMENTS_DIR

# Filenames must look like: C001_chat_transcript_2026-08-14.txt
# Groups: client_id, doc_type (one of three fixed values), date.
FILENAME_PATTERN = re.compile(
    r"^(?P<client_id>[^_]+)_"
    r"(?P<doc_type>chat_transcript|email_thread|survey)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})\.txt$"
)


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """Reads every .txt file in documents_dir and returns a list of dicts
    shaped like {client_id, doc_type, date, text, source_file}."""
    documents = []

    for path in sorted(documents_dir.glob("*.txt")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            print(f"Skipping file with an unexpected name: {path.name}")
            continue

        documents.append(
            {
                "client_id": match.group("client_id"),
                "doc_type": match.group("doc_type"),
                "date": match.group("date"),
                "text": path.read_text(encoding="utf-8"),
                "source_file": path.name,
            }
        )

    return documents


def load_clients(clients_csv: Path = CLIENTS_CSV) -> pd.DataFrame:
    """Reads the client table via pandas."""
    return pd.read_csv(clients_csv)


def _print_summary() -> None:
    clients = load_clients()
    documents = load_documents()

    print(f"Clients: {len(clients)}")
    print(f"Total documents: {len(documents)}")

    counts_by_type: dict[str, int] = {}
    for doc in documents:
        counts_by_type[doc["doc_type"]] = counts_by_type.get(doc["doc_type"], 0) + 1

    for doc_type, count in sorted(counts_by_type.items()):
        print(f"  {doc_type}: {count}")


if __name__ == "__main__":
    _print_summary()
