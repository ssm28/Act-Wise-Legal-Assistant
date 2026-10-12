"""
Embed law sections and store them in a local vector database
---------------------------------------------------------------
Reads the section-level JSON produced by law_text_to_json.py, turns each
section's text into an embedding (a list of numbers representing its
meaning), and stores it in a local Chroma database on disk.

This is what lets your bot later find "which sections are relevant to
this question" by MEANING, not just by keyword matching.

Install (one-time):
    pip install chromadb sentence-transformers

Usage:
    python embed_and_store.py output.json

You can run this on EACH Act's JSON file separately — everything gets
added to the same local database (in a folder called chroma_db/), so your
bot can search across all Acts you've added so far.
"""

from collections import Counter
import sys
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# This is a small, free, open-source embedding model that runs entirely
# on your own machine — no API key, no cost, no internet needed after the
# first download (it downloads itself automatically, ~80MB, the first
# time you run this script).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where the local vector database is stored on disk. It persists between
# runs — running this script again just adds more data to it.
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "law_sections"


def make_id(record: dict) -> str:
    """Build a unique, safe ID for each record, e.g.
    'consumer_protection_act_2019_sec_65'."""
    act_slug = re.sub(r'[^a-z0-9]+', '_', record["act"].lower()).strip('_')
    section_slug = re.sub(r'[^a-z0-9]+', '_', str(record["section"]).lower()).strip('_')
    return f"{act_slug}_sec_{section_slug}"


def load_records(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        print("Usage: python embed_and_store.py <sections.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    records = load_records(json_path)
    print(f"Loaded {len(records)} section records from {json_path}")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          f"(first run downloads it, ~80MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    ids = [make_id(r) for r in records]
    texts = [r["text"] for r in records]
    metadatas = [
        {
            "act": r["act"],
            "chapter": r.get("chapter") or "",
            "section": str(r["section"]),
        }
        for r in records
    ]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Connecting to local database at {CHROMA_DB_PATH}/ ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    from collections import Counter
    counts = Counter(ids)
    bad = [k for k, v in counts.items() if v > 1]
    if bad:
        raise SystemExit(f"Duplicate IDs, fix these in your JSON first: {bad[:10]}")

    # upsert = add new, or update if the same ID already exists — safe to
    # re-run this script if you improve your JSON later.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {len(records)} sections in the '{COLLECTION_NAME}' collection.")
    print(f"Total sections in database now: {collection.count()}")


if __name__ == "__main__":
    main()
