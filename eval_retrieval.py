"""
Retrieval evaluation: Hit@k and MRR
-------------------------------------
Measures how well the SEARCH step finds the right law sections, which is
the number your report needs.

1. Copy eval_set_template.json to eval_set.json and fill in the "expected"
   field for each question: the section(s) a correct answer must come from.
   Aim for 30 to 50 questions covering different topics and Acts.
   Example entry:
     {"question": "...", "expected": [{"act": "Consumer Protection Act, 2019", "section": "35"}]}
2. Run:   python eval_retrieval.py            (default k=4, like the app)
          python eval_retrieval.py 5          (k=5)

Metrics:
  Hit@k : share of questions where AT LEAST ONE expected section appears in
          the top k results.
  MRR   : mean of 1/rank of the first expected section found (rank 1 -> 1.0,
          rank 2 -> 0.5, not found within 10 -> 0). Rewards ranking it HIGH.
"""

import json
import sys

MAX_RANK = 10


def is_match(result, expected) -> bool:
    same_section = str(result["section"]).upper() == str(expected["section"]).upper()
    act = expected.get("act")
    same_act = (not act) or act.lower() in result["act"].lower()
    return same_section and same_act


def evaluate(items, retrieve_fn, k=4):
    """items: list of {"question", "expected"}. retrieve_fn(question, n) -> sections.
    Items with an empty 'expected' list are skipped."""
    rows = []
    for item in items:
        if not item.get("expected"):
            continue
        results = retrieve_fn(item["question"], MAX_RANK)
        rank = next(
            (i + 1 for i, r in enumerate(results)
             if any(is_match(r, e) for e in item["expected"])),
            None,
        )
        rows.append({"question": item["question"], "rank": rank})

    n = len(rows)
    if n == 0:
        return {"n": 0, "hit_at_k": None, "mrr": None, "rows": rows}
    hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    return {"n": n, "hit_at_k": hits / n, "mrr": mrr, "rows": rows}


def main():
    from sentence_transformers import SentenceTransformer
    import chromadb
    from ask_law_bot import (EMBEDDING_MODEL_NAME, CHROMA_DB_PATH,
                             COLLECTION_NAME, retrieve_sections)

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with open("eval_set.json", "r", encoding="utf-8") as f:
        items = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_or_create_collection(COLLECTION_NAME)

    def retrieve(q, n):
        return retrieve_sections(q, model=model, collection=collection, top_k=n)

    result = evaluate(items, retrieve, k=k)
    skipped = len(items) - result["n"]
    if result["n"] == 0:
        print("No questions with an 'expected' section yet. Fill in eval_set.json first.")
        return

    print(f"\nEvaluated {result['n']} questions ({skipped} skipped: no expected section)\n")
    for r in result["rows"]:
        status = f"rank {r['rank']}" if r["rank"] else f"NOT in top {MAX_RANK}"
        print(f"  [{status:>14}]  {r['question'][:80]}")
    print(f"\nHit@{k}: {result['hit_at_k']:.1%}")
    print(f"MRR   : {result['mrr']:.3f}")


if __name__ == "__main__":
    main()
