"""
Calibrate the relevance threshold
----------------------------------
Distances depend on your embedding model and your data, so the cut-off in
config.RELEVANCE_MAX_DISTANCE should be measured, not guessed.

This script prints the best-match distance for a list of questions your
Acts SHOULD cover and a list they should NOT, then suggests a cut-off
between the two groups.

1. EDIT the two lists below so they match the Acts you have loaded.
2. Run:   python calibrate_threshold.py
3. Put the suggested value in config.py:  RELEVANCE_MAX_DISTANCE = <value>
"""

IN_SCOPE = [
    "My new phone stopped working after 3 days and the seller refuses a refund.",
    "An online shop sent me a fake product. Where can I complain?",
    "A company is running a misleading advertisement. What can I do?",
    "What is the time limit to file a consumer complaint?",
    "Which commission should I approach for a complaint about a defective car?",
    "What should i do if a person snached my phone and ran away?",
    "A man is threatening to kill me and my family. What should I do?",
    "A person is blackmailing me with my private photos. How can I get help?",
    "I was harassed by a stranger on the street. Can I file a complaint?",
    "I was scammed by a person pretending to be a bank official. How can I report this?",
]   
OUT_OF_SCOPE = [
    "How do I bake sourdough bread at home?",
    "Who won the cricket world cup in 2011?",
    "What is the capital of France?",
    "Explain how quantum computers work.",
    "Suggest a good workout plan for beginners.",
    "What are the health benefits of meditation?",
    "What is the history of the Roman Empire?",
    "How do I change a flat tire on my car?",
    "What are the best practices for growing tomatoes in a garden?",
]


def suggest_threshold(in_scope_distances, out_scope_distances):
    """Midpoint between the worst in-scope and best out-of-scope distance.
    Returns (suggestion, note)."""
    worst_in, best_out = max(in_scope_distances), min(out_scope_distances)
    if worst_in < best_out:
        return round((worst_in + best_out) / 2, 2), "Groups are cleanly separated."
    return round(worst_in + 0.05, 2), (
        "WARNING: the groups overlap. The suggestion keeps every in-scope "
        "question, but some off-topic ones will get through. Add more varied "
        "questions or consider a better embedding model / reranker."
    )


def main():
    from sentence_transformers import SentenceTransformer
    import chromadb
    from ask_law_bot import (EMBEDDING_MODEL_NAME, CHROMA_DB_PATH,
                             COLLECTION_NAME, retrieve_sections)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_or_create_collection(COLLECTION_NAME)

    def best(q):
        return retrieve_sections(q, model=model, collection=collection, top_k=1)[0]["distance"]

    print("\nIN-SCOPE questions (should have SMALL distances):")
    in_d = [best(q) for q in IN_SCOPE]
    for q, d in zip(IN_SCOPE, in_d):
        print(f"  {d:.3f}  {q}")

    print("\nOUT-OF-SCOPE questions (should have LARGE distances):")
    out_d = [best(q) for q in OUT_OF_SCOPE]
    for q, d in zip(OUT_OF_SCOPE, out_d):
        print(f"  {d:.3f}  {q}")

    value, note = suggest_threshold(in_d, out_d)
    print(f"\nSuggested RELEVANCE_MAX_DISTANCE = {value}\n{note}")


if __name__ == "__main__":
    main()
