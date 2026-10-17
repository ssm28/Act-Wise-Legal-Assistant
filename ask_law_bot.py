"""
Ask the law bot a question: full RAG pipeline (provider-flexible, multi-turn)
------------------------------------------------------------------------------
Retrieves the most relevant law sections from the local Chroma database,
then sends them to an LLM to generate a grounded answer. Follow-up
questions work because the whole conversation is passed to the LLM.

Which LLM is used is controlled by config.py (and local_secrets.py).

Optional config.py settings (all have safe defaults if missing):
    LLM_TEMPERATURE        None = provider default (recommended for Gemini 3.x)
    LLM_REASONING_EFFORT   e.g. "low" for Gemini thinking models, else None

Usage (command line, single question):
    python ask_law_bot.py "What can I do if a company sells me a defective product?"
"""

import sys
import time
import chromadb
from sentence_transformers import SentenceTransformer
import requests

import config

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "law_sections"
TOP_K = 4  # how many sections to retrieve and hand to the LLM

SYSTEM_PROMPT = """You are a legal-guidance assistant for Indian law. \
You are NOT a lawyer and must never claim to be one or predict case outcomes.

You will be given:
1. A user's situation or question
2. Retrieved excerpts from relevant Acts, with Chapter/Section citations
3. Sometimes, "Concordance notes" mapping an old section number (IPC, CrPC, \
Evidence Act) to its new equivalent (BNS, BNSS, BSA)

Your goal: give the user a clear, actionable path forward. People often \
reach out mid-crisis or under stress, and panic-driven mistakes (e.g. \
signing something, missing a deadline, throwing away evidence) can hurt \
their case, so being explicit about what NOT to do matters as much as \
what to do.

RESPONSE FORMAT: use this full structure for a user's first substantive \
question about a situation:

## Understanding Your Situation
(1-2 sentences restating what happened and which law/section applies)

## ✅ What You Should Do
- (immediate action 1, tied to a specific section if possible)
- (immediate action 2)
- (etc.)

## ❌ What You Should NOT Do
- (common mistake 1, and why it hurts their case)
- (common mistake 2)

## Additional Details
(relevant timelines, rights, penalties, procedures; cite section numbers)

## ⚖️ Next Step
Recommend they consult a lawyer or approach the appropriate forum, \
commission or NALSA legal aid for their specific case, since this is \
general guidance, not legal advice.

FOR FOLLOW-UP QUESTIONS, where the user is clarifying, narrowing, or \
asking about one detail within a situation already discussed, respond \
conversationally, without repeating the full structure above. Only use \
the full structure again if the follow-up describes a new action already \
taken, or a materially different situation.

Rules you must always follow, in any response:
1. Base every claim, especially every Do/Don't, strictly on the \
retrieved excerpts provided in the conversation. If the excerpts don't \
cover something, say so plainly rather than guessing.
2. Concordance notes may be used ONLY to explain that an old section \
number corresponds to a new one. Do not add legal content from them.
3. Do not skip the Do's/Don'ts sections on a first-question response, \
even if the situation seems simple.
4. Keep every bullet actionable and specific; no vague statements like \
"know your rights."
5. Always cite the specific Act and Section number(s) supporting each \
point. Only cite sections that appear in the provided excerpts.
6. Keep the tone plain and accessible; the person asking is not a lawyer.
7. Use the conversation history to understand references to earlier \
parts of the discussion (e.g. "what about my second point?").
"""


def retrieve_sections(question: str, model=None, collection=None, top_k=None) -> list:
    """Find the most relevant sections for a question.

    Each result includes a 'distance' (lower = closer match).
    model/collection can be passed in already loaded (the Streamlit UI does
    this so they load only once). top_k overrides TOP_K (used by evaluation).
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)

    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k or TOP_K,
    )

    sections = []
    for i in range(len(results["ids"][0])):
        sections.append({
            "act": results["metadatas"][0][i]["act"],
            "chapter": results["metadatas"][0][i]["chapter"],
            "section": results["metadatas"][0][i]["section"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    return sections


def build_context_block(sections: list) -> str:
    blocks = []
    for s in sections:
        blocks.append(
            f"[{s['act']} — {s['chapter']} — Section {s['section']}]\n{s['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_turn(question: str, context: str, notes: str = "") -> str:
    """The message sent for the CURRENT question: law text + optional notes."""
    return (
        f"Relevant law sections:\n\n{context}\n\n"
        f"{notes}---\n\nUser's question: {question}"
    )


def ask_openai_compatible(messages: list) -> str:
    """Works with Groq, OpenRouter, Gemini's OpenAI-compatible endpoint,
    local Ollama, and any provider that copies OpenAI's chat format.

    `messages` is the conversation so far; the system prompt is prepended.
    """
    url = f"{config.LLM_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 4096,
    }

    # Gemini 3.x guidance: leave temperature at its default (1.0); lowering
    # it can cause looping. So it is only sent if you explicitly set it
    # (sensible for e.g. Llama on Groq: 0.2).
    temperature = getattr(config, "LLM_TEMPERATURE", None)
    if temperature is not None:
        payload["temperature"] = temperature

    # "Thinking" models spend part of max_tokens on hidden reasoning;
    # a low effort leaves more room for the visible answer.
    reasoning_effort = getattr(config, "LLM_REASONING_EFFORT", None)
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Retry transient errors (provider overloaded / rate limited).
    RETRIABLE = {429, 500, 502, 503, 504}
    MAX_ATTEMPTS = 3
    response = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.ok:
            break
        if response.status_code in RETRIABLE and attempt < MAX_ATTEMPTS:
            wait_seconds = 2 ** attempt  # 2s, then 4s
            print(f"Provider returned {response.status_code} (likely temporary); "
                  f"retrying in {wait_seconds}s (attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(wait_seconds)
            continue
        break

    if not response.ok:
        print("\n--- LLM API ERROR ---")
        print(f"URL: {url}")
        print(f"Status: {response.status_code}")
        print(f"Response body: {response.text}")
        print("---------------------\n")
    response.raise_for_status()
    data = response.json()

    if data["choices"][0].get("finish_reason") == "length":
        print("NOTE: the response was cut off by the token limit; "
              "consider raising max_tokens further.")

    return data["choices"][0]["message"]["content"]


def ask_anthropic(messages: list) -> str:
    import anthropic  # only required if you use this provider
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    kwargs = {}
    temperature = getattr(config, "LLM_TEMPERATURE", None)
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL_NAME,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
        **kwargs,
    )
    return response.content[0].text


def ask_llm(messages: list) -> str:
    """messages: the conversation so far, as a list of
    {"role": "user"/"assistant", "content": ...} dicts."""
    if config.LLM_PROVIDER == "openai_compatible":
        return ask_openai_compatible(messages)
    elif config.LLM_PROVIDER == "anthropic":
        return ask_anthropic(messages)
    raise ValueError(
        f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' in config.py; "
        f"use 'openai_compatible' or 'anthropic'."
    )


def main():
    if len(sys.argv) != 2:
        print('Usage: python ask_law_bot.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print("Retrieving relevant sections...")
    sections = retrieve_sections(question)
    for s in sections:
        print(f"  - {s['act']}, Section {s['section']} (distance {s['distance']:.3f})")

    messages = [{"role": "user",
                 "content": build_user_turn(question, build_context_block(sections))}]
    print("\nAsking the LLM...\n")
    print("=" * 70)
    print(ask_llm(messages))
    print("=" * 70)


if __name__ == "__main__":
    main()
