"""
Indian Law Bot: Streamlit UI (chat-based, multi-turn)
------------------------------------------------------
Run it:
    streamlit run app.py

Per-question flow:
    emergency check -> old-law lookup -> retrieval -> relevance gate ->
    LLM answer -> citation check -> log to SQLite -> show answer + feedback

Optional config.py setting:
    RELEVANCE_MAX_DISTANCE  default 1.5 (see calibrate_threshold.py);
                            set to None to switch the gate off.

Optional images (fallback emoji used if missing):
    assets/logo.png, assets/bot_avatar.png
"""

import os
import re
import time
import uuid
import base64

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

import config

# On Streamlit Cloud real secrets come from st.secrets; locally there is
# usually no secrets.toml, so accessing st.secrets raises and we keep the
# values from config.py / local_secrets.py.
try:
    config.LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", config.LLM_PROVIDER)
    config.LLM_API_BASE_URL = st.secrets.get("LLM_API_BASE_URL", config.LLM_API_BASE_URL)
    config.LLM_API_KEY = st.secrets.get("LLM_API_KEY", config.LLM_API_KEY)
    config.LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", config.LLM_MODEL_NAME)
except Exception:
    pass

import guardrails
import storage
from ask_law_bot import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    retrieve_sections,
    build_context_block,
    build_user_turn,
    ask_llm,
)

try:
    RELEVANCE_MAX_DISTANCE = guardrails.parse_distance_setting(
        getattr(config, "RELEVANCE_MAX_DISTANCE", 1.5)
    )
    THRESHOLD_SETTING_INVALID = False
except ValueError:
    RELEVANCE_MAX_DISTANCE = 1.5  # safe fallback; a warning shows in the sidebar
    THRESHOLD_SETTING_INVALID = True

# Storing conversations is OFF unless you set LOG_INTERACTIONS = True in
# config.py. The bot works fully without it.
LOGGING_ENABLED = bool(getattr(config, "LOG_INTERACTIONS", False))

st.set_page_config(page_title="Nyaya Sahayak: Indian Law Bot", page_icon="⚖️", layout="centered")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BOT_AVATAR_PATH = os.path.join(ASSETS_DIR, "bot_avatar.png")
BOT_AVATAR = BOT_AVATAR_PATH if os.path.exists(BOT_AVATAR_PATH) else "⚖️"

LOGGING_OK = False
if LOGGING_ENABLED:
    try:
        storage.init_db()
        LOGGING_OK = True
    except Exception:
        LOGGING_OK = False  # never let a logging problem break the chat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_base64_image(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/{path.rsplit('.', 1)[-1]};base64,{encoded}"


def section_sort_key(section_str):
    m = re.match(r"(\d+)([A-Za-z]*)", str(section_str))
    return (int(m.group(1)), m.group(2)) if m else (999999, str(section_str))


def get_available_acts(collection) -> list:
    data = collection.get(include=["metadatas"])
    return sorted(set(m["act"] for m in data["metadatas"])) if data["metadatas"] else []


def get_act_sections(collection, act_name: str) -> list:
    result = collection.get(where={"act": act_name}, include=["metadatas", "documents"])
    sections = [{"section": m["section"], "chapter": m["chapter"], "text": d}
                for m, d in zip(result["metadatas"], result["documents"])]
    return sorted(sections, key=lambda s: section_sort_key(s["section"]))


def save_feedback(interaction_id, widget_key):
    value = st.session_state.get(widget_key)
    if value is not None and LOGGING_OK:
        try:
            storage.set_feedback(interaction_id, int(value))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Visual design (responsive)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: #232B3A; }
.block-container { padding-top: 0 !important; max-width: 760px; }

.masthead {
    background-color: #14213D; margin: 0 -1rem 1.5rem -1rem;
    padding: 1.5rem; border-bottom: 3px solid #C98A2B;
    display: flex; align-items: center; gap: 1rem;
}
.masthead img { height: 48px; width: 48px; object-fit: contain; border-radius: 8px; flex-shrink: 0; }
.masthead .masthead-emoji-fallback { font-size: 2.2rem; line-height: 1; }
.masthead h1 { font-family: 'IBM Plex Serif', serif; color: #FBF9F4; font-size: 1.85rem;
               font-weight: 600; margin: 0; letter-spacing: 0.01em; }
.masthead p { color: #C9CEDA; margin: 0.3rem 0 0 0; font-size: 0.92rem; }

[data-testid="stChatMessage"] {
    background-color: transparent !important; border-radius: 0 !important;
    box-shadow: none !important; padding: 0.75rem 0 !important; margin-bottom: 0.25rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #14213D; padding-left: 1rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-right: 3px solid #33513A; padding-right: 1rem !important;
    flex-direction: row-reverse; text-align: right;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    text-align: right;
}

.stButton > button, [data-testid="stChatInputSubmitButton"] {
    background-color: #C98A2B; color: #FBF9F4; border: none;
}
.stButton > button:hover, [data-testid="stChatInputSubmitButton"]:hover { background-color: #B37A24; }

[data-testid="stSidebar"] { background-color: #FBF9F4; border-right: 1px solid #E4DFD3; }
.sidebar-acts-heading { font-family: 'IBM Plex Serif', serif; font-size: 0.95rem; color: #14213D;
    margin-top: 1.5rem; margin-bottom: 0.5rem; border-top: 1px solid #E4DFD3; padding-top: 1rem; }

/* Phones and small tablets */
@media (max-width: 640px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .masthead { margin: 0 -0.75rem 1rem -0.75rem; padding: 1rem; gap: 0.75rem; }
    .masthead img { height: 36px; width: 36px; }
    .masthead .masthead-emoji-fallback { font-size: 1.8rem; }
    .masthead h1 { font-size: 1.3rem; }
    .masthead p { font-size: 0.78rem; }
    [data-testid="stChatMessage"] { padding: 0.5rem 0 !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { padding-left: 0.6rem !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { padding-right: 0.6rem !important; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


model = load_embedding_model()
collection = load_collection()
available_acts = get_available_acts(collection)

logo_uri = get_base64_image(LOGO_PATH)
logo_html = (f'<img src="{logo_uri}" alt="logo">' if logo_uri
             else '<span class="masthead-emoji-fallback">⚖️</span>')
st.markdown(f"""
<div class="masthead">
    {logo_html}
    <div>
        <h1>Act Wise</h1>
        <p>Ask in plain language. Answers are grounded in Indian statutes, with citations.</p>
        <p><strong>This is legal information, not legal advice.</strong></p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "view" not in st.session_state:
    st.session_state.view = "chat"
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "all_sections" not in st.session_state:
    st.session_state.all_sections = []  # every section retrieved this conversation

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if st.sidebar.button("🗂️ New Case", use_container_width=True):
    st.session_state.messages = []
    st.session_state.all_sections = []
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.view = "chat"
    st.rerun()

st.sidebar.metric("Sections in database", collection.count())
st.sidebar.markdown('<p class="sidebar-acts-heading">Acts loaded</p>', unsafe_allow_html=True)
if not available_acts:
    st.sidebar.caption("No Acts embedded yet.")
for act_name in available_acts:
    if st.sidebar.button(act_name, key=f"act_btn_{act_name}", use_container_width=True):
        st.session_state.view = f"read_act::{act_name}"
        st.rerun()
st.sidebar.caption("More Acts appear here automatically as they are added.")
if LOGGING_OK:
    st.sidebar.caption(
        "Questions and answers are saved to improve this tool. "
        "Please don't type names, phone numbers or other personal details."
    )
else:
    st.sidebar.caption(
        "This app does not save your conversation. Your question is sent to an "
        "AI service to write the answer, so please avoid personal details."
    )
if THRESHOLD_SETTING_INVALID:
    st.sidebar.warning(
        "RELEVANCE_MAX_DISTANCE in config.py is not a number; using 1.5 instead."
    )


# ---------------------------------------------------------------------------
# Rendering one assistant reply (used for history AND the live turn)
# ---------------------------------------------------------------------------
def render_assistant(msg):
    for flag in msg.get("emergency", []):
        st.error(guardrails.format_emergency_banner(flag), icon="🚨")

    st.markdown(msg["content"])

    if msg.get("unverified"):
        st.warning(
            "Couldn't verify these cited section numbers in the retrieved law text: "
            + ", ".join(msg["unverified"]) + ". Please check them against the Act.",
            icon="⚠️",
        )
    if msg.get("legacy"):
        st.caption("Old-to-new mapping used: " + "; ".join(
            f"{r['code']} {r['old']} → {r['new']} {r['section']}" for r in msg["legacy"]
        ) + " (verify against the official table)")

    if msg.get("sections"):
        best = msg.get("best_distance")
        label = f"{len(msg['sections'])} sections used to answer this"
        if best is not None:
            label += f" (best match distance {best:.2f})"
        with st.expander(label):
            for s in msg["sections"]:
                st.markdown(f"**{s['act']} — {s['chapter']} — Section {s['section']}**")
                st.write(s["text"])
                st.markdown("---")

    if msg.get("interaction_id") and hasattr(st, "feedback"):
        key = f"fb_{msg['interaction_id']}"
        st.feedback("thumbs", key=key, on_change=save_feedback,
                    args=(msg["interaction_id"], key))


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if st.session_state.view.startswith("read_act::"):
    act_name = st.session_state.view.split("::", 1)[1]
    if st.button("← Back to chat"):
        st.session_state.view = "chat"
        st.rerun()
    st.markdown(f"## {act_name}")
    sections = get_act_sections(collection, act_name)
    st.caption(f"{len(sections)} sections")
    current_chapter = None
    for s in sections:
        if s["chapter"] and s["chapter"] != current_chapter:
            current_chapter = s["chapter"]
            st.markdown(f"#### {current_chapter}")
        with st.expander(f"Section {s['section']}"):
            st.write(s["text"])

else:
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                render_assistant(msg)
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

    question = st.chat_input("e.g. What can I do if a shopkeeper refuses to refund a defective product?")

    if question:
        is_first_turn = len(st.session_state.messages) == 0
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Searching relevant law sections..."):
                emergency = guardrails.detect_emergency(question)
                legacy = guardrails.find_legacy_references(question)
                hint = guardrails.legacy_query_hint(legacy)
                sections = retrieve_sections(
                    f"{question} {hint}".strip(), model=model, collection=collection
                )
                best = guardrails.best_distance(sections)
                # The gate applies only to the FIRST message: a follow-up like
                # "what if they refuse?" is vague on its own and would be
                # wrongly blocked. (Query rewriting would fix this properly.)
                in_scope = (not is_first_turn) or guardrails.is_in_scope(
                    sections, RELEVANCE_MAX_DISTANCE
                )

            unverified, latency_ms = [], 0
            if not in_scope:
                answer = guardrails.out_of_scope_message(available_acts)
                shown_sections = []
            else:
                shown_sections = sections
                st.session_state.all_sections.extend(sections)
                llm_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                llm_messages.append({
                    "role": "user",
                    "content": build_user_turn(
                        question,
                        build_context_block(sections),
                        guardrails.legacy_context_note(legacy),
                    ),
                })
                with st.spinner("Thinking..."):
                    started = time.time()
                    try:
                        answer = ask_llm(llm_messages)
                        check = guardrails.verify_citations(
                            answer, st.session_state.all_sections
                        )
                        unverified = check["unverified"]
                    except Exception as e:
                        answer = f"Something went wrong calling the LLM: {e}"
                    latency_ms = int((time.time() - started) * 1000)

            interaction_id = None
            if LOGGING_OK:
                try:
                    interaction_id = storage.log_interaction(
                        st.session_state.session_id, question, answer, sections,
                        best, in_scope, config.LLM_MODEL_NAME, latency_ms,
                        unverified, [f["key"] for f in emergency],
                    )
                except Exception:
                    pass

            msg = {
                "role": "assistant", "content": answer, "sections": shown_sections,
                "best_distance": best, "emergency": emergency, "unverified": unverified,
                "legacy": legacy, "interaction_id": interaction_id,
            }
            render_assistant(msg)

        st.session_state.messages.append(msg)
