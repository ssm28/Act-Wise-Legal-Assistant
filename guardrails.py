"""
Guardrails for the law bot
---------------------------
Four small, independent checks. None of them call an LLM, so they are
fast, free, and easy to test and explain in a viva:

  1. detect_emergency()        - flags urgent situations, shows helplines first
  2. find_legacy_references()  - spots "IPC 420" style references and maps them
                                 to the new BNS / BNSS / BSA section
  3. is_in_scope()             - relevance gate on the best retrieval distance
  4. verify_citations()        - checks cited section numbers really were retrieved
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_MAP_PATH = os.path.join(HERE, "legacy_law_map.json")


# ---------------------------------------------------------------------------
# 1. Emergency detection (keyword based - deliberately simple and explainable)
# ---------------------------------------------------------------------------
# Verify these numbers before your demo; helplines can change.
EMERGENCY_RULES = [
    {
        "key": "self_harm",
        "title": "If you are thinking of harming yourself",
        "patterns": [r"kill myself", r"end my life", r"commit suicide",
                     r"suicidal", r"want to die"],
        "helplines": [("Emergency (all services)", "112"),
                      ("Tele-MANAS (mental health support)", "14416")],
    },
    {
        "key": "violence",
        "title": "If you are in immediate danger",
        "patterns": [r"beat(ing|s)? me", r"hit(ting|s)? me", r"kill me",
                     r"threaten\w* to kill", r"attack(ed|ing)? me",
                     r"locked me", r"kidnap\w*", r"abduct\w*"],
        "helplines": [("Emergency (police / ambulance / fire)", "112")],
    },
    {
        "key": "domestic",
        "title": "Domestic violence or dowry harassment",
        "patterns": [r"domestic violence", r"husband (beats|hits|abuses)",
                     r"in-?laws? (beat|harass|torture)\w*",
                     r"dowry (harass|demand|torture)\w*"],
        "helplines": [("Women Helpline", "181"),
                      ("Emergency", "112")],
    },
    {
        "key": "child",
        "title": "Child abuse or child marriage",
        "patterns": [r"child abuse", r"child marriage",
                     r"abus\w+ (my )?(son|daughter|child)"],
        "helplines": [("Childline", "1098"), ("Emergency", "112")],
    },
    {
        "key": "cyber",
        "title": "Online fraud or cyber crime (report quickly)",
        "patterns": [r"online fraud", r"cyber ?(crime|fraud)", r"upi fraud",
                     r"otp fraud", r"money (was |got )?(stolen|debited)",
                     r"\bhacked\b", r"sextortion", r"morphed (photo|image|video)s?"],
        "helplines": [("National Cyber Crime Helpline", "1930"),
                      ("Report online", "cybercrime.gov.in")],
    },
]
LEGAL_AID_LINE = ("Free legal aid (NALSA)", "15100")


def detect_emergency(text: str) -> list:
    """Return a list of matching emergency categories (empty if none)."""
    found = []
    for rule in EMERGENCY_RULES:
        if any(re.search(p, text, re.IGNORECASE) for p in rule["patterns"]):
            found.append({
                "key": rule["key"],
                "title": rule["title"],
                "helplines": rule["helplines"] + [LEGAL_AID_LINE],
            })
    return found


def format_emergency_banner(flag: dict) -> str:
    lines = [f"**{flag['title']}**"]
    for name, number in flag["helplines"]:
        lines.append(f"- {name}: **{number}**")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Old-law to new-law lookup (IPC -> BNS, CrPC -> BNSS, Evidence Act -> BSA)
# ---------------------------------------------------------------------------
_NAME_PATTERNS = {
    "IPC": r"(?:IPC|I\.P\.C\.?|Indian\s+Penal\s+Code)",
    "CrPC": r"(?:CrPC|Cr\.P\.C\.?|Code\s+of\s+Criminal\s+Procedure)",
    "IEA": r"(?:Indian\s+Evidence\s+Act|Evidence\s+Act)",
}
_SEC = r"(?:sections?|sec\.?|s\.)?\s*(\d{1,3}[A-Za-z]?)"

_legacy_cache = None


def _load_legacy_map() -> dict:
    global _legacy_cache
    if _legacy_cache is None:
        try:
            with open(LEGACY_MAP_PATH, "r", encoding="utf-8") as f:
                _legacy_cache = json.load(f)
        except FileNotFoundError:
            _legacy_cache = {}
    return _legacy_cache


def find_legacy_references(text: str) -> list:
    """Find mentions like 'IPC 420' or 'section 154 CrPC' and look up the
    corresponding new section. Returns a list of dicts."""
    legacy_map = _load_legacy_map()
    refs, seen = [], set()
    for code, name_re in _NAME_PATTERNS.items():
        patterns = [
            re.compile(rf"\b{name_re}\s*[,:\-]?\s*{_SEC}", re.IGNORECASE),
            re.compile(rf"{_SEC}\s*(?:of\s+(?:the\s+)?)?{name_re}", re.IGNORECASE),
        ]
        for pattern in patterns:
            for m in pattern.finditer(text):
                number = m.group(1).upper()
                entry = legacy_map.get(code, {}).get(number)
                if entry and (code, number) not in seen:
                    seen.add((code, number))
                    refs.append({"code": code, "old": number, **entry})
    return refs


def legacy_query_hint(refs: list) -> str:
    """Extra words appended to the retrieval query so the NEW section is found."""
    return " ".join(f"{r['new']} Section {r['section']} {r['topic']}" for r in refs)


def legacy_context_note(refs: list) -> str:
    """A note added to the LLM context so it can connect old and new numbers."""
    if not refs:
        return ""
    lines = [f"- {r['code']} Section {r['old']} corresponds to "
             f"{r['new']} Section {r['section']} ({r['topic']})" for r in refs]
    return "Concordance notes (old-to-new section mapping):\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# 3. Relevance gate
# ---------------------------------------------------------------------------
def parse_distance_setting(value):
    """Turn a config value into a float, or None (gate off).
    Accepts 1.3, "1.3", None, "none", "off" or "". Raises ValueError with a
    readable message for anything else (e.g. a leftover placeholder)."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("", "none", "off"):
            return None
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"RELEVANCE_MAX_DISTANCE must be a number like 1.3, got {value!r}"
            )
    return float(value)


def best_distance(sections: list):
    distances = [s["distance"] for s in sections if s.get("distance") is not None]
    return min(distances) if distances else None


def is_in_scope(sections: list, max_distance) -> bool:
    """True if the closest retrieved section is close enough to trust.
    max_distance=None disables the gate."""
    max_distance = parse_distance_setting(max_distance)
    if max_distance is None:
        return True
    best = best_distance(sections)
    return best is not None and best <= max_distance


def out_of_scope_message(available_acts: list) -> str:
    acts = ", ".join(available_acts) if available_acts else "the Acts loaded so far"
    return (
        "I couldn't find anything relevant to this in the laws I currently have: "
        f"**{acts}**.\n\n"
        "I'd rather say so than guess. You could rephrase with more detail, or "
        "contact your nearest **legal aid clinic (NALSA helpline 15100)** for "
        "guidance on this specific issue."
    )


# ---------------------------------------------------------------------------
# 4. Citation verifier
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(
    r"\b[Ss]ections?\s+"
    r"(\d{1,3}[A-Z]?(?:\(\w+\))*"
    r"(?:\s*(?:,|and|&|or|to)\s*\d{1,3}[A-Z]?(?:\(\w+\))*)*)"
)


def extract_cited_sections(answer: str) -> list:
    """Base section numbers cited in an answer, e.g. 'Section 2(7)' -> '2'."""
    cited = []
    for m in _CITATION_RE.finditer(answer):
        for token in re.split(r"\s*(?:,|and|&|or|to)\s*", m.group(1)):
            base = re.match(r"\d{1,3}[A-Z]?", token.strip())
            if base and base.group(0) not in cited:
                cited.append(base.group(0))
    return cited


def verify_citations(answer: str, retrieved_sections: list) -> dict:
    """Compare section numbers cited in the answer against those retrieved.
    Limitation: compares numbers only, not which Act they belong to."""
    retrieved = {str(s["section"]).upper() for s in retrieved_sections}
    cited = extract_cited_sections(answer)
    unverified = [c for c in cited if c.upper() not in retrieved]
    return {"cited": cited, "unverified": unverified}


# ---------------------------------------------------------------------------
# 5. Exact section lookup ("what does section 63 of BNS say?")
# ---------------------------------------------------------------------------
# Semantic search matches MEANING, so it is poor at exact references like
# "section 63". When a question names a section, we fetch it directly by
# Act + number instead (a metadata lookup, no vector search involved).
_SECTION_WORD = r"(?:sections?|sec\.?|s\.)?"
_LAW_WORDS = re.compile(
    r"\b(act|code|sanhita|adhiniyam|constitution|rules|regulations)\b", re.IGNORECASE
)
_LOOSE_RE = re.compile(
    r"\b(?:sections?|sec\.?|s\.)\s*"
    r"(\d{1,3}[a-z]?(?:\(\w+\))*"
    r"(?:\s*(?:,|and|&|or)\s*\d{1,3}[a-z]?(?:\(\w+\))*)*)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    # "Samhita" and "Sanhita" are both used for the same word.
    return text.lower().replace("samhita", "sanhita")


def build_act_aliases(act_names, extra=None) -> dict:
    """Map every short name people might type to the exact Act name.
    'Bharatiya Nyaya Sanhita, 2023' -> 'bns', 'bharatiya nyaya sanhita';
    'Consumer Protection Act, 2019' -> 'cpa', 'consumer protection act', ...
    extra: optional {act name: [more aliases]}, e.g. from config.ACT_ALIASES."""
    aliases = {}
    for act in act_names:
        base = re.sub(r",?\s*\d{4}\s*$", "", _norm(act)).strip()
        words = re.findall(r"[a-z]+", base)
        if not words:
            continue
        variants = [words]
        if len(words) > 1 and words[-1] == "act":
            variants.append(words[:-1])
        candidates = set()
        for v in variants:
            candidates.add(" ".join(v))
            candidates.add("".join(w[0] for w in v))
        for alias in (extra or {}).get(act, []):
            candidates.add(_norm(alias))
        for c in candidates:
            if len(c) >= 3 and c not in aliases:
                aliases[c] = act
    return aliases


def resolve_act(name: str, act_names, extra=None):
    """'BNS' -> the loaded Act it refers to, or None if it isn't loaded."""
    return build_act_aliases(act_names, extra).get(_norm(name))


def find_direct_section_refs(text: str, act_names, extra=None) -> list:
    """Find explicit section references and pair each with an Act.

    Returns a list of {"act", "number", "explicit"} dicts. 'explicit' is True
    when the user named the Act (so a miss is worth reporting).
    Deliberately returns NOTHING when the reference is risky:
      - the text mentions IPC / CrPC / Evidence Act (handled by the old-law
        table instead), or
      - it names a law that is not loaded ("section 63 of the Companies Act"),
    because a wrong confident match is worse than falling back to search."""
    norm = _norm(text)
    aliases = build_act_aliases(act_names, extra)
    pairs, seen = [], set()

    def add(act, number, explicit):
        key = (act, number.upper())
        if key not in seen:
            seen.add(key)
            pairs.append({"act": act, "number": number.upper(), "explicit": explicit})

    named_acts, attached = [], set()
    if aliases:
        alt = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
        # "BNS 63", "BNS section 63"
        p_after = re.compile(
            rf"\b({alt})\b\s*[,:\-]?\s*{_SECTION_WORD}\s*(\d{{1,3}}[a-z]?)\b")
        # "section 63 of BNS", "63 of the BNS" (also catches "scetion 63 of BNS")
        p_before = re.compile(
            rf"{_SECTION_WORD}\s*(\d{{1,3}}[a-z]?)(?:\s*\(\w+\))*\s+of\s+(?:the\s+)?({alt})\b")
        for m in p_after.finditer(norm):
            add(aliases[m.group(1)], m.group(2), True)
            attached.add(m.group(2).upper())
        for m in p_before.finditer(norm):
            add(aliases[m.group(2)], m.group(1), True)
            attached.add(m.group(1).upper())
        named_acts = list(dict.fromkeys(
            aliases[a] for a in re.findall(rf"\b({alt})\b", norm)))

    # Loose numbers: "section 63" with no Act attached.
    loose = []
    for m in _LOOSE_RE.finditer(norm):
        for token in re.split(r"\s*(?:,|and|&|or)\s*", m.group(1)):
            base = re.match(r"\d{1,3}[a-z]?", token.strip())
            if base and base.group(0).upper() not in attached:
                loose.append(base.group(0))

    legacy_mentioned = any(re.search(p, norm, re.IGNORECASE) for p in _NAME_PATTERNS.values())
    unknown_law = (not named_acts) and bool(_LAW_WORDS.search(norm))
    if legacy_mentioned or unknown_law:
        loose = []

    targets = named_acts or list(act_names)
    for n in loose:
        for act in targets:
            add(act, n, bool(named_acts))
    return pairs


def direct_lookup_note(found: list, missing_explicit: list) -> str:
    """Tells the LLM which sections were fetched by number (and which were
    not found), so it answers as a section explanation."""
    lines = []
    if found:
        wanted = ", ".join(f"{s['act']} Section {s['section']}" for s in found)
        lines.append(f"The user asked about specific sections, fetched directly by number: {wanted}")
    if missing_explicit:
        gone = ", ".join(f"{p['act']} Section {p['number']}" for p in missing_explicit)
        lines.append(f"These requested sections were NOT found in the database: {gone}")
    return ("Lookup notes:\n- " + "\n- ".join(lines) + "\n\n") if lines else ""


def section_not_found_message(missing_explicit: list) -> str:
    gone = ", ".join(f"Section {p['number']} of {p['act']}" for p in missing_explicit)
    return (
        f"I couldn't find **{gone}** in my database. Please check the section "
        "number. You can browse the full text of each Act from the **Acts "
        "loaded** list in the sidebar."
    )
