import streamlit as st
import streamlit.components.v1 as components
import os
import uuid
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from ingest import (
    validate_file,
    validate_file_size,
    extract_text_from_pdf,
    clean_text,
    chunk_text,
    embed_chunks,
    build_vector_store,
    search_similar_chunks,
    generate_answer,
)

load_dotenv()

st.set_page_config(
    page_title="StudyBuddy",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════════════
   RESET & BASE
═══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html {
    font-family: 'Inter', sans-serif;
    /* dvh: dynamic viewport height — shrinks when mobile keyboard opens */
    height: 100dvh;
    overflow: hidden;
}

body {
    height: 100dvh;
    overflow: hidden;
    background: #f0f2f8;
}

/* Thin modern scrollbars — only appear inside .st-key-chat_messages */
* {
    scrollbar-width: thin;
    scrollbar-color: #c5cce8 transparent;
}
*::-webkit-scrollbar { width: 5px; height: 5px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: #c5cce8; border-radius: 10px; }
*::-webkit-scrollbar-thumb:hover { background: #9098be; }

.stApp {
    background: #f0f2f8;
    height: 100dvh;
    overflow: hidden;
}

#MainMenu, footer, header { visibility: hidden; }

/* ═══════════════════════════════════════════════════════
   STREAMLIT STRUCTURAL OVERRIDES
   Goal: make Streamlit's wrapper divs flex-transparent
   so our semantic layout (header / msgs / form) drives geometry.
═══════════════════════════════════════════════════════ */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100dvh !important;
    overflow: hidden !important;
}

/* Top-level two-column row = the full app shell */
[data-testid="stHorizontalBlock"] {
    height: 100dvh !important;
    align-items: stretch !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
    overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════════
   LEFT COLUMN — sidebar panel
   Fixed-height, scrollable internally, never affects chat.
═══════════════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"] > div:first-child {
    background: #ffffff;
    border-right: 1px solid #dde1f0;
    height: 100dvh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 24px 16px !important;
    flex-shrink: 0 !important;
}

/* ═══════════════════════════════════════════════════════
   RIGHT COLUMN — chat shell
   A flex column: header (shrink 0) → msgs (grow) → form (shrink 0)
   The column itself must NOT scroll — only .st-key-chat_messages does.
═══════════════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"] > div:last-child {
    display: flex !important;
    flex-direction: column !important;
    height: 100dvh !important;
    overflow: hidden !important;      /* <-- never let this column scroll */
    padding: 0 !important;
    background: #f0f2f8;
    min-width: 0 !important;
    flex: 1 1 auto !important;
}

[data-testid="stHorizontalBlock"] > div:last-child
  > div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    height: 100dvh !important;
    overflow: hidden !important;
    gap: 0 !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

/* ─── Chat header: flex-shrink:0 so it never collapses ─── */
.chat-header {
    background: #ffffff;
    border-bottom: 1px solid #dde1f0;
    padding: 13px clamp(14px, 3vw, 28px);
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 0 0 auto !important;
    z-index: 10;
}
.chat-header-title { font-size: clamp(13px, 1.8vw, 15px); font-weight: 600; color: #1a2e6e; }
.chat-header-sub   { font-size: clamp(10px, 1.3vw, 12px); color: #9098be; margin-left: auto; white-space: nowrap; }

/* ─── Messages container: the ONE scrollable zone ─── */
/*
   flex: 1 1 0  → takes ALL remaining space between header & form
   min-height: 0 → crucial: without this, flex children ignore overflow
   overflow-y: auto → scroll only here
*/
.st-key-chat_messages {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: clamp(14px, 3vw, 24px) clamp(12px, 4vw, 32px) 16px !important;
    scroll-behavior: smooth !important;
    border: none !important;
    background: transparent !important;
}

/* Strip Streamlit's inner wrapper so it doesn't fight our flex geometry */
.st-key-chat_messages > div[data-testid="stVerticalBlockBorderWrapper"],
.st-key-chat_messages > div[data-testid="stVerticalBlockBorderWrapper"] > div,
.st-key-chat_messages > div[data-testid="stVerticalBlockBorderWrapper"] > div > div {
    height: auto !important;
    min-height: 0 !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* ─── Input form: fixed at bottom, never scrolls ─── */
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
    background: #ffffff !important;
    border: none !important;
    border-top: 1px solid #dde1f0 !important;
    border-radius: 0 !important;
    padding: 12px clamp(12px, 3vw, 28px)
             calc(8px + env(safe-area-inset-bottom, 0px))
             clamp(12px, 3vw, 28px) !important;
    margin: 0 !important;
    flex: 0 0 auto !important;
    width: 100% !important;
}

/* ─── Composer row: wraps input + button in a true flex row ───
   We no longer use st.columns() for the form, so there are no
   Streamlit percentage-width divs to fight. The .composer-row div
   is injected via st.markdown and wraps the two Streamlit elements.
─────────────────────────────────────────────────────────────── */

/* The injected wrapper — flex row, full width */
.composer-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    min-width: 0 !important;
}

/*
  Both the text input widget and the submit button are direct
  children of .composer-row (via their Streamlit element wrappers).
  Input element wrapper → grow to fill space.
  Button element wrapper → shrink to label width only.
*/
.composer-row > div:has(.stTextInput) {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: auto !important;
}

.composer-row > div:has(.stFormSubmitButton),
.composer-row > div:has(button[kind="primaryFormSubmit"]) {
    flex: 0 0 auto !important;
    min-width: fit-content !important;
    width: auto !important;
}

/* The actual <input> — always fill its wrapper, always visible */
.composer-row .stTextInput,
.composer-row .stTextInput > div,
.composer-row .stTextInput > div > div {
    width: 100% !important;
    min-width: 0 !important;
}

.composer-row input {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
    min-width: 0 !important;
    height: 44px !important;
    position: static !important;
}

/* Submit button height */
.composer-row button {
    height: 44px !important;
    white-space: nowrap !important;
}

/*
  Fallback: if the :has() selector isn't supported (older browsers),
  target by child order within the form's stVerticalBlock.
  Input is always first, button always second.
*/
[data-testid="stForm"] > div > div:first-child {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
[data-testid="stForm"] > div > div:last-child {
    flex: 0 0 auto !important;
}

/* ─── Input hint ─── */
.input-hint {
    background: #ffffff;
    font-size: clamp(10px, 1.2vw, 11px);
    color: #b0b8d8;
    text-align: center;
    padding: 3px clamp(12px, 3vw, 28px)
             calc(6px + env(safe-area-inset-bottom, 0px));
    flex: 0 0 auto !important;
}

/* ─── Collapse Streamlit's zero-size autoscroll iframes ─── */
[data-testid="stHorizontalBlock"] > div:last-child .element-container:has(iframe),
[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stIFrame"],
[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stIFrame"] iframe,
[data-testid="stHorizontalBlock"] > div:last-child div:has(> iframe) {
    flex: 0 0 0 !important;
    height: 0 !important; min-height: 0 !important; max-height: 0 !important;
    margin: 0 !important; padding: 0 !important;
    border: none !important; overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════════
   SIDEBAR COMPONENTS
═══════════════════════════════════════════════════════ */
.sb-brand {
    display: flex; align-items: center; gap: 9px;
    margin-bottom: 22px; padding-bottom: 16px;
    border-bottom: 1px solid #eaecf5;
    flex-wrap: wrap;
}
.sb-brand-name { font-size: clamp(14px, 2vw, 17px); font-weight: 700; color: #1a2e6e; }
.sb-brand-tag {
    font-size: 11px; color: #9098be;
    background: #eef0fa; padding: 2px 8px; border-radius: 10px;
    white-space: nowrap;
}
.sb-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.1px;
    text-transform: uppercase; color: #a0a9cc; margin-bottom: 10px;
}
.sb-file-card {
    background: #f5f7fe; border: 1px solid #e0e4f4;
    border-radius: 10px; padding: 11px 13px;
    display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
    min-width: 0;
}
.sb-file-name {
    font-size: clamp(11px, 1.5vw, 13px); font-weight: 600;
    color: #1c1e2e; word-break: break-all;
}
.sb-file-meta { font-size: 11px; color: #9098be; margin-top: 2px; }
.sb-status {
    font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 20px;
    display: inline-block; margin-bottom: 14px;
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sb-status.ready   { background: #e6f4ee; color: #1a6e42; }
.sb-status.waiting { background: #eef0fa; color: #7880ae; }
.sb-divider { border: none; border-top: 1px solid #eaecf5; margin: 18px 0; }
.sb-tips {
    background: #f5f7fe; border: 1px solid #e0e4f4;
    border-radius: 10px; padding: 14px;
    font-size: clamp(11px, 1.4vw, 13px); color: #626a94; line-height: 1.8;
}

/* Mobile hamburger */
.sb-hamburger {
    display: none;
    position: fixed; top: 12px; left: 12px; z-index: 200;
    width: 40px; height: 40px;
    background: #1a2e6e; color: #fff;
    border: none; border-radius: 8px; cursor: pointer;
    font-size: 18px; line-height: 40px; text-align: center;
    box-shadow: 0 2px 8px rgba(26,46,110,0.25);
    transition: background 0.2s;
}
.sb-hamburger:hover { background: #122060; }

.sb-overlay {
    display: none;
    position: fixed; inset: 0; z-index: 150;
    background: rgba(0,0,0,0.35);
    backdrop-filter: blur(2px);
    transition: opacity 0.3s;
    opacity: 0;
    pointer-events: none;
}
.sb-overlay.open { opacity: 1 !important; pointer-events: all !important; }

/* ═══════════════════════════════════════════════════════
   CHAT BUBBLES & CARDS
═══════════════════════════════════════════════════════ */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; color: #9098be; gap: 10px;
    padding: 60px 20px 20px;
    min-height: 200px;
}
.empty-icon  { font-size: clamp(32px, 6vw, 44px); }
.empty-title { font-size: clamp(14px, 2.5vw, 17px); font-weight: 600; color: #6870a0; }
.empty-sub   { font-size: clamp(12px, 1.8vw, 13px); line-height: 1.7; max-width: 320px; }

.q-row { display: flex; justify-content: flex-end; margin-bottom: 4px; }
.q-bubble {
    background: #1a2e6e; color: #fff;
    padding: 11px 16px;
    border-radius: 16px 16px 3px 16px;
    max-width: min(65%, 520px);
    font-size: clamp(13px, 1.8vw, 14px); font-weight: 500; line-height: 1.6;
    word-break: break-word;
}
.q-meta { text-align: right; font-size: 11px; color: #a0a9cc; margin-bottom: 14px; }

.ans-card {
    background: #ffffff;
    border: 1px solid #dde1f0;
    border-radius: 3px 14px 14px 14px;
    padding: clamp(12px, 2.5vw, 18px) clamp(12px, 2.5vw, 20px);
    margin-bottom: 6px;
    max-width: min(90%, 760px);
    word-break: break-word;
}
.ans-meta {
    font-size: 11px; font-weight: 700; color: #1a2e6e;
    letter-spacing: 0.3px; margin-bottom: 13px;
    display: flex; align-items: center; gap: 6px;
}
.ans-dot { width: 6px; height: 6px; border-radius: 50%; background: #1a2e6e; display: inline-block; flex-shrink: 0; }

.ans-body h3 {
    font-size: clamp(12.5px, 1.8vw, 13.5px); font-weight: 700; color: #1a2e6e;
    padding-left: 10px; border-left: 3px solid #c2cbea;
    margin: 14px 0 7px;
}
.ans-body h3:first-child { margin-top: 0; }
.ans-body ul { list-style: none; padding: 0; margin: 0 0 4px; }
.ans-body ul li {
    display: flex; align-items: flex-start; gap: 9px;
    font-size: clamp(12.5px, 1.8vw, 13.5px); color: #2e3356; line-height: 1.72;
    padding: 5px 0; border-bottom: 1px solid #f0f2fa;
}
.ans-body ul li:last-child { border-bottom: none; }
.ans-body ul li::before {
    content: "•"; color: #1a2e6e;
    font-size: 16px; line-height: 1.45;
    flex-shrink: 0; font-weight: 800;
}
.ans-body p { font-size: clamp(12.5px, 1.8vw, 13.5px); color: #2e3356; line-height: 1.78; margin-bottom: 6px; }
.ans-body pre {
    background: #f0f2f8; border: 1px solid #dde1f0;
    border-radius: 7px; padding: 12px;
    font-size: clamp(11px, 1.5vw, 12.5px); color: #1c1e2e;
    overflow-x: auto; margin: 8px 0;
    font-family: 'Courier New', monospace;
    white-space: pre-wrap; word-break: break-word;
}
.ans-body code {
    background: #eef0fa; color: #1a2e6e;
    border-radius: 4px; padding: 1px 5px;
    font-size: 12px; font-family: 'Courier New', monospace;
    word-break: break-all;
}

.turn-sep { border: none; border-top: 1px solid #eaecf5; margin: 18px 0; }

.src-card {
    background: #f5f7fe; border: 1px solid #e0e4f4;
    border-left: 3px solid #1a2e6e; border-radius: 8px;
    padding: 10px 13px; margin-bottom: 8px;
    font-size: clamp(11px, 1.5vw, 12.5px); color: #505880; line-height: 1.65;
    word-break: break-word;
}
.src-num {
    font-size: 10px; font-weight: 700; color: #1a2e6e;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}

/* ═══════════════════════════════════════════════════════
   WIDGET OVERRIDES
═══════════════════════════════════════════════════════ */
.stTextInput > div > div > input {
    background: #f5f7fe !important;
    border: 1.5px solid #d5d9ee !important;
    border-radius: 10px !important;
    color: #1c1e2e !important;
    font-size: clamp(13px, 1.8vw, 14px) !important;
    padding: 10px 15px !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    height: 44px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a2e6e !important;
    box-shadow: 0 0 0 3px rgba(26,46,110,0.09) !important;
    background: #fff !important;
}
.stTextInput > div > div > input::placeholder { color: #b0b8d8 !important; }
.stTextInput { margin-bottom: 0 !important; }

.stButton > button {
    background: #1a2e6e !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    padding: 10px clamp(8px, 2vw, 16px) !important;
    font-weight: 600 !important; font-size: clamp(12px, 1.6vw, 14px) !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important; height: 44px !important;
    white-space: nowrap !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #122060 !important; }

.stFileUploader label { display: none !important; }
.stFileUploader > div {
    background: #f5f7fe !important;
    border: 1.5px dashed #c5cce8 !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] {
    background: #f5f7fe !important;
    border: 1px solid #e0e4f4 !important;
    border-radius: 10px !important;
    margin-top: 4px !important;
    max-width: min(90%, 760px);
}
div[data-testid="stExpander"] summary {
    font-size: 12px !important; color: #6870a0 !important; font-weight: 600 !important;
}

.stSpinner > div { color: #1a2e6e !important; }
.stAlert { border-radius: 10px !important; font-size: 13px !important; }

/* ═══════════════════════════════════════════════════════
   RESPONSIVE BREAKPOINTS
═══════════════════════════════════════════════════════ */
@media (max-width: 1024px) {
    [data-testid="stHorizontalBlock"] > div:first-child {
        padding: 18px 12px !important;
    }
    .q-bubble { max-width: 75%; }
    .ans-card { max-width: 95%; }
}

@media (max-width: 640px) {
    /* Show hamburger + overlay on mobile */
    .sb-hamburger { display: block !important; }
    .sb-overlay   { display: block !important; }

    /* Sidebar becomes off-canvas drawer */
    [data-testid="stHorizontalBlock"] > div:first-child {
        position: fixed !important;
        left: 0; top: 0; bottom: 0;
        z-index: 160 !important;
        width: min(80vw, 300px) !important;
        min-width: 220px !important;
        max-width: 300px !important;
        transform: translateX(-110%) !important;
        transition: transform 0.3s ease !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.15);
        flex: none !important;
        height: 100dvh !important;
        /* Safe area: sidebar top padding */
        padding-top: calc(24px + env(safe-area-inset-top, 0px)) !important;
    }
    [data-testid="stHorizontalBlock"] > div:first-child.sb-open {
        transform: translateX(0) !important;
    }

    /* Chat occupies full width */
    [data-testid="stHorizontalBlock"] > div:last-child {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    .chat-header { padding-left: 60px !important; }
    .q-bubble { max-width: 85%; }
    .ans-card { max-width: 100%; }

    .st-key-chat_messages {
        padding: 14px 12px 12px !important;
    }

    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
        padding: 10px 12px
                 calc(6px + env(safe-area-inset-bottom, 0px))
                 12px !important;
    }

    /* On mobile, tighten the gap between input and button */
    .composer-row {
        gap: 6px !important;
    }
}

@media (max-width: 380px) {
    .chat-header-sub { display: none; }
    .q-bubble { max-width: 90%; font-size: 13px; }
}

@media (min-width: 1800px) {
    .ans-card  { max-width: 860px; }
    .q-bubble  { max-width: 480px; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "session_id": str(uuid.uuid4()),
    "index": None,
    "chunks": [],
    "conversation": [],
    "pdf_ready": False,
    "pdf_name": "",
    "model": None,
    "client": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.model is None:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
if st.session_state.client is None:
    st.session_state.client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Render answer as structured bullets ───────────────────────────────────────
def render_bullets(raw: str) -> str:
    lines = raw.strip().split("\n")
    parts, in_code, code_buf = [], False, []
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            if in_code:
                in_code = False
                parts.append("<pre>" + "\n".join(code_buf) + "</pre>")
                code_buf = []
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        if not s:
            continue
        if s.endswith(":") and len(s) < 90:
            parts.append(f"<h3>{s[:-1]}</h3>")
        elif s.startswith(("-", "*", "•")) or (
            len(s) > 2 and s[0].isdigit() and s[1] in ".)"
        ):
            text = s.lstrip("-*•0123456789.) ").strip()
            parts.append(f"<ul><li>{text}</li></ul>")
        else:
            parts.append(f"<p>{s}</p>")
    return "\n".join(parts).replace("</ul>\n<ul>", "\n")


def run_answer(question):
    if not st.session_state.pdf_ready:
        st.warning("Please upload and process a PDF first.")
        return
    with st.spinner("Generating answer…"):
        try:
            sources = search_similar_chunks(
                question,
                st.session_state.model,
                st.session_state.index,
                st.session_state.chunks,
                k=3,
            )
            answer = generate_answer(question, sources, st.session_state.client)
            st.session_state.conversation.append({
                "question": question,
                "answer": answer,
                "sources": sources,
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


def mobile_sidebar_js():
    """
    Minimal JS — only responsible for:
    1. Hamburger / overlay toggle (mobile sidebar)
    2. Scroll-to-bottom after render
    No height calculations — CSS handles the layout geometry.
    """
    components.html("""
<script>
(function() {
    'use strict';

    var doc = window.parent.document;

    /* ── Scroll to bottom of messages ── */
    function scrollToBottom() {
        var msgs = doc.querySelector('.st-key-chat_messages');
        if (msgs) msgs.scrollTop = msgs.scrollHeight;
    }

    /* ── Mobile hamburger ── */
    function setupMobileSidebar() {
        if (doc.querySelector('#sb-hamburger-btn')) return;

        var btn = doc.createElement('button');
        btn.id = 'sb-hamburger-btn';
        btn.className = 'sb-hamburger';
        btn.setAttribute('aria-label', 'Open sidebar');
        btn.innerHTML = '&#9776;';
        doc.body.appendChild(btn);

        var overlay = doc.createElement('div');
        overlay.className = 'sb-overlay';
        overlay.id = 'sb-overlay';
        doc.body.appendChild(overlay);

        var sidebar = doc.querySelector('[data-testid="stHorizontalBlock"] > div:first-child');

        function open() {
            if (sidebar) sidebar.classList.add('sb-open');
            overlay.classList.add('open');
            btn.innerHTML = '&#10005;';
            btn.setAttribute('aria-label', 'Close sidebar');
        }
        function close() {
            if (sidebar) sidebar.classList.remove('sb-open');
            overlay.classList.remove('open');
            btn.innerHTML = '&#9776;';
            btn.setAttribute('aria-label', 'Open sidebar');
        }

        btn.addEventListener('click', function() {
            sidebar && sidebar.classList.contains('sb-open') ? close() : open();
        });
        overlay.addEventListener('click', close);

        var startX = 0;
        doc.addEventListener('touchstart', function(e) {
            startX = e.touches[0].clientX;
        }, { passive: true });
        doc.addEventListener('touchend', function(e) {
            if (e.changedTouches[0].clientX - startX < -60) close();
        }, { passive: true });
    }

    /* ── Run ── */
    function init() {
        setupMobileSidebar();
        scrollToBottom();
    }

    init();
    setTimeout(init, 250);
    setTimeout(scrollToBottom, 600);
})();
</script>
""", height=0)


# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 3], gap="small")


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown("""
    <div class="sb-brand">
      <span style="font-size:22px;">📖</span>
      <span class="sb-brand-name">StudyBuddy</span>
      <span class="sb-brand-tag">RAG</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.pdf_ready:
        st.markdown(
            f'<div class="sb-status ready">● {st.session_state.pdf_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status waiting">○ No document loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-label">Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")

    if uploaded and not st.session_state.pdf_ready:
        st.markdown(f"""
        <div class="sb-file-card">
          <span style="font-size:18px;">📄</span>
          <div>
            <div class="sb-file-name">{uploaded.name}</div>
            <div class="sb-file-meta">{uploaded.size // 1024} KB</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Process PDF"):
            save_path = f"uploads/{st.session_state.session_id}_{uploaded.name}"
            os.makedirs("uploads", exist_ok=True)
            slot = st.empty()
            try:
                raw = uploaded.getvalue()
                with open(save_path, "wb") as f:
                    f.write(raw)
                slot.caption("⏳ Validating…")
                validate_file(save_path)
                validate_file_size(save_path)
                slot.caption("⏳ Extracting text…")
                text = extract_text_from_pdf(save_path)
                text = clean_text(text)
                slot.caption("⏳ Chunking & embedding…")
                chunks = chunk_text(text)
                embeddings = embed_chunks(chunks, st.session_state.model)
                slot.caption("⏳ Building index…")
                index = build_vector_store(embeddings)
                os.remove(save_path)
                st.session_state.update({
                    "index": index, "chunks": chunks,
                    "pdf_ready": True, "pdf_name": uploaded.name,
                })
                slot.empty()
                st.rerun()
            except Exception as e:
                if os.path.exists(save_path):
                    os.remove(save_path)
                slot.empty()
                st.error(str(e))

    elif st.session_state.pdf_ready:
        st.markdown(f"""
        <div class="sb-file-card">
          <span style="font-size:18px;">✅</span>
          <div>
            <div class="sb-file-name">{st.session_state.pdf_name}</div>
            <div class="sb-file-meta">{len(st.session_state.chunks)} chunks indexed</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load different PDF"):
            for k in ["index", "chunks", "conversation", "pdf_ready", "pdf_name"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-tips">
      Ask specific questions using topic keywords.<br><br>
      Try:<br>
      <em>"List the main points about X"</em><br>
      <em>"Explain how Y works"</em><br>
      <em>"What are the steps for Z?"</em>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Chat
# ════════════════════════════════════════════════════════════════════════════
with right:
    # Fixed header
    st.markdown("""
    <div class="chat-header">
      <span style="font-size:17px;">💬</span>
      <span class="chat-header-title">Chat</span>
      <span class="chat-header-sub">Answers grounded in your document</span>
    </div>
    """, unsafe_allow_html=True)

    # The only scrollable zone — grows to fill remaining space via flex:1 1 0
    msgs = st.container(height=600, key="chat_messages")

    with msgs:
        if not st.session_state.conversation:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">💬</div>
              <div class="empty-title">Nothing here yet</div>
              <div class="empty-sub">
                Upload a PDF on the left and ask your first question below.
                Answers will appear as clear bullet points.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, turn in enumerate(st.session_state.conversation):
                if i > 0:
                    st.markdown('<hr class="turn-sep">', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="q-row"><div class="q-bubble">{turn['question']}</div></div>
                <div class="q-meta">You</div>
                """, unsafe_allow_html=True)

                body = render_bullets(turn["answer"])
                st.markdown(f"""
                <div class="ans-card">
                  <div class="ans-meta"><span class="ans-dot"></span>&nbsp;StudyBuddy</div>
                  <div class="ans-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

                if turn.get("sources"):
                    with st.expander(f"📄 View {len(turn['sources'])} source passage(s)"):
                        for j, chunk in enumerate(turn["sources"], 1):
                            st.markdown(f"""
                            <div class="src-card">
                              <div class="src-num">Passage {j}</div>
                              {chunk}
                            </div>
                            """, unsafe_allow_html=True)

    # Fixed bottom input — single column so Streamlit never injects
    # percentage widths that collapse the input on mobile.
    # CSS (.composer-row) positions input + button side-by-side.
    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown('<div class="composer-row">', unsafe_allow_html=True)
        question = st.text_input(
            "q",
            placeholder="Ask a question… (press Enter or click Ask)",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="input-hint">Press Enter or click Ask · Answers are grounded in your document</div>',
        unsafe_allow_html=True,
    )

    # Lightweight JS — only scroll-to-bottom + mobile sidebar toggle
    mobile_sidebar_js()

    if submitted and question.strip():
        run_answer(question.strip())
    elif submitted:
        st.warning("Please type a question first.")