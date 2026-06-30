import streamlit as st
import streamlit.components.v1 as components
import os
import uuid
import time
import pickle
import faiss
import html as html_lib          # for escaping raw chunk text
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
    merge_vector_stores,
    search_similar_chunks,
    generate_answer,
)
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

load_dotenv()

# ── 1. Page config MUST be the first Streamlit command ─────────────────────
st.set_page_config(
    page_title="StudyBuddy",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Auth setup ────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

oauth2_config = {
    "google": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
    }
}

# ── 3. Login gate — Google OAuth only ───────────────────────────────────────
if not st.session_state.get("authentication_status"):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; }
    html, body {
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        font-family: 'Inter', sans-serif;
        background: #f0f2f8;
    }
    #root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        height: 100vh !important;
        min-height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .stAppHeader, [data-testid="stHeader"],
    #MainMenu, footer, header,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .block-container, [data-testid="block-container"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
        min-height: 100vh !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .login-wrap {
        max-width: 420px;
        width: 100%;
        background: #ffffff;
        border: 1px solid #dde1f0;
        border-radius: 14px;
        padding: 36px 32px;
        text-align: center;
        margin: 0 auto 16px;
    }
    .login-icon { font-size: 40px; margin-bottom: 6px; }
    .login-title { font-size: 22px; font-weight: 700; color: #1a2e6e; margin-bottom: 4px; }
    .login-sub { font-size: 13px; color: #9098be; margin-bottom: 0; }
    div[data-testid="stButton"] button {
        background: #1a2e6e !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        height: 46px !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button:hover { background: #122060 !important; }
    </style>
    <div class="login-wrap">
      <div class="login-icon">📖</div>
      <div class="login-title">StudyBuddy</div>
      <div class="login-sub">Sign in with Google to continue</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        try:
            authenticator.experimental_guest_login(
                "Sign in with Google",
                location="main",
                provider="google",
                oauth2=oauth2_config,
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Google login error: {e}")

    # If login just succeeded during this run, force a full rerun so the
    # login-screen markup never coexists with the main app's CSS/layout.
    if st.session_state.get("authentication_status"):
        st.rerun()

auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if auth_status is False:
    st.stop()
elif auth_status is None:
    st.stop()

# auth_status is True past this point — user is logged in

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════════════
   RESET & BASE — full-viewport, edge-to-edge, no gaps
═══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
    height: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    font-family: 'Inter', sans-serif;
    overflow: hidden;
    background: #f0f2f8;
}

#root,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockSpacer"] {
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    top: 0 !important;
    overflow: hidden !important;
}

/* Kill any default top offset/inset Streamlit applies to the view container */
[data-testid="stAppViewContainer"] {
    position: relative !important;
    inset: 0 !important;
}

* { scrollbar-width: thin; scrollbar-color: #c5cce8 transparent; }
*::-webkit-scrollbar { width: 5px; height: 5px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: #c5cce8; border-radius: 10px; }
*::-webkit-scrollbar-thumb:hover { background: #9098be; }

.stApp {
    background: #f0f2f8;
}

.stAppHeader, [data-testid="stHeader"],
#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* The actual full-bleed flex root: no max-width, no centering, no padding */
.stMainBlockContainer, [data-testid="stMainBlockContainer"],
.stAppViewContainer, [data-testid="stAppViewContainer"],
.stMain, [data-testid="stMain"] {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
    width: 100% !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}

.block-container, [data-testid="block-container"] {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
    width: 100% !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    display: flex !important;
}

[data-testid="stSidebar"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    flex: 0 0 0 !important;
}

/* ═══════════════════════════════════════════════════════
   MAIN LAYOUT — ChatGPT-style: fixed sidebar + flex:1 chat,
   both touching the very top of the viewport, no gaps.
═══════════════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: stretch !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
    overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════════
   LEFT COLUMN — fixed-width sidebar, full height, no top gap
═══════════════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"] > div:first-child {
    background: #ffffff;
    border-right: 1px solid #dde1f0;
    height: 100dvh !important;
    min-height: 100dvh !important;
    max-height: 100dvh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 24px 16px !important;
    margin: 0 !important;
    width: 360px !important;
    min-width: 360px !important;
    max-width: 360px !important;
    flex: 0 0 360px !important;
}

/* ═══════════════════════════════════════════════════════
   RIGHT COLUMN — chat shell, fills all remaining space
═══════════════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"] > div:last-child {
    display: flex !important;
    flex-direction: column !important;
    height: 100dvh !important;
    min-height: 0 !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #f0f2f8;
    min-width: 0 !important;
    flex: 1 1 auto !important;
}

[data-testid="stHorizontalBlock"] > div:last-child
  > [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] > div:last-child
  > div > [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ─── Chat header ─────────────────────────────────────── */
.chat-header {
    background: #ffffff;
    border-bottom: 1px solid #dde1f0;
    padding: 0 clamp(14px, 3vw, 28px);
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 0 0 auto !important;
     height: 52px;
    z-index: 20;
    position: sticky;
    top: 0;
}

.chat-header-menu-slot {
    display: none;
    width: 36px;
    height: 36px;
    flex-shrink: 0;
}

.chat-header-title { font-size: clamp(13px, 1.8vw, 15px); font-weight: 600; color: #1a2e6e; }
.chat-header-sub   { font-size: clamp(10px, 1.3vw, 12px); color: #9098be; margin-left: auto; white-space: nowrap; }

/* ─── Messages container — the ONE scrollable zone ─── */
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

.st-key-chat_messages > [data-testid="stVerticalBlockBorderWrapper"],
.st-key-chat_messages > [data-testid="stVerticalBlockBorderWrapper"] > div,
.st-key-chat_messages > [data-testid="stVerticalBlockBorderWrapper"] > div > div {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* ─── Input form — pinned at bottom ─── */
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
    min-height: 0 !important;
}

/* ─── Composer row ─── */
.composer-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    min-width: 0 !important;
}

.composer-row > div:has(.stTextInput) {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.composer-row > div:has(.stFormSubmitButton),
.composer-row > div:has(button[kind="primaryFormSubmit"]) {
    flex: 0 0 auto !important;
    min-width: fit-content !important;
}
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
.composer-row button { height: 44px !important; white-space: nowrap !important; }

[data-testid="stForm"] > div > div:first-child { flex: 1 1 0 !important; min-width: 0 !important; }
[data-testid="stForm"] > div > div:last-child  { flex: 0 0 auto !important; }

/* ─── Input hint ─── */
.input-hint {
    background: #ffffff;
    font-size: clamp(10px, 1.2vw, 11px);
    color: #b0b8d8;
    text-align: center;
    padding: 4px clamp(12px, 3vw, 28px)
             calc(6px + env(safe-area-inset-bottom, 0px));
    flex: 0 0 auto !important;
}

/* ─── Suppress Streamlit auto-scroll iframes ─── */
[data-testid="stHorizontalBlock"] > div:last-child .element-container:has(iframe),
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stIFrame"],
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stIFrame"] iframe,
[data-testid="stHorizontalBlock"] > div:last-child div:has(> iframe) {
    flex: 0 0 0 !important;
    height: 0 !important; min-height: 0 !important; max-height: 0 !important;
    margin: 0 !important; padding: 0 !important;
    border: none !important; overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════════
   HAMBURGER BUTTON & OVERLAY
═══════════════════════════════════════════════════════ */
.sb-hamburger {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    background: #1a2e6e; color: #fff;
    border: none; border-radius: 8px; cursor: pointer;
    font-size: 17px;
    box-shadow: 0 1px 4px rgba(26,46,110,0.2);
    transition: background 0.2s;
    flex-shrink: 0;
}
.sb-hamburger:hover { background: #122060; }

.sb-overlay {
    display: none;
    position: fixed; inset: 0; z-index: 150;
    background: rgba(0,0,0,0.35); backdrop-filter: blur(2px);
    opacity: 0; pointer-events: none; transition: opacity 0.3s;
}
.sb-overlay.open { opacity: 1 !important; pointer-events: all !important; }

/* ═══════════════════════════════════════════════════════
   SIDEBAR COMPONENTS
═══════════════════════════════════════════════════════ */
.sb-brand {
    display: flex; align-items: center; gap: 9px;
    margin-bottom: 22px; padding-bottom: 16px;
    border-bottom: 1px solid #eaecf5; flex-wrap: wrap;
}
.sb-brand-name { font-size: clamp(14px, 2vw, 17px); font-weight: 700; color: #1a2e6e; }
.sb-brand-tag  {
    font-size: 11px; color: #9098be;
    background: #eef0fa; padding: 2px 8px; border-radius: 10px; white-space: nowrap;
}
.sb-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.1px;
    text-transform: uppercase; color: #a0a9cc; margin-bottom: 10px;
}
.sb-file-card {
    background: #f5f7fe; border: 1px solid #e0e4f4; border-radius: 10px;
    padding: 11px 13px; display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px; min-width: 0;
}
.sb-file-name { font-size: clamp(11px, 1.5vw, 13px); font-weight: 600; color: #1c1e2e; word-break: break-all; }
.sb-file-meta { font-size: 11px; color: #9098be; margin-top: 2px; }

.sb-doc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #1a2e6e; color: #fff;
    font-size: 11px; font-weight: 700;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: 14px;
}

.sb-status {
    font-size: 12px; font-weight: 600; padding: 5px 12px; border-radius: 20px;
    display: inline-block; margin-bottom: 14px;
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sb-status.ready   { background: #e6f4ee; color: #1a6e42; }
.sb-status.waiting { background: #eef0fa; color: #7880ae; }
.sb-divider { border: none; border-top: 1px solid #eaecf5; margin: 18px 0; }
.sb-tips {
    background: #f5f7fe; border: 1px solid #e0e4f4; border-radius: 10px; padding: 14px;
    font-size: clamp(11px, 1.4vw, 13px); color: #626a94; line-height: 1.8;
}

/* ═══════════════════════════════════════════════════════
   CHAT BUBBLES & CARDS
═══════════════════════════════════════════════════════ */
.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; color: #9098be; gap: 10px; padding: 60px 20px 20px; min-height: 200px;
}
.empty-icon  { font-size: clamp(32px, 6vw, 44px); }
.empty-title { font-size: clamp(14px, 2.5vw, 17px); font-weight: 600; color: #6870a0; }
.empty-sub   { font-size: clamp(12px, 1.8vw, 13px); line-height: 1.7; max-width: 320px; }

.q-row { display: flex; justify-content: flex-end; margin-bottom: 4px; }
.q-bubble {
    background: #1a2e6e; color: #fff; padding: 11px 16px;
    border-radius: 16px 16px 3px 16px; max-width: min(65%, 520px);
    font-size: clamp(13px, 1.8vw, 14px); font-weight: 500; line-height: 1.6; word-break: break-word;
}
.q-meta { text-align: right; font-size: 11px; color: #a0a9cc; margin-bottom: 14px; }

.ans-card {
    background: #ffffff; border: 1px solid #dde1f0; border-radius: 3px 14px 14px 14px;
    padding: clamp(12px, 2.5vw, 18px) clamp(12px, 2.5vw, 20px);
    margin-bottom: 6px; max-width: min(90%, 760px); word-break: break-word;
}
.ans-meta {
    font-size: 11px; font-weight: 700; color: #1a2e6e; letter-spacing: 0.3px;
    margin-bottom: 13px; display: flex; align-items: center; gap: 6px;
}
.ans-dot { width: 6px; height: 6px; border-radius: 50%; background: #1a2e6e; display: inline-block; flex-shrink: 0; }

.ans-body h3 {
    font-size: clamp(12.5px, 1.8vw, 13.5px); font-weight: 700; color: #1a2e6e;
    padding-left: 10px; border-left: 3px solid #c2cbea; margin: 14px 0 7px;
}
.ans-body h3:first-child { margin-top: 0; }
.ans-body ul { list-style: none; padding: 0; margin: 0 0 4px; }
.ans-body ul li {
    display: flex; align-items: flex-start; gap: 9px;
    font-size: clamp(12.5px, 1.8vw, 13.5px); color: #2e3356; line-height: 1.72;
    padding: 5px 0; border-bottom: 1px solid #f0f2fa;
}
.ans-body ul li:last-child { border-bottom: none; }
.ans-body ul li::before { content: "•"; color: #1a2e6e; font-size: 16px; line-height: 1.45; flex-shrink: 0; font-weight: 800; }
.ans-body p  { font-size: clamp(12.5px, 1.8vw, 13.5px); color: #2e3356; line-height: 1.78; margin-bottom: 6px; }
.ans-body pre {
    background: #f0f2f8; border: 1px solid #dde1f0; border-radius: 7px; padding: 12px;
    font-size: clamp(11px, 1.5vw, 12.5px); color: #1c1e2e;
    overflow-x: auto; margin: 8px 0; font-family: 'Courier New', monospace;
    white-space: pre-wrap; word-break: break-word;
}
.ans-body code {
    background: #eef0fa; color: #1a2e6e; border-radius: 4px; padding: 1px 5px;
    font-size: 12px; font-family: 'Courier New', monospace; word-break: break-all;
}
.turn-sep { border: none; border-top: 1px solid #eaecf5; margin: 18px 0; }
.src-card {
    background: #f5f7fe; border: 1px solid #e0e4f4; border-left: 3px solid #1a2e6e;
    border-radius: 8px; padding: 10px 13px; margin-bottom: 8px;
    font-size: clamp(11px, 1.5vw, 12.5px); color: #505880; line-height: 1.65; word-break: break-word;
}
.src-num { font-size: 10px; font-weight: 700; color: #1a2e6e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.src-file {
    font-size: 10px; color: #9098be; font-style: italic; margin-bottom: 5px;
}

/* ═══════════════════════════════════════════════════════
   WIDGET OVERRIDES
═══════════════════════════════════════════════════════ */
.stTextInput > div > div > input {
    background: #f5f7fe !important; border: 1.5px solid #d5d9ee !important;
    border-radius: 10px !important; color: #1c1e2e !important;
    font-size: clamp(13px, 1.8vw, 14px) !important; padding: 10px 15px !important;
    font-family: 'Inter', sans-serif !important; width: 100% !important; height: 44px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a2e6e !important; box-shadow: 0 0 0 3px rgba(26,46,110,0.09) !important; background: #fff !important;
}
.stTextInput > div > div > input::placeholder { color: #b0b8d8 !important; }
.stTextInput { margin-bottom: 0 !important; }

.stButton > button {
    background: #1a2e6e !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 10px clamp(8px, 2vw, 16px) !important;
    font-weight: 600 !important; font-size: clamp(12px, 1.6vw, 14px) !important;
    font-family: 'Inter', sans-serif !important; width: 100% !important; height: 44px !important;
    white-space: nowrap !important; transition: background 0.2s !important;
}
.stButton > button:hover { background: #122060 !important; }

[data-testid="stBaseButton-secondary"] {
    background: #fdecea !important; color: #c0392b !important;
    border: 1.5px solid #f0c4c0 !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #fad4d0 !important; border-color: #e08080 !important;
}

.stFileUploader label { display: none !important; }
.stFileUploader > div { background: #f5f7fe !important; border: 1.5px dashed #c5cce8 !important; border-radius: 10px !important; }

div[data-testid="stExpander"] {
    background: #f5f7fe !important; border: 1px solid #e0e4f4 !important;
    border-radius: 10px !important; margin-top: 4px !important; max-width: min(90%, 760px);
}
div[data-testid="stExpander"] summary { font-size: 12px !important; color: #6870a0 !important; font-weight: 600 !important; }
.stSpinner > div { color: #1a2e6e !important; }
.stAlert { border-radius: 10px !important; font-size: 13px !important; }

/* ═══════════════════════════════════════════════════════
   RESPONSIVE BREAKPOINTS
═══════════════════════════════════════════════════════ */

@media (max-width: 767.98px) {
    .chat-header-menu-slot { display: flex !important; }
    .sb-overlay { display: block !important; }

    .chat-header { padding: 0 14px; gap: 10px; }

    [data-testid="stHorizontalBlock"] > div:first-child {
        position: fixed !important;
        left: 0; top: 0; bottom: 0; z-index: 160 !important;
        width: min(80vw, 300px) !important;
        min-width: 240px !important; max-width: 300px !important;
        transform: translateX(-110%) !important;
        transition: transform 0.3s ease !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.18);
        flex: none !important; height: 100dvh !important;
        overflow-y: auto !important;
        padding: calc(20px + env(safe-area-inset-top, 0px)) 16px 20px !important;
    }
    [data-testid="stHorizontalBlock"] > div:first-child.sb-open {
        transform: translateX(0) !important;
    }

    [data-testid="stHorizontalBlock"] > div:last-child {
        width: 100% !important; flex: 1 1 100% !important;
    }

    .q-bubble { max-width: 86%; }
    .ans-card  { max-width: 100%; }
    .st-key-chat_messages { padding: 14px 12px 12px !important; }

    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
        padding: 10px 12px calc(6px + env(safe-area-inset-bottom, 0px)) 12px !important;
    }
    .composer-row { gap: 8px !important; }
}

@media (max-width: 375.98px) {
    .chat-header { height: 48px; padding: 0 10px; gap: 7px; }
    .chat-header-title { font-size: 13px; }
    .chat-header-sub { display: none; }

    .st-key-chat_messages { padding: 10px 9px 10px !important; }

    .q-bubble {
        max-width: 92%;
        font-size: 13px;
        padding: 9px 12px;
    }
    .ans-card {
        max-width: 100%;
        padding: 10px 11px;
    }
    .ans-body p,
    .ans-body ul li { font-size: 12.5px; }

    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
        padding: 8px 9px calc(4px + env(safe-area-inset-bottom, 0px)) 9px !important;
    }
    .composer-row { gap: 6px !important; }
    .composer-row input { height: 40px !important; font-size: 13px !important; }
    .composer-row button { height: 40px !important; font-size: 12px !important; padding: 0 10px !important; }

    .input-hint { font-size: 10px; padding: 3px 9px calc(4px + env(safe-area-inset-bottom, 0px)); }

    [data-testid="stHorizontalBlock"] > div:first-child {
        width: min(85vw, 280px) !important;
        min-width: 220px !important;
        padding: calc(16px + env(safe-area-inset-top, 0px)) 13px 16px !important;
    }
    .sb-brand-name { font-size: 15px; }
    .sb-tips { font-size: 11.5px; padding: 11px; }
}

@media (max-width: 375.98px) and (max-height: 680px) {
    .empty-state { padding: 30px 16px 16px; gap: 7px; }
    .empty-icon { font-size: 28px; }
    .empty-title { font-size: 14px; }
    .empty-sub { font-size: 12px; }

    .chat-header { height: 44px; }
    .turn-sep { margin: 12px 0; }

    .input-hint { padding: 2px 9px calc(2px + env(safe-area-inset-bottom, 0px)); font-size: 9.5px; }
}

@media (min-width: 376px) and (max-width: 575.98px) {
    .chat-header-sub { display: none; }
    .q-bubble { max-width: 90%; font-size: 13px; }
    .st-key-chat_messages { padding: 12px 10px 10px !important; }
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
        padding: 8px 10px calc(4px + env(safe-area-inset-bottom, 0px)) 10px !important;
    }
}

@media (min-width: 576px) and (max-width: 767.98px) {
    .chat-header-sub { display: none; }
    .q-bubble { max-width: 82%; }
    .ans-card  { max-width: 98%; }
}

@media (min-width: 768px) and (max-width: 991.98px) {
    [data-testid="stHorizontalBlock"] > div:first-child {
        padding: 20px 12px !important;
        width: 220px !important;
        min-width: 220px !important;
        max-width: 220px !important;
        flex: 0 0 220px !important;
    }

    .chat-header-menu-slot { display: none !important; }
    .sb-overlay { display: none !important; }
    [data-testid="stHorizontalBlock"] > div:first-child {
        position: relative !important;
        transform: none !important;
        box-shadow: none !important;
        height: 100dvh !important;
    }

    .chat-header { padding: 0 16px; }
    .chat-header-sub { font-size: 11px; }

    .q-bubble { max-width: 72%; }
    .ans-card  { max-width: 94%; }
    .st-key-chat_messages { padding: 16px 14px 14px !important; }

    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
        padding: 10px 16px calc(6px + env(safe-area-inset-bottom, 0px)) 16px !important;
    }

    .sb-brand-name { font-size: 15px; }
    .sb-tips { font-size: 12px; }
}

@media (min-width: 992px) and (max-width: 1199.98px) {
    [data-testid="stHorizontalBlock"] > div:first-child {
        padding: 22px 14px !important;
        flex: 0 0 280px !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }

    .chat-header { padding: 0 20px; }
    .q-bubble { max-width: 68%; }
    .ans-card  { max-width: 90%; }
    .st-key-chat_messages { padding: 18px 20px 14px !important; }
}

@media (min-width: 1024px) and (max-width: 1366px) and (pointer: coarse) {
    [data-testid="stHorizontalBlock"] > div:first-child {
        flex: 0 0 300px !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        padding: 24px 16px !important;
    }

    .q-bubble { max-width: 60%; }
    .ans-card  { max-width: 86%; }
    .st-key-chat_messages { padding: 20px 24px 16px !important; }

    .composer-row input { height: 46px !important; }
    .composer-row button { height: 46px !important; }
}

@media (min-width: 1020px) and (max-width: 1030px) and (max-height: 780px) and (pointer: coarse) {
    .chat-header { height: 48px; }
    .st-key-chat_messages { padding: 14px 18px 12px !important; }
}

@media (min-width: 1200px) and (max-width: 1399.98px) {
    [data-testid="stHorizontalBlock"] > div:first-child { padding: 24px 16px !important; }
    .q-bubble { max-width: min(65%, 520px); }
    .ans-card  { max-width: min(90%, 760px); }
}

@media (min-width: 1400px) {
    [data-testid="stHorizontalBlock"] > div:first-child { padding: 28px 20px !important; }
    .ans-card  { max-width: 860px; }
    .q-bubble  { max-width: 500px; }
    .st-key-chat_messages { padding: 28px 48px 20px !important; }
}

@media (min-width: 1800px) {
    .ans-card  { max-width: 960px; }
    .q-bubble  { max-width: 560px; }
    .st-key-chat_messages { padding: 32px 64px 24px !important; }
}

</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "session_id": username,        # tied to the logged-in user, not a random UUID
    "index": None,
    "chunks": [],          # list[dict]  {"text": ..., "source": ...}
    "pdf_names": [],       # ordered list of loaded filenames
    "conversation": [],
    "model": None,
    "client": None,
    "last_active": time.time(),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Idle timeout check ──────────────────────────────────────────────────────
IDLE_TIMEOUT_SECONDS = 30 * 60
if time.time() - st.session_state.last_active > IDLE_TIMEOUT_SECONDS:
    for k in ["index", "chunks", "pdf_names", "conversation"]:
        st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
    st.warning("Session expired due to inactivity. Please log in again.")
    st.stop()
st.session_state.last_active = time.time()

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

st.session_state.model = load_embedding_model()
st.session_state.client = load_groq_client()

# ── Per-user persistent storage ──────────────────────────────────────────────
USER_STORE_ROOT = "faiss_store"

def _user_dir(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
    path = os.path.join(USER_STORE_ROOT, safe_id)
    os.makedirs(path, exist_ok=True)
    return path

def save_user_store():
    user_dir = _user_dir(username)
    if st.session_state.index is not None:
        faiss.write_index(st.session_state.index, os.path.join(user_dir, "index.faiss"))
        with open(os.path.join(user_dir, "chunks.pkl"), "wb") as f:
            pickle.dump({
                "chunks": st.session_state.chunks,
                "pdf_names": st.session_state.pdf_names,
            }, f)
    else:
        clear_user_store()

def load_user_store():
    user_dir = _user_dir(username)
    index_path = os.path.join(user_dir, "index.faiss")
    chunks_path = os.path.join(user_dir, "chunks.pkl")
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            st.session_state.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                data = pickle.load(f)
                st.session_state.chunks = data.get("chunks", [])
                st.session_state.pdf_names = data.get("pdf_names", [])
        except Exception:
            clear_user_store()

def clear_user_store():
    user_dir = _user_dir(username)
    for fname in ("index.faiss", "chunks.pkl"):
        fpath = os.path.join(user_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

if not st.session_state.get("_store_loaded"):
    load_user_store()
    st.session_state._store_loaded = True

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
        elif s.startswith(("-", "*", "•")) or (len(s) > 2 and s[0].isdigit() and s[1] in ".)"):
            text = s.lstrip("-*•0123456789.) ").strip()
            parts.append(f"<ul><li>{text}</li></ul>")
        else:
            parts.append(f"<p>{s}</p>")
    return "\n".join(parts).replace("</ul>\n<ul>", "\n")


def run_answer(question):
    if not st.session_state.pdf_names:
        st.warning("Please upload and process a PDF first.")
        return
    with st.spinner("Generating answer…"):
        try:
            sources = search_similar_chunks(
                question, st.session_state.model,
                st.session_state.index, st.session_state.chunks, k=3,
            )
            answer = generate_answer(
                question,
                [s["text"] for s in sources],
                st.session_state.client,
            )
            st.session_state.conversation.append({
                "question": question,
                "answer": answer,
                "sources": sources,
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


def process_pdf(uploaded_file):
    name = uploaded_file.name
    if name in st.session_state.pdf_names:
        st.warning(f'"{name}" is already loaded.')
        return False

    save_path = f"uploads/{st.session_state.session_id}_{name}"
    os.makedirs("uploads", exist_ok=True)
    slot = st.empty()
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        slot.caption("⏳ Validating…")
        validate_file(save_path)
        validate_file_size(save_path)

        slot.caption("⏳ Extracting text…")
        text = clean_text(extract_text_from_pdf(save_path))

        slot.caption("⏳ Chunking & embedding…")
        raw_chunks = chunk_text(text)
        tagged_chunks = [{"text": c, "source": name} for c in raw_chunks]
        embeddings = embed_chunks(raw_chunks, st.session_state.model)

        slot.caption("⏳ Building / merging index…")
        new_index = build_vector_store(embeddings)

        if st.session_state.index is None:
            st.session_state.index = new_index
        else:
            st.session_state.index = merge_vector_stores(
                st.session_state.index, new_index
            )

        st.session_state.chunks.extend(tagged_chunks)
        st.session_state.pdf_names.append(name)

        save_user_store()

        os.remove(save_path)
        slot.empty()
        return True
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        slot.empty()
        st.error(str(e))
        return False


def inject_js():
    components.html("""
<script>
(function () {
    'use strict';
    var doc = window.parent.document;

    function scrollToBottom() {
        var msgs = doc.querySelector('.st-key-chat_messages');
        if (msgs) msgs.scrollTop = msgs.scrollHeight;
    }

    function setup() {
        if (doc.querySelector('#sb-hamburger-btn')) {
            scrollToBottom();
            return;
        }

        var overlay = doc.createElement('div');
        overlay.className = 'sb-overlay';
        overlay.id = 'sb-overlay';
        doc.body.appendChild(overlay);

        var btn = doc.createElement('button');
        btn.id = 'sb-hamburger-btn';
        btn.className = 'sb-hamburger';
        btn.setAttribute('aria-label', 'Open sidebar');
        btn.innerHTML = '&#9776;';

        var slot = doc.querySelector('.chat-header-menu-slot');
        if (slot) {
            slot.appendChild(btn);
        } else {
            var hdr = doc.querySelector('.chat-header');
            if (hdr) hdr.prepend(btn);
        }

        var sidebar = doc.querySelector(
            '[data-testid="stHorizontalBlock"] > div:first-child'
        );

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

        btn.addEventListener('click', function () {
            sidebar && sidebar.classList.contains('sb-open') ? close() : open();
        });
        overlay.addEventListener('click', close);

        var startX = 0;
        doc.addEventListener('touchstart', function (e) {
            startX = e.touches[0].clientX;
        }, { passive: true });
        doc.addEventListener('touchend', function (e) {
            if (e.changedTouches[0].clientX - startX < -60) close();
        }, { passive: true });

        scrollToBottom();
    }

    setup();
    setTimeout(setup,          300);
    setTimeout(scrollToBottom, 700);
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

    if st.button("Logout", key="logout_btn"):
        authenticator.logout()
        st.rerun()

    if st.session_state.pdf_names:
        n = len(st.session_state.pdf_names)
        total_chunks = len(st.session_state.chunks)
        st.markdown(
            f'<div class="sb-doc-badge">📚 {n} document{"s" if n > 1 else ""} · {total_chunks} chunks</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sb-status waiting">○ No documents loaded</div>', unsafe_allow_html=True)

    if st.session_state.pdf_names:
        st.markdown('<div class="sb-label">Loaded Documents</div>', unsafe_allow_html=True)
        for fname in st.session_state.pdf_names:
            n_chunks = sum(1 for c in st.session_state.chunks if c["source"] == fname)
            st.markdown(f"""
            <div class="sb-file-card">
              <span style="font-size:18px;">✅</span>
              <div>
                <div class="sb-file-name">{fname}</div>
                <div class="sb-file-meta">{n_chunks} chunks</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    label = "Add PDFs" if st.session_state.pdf_names else "Document"
    st.markdown(f'<div class="sb-label">{label}</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "pdf",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    new_files = [f for f in uploaded_files if f.name not in st.session_state.pdf_names]

    if new_files:
        for f in new_files:
            st.markdown(f"""
            <div class="sb-file-card">
              <span style="font-size:18px;">📄</span>
              <div>
                <div class="sb-file-name">{f.name}</div>
                <div class="sb-file-meta">{f.size // 1024} KB · pending</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button(f"Process {len(new_files)} PDF{'s' if len(new_files) > 1 else ''}"):
            any_ok = False
            for f in new_files:
                if process_pdf(f):
                    any_ok = True
            if any_ok:
                st.rerun()
    elif uploaded_files:
        st.info("All selected files are already loaded.")

    if st.session_state.pdf_names:
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        if st.button("🗑 Clear all documents", key="clear_btn", type="secondary"):
            st.session_state.index        = None
            st.session_state.chunks       = []
            st.session_state.pdf_names    = []
            st.session_state.conversation = []
            clear_user_store()
            st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-tips">
      Load multiple PDFs and ask questions across all of them.<br><br>
      Try:<br>
      <em>"Compare X across both documents"</em><br>
      <em>"What does doc 2 say about Y?"</em><br>
      <em>"List all points about Z"</em>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Chat
# ════════════════════════════════════════════════════════════════════════════
with right:
    if st.session_state.pdf_names:
        doc_label = ", ".join(st.session_state.pdf_names)
        if len(doc_label) > 48:
            doc_label = doc_label[:45] + "…"
        sub_text = f"Searching: {doc_label}"
    else:
        sub_text = "Answers grounded in your documents"

    st.markdown(f"""
    <div class="chat-header">
      <div class="chat-header-menu-slot"></div>
      <span style="font-size:17px;">💬</span>
      <span class="chat-header-title">Chat</span>
      <span class="chat-header-sub">{sub_text}</span>
    </div>
    """, unsafe_allow_html=True)

    msgs = st.container(height=600, key="chat_messages")
    with msgs:
        if not st.session_state.conversation:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">💬</div>
              <div class="empty-title">Nothing here yet</div>
              <div class="empty-sub">Upload one or more PDFs on the left and ask your first question below.
                Answers are drawn from all loaded documents.</div>
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
                        for j, src in enumerate(turn["sources"], 1):
                            text  = src["text"]   if isinstance(src, dict) else src
                            fname = src.get("source", "") if isinstance(src, dict) else ""
                            safe_text  = html_lib.escape(text)
                            safe_fname = html_lib.escape(fname)
                            st.markdown(f"""
                            <div class="src-card">
                              <div class="src-num">Passage {j}</div>
                              <div class="src-file">📄 {safe_fname}</div>
                              <div style="white-space:pre-wrap">{safe_text}</div>
                            </div>
                            """, unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown('<div class="composer-row">', unsafe_allow_html=True)
        question = st.text_input("q", placeholder="Ask a question across all loaded documents…", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask →", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="input-hint">Press Enter or click Ask · Answers are grounded in your documents</div>',
        unsafe_allow_html=True,
    )

    inject_js()

    if submitted and question.strip():
        run_answer(question.strip())
    elif submitted:
        st.warning("Please type a question first.")