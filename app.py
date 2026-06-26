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

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    height: 100%;
    overflow: hidden;
}

.stApp {
    background: #f0f2f8;
    height: 100vh;
    overflow: hidden;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
    overflow: hidden !important;
}

[data-testid="stHorizontalBlock"] {
    height: 100vh !important;
    align-items: stretch !important;
    gap: 0 !important;
}

/* ── LEFT column ── */
[data-testid="stHorizontalBlock"] > div:first-child {
    background: #ffffff;
    border-right: 1px solid #dde1f0;
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 24px 16px !important;
    flex-shrink: 0;
}

/* ── RIGHT column — flex column, owns full height ── */
[data-testid="stHorizontalBlock"] > div:last-child {
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    background: #f0f2f8;
}

/* The right column's inner vertical block also needs to be a flex column
   so the header / messages / input bar stack and share height correctly. */
[data-testid="stHorizontalBlock"] > div:last-child > div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* Sidebar pieces */
.sb-brand {
    display: flex; align-items: center; gap: 9px;
    margin-bottom: 22px; padding-bottom: 16px;
    border-bottom: 1px solid #eaecf5;
}
.sb-brand-name { font-size: 17px; font-weight: 700; color: #1a2e6e; }
.sb-brand-tag {
    font-size: 11px; color: #9098be;
    background: #eef0fa; padding: 2px 8px; border-radius: 10px;
}
.sb-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.1px;
    text-transform: uppercase; color: #a0a9cc; margin-bottom: 10px;
}
.sb-file-card {
    background: #f5f7fe; border: 1px solid #e0e4f4;
    border-radius: 10px; padding: 11px 13px;
    display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
}
.sb-file-name { font-size: 13px; font-weight: 600; color: #1c1e2e; word-break: break-all; }
.sb-file-meta { font-size: 11px; color: #9098be; margin-top: 2px; }
.sb-status {
    font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 20px;
    display: inline-block; margin-bottom: 14px;
}
.sb-status.ready   { background: #e6f4ee; color: #1a6e42; }
.sb-status.waiting { background: #eef0fa; color: #7880ae; }
.sb-divider { border: none; border-top: 1px solid #eaecf5; margin: 18px 0; }
.sb-tips {
    background: #f5f7fe; border: 1px solid #e0e4f4;
    border-radius: 10px; padding: 14px;
    font-size: 13px; color: #626a94; line-height: 1.8;
}

/* Chat header — fixed top of right column */
.chat-header {
    background: #ffffff;
    border-bottom: 1px solid #dde1f0;
    padding: 13px 28px;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
}
.chat-header-title { font-size: 15px; font-weight: 600; color: #1a2e6e; }
.chat-header-sub   { font-size: 12px; color: #9098be; margin-left: auto; }

/* ── SCROLLABLE MESSAGES ──
   This targets the *real* native Streamlit container created with
   st.container(height=..., key="chat_messages"). Because it's a genuine
   bounded container (not a markdown div spanning multiple calls), the
   messages rendered inside it are actually nested in the DOM and overflow works. */
.st-key-chat_messages,
[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 24px 32px 16px !important;
    scroll-behavior: smooth !important;
    border: none !important;
    background: transparent !important;
}
.st-key-chat_messages > div,
[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stVerticalBlockBorderWrapper"] > div {
    gap: 0 !important;
}

/* ── INPUT BAR ──
   No wrapping div here — Streamlit splits content across separate
   st.markdown() calls into DOM siblings, not nested children (the same
   issue that originally broke the chat-scroll div). Instead, the form
   widget itself is styled directly so it sits flush at the bottom,
   sized only to its own content, with no leftover whitespace. */
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] {
    background: #ffffff !important;
    border: none !important;
    border-top: 1px solid #dde1f0 !important;
    border-radius: 0 !important;
    padding: 12px 28px 6px !important;
    margin: 0 !important;
    flex: 0 0 auto !important;
}
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    align-items: center !important;
}
[data-testid="stHorizontalBlock"] > div:last-child div:has(> iframe) {
    flex: 0 0 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.stTextInput input {
    height: 44px !important;
}

/* Empty state */
.empty-state {
    height: 100%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; color: #9098be; gap: 10px;
}
.empty-icon  { font-size: 44px; }
.empty-title { font-size: 17px; font-weight: 600; color: #6870a0; }
.empty-sub   { font-size: 13px; line-height: 1.7; max-width: 320px; }

/* Question */
.q-row { display: flex; justify-content: flex-end; margin-bottom: 4px; }
.q-bubble {
    background: #1a2e6e; color: #fff;
    padding: 11px 16px;
    border-radius: 16px 16px 3px 16px;
    max-width: 65%;
    font-size: 14px; font-weight: 500; line-height: 1.6;
}
.q-meta { text-align: right; font-size: 11px; color: #a0a9cc; margin-bottom: 14px; }

/* Answer */
.ans-card {
    background: #ffffff;
    border: 1px solid #dde1f0;
    border-radius: 3px 14px 14px 14px;
    padding: 18px 20px;
    margin-bottom: 6px;
    max-width: 90%;
}
.ans-meta {
    font-size: 11px; font-weight: 700; color: #1a2e6e;
    letter-spacing: 0.3px; margin-bottom: 13px;
    display: flex; align-items: center; gap: 6px;
}
.ans-dot { width: 6px; height: 6px; border-radius: 50%; background: #1a2e6e; display: inline-block; }

.ans-body h3 {
    font-size: 13.5px; font-weight: 700; color: #1a2e6e;
    padding-left: 10px; border-left: 3px solid #c2cbea;
    margin: 14px 0 7px;
}
.ans-body h3:first-child { margin-top: 0; }
.ans-body ul { list-style: none; padding: 0; margin: 0 0 4px; }
.ans-body ul li {
    display: flex; align-items: flex-start; gap: 9px;
    font-size: 13.5px; color: #2e3356; line-height: 1.72;
    padding: 5px 0; border-bottom: 1px solid #f0f2fa;
}
.ans-body ul li:last-child { border-bottom: none; }
.ans-body ul li::before {
    content: "•"; color: #1a2e6e;
    font-size: 16px; line-height: 1.45;
    flex-shrink: 0; font-weight: 800;
}
.ans-body p { font-size: 13.5px; color: #2e3356; line-height: 1.78; margin-bottom: 6px; }
.ans-body pre {
    background: #f0f2f8; border: 1px solid #dde1f0;
    border-radius: 7px; padding: 12px;
    font-size: 12.5px; color: #1c1e2e;
    overflow-x: auto; margin: 8px 0;
    font-family: 'Courier New', monospace;
}
.ans-body code {
    background: #eef0fa; color: #1a2e6e;
    border-radius: 4px; padding: 1px 5px;
    font-size: 12px; font-family: 'Courier New', monospace;
}

.turn-sep { border: none; border-top: 1px solid #eaecf5; margin: 18px 0; }

.src-card {
    background: #f5f7fe; border: 1px solid #e0e4f4;
    border-left: 3px solid #1a2e6e; border-radius: 8px;
    padding: 10px 13px; margin-bottom: 8px;
    font-size: 12.5px; color: #505880; line-height: 1.65;
}
.src-num {
    font-size: 10px; font-weight: 700; color: #1a2e6e;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}

/* Input hint — sits directly under the form, no separate wrapper needed */
.input-hint {
    background: #ffffff;
    font-size: 11px; color: #b0b8d8; text-align: center;
    padding: 0 28px 12px;
}

/* Widget overrides */
.stTextInput > div > div > input {
    background: #f5f7fe !important;
    border: 1.5px solid #d5d9ee !important;
    border-radius: 10px !important;
    color: #1c1e2e !important;
    font-size: 14px !important;
    padding: 10px 15px !important;
    font-family: 'Inter', sans-serif !important;
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
    padding: 10px 16px !important;
    font-weight: 600 !important; font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important; height: 44px !important;
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
    max-width: 90%;
}
div[data-testid="stExpander"] summary {
    font-size: 12px !important; color: #6870a0 !important; font-weight: 600 !important;
}

.stSpinner > div { color: #1a2e6e !important; }
.stAlert { border-radius: 10px !important; font-size: 13px !important; }
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


def autoscroll_chat():
    """Executes real JS (st.markdown's <script> tags do not run) to scroll
    the native chat container to the bottom after every rerun."""
    components.html(
        """
        <script>
        (function() {
            function scrollIt() {
                var doc = window.parent.document;
                var el = doc.querySelector('.st-key-chat_messages');
                if (el) { el.scrollTop = el.scrollHeight; }
            }
            scrollIt();
            setTimeout(scrollIt, 50);
            setTimeout(scrollIt, 250);
        })();
        </script>
        """,
        height=0,
    )


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
        st.markdown(f'<div class="sb-status ready">● {st.session_state.pdf_name}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-status waiting">○ No document loaded</div>',
                    unsafe_allow_html=True)

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

    # ── REAL scrollable container ──
    # st.container(height=..., key=...) creates a genuine bounded DOM node
    # that Streamlit renders all nested elements *inside*. The CSS rule for
    # ".st-key-chat_messages" above turns that node into the flexible,
    # internally-scrolling chat panel. The height value passed here is just
    # a placeholder — the CSS `flex: 1 1 auto` overrides it so the panel
    # actually fills whatever space is left between the header and input bar.
    msgs = st.container(height=300, key="chat_messages")

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

    # ── Input bar: just the form + hint, no wrapping div (that pattern is
    # exactly what broke the original chat-scroll container). The .stForm
    # CSS rule above gives it its flush-bottom, white background look. ──
    with st.form(key="chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "q", placeholder="Ask a question… (press Enter or click Ask)",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("Ask →", use_container_width=True)

    st.markdown(
        '<div class="input-hint">Press Enter or click Ask · Answers are grounded in your document</div>',
        unsafe_allow_html=True,
    )

    autoscroll_chat()

    if submitted and question.strip():
        run_answer(question.strip())
    elif submitted:
        st.warning("Please type a question first.")