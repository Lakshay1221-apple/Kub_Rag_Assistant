import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- AVATARS ---
AI_AVATAR = "assistant"
USER_AVATAR = "user"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 60
EXAMPLE_PROMPTS = [
    "How do Kubernetes CronJobs work?",
    "Summarize pod autoscaling from the docs.",
    "What is the recommended way to monitor Jobs?",
    "Compare Jobs and CronJobs.",
]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --kub-bg: #f7f9fc;
            --kub-panel: #ffffff;
            --kub-ink: #172033;
            --kub-muted: #667085;
            --kub-line: #d8dee9;
            --kub-blue: #246bfe;
            --kub-cyan: #00a3b5;
            --kub-green: #11845b;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(36, 107, 254, 0.10), transparent 28rem),
                linear-gradient(180deg, #fbfcff 0%, var(--kub-bg) 100%);
            color: var(--kub-ink);
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div,
        .stMarkdown,
        .stMarkdown * {
            color: var(--kub-ink);
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 1.4rem;
            padding-bottom: 7rem;
        }

        .hero {
            border: 1px solid var(--kub-line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: var(--kub-blue);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .hero h1 {
            color: var(--kub-ink);
            font-size: 2rem;
            line-height: 1.15;
            margin: 0;
        }

        .hero p {
            color: var(--kub-muted);
            font-size: 0.98rem;
            margin: 0.45rem 0 0;
            max-width: 760px;
        }

        .metric-row {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.85rem 0 1.15rem;
        }

        .metric {
            background: var(--kub-panel);
            border: 1px solid var(--kub-line);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }

        .metric span {
            color: var(--kub-muted);
            display: block;
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }

        .metric strong {
            color: var(--kub-ink);
            font-size: 1rem;
        }

        .empty-state {
            background: #ffffff;
            border: 1px dashed #b7c4d8;
            border-radius: 8px;
            margin: 1rem 0;
            padding: 1.1rem;
        }

        .empty-state h3 {
            color: var(--kub-ink);
            font-size: 1rem;
            margin: 0 0 0.35rem;
        }

        .empty-state p {
            color: var(--kub-muted);
            margin: 0;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(216, 222, 233, 0.86);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.65rem;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] code,
        div[data-testid="stChatMessage"] pre,
        div[data-testid="stChatMessage"] div {
            color: var(--kub-ink);
        }

        div[data-testid="stChatMessage"] p {
            font-size: 0.98rem;
            line-height: 1.65;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #f0f7ff;
            border-color: #b9d4ff;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: #ffffff;
            border-color: #d9e2ef;
        }

        div[data-testid="stStatusWidget"] {
            border-radius: 8px;
            border: 1px solid var(--kub-line);
            background: #ffffff;
        }

        div[data-testid="stStatusWidget"] *,
        div[data-testid="stExpander"] *,
        div[data-testid="stAlert"] * {
            color: var(--kub-ink);
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid #cfd8e6;
            font-weight: 650;
            min-height: 2.35rem;
        }

        .stTextInput input,
        .stChatInput textarea,
        div[data-testid="stChatInput"] textarea {
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid #c4cede;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
            color: var(--kub-ink);
        }

        div[data-testid="stChatInput"] {
            background: rgba(247, 249, 252, 0.92);
        }

        div[data-testid="stChatInput"] button {
            border-radius: 8px;
        }

        @media (max-width: 760px) {
            .metric-row {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def compact_source_title(source: object, index: int) -> str:
    text = str(source).replace("\n", " ").strip()
    if not text:
        return f"Source {index}"
    return f"Source {index}: {text[:92]}{'...' if len(text) > 92 else ''}"


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


inject_theme()


# --- SIDEBAR ---
with st.sidebar:
    st.title("Agent Console")
    st.caption("Runtime controls and session context")

    st.divider()
    st.markdown("**Backend**")
    st.code(BACKEND_URL, language=None)
    st.markdown("**Tracing**")
    if LOGFIRE_STATUS == "Connected & Tracing":
        st.success(LOGFIRE_STATUS)
    else:
        st.warning(LOGFIRE_STATUS)

    st.divider()
    st.markdown("**Memory**")
    st.caption(f"Session: {st.session_state.session_id[:8]}")
    st.caption(f"Messages: {len(st.session_state.messages)}")

    if st.button("Clear chat and memory", width="stretch", type="primary"):
        logfire.warn(f"Memory wipe triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()
    st.markdown("**Try asking**")
    for example in EXAMPLE_PROMPTS:
        if st.button(example, width="stretch"):
            st.session_state.pending_prompt = example
            st.rerun()

# --- MAIN CHAT ---
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Kubernetes Documentation Assistant</div>
        <h1>Enterprise Agentic RAG</h1>
        <p>Ask grounded questions across your indexed docs. The assistant plans, retrieves, reranks, and answers with source context when retrieval is needed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric"><span>Backend</span><strong>{BACKEND_URL.replace("http://", "")}</strong></div>
        <div class="metric"><span>Session</span><strong>{st.session_state.session_id[:8]}</strong></div>
        <div class="metric"><span>Conversation</span><strong>{len(st.session_state.messages)} messages</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Start with a documentation question</h3>
            <p>Use the sidebar prompts or ask something specific about Jobs, CronJobs, autoscaling, monitoring, or architecture.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
typed_prompt = st.chat_input("Ask about your documentation...")
prompt = st.session_state.pending_prompt or typed_prompt

if prompt:
    st.session_state.pending_prompt = None
    # START TRACE: User Interaction
    with logfire.span("User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("Planning retrieval workflow...", expanded=False) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("Calling RAG Backend"):
                        # Get backend URL from env, or default to local if not set
                        url = f"{BACKEND_URL}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(
                            url,
                            json=payload,
                            timeout=REQUEST_TIMEOUT_SECONDS,
                        )
                        response.raise_for_status()
                        data = response.json()

                    status.update(label="Answer synthesized", state="complete", expanded=False)
                except Exception as e:
                    logfire.error(f"UI-Backend connection failed: {e}")
                    status.update(label="Connection failed", state="error")
                    st.error(
                        "I could not reach the RAG backend. Check that FastAPI is running and that BACKEND_URL is correct."
                    )
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            words = full_answer.split(" ")
            if len(words) > 1:
                curr_text = ""
                for index in range(0, len(words), 6):
                    curr_text = " ".join(words[: index + 6])
                    answer_placeholder.markdown(curr_text + " ▌")
                    time.sleep(0.015)

            answer_placeholder.markdown(full_answer)

            steps = data.get("thought_process", [])
            sources = data.get("sources", [])
            if steps or sources:
                with st.expander("Reasoning and retrieved context"):
                    if steps:
                        st.markdown("**Agent trace**")
                        for step in steps:
                            st.write(f"- {step}")
                    else:
                        st.caption("No planner trace returned.")

                    if sources:
                        st.markdown("**Sources**")
                        for i, source in enumerate(sources):
                            with st.expander(compact_source_title(source, i + 1)):
                                st.markdown(str(source))
                    else:
                        st.caption("No external source chunks were returned for this answer.")

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
