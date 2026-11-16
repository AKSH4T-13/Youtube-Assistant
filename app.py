import streamlit as st

import importlib
import inspect
import extractors
import main as backend

# Streamlit reruns this file but can retain older imported backend modules.
# Refresh only outdated signatures, preserving loaded models on normal reruns.
if "force_ocr" not in inspect.signature(extractors.extract_fallback_content).parameters:
    importlib.reload(extractors)
if ("force_ocr" not in inspect.signature(backend.initialize_rag).parameters
        or backend.extract_fallback_content is not extractors.extract_fallback_content):
    importlib.reload(backend)

initialize_rag = backend.initialize_rag
ask_question = backend.ask_question


st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="▶️",
    layout="centered"
)


# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #ffffff;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

h1 {
    font-weight: 700;
    color: #111111;
}

.subtitle {
    color: #777777;
    font-size: 16px;
    margin-bottom: 30px;
}

.stButton button {
    border-radius: 8px;
    height: 42px;
    font-weight: 600;
}

div[data-testid="stChatMessage"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# Session state initialization
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_url" not in st.session_state:
    st.session_state.video_url = ""


# Header
st.title("YouTube RAG Assistant")

st.markdown(
    '<p class="subtitle">Ask questions about any YouTube video using AI.</p>',
    unsafe_allow_html=True
)


# Video URL input
st.subheader("Add YouTube Video")

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="Paste your YouTube video link here...",
    key="youtube_input"
)


extraction_mode = st.selectbox("Content extraction", ["Automatic", "Slides only (OCR)"])

with st.expander("Slide reading settings"):
    ocr_language = st.selectbox("Slide language", ["English", "Hindi + English"])
    ocr_interval = st.select_slider("Read a frame every (seconds)", options=[2, 3, 5], value=5)
    st.caption("English uses a smaller recognition model. Longer intervals process fewer frames but may miss short slides.")

if st.button("Load Video", use_container_width=True):

    if not youtube_url.strip():

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            with st.status("Loading video...", expanded=True) as status:

                progress = st.empty()
                details = {}

                rag_chain, chat_history, chunk_count = initialize_rag(
                    youtube_url, progress_callback=progress.text,
                    ocr_languages=("en",) if ocr_language == "English" else ("en", "hi"),
                    ocr_interval=ocr_interval,
                    extraction_details=details,
                    force_ocr=extraction_mode == "Slides only (OCR)",
                )

                st.session_state.rag_chain = rag_chain

                st.session_state.chat_history = chat_history

                st.session_state.video_url = youtube_url

                st.session_state.messages = []
                st.session_state.extraction_details = details

                status.update(label="Video ready", state="complete", expanded=False)

            st.success(
                f"Video loaded successfully! {chunk_count} chunks created."
            )

        except Exception as e:

            st.error(f"Error loading video: {str(e)}")


# Chat section
if st.session_state.rag_chain:

    details = st.session_state.get("extraction_details", {})
    st.caption(f"Extraction source: {details.get('source', 'Reload video to see source')}")
    if details.get("preview"):
        with st.expander("Extracted text preview"):
            st.text(details["preview"])

    st.divider()

    st.subheader("Chat with your video")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    question = st.chat_input(
        "Ask something about the video..."
    )


    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = ask_question(
                        st.session_state.rag_chain,
                        st.session_state.chat_history,
                        question
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(f"Something went wrong: {str(e)}")


    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.session_state.chat_history.clear()

        st.rerun()
