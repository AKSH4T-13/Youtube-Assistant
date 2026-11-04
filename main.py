import os
import re

from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


def get_video_id(youtube_url: str) -> str:

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
        r"(?:youtube\.com/embed/)([^?&]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)

        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL")


def get_video_transcript(video_id: str) -> str:

    try:

        transcript_data = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        transcript = " ".join(
            item.text if hasattr(item, "text") else item["text"]
            for item in transcript_data
        )

        if not transcript.strip():
            raise RuntimeError("Transcript is empty.")

        return transcript

    except TranscriptsDisabled:
        raise RuntimeError(
            "Captions are disabled for this video."
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch transcript: {exc}"
        )


def create_chunks(transcript: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    if not chunks:
        raise RuntimeError("No transcript chunks were created.")

    return chunks


def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks, embeddings):

    return FAISS.from_documents(
        chunks,
        embeddings
    )


def create_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
            "lambda_mult": 0.7
        }
    )


def create_llm():

    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is missing")

    if not GEMINI_MODEL:
        raise EnvironmentError("GEMINI_MODEL is missing")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )


def create_rag_chain(retriever, llm, chat_history):

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant answering questions about a YouTube video.

        Use the video context as the primary source of information.

        Rules:
        1. If the answer is available in the video context, use it.
        2. If the context does not contain the answer, use general knowledge.
        3. Never claim general knowledge was stated in the video.
        4. Use conversation history for follow-up questions.
        5. Give clear, accurate and helpful answers.

        Conversation History:
        {history}

        Video Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    def format_docs(documents):

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    def format_history():

        return "\n".join(
            f"{message.type}: {message.content}"
            for message in chat_history.messages
        )

    def retrieve_context(question):

        documents = retriever.invoke(question)

        return format_docs(documents)

    chain = (
        {
            "history": lambda x: format_history(),
            "context": lambda x: retrieve_context(x),
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def initialize_rag(youtube_url: str):

    video_id = get_video_id(youtube_url)

    transcript = get_video_transcript(video_id)

    chunks = create_chunks(transcript)

    embeddings = create_embeddings()

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    retriever = create_retriever(vector_store)

    llm = create_llm()

    chat_history = InMemoryChatMessageHistory()

    rag_chain = create_rag_chain(
        retriever,
        llm,
        chat_history
    )

    return rag_chain, chat_history, len(chunks)


def ask_question(rag_chain, chat_history, question):

    answer = rag_chain.invoke(question)

    chat_history.add_message(
        HumanMessage(content=question)
    )

    chat_history.add_message(
        AIMessage(content=answer)
    )

    return answer

