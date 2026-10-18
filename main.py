import os

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


if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is missing")

if not GEMINI_MODEL:
    raise EnvironmentError("GEMINI_MODEL is missing")

# transcript nikaalo from video id
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

        return transcript

    except TranscriptsDisabled:
        raise RuntimeError(
            "Captions are disabled for this video."
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch transcript: {exc}"
        )

# if transcript is too big, create chunks of 1000
def create_chunks(transcript: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    if not chunks:
        raise RuntimeError("No transcript chunks were created.")

    return chunks

# vector embeddings banao ab
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store

# top 4 relevant + diverse chunks retrieve honge
def create_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
            "lambda_mult": 0.7
        }
    )

    return retriever

# initialize LLM
def create_llm():

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )

    return llm

# chain bana do to connect all dots
def create_rag_chain(retriever, llm, chat_history):
    prompt = ChatPromptTemplate.from_template(
        """
    You are an AI assistant answering questions about a YouTube video.

    Use the video context as the primary source of information.

    Rules:
    1. If the answer is available in the video context, use the video context.
    2. If the video context does not contain the answer, you may use your
       general knowledge to provide a helpful answer.
    3. Never claim that information from general knowledge was stated in the video.
    4. Use the conversation history to understand follow-up questions.
    5. If the question is ambiguous, use the conversation history to understand
       what the user is referring to.
    6. Give a clear, accurate and helpful answer.

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

chat_history = InMemoryChatMessageHistory()

def main():

    # YouTube video ID
    youtube_url = input("Enter YouTube video URL: ")
    if "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = youtube_url.split("v=")[1].split("&")[0]

    print("Fetching transcript...")

    transcript = get_video_transcript(video_id)

    print(
        f"Transcript fetched: {len(transcript)} characters"
    )


    # Create chunks
    print("Creating chunks...")

    chunks = create_chunks(transcript)

    print(
        f"Created {len(chunks)} chunks"
    )


    # Create FAISS vector store
    print("Creating vector store...")

    vector_store = create_vector_store(chunks)

    print("Vector store created")


    # Create retriever
    retriever = create_retriever(vector_store)

    print("Retriever ready")


    # Create Gemini
    print("Loading Gemini...")

    llm = create_llm()

    print("Gemini ready")


    # Create RAG chain
    rag_chain = create_rag_chain(
        retriever,
        llm,
        chat_history
    )

    print("RAG chain ready")


    # Ask question
    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break

        answer = rag_chain.invoke(question)

        chat_history.add_message(
            HumanMessage(content=question)
        )

        chat_history.add_message(
            AIMessage(content=answer)
        )

        print("\nAI:")
        print(answer)

if __name__ == "__main__":
    main()
