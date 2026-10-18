# YouTube RAG Assistant

A conversational Retrieval-Augmented Generation (RAG) application that allows users to ask questions about YouTube videos and have context-aware conversations about their content.

The application extracts a video's transcript, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves relevant information using MMR, and generates answers using Google's Gemini model.

## Features

- YouTube transcript extraction
- Recursive text chunking
- Local HuggingFace embeddings
- FAISS vector database
- MMR-based retrieval
- Gemini-powered question answering
- Conversational memory
- Follow-up questions
- Context-aware responses
- General-knowledge fallback when the answer is not available in the video context
- Secure API key management using `.env`

## Tech Stack

- **Python**
- **LangChain**
- **Google Gemini**
- **FAISS**
- **HuggingFace Sentence Transformers**
- **YouTube Transcript API**
- **python-dotenv**

## How It Works

```text
YouTube Video
      ↓
Transcript Extraction
      ↓
Text Chunking
      ↓
HuggingFace Embeddings
      ↓
FAISS Vector Store
      ↓
MMR Retriever
      ↓
Conversation History + Retrieved Context
      ↓
Gemini
      ↓
Answer