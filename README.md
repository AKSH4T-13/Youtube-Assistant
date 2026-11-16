# YouTube RAG Assistant

A conversational Retrieval-Augmented Generation (RAG) application that allows users to ask questions about YouTube videos and have context-aware conversations about their content.

The application extracts a video's transcript, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves relevant information using MMR, and generates answers using a Groq-hosted model.

## Features

- YouTube transcript extraction
- Recursive text chunking
- Local HuggingFace embeddings
- FAISS vector database
- MMR-based retrieval
- Groq-powered question answering
- Conversational memory
- Follow-up questions
- Context-aware responses
- Answers grounded in extracted video content, with explicit notices for missing information
- Secure API key management using `.env`

## Tech Stack

- **Python**
- **LangChain**
- **Groq**
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
Groq
      ↓
Answer
```

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Set `GROQ_API_KEY` in `.env`. Optionally set `GROQ_MODEL`; the default is `openai/gpt-oss-120b`.

## Content extraction fallback

Try English YouTube captions first. If unavailable, download audio with yt-dlp
and transcribe locally with faster-whisper (multilingual `base`, CPU int8, voice
activity detection). If speech is empty or extraction fails, download video and
run English/Hindi EasyOCR on a frame every **5 seconds** by default, without deduplication.
The extracted text and timestamps feed the existing RAG. All methods failing
produces a readable error in the UI.

Models download on first use and are reused in the running process. Optional
`WHISPER_MODEL` in `.env` selects another Whisper model. Temporary media downloads
are cleaned up after extraction. No standalone FFmpeg is needed: faster-whisper
uses PyAV, and downloads use single streams without merging.

CPU processing of long videos, especially OCR, can be slow. The UI shows the current
stage and OCR timestamp. Slides shorter than the selected sampling interval may be missed. OCR reads
text, not diagrams. OCR is skipped when captions or speech succeed, including for
silent sections inside narrated videos. Speech recognition may misrecognize noise.
Private/deleted/download-blocked videos can still fail. Keep yt-dlp updated if
YouTube download requirements change.

Sound-cue-only captions such as [Music] and [Applause] trigger extraction fallback.
Answers must use extracted context; missing information is explicitly reported rather
than filled from general knowledge. A missing retrieved passage does not establish
that a fact is absent from the entire video.
