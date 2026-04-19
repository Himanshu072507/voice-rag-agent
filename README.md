Goal: Build a multi-agent voice RAG system where users upload a PDF, chat with it via a GPT-style UI, and receive answers as both text and spoken audio.

Architecture: Four specialized LangChain LCEL agents in a sequential pipeline — PDF Ingestion → Retrieval → Answer Generation → TTS. FastAPI backend on Railway with ChromaDB vector store; Next.js 15 frontend on Vercel.

Tech Stack

| Layer         | Technology                                 |
| ------------- | ------------------------------------------ |
| Frontend      | Next.js 15, shadcn/ui, wavesurfer.js       |
| Backend       | FastAPI, Python 3.11                       |
| RAG Framework | LangChain LCEL                             |
| LLM           | Groq llama-3.3-70b-versatile (free)        |
| Embeddings    | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector Store  | ChromaDB                                   |
| TTS           | Browser Web Speech API (free, built-in)    |
| PDF Parsing   | PyMuPDF                                    |
| Hosting       | Localhost only                             |

Agent Pipeline

| Agent             | Input                  | Output              |
| ----------------- | ---------------------- | ------------------- |
| PDF Ingestion     | PDF bytes + session_id | ChromaDB collection |
| Retrieval         | Query + session_id     | Top-5 text chunks   |
| Answer Generation | Query + chunks         | Answer text         |
| TTS               | Answer text            | MP3 file path       |
