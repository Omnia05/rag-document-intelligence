# RAG Document Intelligence

An AI-powered **Retrieval-Augmented Generation (RAG)** system that transforms a collection of academic documents into an intelligent, searchable knowledge assistant.

The system supports multiple document formats, performs document preprocessing and token-aware chunking, stores embeddings in a session-specific Chroma vector database, retrieves relevant context, and generates conversational answers using an LLM through OpenRouter.

---

##  Features

### Multi-format document ingestion
  - PDF
  - Markdown (`.md`)
  - Text (`.txt`)
  - Word documents (`.docx`)

### Recursive directory processing
  - Process an entire knowledge-base directory
  - Automatically discover documents inside subdirectories
  - Avoid duplicate document ingestion

### File-size validation
  - Configurable maximum file size
  - Oversized or unreadable files are skipped and reported

### Configurable text chunking
  - Adjustable chunk size
  - Adjustable chunk overlap
  - Token-aware filtering to control the amount of context sent to the LLM

### Token-aware context management
  - Maximum total token budget
  - Maximum individual chunk size
  - Prevents excessively large chunks from entering the retrieval pipeline

### Semantic embeddings
  - Uses `sentence-transformers/all-MiniLM-L6-v2`
  - Converts document chunks into vector representations

### Chroma vector database
  - Stores document embeddings for semantic retrieval
  - Uses a separate temporary vector store for each user session

### Conversational RAG
  - Maintains conversation history
  - Retrieves relevant document chunks for each question
  - Generates answers using an OpenRouter-hosted LLM

### Source attribution
  - Displays the documents used to generate an answer
  - Shows task/category information
  - Shows PDF page numbers when available
  - Supports source display for non-PDF documents

### Session management
  - Generates a unique session for each processing run
  - Automatically cleans up stale sessions
  - Refreshes active sessions when the chatbot is used

### Interactive Gradio interface
  - Document processing configuration
  - Processing status and detailed logs
  - Configurable retrieval depth (`Top-K`)
  - Conversational chatbot
  - Chat-history reset

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │    Gradio Web UI     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Document Ingestion  │
                         │ PDF / MD / TXT / DOCX│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Preprocessing &     │
                         │  Token Filtering     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Text Chunking      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌───────────────────────┐
                         │ HuggingFace Embeddings│
                         │ all-MiniLM-L6-v2      │
                         └──────────┬────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Chroma Vector Store │
                         └──────────┬───────────┘
                                    │
                           Semantic Retrieval
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Conversational       │
                         │ Retrieval Chain      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ OpenRouter LLM       │
                         │ GPT-OSS-120B         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Answer + Sources     │
                         └──────────────────────┘
```

---

## Project Structure

```text
rag-document-intelligence/
│
├── app/
│   ├── __init__.py
│   └── gradio_app.py
│
├── src/
│   ├── __init__.py
│   │
│   └── rag/
│       ├── __init__.py
│       ├── config.py
│       ├── session.py
│       ├── ingestion.py
│       ├── preprocessing.py
│       ├── embeddings.py
│       ├── vectorstore.py
│       ├── chat.py
│       └── pipeline.py
│
│── notebooks/
│   └── baseline.ipynb
│ 
├── .env.example
├── .gitignore
├── README.md
└── LICENSE

```

---

## RAG Pipeline

The application follows the following processing pipeline:

### 1. Document Ingestion

The user provides a directory containing academic documents.

Supported formats:

```text
.pdf
.md
.txt
.docx
```

The ingestion module:

- Recursively discovers supported files
- Checks file size
- Loads documents using format-specific LangChain loaders
- Adds useful metadata
- Records files that could not be processed

Example metadata:

- file_name
- file_path
- file_extension
- doc_type
- page
- page_label
- total_pages

### 2. Document Preprocessing

Documents are divided into manageable chunks using a configurable:
- Chunk size
- Chunk overlap

The pipeline also performs token-based filtering.

Two limits are enforced:
- Maximum total tokens: 250,000
- Maximum tokens per chunk: 8,000

This prevents excessively large context from entering the retrieval pipeline.

### 3. Embedding Generation

Each document chunk is converted into a vector representation using:
```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings capture the semantic meaning of the text and allow relevant chunks to be retrieved based on meaning rather than exact keyword matches.

### 4. Vector Storage

The generated embeddings are stored in a Chroma vector database.

Each processing session receives its own temporary vector-store directory.

```text
Temporary Directory
        │
        └── rag_sessions/
              │
              ├── <session-id-1>/
              ├── <session-id-2>/
              └── <session-id-3>/
```

This prevents different sessions from sharing document embeddings.

### 5. Semantic Retrieval

When the user asks a question, the system retrieves the most relevant document chunks from Chroma.
The number of retrieved chunks is configurable through the Gradio interface.

Default: Top-K = 25

### 6. Conversation Generation

The retrieved context is passed to a conversational retrieval chain.
The system maintains conversation history using: ConversationBufferMemory

The language model is accessed through OpenRouter.
Default model:
```text
openai/gpt-oss-120b
```

### 7. Source Attribution

The system returns both:
- Answer
- Source Documents

Retrieved source metadata is formatted into readable references.

--- 

## User Interface

The application is built using Gradio.
The interface is divided into two main sections.

### Configuration

Users can configure:
- Knowledge Base Directory
- Maximum File Size
- Chunk Size
- Chunk Overlap
- Number of retrieved chunks

The interface also displays:
- Processing status
- Detailed ingestion logs
- Number of documents loaded
- Number of chunks created
- Number of skipped files

### Chat

After processing the documents, users can ask questions through the conversational interface.

The chatbot:
- Receives the user's question
- Retrieves relevant document chunks
- Sends the retrieved context to the LLM
- Generates an answer
- Displays the supporting source documents

Users can also clear the conversation history.

--- 

## Configuration

Create a `.env` file in the project root. 
A `.env.example` file is provided for reference

--- 

## Installation
### 1. Clone the repository
### 2. Create a virtual environment
```text
python3 -m venv .venv
```
Activate it:

- macOS / Linux
```text
source .venv/bin/activate
```
- Windows
```text
.venv\Scripts\activate
```
### 3. Install dependencies

### 4. Configure the API key

---
## Running the Application
From the project root:
```text
python -m app.gradio_app
```
The Gradio interface will start locally.

--- 

## Example Workflow

Suppose the knowledge base contains:

```text
academic_documents/
│
├── task1/
│   └── analysis.pdf
│
├── task2/
│   ├── partA/
│   │   └── README.md
│   │
│   └── partB/
│       └── analysis.pdf
│
└── task3/
    └── analysis.pdf
```

The application automatically discovers the supported documents.
Example processing output:
```text
=== Document Ingestion ===

Processing folder: task1
  Documents loaded: 6
    - analysis.pdf: 6 document(s)

Processing folder: task2
  Documents loaded: 10
    - analysis.pdf: 9 document(s)
    - README.md: 1 document(s)

Processing folder: task3
  Documents loaded: 5
    - analysis.pdf: 5 document(s)
```
The documents are then processed through:
```text
Loaded Documents
       ↓
Chunks
       ↓
Embeddings
       ↓
Chroma
       ↓
Retriever
       ↓
LLM
       ↓
Answer + Sources
```
--- 
## Technologies Used

```text
| Component            | Technology                        |
| -------------------- | --------------------------------- |
| Programming Language | Python                            |
| UI                   | Gradio                            |
| LLM                  | OpenRouter                        |
| LLM Model            | GPT-OSS-120B                      |
| RAG Framework        | LangChain                         |
| Vector Database      | Chroma                            |
| Embeddings           | HuggingFace Sentence Transformers |
| Embedding Model      | all-MiniLM-L6-v2                  |
| PDF Processing       | PyPDF                             |
| DOCX Processing      | docx2txt                          |
| Configuration        | python-dotenv                     |
| Tokenization         | tiktoken                          |
```
--- 
## Core Modules

### `config.py`

Centralizes:
- API configuration
- LLM configuration
- Embedding configuration
- Retrieval defaults
- Token limits

### `session.py`

Handles:
- Session creation
- Temporary session directories
- Session cleanup
- Stale-session cleanup
- Session initialization

### `ingestion.py`

Responsible for:
- File discovery
- Supported-format validation
- File-size validation
- Document loading
- Metadata enrichment
- Error handling

### `preprocessing.py`

Responsible for:
- Token counting
- Token-based filtering
- Document chunking
- Chunk-size control
- Total-token limits

### `embeddings.py`

Creates the HuggingFace embedding model used by the vector store.

### `vectorstore.py`

Creates the Chroma vector database from processed document chunks.

### `chat.py`

Handles:
- LLM initialization
- Conversational retrieval
- Conversation memory
- Source-document retrieval
- Source formatting
- Conversation reset

### `pipeline.py`

Coordinates the complete workflow:

```text
Ingestion
   ↓
Preprocessing
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Store
   ↓
Conversational Retrieval
```

### `gradio_app.py`

Provides the interactive web interface and connects the UI to the RAG pipeline.

--- 
## Error Handling

The application handles several common failure cases:

- Missing OpenRouter API key
- Unsupported file formats
- Files exceeding the configured size limit
- Document loading failures
- Empty document collections
- Empty chunk collections
- Missing active conversation
- LLM/retrieval errors

Skipped files and their reasons are displayed in the processing log.

---

## License

This project is intended for educational and research purposes.