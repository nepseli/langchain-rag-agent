# Financial Document RAG Agent

A retrieval-augmented generation (RAG) agent for querying financial documents — invoices, contracts, and financial statements — through a conversational chat interface. The agent handles numeric-heavy documents with tables, performs calculations, compares figures across multiple documents, and asks clarifying questions when a query is ambiguous.

---

## Intention

Most financial documents contain a mix of prose, structured tables, and image-based content (logos, stamps, scanned headers). Standard RAG pipelines treat all of this as flat text and struggle with numeric reasoning. This project addresses that by:

- Extracting tables as structured data (not just text) and executing pandas code against them for reliable calculations
- Supplementing PDF text extraction with OCR so image-rendered content (company names in logos, letterheads) is captured
- Using HyDE (Hypothetical Document Embeddings) to improve retrieval on domain-specific financial vocabulary
- Detecting ambiguous queries ("what is this invoice about?") when multiple documents are loaded and asking for clarification before answering

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Orchestration | LangGraph (StateGraph) |
| Chains / Tools | LangChain + PythonREPLTool |
| Vector store | ChromaDB (persistent) |
| Metadata / tables | SQLite |
| PDF parsing | pdfplumber + PyMuPDF |
| OCR | Tesseract via pytesseract |
| Office formats | python-docx, pandas, openpyxl |
| UI | Streamlit |
| Config | pydantic-settings + `.env` |

---

## Project Structure

```
rag_agent/
├── app/
│   └── streamlit_app.py        # Streamlit chat UI
├── core/
│   ├── agent/
│   │   ├── graph.py            # LangGraph StateGraph definition
│   │   ├── nodes.py            # Node functions (classify, retrieve, generate, …)
│   │   ├── prompts.py          # All prompt templates
│   │   └── state.py            # RAGState TypedDict
│   ├── ingestion/
│   │   ├── chunker.py          # Text and table chunking
│   │   ├── document_loader.py  # PDF / DOCX / Excel / CSV loaders
│   │   ├── ocr.py              # OCR supplement for image-heavy pages
│   │   ├── pipeline.py         # Ingestion orchestrator
│   │   └── table_extractor.py  # Table extraction (dual markdown + JSON)
│   └── retrieval/
│       ├── metadata_store.py   # SQLite operations
│       └── vector_store.py     # ChromaDB wrapper
├── db/
│   └── migrations/
│       └── 001_initial.sql     # Schema for documents, tables, chat sessions
├── models/
│   └── schemas.py              # Pydantic models (Document, Chunk, SourceRef, …)
├── config.py                   # Central config via pydantic-settings
├── requirements.txt
└── .env                        # Your secrets (never committed)
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Tesseract OCR binary (required for reading logos and image-embedded text)
  - **Windows:** download from https://github.com/UB-Mannheim/tesseract/wiki and install to the default path
  - **macOS:** `brew install tesseract`
  - **Linux:** `sudo apt-get install tesseract-ocr`

### 2. Clone and create a virtual environment

```bash
git clone https://github.com/nepseli/langchain-rag-agent.git
cd langchain-rag-agent

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (never commit this):

```env
OPENAI_API_KEY=sk-...

# Optional overrides (defaults shown)
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
DB_PATH=data/rag.db
CHROMA_PATH=data/chroma_db
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=12
```

### 5. Run

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

1. **Upload documents** — drag PDFs, DOCX, Excel, or CSV files into the sidebar. Each file is parsed, OCR'd if needed, chunked, and indexed into ChromaDB. Progress is shown inline.
2. **Ask questions** — type in the chat box. Examples:
   - "What is the total amount on invoice INV-00002?"
   - "Sum all invoice totals across all documents"
   - "Compare the payment terms in the two contracts"
   - "What company issued the proforma invoice?"
3. **Clarification** — if your query is ambiguous (e.g. "what is this invoice about?" when multiple documents are loaded), the agent asks which document you mean before proceeding.
4. **Delete documents** — use the 🗑 button next to each document, or "Delete all docs" to reset.
5. **Clear chat** — resets the conversation history without affecting indexed documents.

---

## Salient Features

**Hybrid OCR ingestion** — every PDF page that contains embedded images is rendered at 200 DPI via PyMuPDF and passed through Tesseract. The OCR output is merged with pdfplumber's vector text extraction, so company names, logos, and stylised headers are captured.

**Dual-representation tables** — tables are stored as both GitHub-flavoured markdown (for embedding and display) and raw JSON (for pandas execution). This means the agent can both retrieve and compute against the same table.

**Python REPL for calculations** — when a query requires arithmetic, the agent generates pandas code and executes it via `PythonREPLTool` rather than asking the LLM to compute. This eliminates arithmetic hallucinations.

**HyDE retrieval** — before searching ChromaDB, the agent generates a hypothetical answer to the query. The embedding of this hypothetical answer retrieves more relevant chunks than embedding the raw question, particularly for financial vocabulary.

**Comparative fan-out** — for cross-document queries, the retriever samples the top-N chunks from each indexed document independently, ensuring no single verbose document crowds out the others.

**Ambiguity detection** — a `check_clarity` node runs before retrieval. If the query uses vague references ("this invoice", "the document") and multiple documents are indexed, the agent lists the available files and asks the user to specify — instead of guessing.

**Per-doc chunk cap** — a configurable ceiling (`max_chunks_per_doc`) prevents one large document from dominating the context window on factual and numeric queries.

---

## Agent Graph

```
check_clarity
  ├── [ambiguous] → clarify → END
  └── [clear]     → classify_query → rewrite_query → retrieve
                        ├── [has tables + numeric/calc/comparative]
                        │     → fetch_table_data
                        │           ├── [calculation] → execute_calculation → generate → END
                        │           └── [numeric]                           → generate → END
                        └── [factual / general]                             → generate → END
```

---

## Next Steps

- **Authentication** — add user login so multiple users can maintain separate document sets and chat histories
- **Streaming responses** — wire up LangGraph streaming mode and Streamlit's `st.write_stream` so answers appear token by token
- **Re-ranking** — add a cross-encoder re-ranker (e.g. `ms-marco-MiniLM`) between retrieval and generation to improve chunk relevance ordering
- **Multi-modal support** — extend OCR to handle scanned-only PDFs end-to-end, and add support for images embedded in DOCX files
- **Metadata filtering** — allow users to filter retrieval by document type, date range, or vendor name using ChromaDB's `where` clause
- **Evaluation harness** — build a small benchmark of question/answer pairs over known documents and measure retrieval recall and answer correctness automatically
- **Export** — add a button to export the current chat session as a PDF or markdown report with all citations

---

## License

MIT
