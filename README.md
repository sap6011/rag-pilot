# rag-pilot

A from-scratch Retrieval-Augmented Generation system built over university coursework notes (PDFs, pptx, docx, ipynb, Rmd). Built to learn production RAG mechanics by measurement-driven iteration. Every improvement is justified by an eval metric.

## Why from scratch

Most RAG tutorials start with LangChain or LlamaIndex and end with a working chatbot. The problem is you learn the framework's API but not what it's actually doing. In this project, I deliberately build the same system three times:

1. **Phase 1-3:** Raw Python. No framework. Loaders, chunker, embedder, vector store, retrieval and generation coded manually.
2. **Phase 4 (planned):** Rebuild in LangChain. See how the framework helps and where are the constrains.
3. **Phase 5 (planned):** Extend into a stateful agent with LangGraph.

The point was to understand the *abstractions of the framework* before adopting them.

## Architecture

PDF / pptx / docx / ipynb / Rmd
↓
loaders.py        (format-specific text extraction + metadata)
↓
vlm_captioner.py  (qwen3-vl:4b captions image-heavy pages)
↓
chunker.py        (recursive 800-char splits, 100-char overlap)
↓
sentence-transformers/all-MiniLM-L6-v2  (384-dim embeddings)
↓
ChromaDB          (HNSW vector index, persisted to disk)
↓
retriever         (top-k cosine similarity)
↓
llama3.2:3b       (grounded answer generation via Ollama)
↓
answer + cited sources

Two LLMs and two roles: a small text model (`llama3.2:3b`) for answering and a small vision model (`qwen3-vl:4b`) for one-time captioning of diagrams during ingestion. The embedder is a third specialized model. Each component does its own thing.

## Results

Eval over 21 questions across 6 categories, k=5. Same corpus, same eval set, two configurations:

|                   | Baseline (text only) | + VLM captions |
|-------------------|---------------------:|---------------:|
| Recall@5          | 95%                  | **100%**       |
| MRR               | 0.835                | 0.853          |
| Keyword coverage  | 67%                  | **80%**        |

Breakdown by category (keyword coverage):

| Category      | Baseline | + VLM captions | Notes |
|---------------|---------:|---------------:|-------|
| case_study    | 67%      | 83%            |       |
| comparison    | 100%     | 100%           |       |
| definition    | 80%      | 74%            |       |
| image_heavy   | **7%**   | **93%**        | Targeted fix: 14x improvement |
| multi_chunk   | 40%      | 20%            | Known issue: title slides outrank content slides |
| out_of_scope  | 100% (refusal) | 100% (refusal) | Refusal accuracy is 10% |

### Story behind the image_heavy fix

The baseline eval showed image-heavy slides (diagrams) had 100% retrieval recall but only 6.67% answer coverage. Diagnosis: pypdf extracts the slide *title* but not the content *inside* the diagram. The vector match was matching titles, but the answer had nothing real to say.

Fix: during ingestion, for any page with <200 chars of extracted text, render the page and ask `qwen3-vl:4b` to caption it. Append the caption to the page text. The rest of the pipeline (chunking, embedding, retrieval, generation) is unchanged — the new text is just text.

Cost: ingestion goes from ~5s to ~90s for a 42-page PDF (one-time). Query latency is unaffected.

Result: image_heavy answer coverage 6.67% → 93.33%.

### The multi_chunk story

The "regression" from 40% → 20% on the one multi_chunk question turned out not to be a regression at all — the answer was correct ("all stakeholders are responsible, the data scientist is key") but the keywords list demanded enumeration of specific roles (managers, analysts, IT developers) that the model didn't bother to list. The underlying retrieval problem (page 24 with the actual stakeholder list ranks 4th out of 5 because shorter title slides outrank it) was already present in the baseline and is the next planned improvement.

**Lesson:** Always inspect per-question JSON before drawing conclusions from aggregate moves.

## What's next

- [ ] **Title-slide ranking fix** — merge title-only slides with the following content slide during chunking. Should lift definition MRR and the multi_chunk question.
- [ ] **Hybrid search (BM25 + vector with RRF fusion)** — biggest expected win for proper-noun queries.
- [ ] **Cross-encoder reranker** — bge-reranker-v2-m3 over top-20 candidates.
- [ ] **FastAPI endpoint** — `/query` and `/ingest` HTTP wrappers.
- [ ] **LangChain rebuild** — same system, same eval, framework version.
- [ ] **LangGraph extension** — turn it into a stateful agent that can ask clarifying questions, decide whether to retrieve, and chain multi-step lookups.

## How to run

Requires Python 3.11+, [Ollama](https://ollama.com), and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) on Windows (added to PATH).

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate           # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt  # (or pip install chromadb sentence-transformers ollama pypdf python-pptx python-docx nbformat tqdm pdf2image)

# Pull local models
ollama pull llama3.2:3b
ollama pull qwen3-vl:4b

# Drop course files into data/raw/, then ingest
python -m ingest.ingest

# Ask questions
python -m rag.pipeline "What is the Capability Maturity Model?"

# Run the eval
python -m eval.run_eval

# Compare runs
python -m eval.compare_runs
```

## Tech stack

- **LLM (text):** llama3.2:3b via Ollama (~2 GB, runs on CPU)
- **VLM (vision):** qwen3-vl:4b via Ollama (~3 GB, ingestion-only)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU)
- **Vector store:** ChromaDB with HNSW index (persisted to disk)
- **PDF rendering:** pdf2image + Poppler
- **API (planned):** FastAPI
- **Frameworks (planned):** LangChain, LangGraph

## Project structure

rag-pilot/
├── ingest/
│   ├── loaders.py         # per-format text extraction
│   ├── chunker.py         # recursive character splitter
│   ├── vlm_captioner.py   # qwen3-vl captioning for image-heavy pages
│   └── ingest.py          # orchestrates ingestion
├── rag/
│   └── pipeline.py        # retrieve → format context → generate
├── eval/
│   ├── testset.jsonl      # 21 test cases across 6 categories
│   ├── run_eval.py        # measures recall@k, MRR, keyword coverage, refusal
│   ├── compare_runs.py    # diff the two most recent runs
│   └── runs/              # versioned per-run JSON snapshots (gitignored)
├── api/                   # FastAPI (planned)
├── data/
│   ├── raw/               # source files (gitignored)
│   └── chroma/            # vector store (gitignored)
└── README.md

## Design principles

- **Each module has one reason to change.** Loaders change when file formats change. The captioner changes when vision strategy changes. The chunker changes when chunking strategy changes. They don't bleed into each other.
- **Measurement before improvement.** Every change is justified by a metric that's expected to move, and verified by re-running the eval.
- **Local-first.** Everything runs on a laptop with no API keys. Lets you understand the full stack and iterate without rate limits.
- **Versioned evals.** Every eval run produces a timestamped JSON with config metadata and per-question results. No silent overwrites; every metric movement is traceable to a specific config.