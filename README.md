# Personal RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about my profile, projects, achievements, and experience using semantic search and Large Language Models.

The system processes personal documents, converts them into embeddings, retrieves relevant information using cosine similarity, and generates responses using Gemini.

---

## Features

* Document ingestion from TXT and PDF files
* Automatic text chunking
* Semantic search using embeddings
* Cosine similarity based retrieval
* Question answering using Gemini API
* Metadata tracking for source identification

---

## Project Structure

```
my-rag/
│
├── data/
│   ├── resume.txt
│   ├── projects.txt
│   ├── achievements.txt
│   └── certificates.pdf
│
└── build_knowledge.py
```

---

## Architecture

```
Personal Documents
(resume, projects, achievements, certificates)
                    │
                    ▼
           Document Loading
      (TXT + PDF extraction)
                    │
                    ▼
             Text Chunking
      (500 characters + overlap)
                    │
                    ▼
          Embedding Generation
     (FastEmbed + BAAI/bge-small-en-v1.5)
                    │
                    ▼
            Vector Storage
              (NumPy Array)
                    │
                    ▼
         Cosine Similarity Search
                    │
                    ▼
          Top Relevant Chunks
                    │
                    ▼
            Gemini Generation
                    │
                    ▼
             Final Response
```

---

## Tech Stack

### Backend

* Python

### Libraries Used

#### pathlib

Used for directory traversal and file handling.

Purpose:

* Discover files automatically
* Simplify path operations

#### PyPDF2

Used to extract text from PDF documents.

Purpose:

* Process certificates and PDF knowledge files

#### FastEmbed

Used to generate semantic embeddings.

Embedding Model:

```
BAAI/bge-small-en-v1.5
```

Purpose:

* Convert text into vector representations
* Enable semantic retrieval

#### NumPy

Used for vector storage and similarity calculations.

Purpose:

* Store embeddings efficiently
* Compute cosine similarity
* Retrieve Top-K relevant chunks

#### Gemini API

Used as the answer generation layer.

Purpose:

* Generate human-readable responses
* Answer questions using retrieved context

---

## Retrieval Pipeline

1. Load documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in NumPy
5. Embed incoming user question
6. Compute cosine similarity
7. Retrieve Top-K relevant chunks
8. Send retrieved context to Gemini
9. Generate final answer

---

## Example Questions

```
Which project uses OCR?

What machine learning algorithm is used in crop recommendation?

Which project was built independently?

Tell me about revenue leakage detection.
```

---

## Example Workflow

Question:

```
Which project uses OCR?
```

Retrieved Context:

```
Gate Scanner

Technologies:
HTML
CSS
JavaScript
OCR using Tesseract
```

Generated Answer:

```
Gate Scanner uses OCR with Tesseract to scan student ID cards and store student details.
```

---


## Learning Outcomes

In this project:

* Retrieval-Augmented Generation architecture
* Embedding generation
* Semantic search
* Cosine similarity
* Chunking strategies
* Prompt engineering
* Context grounding techniques
* End-to-end AI pipeline development

---

## Author

Udaya Krishna
Computer Science Engineering
AI/ML Enthusiast
