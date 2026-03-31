# RAG Context-Length Evaluation Pipeline

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) pipeline developed as part of a Master's thesis. The system is designed to evaluate how varying context length affects the performance of large language models (LLMs).

## Overview

The pipeline processes PDF documents, extracts textual and tabular content using OCR, and builds a retrieval system based on dense vector embeddings. Retrieved context is then used to generate responses via a large language model (Gemini), enabling controlled evaluation of context-length sensitivity.

## Pipeline Components

### 1. Document Processing
- PDF to image conversion using `pdf2image`
- OCR-based text and table extraction using `DocTR`

### 2. Text Preprocessing
- Cleaning and structuring extracted content
- Chunking with overlap to preserve semantic continuity

### 3. Embedding Generation
- Sentence embeddings generated using `sentence-transformers` (MiniLM model)

### 4. Vector Indexing
- FAISS index construction for efficient similarity search

### 5. Retrieval Mechanism
- Top-k relevant chunks retrieved based on query embedding

### 6. Prompt Construction
- Context injection with strict instruction constraints to ensure grounded responses

### 7. LLM Inference
- Response generation using Google's Gemini model via `google-generativeai`

## Key Features

- End-to-end RAG pipeline implementation
- OCR-based document understanding (including tables)
- Context-length control via chunk retrieval
- Deterministic instruction-based prompting
- Designed for academic evaluation and reproducibility


