# libs/knowledge_ingest.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import psycopg2
import psycopg2.extras

from .db import get_conn
from .user_memory import embed_text  # reuse same embedding logic


EMBED_DIM = 1536  # must match your pgvector dimension


# ---------- text extraction ----------

def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF using pypdf library."""
    try:
        from pypdf import PdfReader  # pip install pypdf
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts)
    except ImportError:
        print("[ERROR] pypdf not installed. Run: pip install pypdf")
        return ""


def extract_text_from_txt(path: Path) -> str:
    """Extract text from plain text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_md(path: Path) -> str:
    """Extract text from markdown file (treated as plain text)."""
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    You can later replace with sentence/paragraph-aware chunking.
    """
    text = text.strip()
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------- DB insert helpers ----------

def insert_document(conn, source: str, uri: str, title: str, doc_type: str, meta: Dict[str, Any]) -> int:
    """Insert a document record and return its ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO knowledge_documents (source, uri, title, doc_type, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            RETURNING id
            """,
            (source, uri, title, doc_type, psycopg2.extras.Json(meta)),
        )
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


def insert_chunks(conn, document_id: int, chunks: List[str]) -> None:
    """Insert text chunks with embeddings for a document."""
    with conn.cursor() as cur:
        for idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            cur.execute(
                """
                INSERT INTO knowledge_chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                (document_id, idx, chunk, emb),
            )
    conn.commit()


# ---------- high-level ingestion ----------

def ingest_file(conn, path: Path, source: str) -> None:
    """Ingest a single file (PDF/TXT/MD) into the knowledge base."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
        doc_type = "pdf"
    elif ext in (".txt", ".text"):
        text = extract_text_from_txt(path)
        doc_type = "txt"
    elif ext in (".md", ".markdown"):
        text = extract_text_from_md(path)
        doc_type = "md"
    else:
        print(f"[skip] Unsupported file type: {path}")
        return

    if not text.strip():
        print(f"[skip] No text extracted from: {path}")
        return

    chunks = chunk_text(text)
    if not chunks:
        print(f"[skip] No chunks for: {path}")
        return

    title = path.stem
    uri = str(path.resolve())
    meta = {"filename": path.name}

    print(f"[ingest] {path} -> {len(chunks)} chunks")

    doc_id = insert_document(conn, source=source, uri=uri, title=title, doc_type=doc_type, meta=meta)
    insert_chunks(conn, doc_id, chunks)


def ingest_folder(root: Path, source: str) -> None:
    """Ingest all supported files from a folder recursively."""
    conn = get_conn()
    try:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() in (".pdf", ".txt", ".text", ".md", ".markdown"):
                    ingest_file(conn, p, source=source)
    finally:
        conn.close()
