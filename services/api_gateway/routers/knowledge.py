# services/api_gateway/routers/knowledge.py
"""
Admin API endpoints for knowledge base management.
Supports uploading documents for ingestion.
"""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from libs.db import get_conn
from libs.knowledge_ingest import ingest_file

router = APIRouter(prefix="/admin/knowledge", tags=["knowledge"])

UPLOAD_DIR = Path(os.getenv("KNOWLEDGE_UPLOAD_DIR", "data/uploads"))


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), source: str = "manual_upload"):
    """
    Upload a single PDF/TXT/MD file and ingest it into the knowledge base.
    
    Args:
        file: The uploaded file
        source: Source label for tracking (default: "manual_upload")
        
    Returns:
        {"status": "ok", "filename": str}
    """
    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename

    # Save file to disk
    with dest.open("wb") as f:
        content = await file.read()
        f.write(content)

    # Ingest using the same logic as folder ingestion
    conn = get_conn()
    try:
        ingest_file(conn, dest, source=source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        conn.close()

    return {"status": "ok", "filename": file.filename, "path": str(dest)}


@router.get("/stats")
def get_knowledge_stats():
    """
    Get statistics about the knowledge base.
    
    Returns:
        {
            "total_documents": int,
            "total_chunks": int,
            "sources": [{"source": str, "count": int}, ...]
        }
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Total documents
            cur.execute("SELECT COUNT(*) FROM knowledge_documents")
            total_docs = cur.fetchone()[0]
            
            # Total chunks
            cur.execute("SELECT COUNT(*) FROM knowledge_chunks")
            total_chunks = cur.fetchone()[0]
            
            # By source
            cur.execute("""
                SELECT source, COUNT(*) as count 
                FROM knowledge_documents 
                GROUP BY source 
                ORDER BY count DESC
            """)
            sources = [{"source": row[0], "count": row[1]} for row in cur.fetchall()]
    
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "sources": sources
    }
