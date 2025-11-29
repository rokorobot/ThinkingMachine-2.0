#!/usr/bin/env python3
# scripts/ingest_knowledge.py
"""
Folder-based knowledge ingestion script.

Usage:
    python scripts/ingest_knowledge.py --path data/docs/ --source "tm_core_kb"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.knowledge_ingest import ingest_folder


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    parser.add_argument("--path", required=True, help="Folder containing PDFs / text files")
    parser.add_argument("--source", default="local_corpus", help="Source label for knowledge_documents.source")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Path does not exist or is not a directory: {root}")

    print(f"Ingesting documents from: {root}")
    print(f"Source label: {args.source}")
    
    ingest_folder(root, source=args.source)
    
    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
