"""
Utility script — Ingest all documents in the knowledge base into ChromaDB.
Run once before starting the server (or call POST /ingest endpoint instead).

Usage:
    python ingest.py
    python ingest.py --docs-dir ./data/knowledge_base
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest farming knowledge base into ChromaDB")
    parser.add_argument(
        "--docs-dir",
        default="./data/knowledge_base",
        help="Path to the directory containing knowledge base documents",
    )
    args = parser.parse_args()

    logger.info("Starting knowledge base ingestion from: %s", args.docs_dir)

    from rag_pipeline import get_rag_pipeline
    pipeline = get_rag_pipeline()
    chunks = pipeline.ingest_documents(args.docs_dir)

    if chunks > 0:
        logger.info("✅ Successfully ingested %d chunks into ChromaDB.", chunks)
    else:
        logger.warning("⚠️  No documents were ingested. Check that %s contains .txt/.pdf/.csv/.docx files.", args.docs_dir)

    return 0 if chunks >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
