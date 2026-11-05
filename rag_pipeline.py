"""
RAG Pipeline — with graceful fallback when langchain/chromadb are not yet installed.
Uses ChromaDB + sentence-transformers when available; falls back to simple keyword search.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SimpleDocument:
    """Minimal document container (replaces LangChain Document)."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class RAGPipeline:
    """
    RAG pipeline with two modes:
    1. Full mode  — ChromaDB + sentence-transformers (when installed)
    2. Simple mode — keyword-based retrieval from in-memory text chunks
    """

    def __init__(self):
        self.vector_db_path = settings.vector_db_path
        self._vectorstore = None
        self._use_full_rag = False
        self._chunks: List[SimpleDocument] = []   # fallback in-memory store
        self._try_init_full_rag()

    def _try_init_full_rag(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            self._HuggingFaceEmbeddings = HuggingFaceEmbeddings
            self._Chroma = Chroma
            self._use_full_rag = True
            logger.info("Full RAG mode active (ChromaDB + sentence-transformers)")
        except ImportError:
            self._use_full_rag = False
            logger.info("Simple RAG mode active (keyword search — install requirements.txt for full RAG)")

    def _get_splitter(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _get_vectorstore(self):
        if self._vectorstore:
            return self._vectorstore
        os.makedirs(self.vector_db_path, exist_ok=True)
        embeddings = self._HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = self._Chroma(
            collection_name="farming_knowledge",
            embedding_function=embeddings,
            persist_directory=self.vector_db_path,
        )
        return self._vectorstore

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_documents(self, docs_dir: str = "./data/knowledge_base") -> int:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            logger.warning("Knowledge base directory not found: %s", docs_dir)
            return 0

        raw_texts = []
        for file_path in docs_path.rglob("*"):
            if file_path.suffix.lower() in {".txt", ".md"}:
                try:
                    text = file_path.read_text(encoding="utf-8")
                    raw_texts.append((text, str(file_path)))
                    logger.info("Loaded: %s", file_path)
                except Exception as exc:
                    logger.error("Failed to load %s: %s", file_path, exc)

        if not raw_texts:
            logger.warning("No .txt/.md documents found in %s", docs_dir)
            return 0

        if self._use_full_rag:
            return self._ingest_full_rag(raw_texts)
        else:
            return self._ingest_simple(raw_texts)

    def _ingest_full_rag(self, raw_texts) -> int:
        from langchain.schema import Document
        splitter = self._get_splitter()
        all_docs = [Document(page_content=t, metadata={"source": s}) for t, s in raw_texts]
        chunks = splitter.split_documents(all_docs)
        vs = self._get_vectorstore()
        vs.add_documents(chunks)
        vs.persist()
        logger.info("ChromaDB: ingested %d chunks", len(chunks))
        return len(chunks)

    def _ingest_simple(self, raw_texts) -> int:
        """Split on double-newlines and store in memory."""
        self._chunks = []
        for text, source in raw_texts:
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            for para in paragraphs:
                self._chunks.append(SimpleDocument(page_content=para, metadata={"source": source}))
        logger.info("Simple RAG: stored %d chunks in memory", len(self._chunks))
        return len(self._chunks)

    def ingest_text(self, text: str, metadata: dict = None) -> int:
        if self._use_full_rag:
            from langchain.schema import Document
            splitter = self._get_splitter()
            docs = splitter.split_documents([Document(page_content=text, metadata=metadata or {})])
            self._get_vectorstore().add_documents(docs)
            return len(docs)
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            for p in paragraphs:
                self._chunks.append(SimpleDocument(page_content=p, metadata=metadata or {}))
            return len(paragraphs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = None) -> List[SimpleDocument]:
        k = top_k or settings.top_k_retrieval
        if self._use_full_rag:
            try:
                results = self._get_vectorstore().similarity_search(query, k=k)
                return results
            except Exception as exc:
                logger.error("ChromaDB retrieval error: %s — falling back to simple", exc)

        return self._simple_retrieve(query, k)

    def _simple_retrieve(self, query: str, k: int) -> List[SimpleDocument]:
        """Keyword-overlap retrieval."""
        if not self._chunks:
            return []
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            chunk_words = set(chunk.page_content.lower().split())
            score = len(query_words & chunk_words)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def format_context(self, documents: List[SimpleDocument]) -> str:
        if not documents:
            return "No relevant information found in knowledge base."
        parts = []
        for i, doc in enumerate(documents, 1):
            source = Path(doc.metadata.get("source", "knowledge base")).name
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


_rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
