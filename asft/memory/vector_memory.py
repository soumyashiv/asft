"""
Vector Memory — Adapter layer supporting ChromaDB, FAISS, and Qdrant.
Switch backends via config alone with no changes to core architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    doc: VectorDocument
    score: float
    distance: float


from asft.core.interfaces import IMemoryStore, MemoryQueryResult  # noqa: E402


# ChromaDB is no longer the primary. Kept for backwards compatibility.
class ChromaDBBackend(IMemoryStore):
    def __init__(
        self, persist_dir: str, collection_name: str, host: str | None = None, port: int = 8000
    ):
        import chromadb

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB backend: collection=%s", collection_name)

    async def add(
        self, content: str, metadata: dict | None = None, vector: list[float] | None = None
    ) -> str:
        import uuid

        doc_id = str(uuid.uuid4())
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[vector] if vector else None,
            metadatas=[metadata or {}],
        )
        return doc_id

    async def update(
        self,
        item_id: str,
        content: str,
        metadata: dict | None = None,
        vector: list[float] | None = None,
    ) -> bool:
        self._collection.upsert(
            ids=[item_id],
            documents=[content],
            embeddings=[vector] if vector else None,
            metadatas=[metadata or {}],
        )
        return True

    async def delete(self, item_id: str) -> bool:
        self._collection.delete(ids=[item_id])
        return True

    async def search(self, query_vector: list[float], top_k: int = 5) -> list[MemoryQueryResult]:
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)
        out = []
        for i, _doc_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][i]
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            text = results["documents"][0][i] if results["documents"] else ""
            out.append(
                MemoryQueryResult(
                    source="chroma",
                    content=text,
                    confidence=1.0 - dist,
                    metadata=meta,
                )
            )
        return out

    async def batch_insert(
        self,
        contents: list[str],
        metadatas: list[dict] | None = None,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        import uuid

        ids = [str(uuid.uuid4()) for _ in contents]
        self._collection.upsert(
            ids=ids,
            documents=contents,
            embeddings=vectors if vectors else None,
            metadatas=metadatas if metadatas else [{} for _ in contents],
        )
        return ids

    async def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Wraps sentence-transformers for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model: %s (dim=%d, device=%s)", model_name, self.dim, device)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# VectorMemory — public interface
# ---------------------------------------------------------------------------


class VectorMemory:
    """
    Unified vector memory with pluggable backend.
    Backend is selected via config: "chromadb" | "faiss" | "qdrant"
    """

    def __init__(
        self,
        backend: str = "qdrant",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        **backend_kwargs,
    ):
        self._embedder = EmbeddingModel(embedding_model, embedding_device)
        dim = self._embedder.dim

        if backend == "chromadb":
            self._backend: IMemoryStore = ChromaDBBackend(**backend_kwargs)
        elif backend == "faiss":
            from asft.memory.backends.faiss_adapter import FaissBackend

            self._backend = FaissBackend(dim=dim)
        elif backend == "qdrant":
            from asft.memory.backends.qdrant import QdrantBackend

            self._backend = QdrantBackend(
                collection_name=backend_kwargs.get("collection_name", "asft_memory")
            )
        else:
            raise ValueError(
                f"Unknown vector backend: {backend!r}. Choose: chromadb, faiss, qdrant"
            )

        logger.info("VectorMemory: backend=%s", backend)

    async def add_text(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        embedding = self._embedder.encode_one(text)
        await self._backend.add(content=text, metadata=metadata, vector=embedding)

    async def add_texts(
        self, texts: list[tuple[str, str]], metadata: list[dict] | None = None
    ) -> None:
        """Add multiple (id, text) pairs."""
        all_texts = [t for _, t in texts]
        embeddings = self._embedder.encode(all_texts)
        await self._backend.batch_insert(
            contents=all_texts,
            metadatas=metadata,
            vectors=embeddings,
        )

    async def search(
        self, query: str, top_k: int = 10, filter: dict | None = None
    ) -> list[MemoryQueryResult]:
        query_emb = self._embedder.encode_one(query)
        # Note: IMemoryStore interface doesn't natively take filter.
        # But MemoryQueryResult has all the metadata.
        return await self._backend.search(query_vector=query_emb, top_k=top_k)

    async def delete(self, ids: list[str]) -> None:
        for item_id in ids:
            await self._backend.delete(item_id)

    @classmethod
    def from_config(cls, cfg) -> VectorMemory:
        """Build VectorMemory from MemoryConfig."""
        kwargs: dict[str, Any] = {}
        backend = cfg.vector_backend
        if backend == "chromadb":
            kwargs = {
                "persist_dir": cfg.chroma_persist_dir,
                "collection_name": (
                    cfg.vector_collection_name
                    if hasattr(cfg, "vector_collection_name")
                    else "asft_memory"
                ),
                "host": cfg.chroma_host,
                "port": cfg.chroma_port,
            }
        elif backend == "faiss":
            kwargs = {"index_path": cfg.faiss_index_path, "index_type": cfg.faiss_index_type}
        elif backend == "qdrant":
            kwargs = {
                "host": cfg.qdrant_host,
                "port": cfg.qdrant_port,
                "collection_name": cfg.qdrant_collection,
            }
        return cls(
            backend=backend,
            embedding_model=cfg.embedding_model,
            embedding_device=cfg.embedding_device,
            **kwargs,
        )
