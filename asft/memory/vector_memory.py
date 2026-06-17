"""
Vector Memory — Adapter layer supporting ChromaDB, FAISS, and Qdrant.
Switch backends via config alone with no changes to core architecture.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    doc: VectorDocument
    score: float
    distance: float


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class VectorBackend(ABC):
    """Base interface all vector backends must implement."""

    @abstractmethod
    def add(self, docs: List[VectorDocument]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10,
               filter: Optional[Dict] = None) -> List[SearchResult]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class ChromaDBBackend(VectorBackend):
    def __init__(self, persist_dir: str, collection_name: str,
                 host: Optional[str] = None, port: int = 8000):
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

    def add(self, docs: List[VectorDocument]) -> None:
        if not docs:
            return
        self._collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            embeddings=[d.embedding for d in docs if d.embedding is not None] or None,
            metadatas=[d.metadata for d in docs],
        )

    def search(self, query_embedding: List[float], top_k: int = 10,
               filter: Optional[Dict] = None) -> List[SearchResult]:
        kwargs: Dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
        if filter:
            kwargs["where"] = filter
        results = self._collection.query(**kwargs)
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][i]
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            text = results["documents"][0][i] if results["documents"] else ""
            out.append(SearchResult(
                doc=VectorDocument(id=doc_id, text=text, metadata=meta),
                score=1.0 - dist,
                distance=dist,
            ))
        return out

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class FAISSBackend(VectorBackend):
    def __init__(self, index_path: str, index_type: str = "Flat", dim: int = 384):
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss: pip install faiss-cpu  or  pip install asft[faiss]")
        import faiss
        self._faiss = faiss
        self._index_path = index_path
        self._dim = dim
        self._docs: Dict[int, VectorDocument] = {}
        self._id_map: Dict[str, int] = {}  # str_id → faiss int_id
        self._next_id = 0

        if index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, min(100, max(1, 1)))
        elif index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self._index = faiss.IndexFlatIP(dim)  # Flat inner-product (cosine after normalize)
        logger.info("FAISS backend: type=%s dim=%d", index_type, dim)

    def add(self, docs: List[VectorDocument]) -> None:
        if not docs:
            return
        vecs = []
        for doc in docs:
            if doc.embedding is None:
                continue
            arr = np.array(doc.embedding, dtype=np.float32)
            arr /= (np.linalg.norm(arr) + 1e-9)  # normalize for cosine
            vecs.append(arr)
            faiss_id = self._next_id
            self._next_id += 1
            self._id_map[doc.id] = faiss_id
            self._docs[faiss_id] = doc
        if vecs:
            matrix = np.stack(vecs)
            self._index.add(matrix)

    def search(self, query_embedding: List[float], top_k: int = 10,
               filter: Optional[Dict] = None) -> List[SearchResult]:
        arr = np.array(query_embedding, dtype=np.float32)
        arr /= (np.linalg.norm(arr) + 1e-9)
        scores, indices = self._index.search(arr.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self._docs:
                continue
            doc = self._docs[idx]
            results.append(SearchResult(doc=doc, score=float(score), distance=1.0 - float(score)))
        return results

    def delete(self, ids: List[str]) -> None:
        logger.warning("FAISS flat index does not support deletion; marking as removed in doc store.")
        for sid in ids:
            fid = self._id_map.pop(sid, None)
            if fid is not None:
                self._docs.pop(fid, None)

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()
        self._id_map.clear()
        self._next_id = 0
        self._index.reset()


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------

class QdrantBackend(VectorBackend):
    def __init__(self, host: str, port: int, collection_name: str, dim: int = 384):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install qdrant-client: pip install asft[qdrant]")
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection_name
        self._dim = dim
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        logger.info("Qdrant backend: %s:%d/%s", host, port, collection_name)

    def add(self, docs: List[VectorDocument]) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=d.id, vector=d.embedding, payload={"text": d.text, **d.metadata})
            for d in docs if d.embedding
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_embedding: List[float], top_k: int = 10,
               filter: Optional[Dict] = None) -> List[SearchResult]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=top_k,
        )
        results = []
        for h in hits:
            text = h.payload.get("text", "") if h.payload else ""
            meta = {k: v for k, v in (h.payload or {}).items() if k != "text"}
            results.append(SearchResult(
                doc=VectorDocument(id=str(h.id), text=text, metadata=meta),
                score=h.score,
                distance=1.0 - h.score,
            ))
        return results

    def delete(self, ids: List[str]) -> None:
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=ids),
        )

    def count(self) -> int:
        return self._client.get_collection(self._collection).points_count

    def clear(self) -> None:
        self._client.delete_collection(self._collection)


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

    def encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def encode_one(self, text: str) -> List[float]:
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
        backend: str = "chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        **backend_kwargs,
    ):
        self._embedder = EmbeddingModel(embedding_model, embedding_device)
        dim = self._embedder.dim

        if backend == "chromadb":
            self._backend: VectorBackend = ChromaDBBackend(**backend_kwargs)
        elif backend == "faiss":
            self._backend = FAISSBackend(dim=dim, **backend_kwargs)
        elif backend == "qdrant":
            self._backend = QdrantBackend(dim=dim, **backend_kwargs)
        else:
            raise ValueError(f"Unknown vector backend: {backend!r}. Choose: chromadb, faiss, qdrant")

        logger.info("VectorMemory: backend=%s", backend)

    def add_text(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        embedding = self._embedder.encode_one(text)
        doc = VectorDocument(id=doc_id, text=text, embedding=embedding, metadata=metadata or {})
        self._backend.add([doc])

    def add_texts(self, texts: List[Tuple[str, str]], metadata: Optional[List[Dict]] = None) -> None:
        """Add multiple (id, text) pairs."""
        all_texts = [t for _, t in texts]
        embeddings = self._embedder.encode(all_texts)
        docs = [
            VectorDocument(
                id=doc_id, text=text,
                embedding=emb,
                metadata=(metadata[i] if metadata else {}),
            )
            for i, ((doc_id, text), emb) in enumerate(zip(texts, embeddings))
        ]
        self._backend.add(docs)

    def search(self, query: str, top_k: int = 10,
               filter: Optional[Dict] = None) -> List[SearchResult]:
        query_emb = self._embedder.encode_one(query)
        return self._backend.search(query_emb, top_k=top_k, filter=filter)

    def delete(self, ids: List[str]) -> None:
        self._backend.delete(ids)

    def count(self) -> int:
        return self._backend.count()

    def clear(self) -> None:
        self._backend.clear()

    @classmethod
    def from_config(cls, cfg) -> "VectorMemory":
        """Build VectorMemory from MemoryConfig."""
        kwargs: Dict[str, Any] = {}
        backend = cfg.vector_backend
        if backend == "chromadb":
            kwargs = {
                "persist_dir": cfg.chroma_persist_dir,
                "collection_name": cfg.vector_collection_name
                if hasattr(cfg, "vector_collection_name") else "asft_memory",
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
