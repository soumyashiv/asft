from typing import Dict, List, Optional
import uuid

from asft.core.interfaces import IMemoryStore, MemoryQueryResult

class QdrantBackend(IMemoryStore):
    """Qdrant-based vector memory backend."""

    def __init__(self, collection_name: str = "asft_memory", location: str = ":memory:"):
        import qdrant_client
        self.client = qdrant_client.QdrantClient(location=location)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.http.models import Distance, VectorParams
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Default to 384 dimensions for all-MiniLM-L6-v2
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    async def add(self, content: str, metadata: Optional[Dict] = None, vector: Optional[List[float]] = None) -> str:
        """Add a single item. Note: Qdrant requires vectors, we assume they are passed or computed."""
        from qdrant_client.http.models import PointStruct
        point_id = str(uuid.uuid4())
        # In a real scenario, the embedding is computed by the MemoryManager before calling backend
        if not vector:
            vector = [0.0] * 384
            
        point = PointStruct(id=point_id, vector=vector, payload={"content": content, **(metadata or {})})
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return point_id

    async def update(self, item_id: str, content: str, metadata: Optional[Dict] = None, vector: Optional[List[float]] = None) -> bool:
        """Update an item."""
        from qdrant_client.http.models import PointStruct
        if not vector:
            vector = [0.0] * 384
        point = PointStruct(id=item_id, vector=vector, payload={"content": content, **(metadata or {})})
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return True

    async def delete(self, item_id: str) -> bool:
        """Delete an item."""
        self.client.delete(collection_name=self.collection_name, points_selector=[item_id])
        return True

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[MemoryQueryResult]:
        """Search similar items."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [
            MemoryQueryResult(
                source="qdrant",
                content=r.payload.get("content", ""),
                confidence=r.score,
                metadata={k: v for k, v in r.payload.items() if k != "content"}
            )
            for r in results
        ]

    async def batch_insert(self, contents: List[str], metadatas: Optional[List[Dict]] = None, vectors: Optional[List[List[float]]] = None) -> List[str]:
        """Insert multiple items."""
        from qdrant_client.http.models import PointStruct
        points = []
        ids = []
        for i, content in enumerate(contents):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            meta = metadatas[i] if metadatas else {}
            vec = vectors[i] if vectors else [0.0] * 384
            points.append(PointStruct(id=point_id, vector=vec, payload={"content": content, **meta}))
            
        self.client.upsert(collection_name=self.collection_name, points=points)
        return ids

    async def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
