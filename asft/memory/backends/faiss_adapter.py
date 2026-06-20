import uuid

import numpy as np

from asft.core.interfaces import IMemoryStore, MemoryQueryResult


class FaissBackend(IMemoryStore):
    """FAISS-based vector memory backend (in-memory local)."""

    def __init__(self, dimension: int = 384):
        import faiss

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        # FAISS only maps int64 to vectors. We need an ID mapping.
        self.id_to_vector_id: dict[str, int] = {}
        self.vector_id_to_data: dict[int, dict] = {}
        self._next_id = 0

    async def add(
        self, content: str, metadata: dict | None = None, vector: list[float] | None = None
    ) -> str:
        if not vector:
            vector = [0.0] * self.dimension

        vec_np = np.array([vector], dtype=np.float32)
        self.index.add(vec_np)

        vec_id = self._next_id
        self._next_id += 1

        point_id = str(uuid.uuid4())
        self.id_to_vector_id[point_id] = vec_id
        self.vector_id_to_data[vec_id] = {
            "id": point_id,
            "content": content,
            "metadata": metadata or {},
        }

        return point_id

    async def update(
        self,
        item_id: str,
        content: str,
        metadata: dict | None = None,
        vector: list[float] | None = None,
    ) -> bool:
        # FAISS Flat index doesn't support easy updates.
        # We delete and re-add.
        await self.delete(item_id)
        # Note: We generate a new UUID internally but we should keep the same ID to respect the interface.
        # Actually, IndexFlatL2 delete is O(N). Let's implement a naive mark-as-deleted.
        if item_id in self.id_to_vector_id:
            vec_id = self.id_to_vector_id[item_id]
            self.vector_id_to_data[vec_id]["content"] = content
            self.vector_id_to_data[vec_id]["metadata"] = metadata or {}
            # If vector changed, we can't easily update IndexFlatL2 without recreating.
            # In production FAISS setups, IndexIVFFlat + remove_ids is used.
            # For simplicity in this fallback adapter, we just update the payload.
            return True
        return False

    async def delete(self, item_id: str) -> bool:
        if item_id in self.id_to_vector_id:
            vec_id = self.id_to_vector_id[item_id]
            del self.id_to_vector_id[item_id]
            del self.vector_id_to_data[vec_id]
            return True
        return False

    async def search(self, query_vector: list[float], top_k: int = 5) -> list[MemoryQueryResult]:
        if self.index.ntotal == 0:
            return []

        vec_np = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(vec_np, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.vector_id_to_data:
                data = self.vector_id_to_data[idx]
                results.append(
                    MemoryQueryResult(
                        source="faiss",
                        content=data["content"],
                        confidence=1.0 / (1.0 + distances[0][i]),  # Convert L2 to roughly 0-1
                        metadata=data["metadata"],
                    )
                )
        return results

    async def batch_insert(
        self,
        contents: list[str],
        metadatas: list[dict] | None = None,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        if not vectors:
            vectors = [[0.0] * self.dimension for _ in contents]

        vec_np = np.array(vectors, dtype=np.float32)
        self.index.add(vec_np)

        ids = []
        for i, content in enumerate(contents):
            vec_id = self._next_id
            self._next_id += 1
            point_id = str(uuid.uuid4())
            self.id_to_vector_id[point_id] = vec_id
            self.vector_id_to_data[vec_id] = {
                "id": point_id,
                "content": content,
                "metadata": metadatas[i] if metadatas else {},
            }
            ids.append(point_id)

        return ids

    async def health_check(self) -> bool:
        return True
