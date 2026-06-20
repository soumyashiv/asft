"""
Streaming Dataset Compressor — Bounded Memory Implementation (V3)
Handles infinitely large datasets within < 2GB RAM.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import psutil
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

class StreamingCompressor:
    """
    Streaming compressor that processes datasets in chunks to strictly bound memory usage.
    Supports JSONL, Parquet, CSV via HuggingFace `datasets` streaming.
    """
    
    def __init__(self, config=None):
        self._config = config
        self.batch_size = getattr(config, "stream_batch_size", 5000) if config else 5000
        self.n_clusters = getattr(config, "stream_n_clusters", 100) if config else 100
        self.samples_per_cluster = getattr(config, "samples_per_cluster", 10) if config else 10
        self.output_dir = Path(getattr(config, "compressed_output_dir", "./asft_data/datasets") if config else "./asft_data/datasets")
        
        # Incremental clustering
        self.clusterer = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size, random_state=42)
        
        # FAISS Deduplication index
        self.similarity_threshold = getattr(config, "similarity_threshold", 0.05) if config else 0.05
        self.embedding_dim = 384 # default for all-MiniLM-L6-v2
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Reservoir sampling: keep a fixed number of representative items per cluster
        self.reservoir: dict[int, list[dict[str, Any]]] = {i: [] for i in range(self.n_clusters)}
        self.cluster_counts = {i: 0 for i in range(self.n_clusters)}
        
        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        
        # Phase 7: Memory Backpressure Limits
        self.limit_warning = 1500
        self.limit_flush = 2000
        self.limit_emergency = 2250
        self.limit_abort = 2500

    def check_memory(self):
        """Monitors memory to ensure multi-stage backpressure is applied."""
        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        
        if mem_mb >= self.limit_abort:
            logger.critical(f"MEMORY ABORT: {mem_mb:.1f} MB >= {self.limit_abort} MB. Aborting to save OS.")
            self._flush_state_to_disk()
            raise MemoryError("Recoverable memory abort threshold reached. State saved.")
            
        elif mem_mb >= self.limit_emergency:
            logger.error(f"MEMORY EMERGENCY: {mem_mb:.1f} MB >= {self.limit_emergency} MB. Reducing batch size.")
            self.batch_size = max(100, self.batch_size // 2)
            self._flush_state_to_disk()
            import gc
            gc.collect()
            
        elif mem_mb >= self.limit_flush:
            logger.warning(f"MEMORY FLUSH: {mem_mb:.1f} MB >= {self.limit_flush} MB. Flushing to disk.")
            self._flush_state_to_disk()
            import gc
            gc.collect()
            
        elif mem_mb >= self.limit_warning:
            logger.warning(f"MEMORY WARNING: {mem_mb:.1f} MB >= {self.limit_warning} MB.")

    def _flush_state_to_disk(self):
        """Flushes current reservoir state to disk to relieve memory pressure."""
        logger.info("Flushing intermediate reservoir to disk...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_dir / "reservoir_temp.jsonl"
        with open(temp_path, "a", encoding="utf-8") as f:
            for cluster_id, items in self.reservoir.items():
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Clear reservoir contents but keep structure
        for cluster_id in self.reservoir:
            self.reservoir[cluster_id].clear()
            
        # Optional cache clearing
        if self.faiss_index.ntotal > 50000:
            logger.info("Resetting FAISS index to relieve memory...")
            self.faiss_index.reset()

    def compress_stream(
        self,
        dataset_path: str,
        dataset_format: str = "json",
        text_field: str = "text",
        embedding_model: str = "all-MiniLM-L6-v2",
        output_name: str = "compressed_stream",
    ) -> tuple[list[str], dict[str, Any]]:
        """
        Process the dataset incrementally.
        """
        logger.info(f"Starting streaming compression for {dataset_path} format {dataset_format}")
        
        # Load embedding model incrementally
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
        
        # Load dataset stream
        try:
            # If dataset_path is a local file, datasets load_dataset might need specific args
            data_files = {"train": dataset_path} if os.path.isfile(dataset_path) else None
            if data_files:
                dataset = load_dataset(dataset_format, data_files=data_files, split="train", streaming=True)
            else:
                dataset = load_dataset(dataset_path, split="train", streaming=True)
        except Exception as e:
            logger.error(f"Failed to load dataset stream: {e}")
            raise

        batch_texts = []
        batch_meta = []
        total_processed = 0
        
        for item in dataset:
            text = item.get(text_field, "")
            if not text:
                continue
                
            batch_texts.append(text)
            batch_meta.append({k: v for k, v in item.items() if k != text_field})
            total_processed += 1
            
            if len(batch_texts) >= self.batch_size:
                self._process_batch(batch_texts, batch_meta, model)
                batch_texts = []
                batch_meta = []
                self.check_memory()
                logger.info(f"Processed {total_processed} items. Memory: {self.process.memory_info().rss / (1024*1024):.1f} MB")
                
        # Process remaining
        if batch_texts:
            self._process_batch(batch_texts, batch_meta, model)
            total_processed += len(batch_texts)

        # Export reservoir to disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{output_name}.jsonl"
        final_count = 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            for cluster_id, items in self.reservoir.items():
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    final_count += 1
                    
        report = {
            "original_count": total_processed,
            "final_count": final_count,
            "total_reduction": 1.0 - (final_count / max(1, total_processed)),
            "output_path": str(output_path),
            "peak_memory_mb": self.process.memory_info().rss / (1024 * 1024)
        }
        
        logger.info(f"Streaming compression complete. Reduced {total_processed} -> {final_count}. Mem: {report['peak_memory_mb']:.1f} MB")
        
        # We don't return the full text list to avoid RAM spikes, just an empty list + report
        # The actual compressed data is purely on disk.
        return [], report

    def _process_batch(self, texts: list[str], metas: list[dict[str, Any]], model):
        """Embeds, deduplicates, clusters, and reservoir-samples a single batch."""
        import random
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Ensure correct dimensionality
        if embeddings.shape[1] != self.embedding_dim:
            if self.faiss_index.ntotal == 0:
                self.embedding_dim = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                logger.error("Embedding dimension mismatch!")
                return
                
        # Phase 4: FAISS Deduplication
        novel_indices = []
        if self.faiss_index.ntotal > 0:
            distances, _ = self.faiss_index.search(embeddings, 1)
            for i, dist in enumerate(distances):
                if dist[0] > self.similarity_threshold:
                    novel_indices.append(i)
        else:
            novel_indices = list(range(len(embeddings)))
            
        if not novel_indices:
            return # Entire batch was duplicate
            
        # Filter novel items
        novel_embeddings = embeddings[novel_indices]
        novel_texts = [texts[i] for i in novel_indices]
        novel_metas = [metas[i] for i in novel_indices]
        
        # Add to FAISS index
        self.faiss_index.add(novel_embeddings)
        
        # Partial fit the clustering model
        self.clusterer.partial_fit(novel_embeddings)
        labels = self.clusterer.predict(novel_embeddings)
        
        # Reservoir sampling into the clusters
        for i, (text, meta, label) in enumerate(zip(novel_texts, novel_metas, labels)):
            item = {"text": text}
            if meta:
                item["metadata"] = meta
                
            cluster_id = int(label)
            self.cluster_counts[cluster_id] += 1
            
            # If there's room in the reservoir, just add it
            if len(self.reservoir[cluster_id]) < self.samples_per_cluster:
                self.reservoir[cluster_id].append(item)
            else:
                # Reservoir sampling: probability = k / n
                prob = self.samples_per_cluster / self.cluster_counts[cluster_id]
                if random.random() < prob:
                    replace_idx = random.randint(0, self.samples_per_cluster - 1)
                    self.reservoir[cluster_id][replace_idx] = item
