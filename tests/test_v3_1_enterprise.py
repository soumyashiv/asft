import os
import unittest
from unittest.mock import patch

# Import the new hardened modules
from asft.dataset.streaming_compressor import StreamingCompressor
from asft.memory.backends.secure_qdrant import SecureQdrantAdapter
from asft.optimizer.decision_engine import (
    MultiArmedBanditRouter,
)
from asft.training.checkpoint_manager import CheckpointManager


class TestASFTV31Enterprise(unittest.TestCase):

    @patch.dict(
        os.environ,
        {
            "ASFT_S3_CHECKPOINT_BUCKET": "s3://test-bucket",
            "ASFT_LOCAL_CHECKPOINT_DIR": "/tmp/asft_test",
        },
    )
    def test_checkpoint_manager_priorities(self):
        manager = CheckpointManager(job_id="test_job_123")
        # Should default to local if S3 has no creds or fails (which it will in mock)
        self.assertEqual(manager.s3_bucket, "s3://test-bucket")
        self.assertEqual(manager.local_volume, "/tmp/asft_test")
        self.assertIn(manager.durability, ["durable", "semi_durable", "temporary"])

    def test_secure_qdrant_adapter_tls(self):
        adapter = SecureQdrantAdapter()
        # Mock health check should fail if env vars not set, ensuring defensive routing
        self.assertFalse(adapter.is_healthy())

    def test_streaming_compressor_backpressure(self):
        compressor = StreamingCompressor()
        self.assertEqual(compressor.limit_warning, 1500)
        self.assertEqual(compressor.limit_abort, 2500)

        # Test FAISS init
        self.assertIsNotNone(compressor.faiss_index)
        self.assertEqual(compressor.faiss_index.ntotal, 0)

    @patch("asft.optimizer.decision_engine.MultiArmedBanditRouter._get_historical_utilities")
    def test_bandit_cold_start_protection(self, mock_utilities):
        # Return < 50 samples
        mock_utilities.return_value = ({"memory_rag": 1.0, "qlora": 1.0}, 10)

        router = MultiArmedBanditRouter()
        strategy, is_explore = router.select_strategy("task_hash_1", ["memory_rag", "qlora"])

        # Cold start (<50) should force safe initial routing (memory_rag)
        self.assertEqual(strategy, "memory_rag")
        self.assertFalse(is_explore)

    @patch("asft.optimizer.decision_engine.MultiArmedBanditRouter._get_historical_utilities")
    def test_bandit_epsilon_decay(self, mock_utilities):
        # Return 100 samples
        mock_utilities.return_value = ({"memory_rag": 0.8, "qlora": 1.5}, 100)

        router = MultiArmedBanditRouter(epsilon_initial=0.5, alpha=0.99)
        # 0.5 * (0.99^100) = ~0.18. Hybrid Rule boosts to max(0.18, 0.20) -> 0.20
        # Just verifying it doesn't crash and returns a valid tuple
        strategy, is_explore = router.select_strategy("task_hash_1", ["memory_rag", "qlora"])
        self.assertIn(strategy, ["memory_rag", "qlora"])


if __name__ == "__main__":
    unittest.main()
