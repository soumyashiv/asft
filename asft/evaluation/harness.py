import logging
from typing import Any

try:
    import lm_eval
except ImportError:
    lm_eval = None

from asft.core.interfaces import IEvaluationHarness

logger = logging.getLogger(__name__)

class LmEvalHarnessAdapter(IEvaluationHarness):
    """
    Adapter for EleutherAI's lm-evaluation-harness.
    """
    
    def __init__(self):
        if lm_eval is None:
            raise ImportError("lm_eval package is required. Install with: pip install lm-eval")

    async def evaluate(self, model_path: str, tasks: list[str]) -> dict[str, Any]:
        """
        Run lm-evaluation-harness tasks on the specified model.
        Note: This is an async wrapper around a blocking synchronous operation.
        For production, this should ideally be dispatched to Celery or an executor.
        """
        logger.info(f"Starting lm-eval on model {model_path} for tasks: {tasks}")
        
        try:
            # We use simple evaluation call. Real integration might need model_args, batch_size, etc.
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_path}",
                tasks=tasks,
                batch_size="auto",
                limit=None,
            )
            
            # Format results
            summary = {}
            if "results" in results:
                for task_name, metrics in results["results"].items():
                    summary[task_name] = metrics
                    
            logger.info("Evaluation completed successfully.")
            return summary
        except Exception as e:
            logger.exception("Evaluation failed")
            raise RuntimeError(f"lm-eval execution failed: {e}")
