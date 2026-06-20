import os
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

class CheckpointManager(TrainerCallback):
    """
    Enterprise-grade Checkpoint Manager for ASFT training jobs.
    Hooks into HuggingFace PEFT/SFTTrainer to backup checkpoints securely.
    """
    def __init__(self, job_id: str):
        self.job_id = job_id
        
        # Determine Storage Priority
        self.s3_bucket = os.getenv("ASFT_S3_CHECKPOINT_BUCKET")
        self.local_volume = os.getenv("ASFT_LOCAL_CHECKPOINT_DIR", "./asft_data/checkpoints")
        self.emergency_dir = "/tmp/asft_checkpoints"
        
        self.active_backend = self._determine_backend()
        self.durability = "temporary"
        if self.active_backend.startswith("s3://"):
            self.durability = "durable"
        elif self.active_backend == self.local_volume:
            self.durability = "semi_durable"
            
        logger.info(f"CheckpointManager initialized for job {job_id} with backend {self.active_backend} ({self.durability})")

    def _determine_backend(self) -> str:
        if self.s3_bucket:
            try:
                import boto3
                s3 = boto3.client('s3')
                # Light health check
                s3.head_bucket(Bucket=self.s3_bucket.replace("s3://", "").split("/")[0])
                return self.s3_bucket
            except Exception as e:
                logger.warning(f"S3 backend unavailable: {e}. Falling back to Priority 2.")
                
        if self.local_volume:
            try:
                os.makedirs(self.local_volume, exist_ok=True)
                return self.local_volume
            except Exception as e:
                logger.warning(f"Local volume unavailable: {e}. Falling back to Emergency Priority 3.")
                
        os.makedirs(self.emergency_dir, exist_ok=True)
        return self.emergency_dir

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Triggered when Trainer saves a checkpoint."""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        source_path = os.path.join(args.output_dir, checkpoint_folder)
        if os.path.exists(source_path):
            self._backup_checkpoint(source_path, checkpoint_folder)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Triggered at end of training to save the final model."""
        if os.path.exists(args.output_dir):
            self._backup_checkpoint(args.output_dir, "final")

    def on_exception(self, args, state: TrainerState, control: TrainerControl, exception=None, **kwargs):
        """Triggered on exception (e.g. OOM, node crash signal). Try to rescue the last weights."""
        logger.error(f"Trainer exception caught: {exception}. Attempting emergency backup.")
        if state.best_model_checkpoint and os.path.exists(state.best_model_checkpoint):
            self._backup_checkpoint(state.best_model_checkpoint, "rescue")

    def _backup_checkpoint(self, source_path: str, checkpoint_name: str):
        try:
            if self.active_backend.startswith("s3://"):
                self._upload_to_s3(source_path, checkpoint_name)
            else:
                self._copy_to_local(source_path, checkpoint_name)
        except Exception as e:
            logger.error(f"Failed to backup checkpoint {checkpoint_name}: {e}")

    def _upload_to_s3(self, source_path: str, checkpoint_name: str):
        import boto3
        s3 = boto3.client('s3')
        bucket = self.s3_bucket.replace("s3://", "").split("/")[0]
        prefix = f"jobs/{self.job_id}/{checkpoint_name}"
        
        for root, dirs, files in os.walk(source_path):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, source_path)
                s3_key = f"{prefix}/{rel_path}"
                s3.upload_file(local_file, bucket, s3_key)
        logger.info(f"Successfully uploaded {checkpoint_name} to S3.")

    def _copy_to_local(self, source_path: str, checkpoint_name: str):
        dest_path = os.path.join(self.active_backend, self.job_id, checkpoint_name)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(source_path, dest_path)
        logger.info(f"Successfully copied {checkpoint_name} to {dest_path}.")

    @classmethod
    def get_latest_checkpoint(cls, job_id: str) -> Optional[str]:
        """Utility to retrieve the latest checkpoint for resumption."""
        s3_bucket = os.getenv("ASFT_S3_CHECKPOINT_BUCKET")
        local_volume = os.getenv("ASFT_LOCAL_CHECKPOINT_DIR", "./asft_data/checkpoints")
        
        # Check Local first for simplicity in this implementation
        job_dir = os.path.join(local_volume, job_id)
        if os.path.exists(job_dir):
            checkpoints = [d for d in os.listdir(job_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                return os.path.join(job_dir, checkpoints[-1])
                
        return None
