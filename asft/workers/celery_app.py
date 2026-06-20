import logging

from celery import Celery

from asft.core.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery(
    "asft_workers",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["asft.workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 24, # 24 hour max limit for training
    worker_prefetch_multiplier=1, # Don't prefetch long-running training jobs
    task_acks_late=True, # Acknowledge task after it has finished
)

logger.info("Celery App initialized with broker: %s", settings.celery_broker_url)
