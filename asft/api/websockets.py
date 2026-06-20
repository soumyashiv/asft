import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # Map job_id to a list of connected websockets
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info("Client connected to job %s. Total connections: %d", job_id, len(self.active_connections[job_id]))

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
            logger.info("Client disconnected from job %s", job_id)

    async def broadcast(self, job_id: str, message: dict[str, Any]):
        if job_id in self.active_connections:
            websockets = self.active_connections[job_id]
            tasks = [ws.send_json(message) for ws in websockets]
            await asyncio.gather(*tasks, return_exceptions=True)

manager = ConnectionManager()

@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            # We wait for any messages from client, maybe heartbeats
            data = await websocket.receive_text()
            # Can process incoming messages if necessary
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

import redis.asyncio as redis

from asft.core.settings import get_settings


async def redis_listener():
    """Background task to listen to Redis and broadcast to WebSockets."""
    settings = get_settings()
    try:
        r = redis.from_url(settings.celery_broker_url)
        pubsub = r.pubsub()
        await pubsub.subscribe("job_events")
        
        logger.info("Started Redis PubSub listener for WebSockets.")
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    job_id = data.get("job_id")
                    if job_id:
                        await manager.broadcast(job_id, data)
                except Exception as e:
                    logger.error("Error processing redis message: %s", e)
    except Exception as e:
        logger.error("Redis listener failed to start: %s", e)
