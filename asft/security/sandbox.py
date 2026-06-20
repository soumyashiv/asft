import logging
import asyncio
import uuid
import time
from typing import Optional

try:
    import docker
except ImportError:
    docker = None

from asft.core.interfaces import ISandbox
from asft.core.settings import get_settings

logger = logging.getLogger(__name__)


class DockerSandbox(ISandbox):
    """
    Secure code execution sandbox using Docker.
    Requires 'docker' python package and Docker daemon running.
    """

    def __init__(self, image: str = "python:3.10-slim", memory_limit: str = "128m"):
        if docker is None:
            raise ImportError("The 'docker' package is required for DockerSandbox.")
        self.image = image
        self.memory_limit = memory_limit
        self.client = docker.from_env()
        self.active_containers = []

    async def execute(self, code: str, timeout: int = 5) -> str:
        """Execute python code inside a disposable Docker container."""
        # Clean up code string
        code = code.replace('"', '\\"')
        command = f'python -c "{code}"'
        
        container_name = f"asft_sandbox_{uuid.uuid4().hex[:8]}"
        
        try:
            # Run container detached
            container = self.client.containers.run(
                self.image,
                command=command,
                name=container_name,
                detach=True,
                network_mode="none",  # No network access
                mem_limit=self.memory_limit,
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],     # Drop all capabilities
            )
            self.active_containers.append(container)
            
            # Wait for execution asynchronously
            start_time = time.time()
            while container.status in ('created', 'running'):
                if time.time() - start_time > timeout:
                    container.kill()
                    return f"ExecutionTimeout: Code ran longer than {timeout} seconds."
                await asyncio.sleep(0.1)
                container.reload()
                
            logs = container.logs().decode('utf-8')
            return logs
        except Exception as e:
            logger.exception("Sandbox execution failed")
            return f"ExecutionError: {str(e)}"
        finally:
            try:
                container.remove(force=True)
                if container in self.active_containers:
                    self.active_containers.remove(container)
            except Exception:
                pass

    async def terminate(self) -> None:
        """Force terminate all active sandbox containers."""
        for container in self.active_containers:
            try:
                container.remove(force=True)
            except Exception as e:
                logger.warning("Failed to remove container %s: %s", container.name, e)
        self.active_containers.clear()

    async def health_check(self) -> bool:
        """Check if Docker daemon is responsive."""
        try:
            return self.client.ping()
        except Exception:
            return False
