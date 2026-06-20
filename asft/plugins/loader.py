import importlib
import inspect
import logging
import pkgutil

from asft.core.interfaces import IMemoryStore, ISkillPack, ITrainer
from asft.core.registry import Registry

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Dynamically loads ASFT plugins from entry points or designated directories.
    Registers compliant classes into the core Registry.
    """

    def __init__(self, registry: Registry):
        self.registry = registry

    def load_from_package(self, package_name: str) -> None:
        """
        Scan a package for ASFT interfaces and register them.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.warning("Failed to import plugin package %s: %s", package_name, e)
            return

        prefix = package.__name__ + "."
        for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
            if not ispkg:
                self._load_module(modname)

    def _load_module(self, modname: str) -> None:
        try:
            module = importlib.import_module(modname)
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    self._register_if_compliant(obj)
        except Exception as e:
            logger.error("Error loading module %s: %s", modname, e)

    def _register_if_compliant(self, cls: type) -> None:
        try:
            if issubclass(cls, ISkillPack) and cls is not ISkillPack:
                instance = cls()
                self.registry.register("skill", instance.name, instance)
                logger.info("Registered skill plugin: %s", instance.name)

            elif issubclass(cls, IMemoryStore) and cls is not IMemoryStore:
                # Memory stores usually need initialization args, so we might just register the class
                # But registry pattern in ASFT usually holds instances. We'll register class for now.
                self.registry.register("memory_backend", cls.__name__.lower(), cls)
                logger.info("Registered memory plugin: %s", cls.__name__)

            elif issubclass(cls, ITrainer) and cls is not ITrainer:
                instance = cls()
                # Trainers usually don't have a strict name property in the interface,
                # but we can use class name
                self.registry.register("trainer", cls.__name__.lower(), instance)
                logger.info("Registered trainer plugin: %s", cls.__name__)

        except TypeError:
            pass  # Abstract class or similar
        except Exception as e:
            logger.error("Failed to register plugin class %s: %s", cls.__name__, e)
