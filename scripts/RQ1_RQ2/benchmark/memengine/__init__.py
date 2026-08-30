"""Memory engine public API with lazy optional backend imports."""

from importlib import import_module

from .config.Config import MemoryConfig


_MEMORY_CLASSES = {
    "FUMemory": ("FUMemory", "FUMemory"),
    "STMemory": ("STMemory", "STMemory"),
    "LTMemory": ("LTMemory", "LTMemory"),
    "GAMemory": ("GAMemory", "GAMemory"),
    "MGMemory": ("MGMemory", "MGMemory"),
    "RFMemory": ("RFMemory", "RFMemory"),
    "MMMemory": ("MMMemory", "MMMemory"),
    "MMFUMemory": ("MMFUMemory", "MMFUMemory"),
    "NGMemory": ("NGMemory", "NGMemory"),
    "AUGUSTUSMemory": ("AUGUSTUSMemory", "AUGUSTUSMemory"),
    "UniversalRAGMemory": ("UniversalRAGMemory", "UniversalRAGMemory"),
    "AMemMemory": ("AMemMemory", "AMemMemory"),
    "MemoryOSMemory": ("MemoryOSMemory", "MemoryOSMemory"),
}


def __getattr__(name):
    """Load optional memory backends only when a caller selects one."""

    if name not in _MEMORY_CLASSES:
        raise AttributeError(name)
    module_name, class_name = _MEMORY_CLASSES[name]
    value = getattr(import_module(f".memory.{module_name}", __name__), class_name)
    globals()[name] = value
    return value


__all__ = ["MemoryConfig", *_MEMORY_CLASSES]
