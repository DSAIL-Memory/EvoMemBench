"""
Memory providers for different frameworks.
Each import is guarded so that a missing optional dependency does not prevent
the rest of the providers from loading.
"""

try:
    from .agent_kb_provider import AgentKBProvider
except Exception:
    AgentKBProvider = None  # type: ignore[assignment,misc]

try:
    from .skillweaver_provider import SkillWeaverProvider
except Exception:
    SkillWeaverProvider = None  # type: ignore[assignment,misc]

try:
    from .agent_workflow_memory_provider import AgentWorkflowMemoryProvider
except Exception:
    AgentWorkflowMemoryProvider = None  # type: ignore[assignment,misc]

try:
    from .lightweight_memory_provider import LightweightMemoryProvider
except Exception:
    LightweightMemoryProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "AgentKBProvider",
    "SkillWeaverProvider",
    "AgentWorkflowMemoryProvider",
    "LightweightMemoryProvider",
]
