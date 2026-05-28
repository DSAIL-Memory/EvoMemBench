"""
Configuration for EvolveLab memory system
"""

from typing import Dict, Any
import os
from .memory_types import MemoryType

STORAGE_BASE_DIR = "./storage"

DEFAULT_CONFIG = {
    "EvolveLab": {
        "default_top_k": 3,
        "active_provider": "agent_kb",
        "storage_base_dir": STORAGE_BASE_DIR,
    },

    "providers": {
        MemoryType.AGENT_KB: {
            "kb_database_path": os.path.join(STORAGE_BASE_DIR, "agent_kb", "agent_kb_database.json"),
            "top_k": 3,
            "search_weights": {'text': 0.5, 'semantic': 0.5},
        },

        MemoryType.SKILLWEAVER: {
            "skills_file_path": os.path.join(STORAGE_BASE_DIR, "skillweaver", "skillweaver_generated_skills.py"),
            "skills_dir": os.path.join(STORAGE_BASE_DIR, "skillweaver"),
        },

        MemoryType.LIGHTWEIGHT_MEMORY: {
            "model": None,
            "storage_dir": "./storage/lightweight_memory",
            "max_strategic_memories": 30,
            "max_operational_memories": 30,
            "max_shortterm_items": 15,
            "shortterm_provision_interval": 5,
            "top_k_longterm": 3,
            "enable_longterm_provision": False,
        },

        MemoryType.AGENT_WORKFLOW_MEMORY: {
            "store_path": "./storage/agent_workflow_memory/workflow_memory.json",
            "top_k": 1,
            "enable_induction": True,
        },
    }
}


def get_memory_config(provider_type: MemoryType) -> Dict[str, Any]:
    return DEFAULT_CONFIG["providers"].get(provider_type, {}).copy()


def get_evolve_lab_config() -> Dict[str, Any]:
    return DEFAULT_CONFIG["EvolveLab"].copy()
