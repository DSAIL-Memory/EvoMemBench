"""
Configuration for EvolveLab memory system
"""

from typing import Dict, Any
import os
from .memory_types import MemoryType

# Default configuration for different memory providers
STORAGE_BASE_DIR = "./storage"

DEFAULT_CONFIG = {
    "EvolveLab": {
        "default_top_k": 3,
        "active_provider": "agent_kb",  # Default active provider
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
        
        MemoryType.MOBILEE: {
            "tips_file_path": os.path.join(STORAGE_BASE_DIR, "mobilee", "tips", "tips.json"),
            "shortcuts_file_path": os.path.join(STORAGE_BASE_DIR, "mobilee", "shortcuts", "shortcuts.json"),
        },
        
        MemoryType.EXPEL: {
            "insights_file_path": os.path.join(STORAGE_BASE_DIR, "expel", "insights.json"),
            "success_trajectories_file_path": os.path.join(STORAGE_BASE_DIR, "expel", "success_trajectories.json"),
            "top_k": 3,
            "search_weights": {'text': 0.3, 'semantic': 0.7},
            # embedding model id for sentence-transformers (optional override)
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
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
        MemoryType.CEREBRA_FUSION_MEMORY: {
            "model": None,
            "storage_dir": "./storage/cerebra_fusion_memory",
            "db_path": "./storage/cerebra_fusion_memory/cf_database.json",
            "model_cache_dir": "./storage/models",
            "top_k": 3,
            "search_weights": {"text": 0.2, "semantic": 0.8},
            "min_score": 0.22,
            "min_score_in_phase": 0.22,
            "semantic_edge_threshold": 0.75,
            "max_neighbors_expand": 3,
            "enable_graph_expansion": True,
            "enable_tool_memory": True,
            "tools_storage_path": "./storage/cerebra_fusion_memory/tools_storage.py",
            "max_tool_candidates": 3,
            "consolidation_interval": 50,
        },
        MemoryType.DILU: {
            "db_path": "./storage/dilu/dilu_memory.json",      
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": "./storage/models"
        },
        MemoryType.GENERATIVE: {
            "db_path": "./storage/generative/generative_memory.json",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": "./storage/models"
        },
        MemoryType.VOYAGER: {
            "db_path": "./storage/voyager/voyager_memory.json",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": "./storage/models"
        },
        MemoryType.MEMP: {
            "store_path": os.path.join(STORAGE_BASE_DIR, "memp"),
            "records_file": "procedural_records.json",
        },
        MemoryType.DYNAMIC_CHEATSHEET: {
            "store_path": "./storage/dynamic_cheatsheet",
            "records_file": "dynamic_cheatsheet.json",
            "cheatsheet_file": "global_cheatsheet.txt",
            "top_k": 1,
        },
        MemoryType.AGENT_WORKFLOW_MEMORY: {
            "store_path": "./storage/agent_workflow_memory/workflow_memory.json",
            "index_dir": "./storage/agent_workflow_memory/index",
            "top_k": 1,
            "enable_induction": True,
        },
        MemoryType.EVOLVER: {
            "store_path": "./storage/evolver",
            "records_file": "principle_records.json",
            "search_top_k": 1,
            "max_pos_examples": 1,
            "max_neg_examples": 1,
            "prune_threshold": 0.3,
        },
        MemoryType.BM25: {
            "top_k": 10,
            "return_on_in": False,
            "memory_dir": os.path.join(STORAGE_BASE_DIR, "bm25"),
            "chunk_size": 1024,
        },
        MemoryType.QWEN3_EMBEDDING: {
            "top_k": 10,
            "return_on_in": False,
            "memory_dir": os.path.join(STORAGE_BASE_DIR, "qwen3_embedding"),
            "chunk_size": 1024,
            "embed_model": "Qwen/Qwen3-Embedding-4B",
            # api_key / base_url read from QWEN3_EMBED_API_KEY / QWEN3_EMBED_BASE_URL at init time
            "embed_api_key": None,
            "embed_base_url": None,
        },
        MemoryType.GRAPHRAG: {
            "top_k": 10,
            "return_on_in": False,
            "memory_dir": os.path.join(STORAGE_BASE_DIR, "graphrag"),
            "chunk_size": 1024,
            "embed_model": "text-embedding-v4",
            # embed api_key / base_url read from DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL at init time
            "embed_api_key": None,
            "embed_base_url": None,
            # LLM for concept extraction; None → reads BATCH_MODEL env var at init time
            "extract_model": None,
        },
        MemoryType.MEM0: {
            "qdrant_path": os.path.join(STORAGE_BASE_DIR, "mem0", "qdrant"),
            "collection_name": "mem0_xbench",
            "top_k": 3,
            "return_on_in": False,
            "user_id": "benchmark_user",
            # LLM/embed credentials read from os.environ at provider init time
            "llm": {"model": None, "api_key": None, "base_url": None},
            "embed": {"model": "text-embedding-v4", "api_key": None, "base_url": None},
        },
        MemoryType.ACE: {
            "playbook_path": os.path.join(STORAGE_BASE_DIR, "ace", "playbook.txt"),
            "initial_playbook_path": "EvolveLab/providers/ace_assets/prompts/initial_playbook.txt",
            "reflector_prompt_path": "EvolveLab/providers/ace_assets/prompts/reflector_prompt.txt",
            "curator_prompt_path": "EvolveLab/providers/ace_assets/prompts/curator_prompt.txt",
            "use_reflector": True,
            "return_on_in": False,
            # LLM credentials read from os.environ at provider init time
            "llm": {"model": None, "api_key": None, "base_url": None},
        },
        MemoryType.REASONING_BANK: {
            "bank_path": os.path.join(STORAGE_BASE_DIR, "reasoning_bank", "bank.jsonl"),
            "embeddings_path": os.path.join(STORAGE_BASE_DIR, "reasoning_bank", "embeddings.jsonl"),
            "top_k": 1,
            "max_items_per_trajectory": 3,
            "return_on_in": False,
            "use_llm_judge": False,
            "llm": {"model": None, "api_key": None, "base_url": None},
            "embed": {"model": "text-embedding-v4", "api_key": None, "base_url": None},
        },
        MemoryType.A_MEM: {
            "db_path": os.path.join(STORAGE_BASE_DIR, "a_mem", "a_mem_memory.json"),
            "top_k": 3,
            "evo_threshold": 99999,
            "return_on_in": False,
            "llm":   {"api_provider": "openai", "model": None, "api_key": None, "base_url": None},
            "embed": {"model": "text-embedding-v4", "api_key": None, "base_url": None},
        },
        MemoryType.MEMOS: {
            "qdrant_path": os.path.join(STORAGE_BASE_DIR, "memos", "qdrant"),
            "collection_name": "memos_xbench",
            "vector_dimension": 1024,
            "top_k": 3,
            "return_on_in": False,
            "llm":   {"model": None, "api_key": None, "base_url": None},
            "embed": {"model": "text-embedding-v4", "api_key": None, "base_url": None},
        },
        MemoryType.MEMORY_OS: {
            "storage_path": os.path.join(STORAGE_BASE_DIR, "memory_os"),
            "user_id": "benchmark_user",
            "top_k": 3,
            "return_on_in": False,
            "llm_model": None,
            "embedding_model": "text-embedding-v4",
            "short_term_capacity": 10,
        },
        # add new memory type upside this line
}
}


def get_memory_config(provider_type: MemoryType) -> Dict[str, Any]:
    """Get configuration for a specific memory provider"""
    return DEFAULT_CONFIG["providers"].get(provider_type, {}).copy()


def get_evolve_lab_config() -> Dict[str, Any]:
    """Get configuration for the EvolveLab memory system"""
    return DEFAULT_CONFIG["EvolveLab"].copy()


def compute_storage_overrides(provider_type: MemoryType, root: str) -> Dict[str, Any]:
    """Map a single isolation root directory to the config keys each provider actually reads.

    Used by run_pipeline.py's --memory_storage_root flag so that --both-orders can run
    two orderings in parallel with fully isolated per-ordering storage, even for providers
    that don't read 'memory_dir' (bm25/qwen3/graphrag do; these don't).
    """
    p = provider_type
    if p == MemoryType.AGENT_KB:
        return {"kb_database_path": os.path.join(root, "agent_kb_database.json")}
    if p == MemoryType.AGENT_WORKFLOW_MEMORY:
        return {"store_path": os.path.join(root, "workflow_memory.json"),
                "index_dir":  os.path.join(root, "index")}
    if p == MemoryType.SKILLWEAVER:
        return {"skills_dir": root,
                "skills_file_path": os.path.join(root, "skillweaver_generated_skills.py")}
    if p == MemoryType.LIGHTWEIGHT_MEMORY:
        return {"storage_dir": root}
    if p == MemoryType.MEM0:
        return {"qdrant_path": os.path.join(root, "qdrant")}
    if p == MemoryType.ACE:
        return {"playbook_path": os.path.join(root, "playbook.txt")}
    if p == MemoryType.REASONING_BANK:
        return {"bank_path": os.path.join(root, "bank.jsonl"),
                "embeddings_path": os.path.join(root, "embeddings.jsonl")}
    if p == MemoryType.EVOLVER:
        return {"store_path": root}
    if p == MemoryType.A_MEM:
        return {"db_path": os.path.join(root, "a_mem_memory.json")}
    if p == MemoryType.MEMOS:
        return {"qdrant_path": os.path.join(root, "qdrant")}
    if p == MemoryType.MEMORY_OS:
        return {"storage_path": root}
    # bm25 / qwen3_embedding / graphrag and any future provider that reads memory_dir
    return {"memory_dir": root}