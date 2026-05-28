#!/usr/bin/env python
# coding=utf-8

"""
Phase 1: Analysis phase for memory evolution
Analyzes task trajectories to identify memory system issues
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from base_agent import AnalysisAgent
from ..config import ANALYSIS_MAX_STEPS


class PhaseAnalyzer:
    """
    Handles trajectory analysis phase
    
    Analyzes task execution logs to understand current memory system
    performance and identify improvement opportunities
    """
    
    def __init__(self, analysis_model_id: str, task_logs_dir: str, default_provider: Optional[str] = "agent_kb"):
        """
        Initialize analyzer
        
        Args:
            analysis_model_id: Model ID for analysis agent
            task_logs_dir: Directory containing task execution logs
            default_provider: Provider name to use as template reference (required)
        """
        import os
        from FlashOAgents import OpenAIServerModel
        
        if not default_provider or default_provider.strip() == "":
            raise ValueError("default_provider is required and cannot be empty")
        
        self.analysis_model_id = analysis_model_id
        self.base_model = OpenAIServerModel(
            model_id=analysis_model_id,
            api_base=os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE")),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.task_logs_dir = Path(task_logs_dir)
        self.default_provider = default_provider
    
    def run_analysis(self, max_steps: int = ANALYSIS_MAX_STEPS) -> Dict:
        """
        Run trajectory analysis
        
        Args:
            max_steps: Maximum steps for analysis agent
            
        Returns:
            Analysis result with agent output
        """
        print(f"[Analyze] Starting trajectory analysis")
        print(f"  Task logs: {self.task_logs_dir}")
        print(f"  Template provider: {self.default_provider}")
        
        if self.base_model is None:
            raise ValueError("base_model is required for analysis phase")
        
        # Create analysis agent
        analysis_agent = AnalysisAgent(
            model=self.base_model,
            task_logs_dir=str(self.task_logs_dir),
            max_steps=max_steps
        )
        
        # Collect task statistics
        stats = self._collect_task_stats(self.task_logs_dir)
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(stats, self.default_provider, str(self.task_logs_dir))
        
        # Run analysis
        print(f"[Analyze] Running AnalysisAgent...")
        result = analysis_agent(prompt)
        
        print(f"[Analyze] Analysis complete")
        
        return {
            "success": True,
            "agent_result": result,
            "stats": stats
        }
    
    def _collect_task_stats(self, task_logs_dir: Path) -> Dict:
        """
        Collect statistics from task logs with flexible success detection
        
        Args:
            task_logs_dir: Path to task logs directory
            
        Returns:
            Statistics dictionary with task summaries
        """
        stats = {
            "total_tasks": 0,
            "correct_tasks": 0,
            "successful_tasks": 0,
            "task_summaries": []
        }
        
        for task_file in task_logs_dir.glob("*.json"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                task_id = task_file.stem
                
                # Flexible success detection: try multiple field names and formats
                is_correct = self._is_task_correct(data)
                status = data.get("status", "unknown")
                
                stats["total_tasks"] += 1
                if is_correct:
                    stats["correct_tasks"] += 1
                if status == "success":
                    stats["successful_tasks"] += 1
                
                stats["task_summaries"].append({
                    "task_id": task_id,
                    "question": data.get("question", "")[:100],
                    "is_correct": is_correct,
                    "success": status == "success"
                })
            except Exception as e:
                print(f"  Warning: Failed to load {task_file}: {e}")
        
        return stats
    
    def _is_task_correct(self, data: Dict) -> bool:
        """
        Determine if a task is correct using multiple possible field names and formats
        
        Supports:
        - judgement: "correct" / "incorrect"
        - score: 1 / 0
        - is_correct: True / False
        - correct: True / False
        - success: True / False (as fallback)
        
        Args:
            data: Task data dictionary
            
        Returns:
            True if task is correct, False otherwise
        """
        # Check 'judgement' field (string)
        if "judgement" in data:
            judgement = data["judgement"]
            if isinstance(judgement, str):
                return judgement.lower() == "correct"
            elif isinstance(judgement, bool):
                return judgement
        
        # Check 'score' field (numeric: 1 = correct, 0 = wrong)
        if "score" in data:
            score = data["score"]
            if isinstance(score, (int, float)):
                return score > 0
            elif isinstance(score, str):
                try:
                    return float(score) > 0
                except ValueError:
                    pass
        
        # Check 'is_correct' field (boolean)
        if "is_correct" in data:
            is_correct = data["is_correct"]
            if isinstance(is_correct, bool):
                return is_correct
            elif isinstance(is_correct, str):
                return is_correct.lower() in ["true", "1", "yes"]
        
        # Check 'correct' field (boolean)
        if "correct" in data:
            correct = data["correct"]
            if isinstance(correct, bool):
                return correct
            elif isinstance(correct, str):
                return correct.lower() in ["true", "1", "yes"]
        
        # Check 'success' field as last resort (boolean)
        if "success" in data:
            success = data["success"]
            if isinstance(success, bool):
                return success
            elif isinstance(success, str):
                return success.lower() in ["true", "1", "yes"]
        
        # Default: assume incorrect if no recognizable field found
        return False
    
    def _find_memory_database_files(self, provider_name: str) -> list:
        """
        Find all text-based memory database files for the given provider
        
        Handles common naming variations like:
        - provider_name -> storage/provider_name/
        - provider_name_memory -> storage/provider_name/
        - provider_name -> storage/provider_name_memory/
        
        Args:
            provider_name: Name of the memory provider (e.g., "agent_kb", "cerebra_fusion_memory")
            
        Returns:
            List of file paths relative to workspace root
        """
        # Try multiple possible directory names
        possible_dirs = [
            Path("storage") / provider_name,
        ]
        
        if provider_name.endswith("_memory"):
            base_name = provider_name[:-7]
            possible_dirs.append(Path("storage") / base_name)
        else:
            possible_dirs.append(Path("storage") / f"{provider_name}_memory")
        storage_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                storage_dir = dir_path
                break
        
        if storage_dir is None:
            return []
        
        # Find all text-based files (json, jsonl, py, txt, etc.)
        text_extensions = {'.json', '.jsonl', '.py', '.txt', '.md', '.yaml', '.yml', '.toml'}
        memory_files = []
        
        for file_path in storage_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                # Get relative path from workspace root
                try:
                    rel_path = file_path.relative_to(Path.cwd())
                    memory_files.append(str(rel_path))
                except ValueError:
                    # If relative_to fails, use absolute path
                    memory_files.append(str(file_path))
        
        return sorted(memory_files)
    
    def _load_prompt_template(self) -> str:
        """
        Load analysis prompt template from YAML file
        
        Returns:
            Prompt template string
        """
        # Load prompt template
        # prompts directory is at MemEvolve/prompts/, not MemEvolve/phases/prompts/
        prompt_file = Path(__file__).parent.parent / "prompts" / "analysis_prompt.yaml"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get("prompt_template", "")
        except Exception as e:
            print(f"  Error: Failed to load prompt template from {prompt_file}: {e}")
            raise
    
    def _build_analysis_prompt(self, stats: Dict, default_provider: str, task_logs_dir: str) -> str:
        """
        Build prompt for analysis phase focusing on three core memory operations
        
        Args:
            stats: Task statistics
            default_provider: Provider name to analyze
            task_logs_dir: Path to task logs
            
        Returns:
            Analysis prompt string
        """
        # Show overview of all tasks
        overview = "\n".join([
            f"{i+1}. {'✓ CORRECT' if t['is_correct'] else '✗ WRONG'} [{t['task_id']}] {t['question'][:100]}..."
            for i, t in enumerate(stats["task_summaries"])
        ])
        
        # Get provider template for reference
        provider_template = ""
        try:
            # Get the correct module name from PROVIDER_MAPPING
            from EvolveLab.memory_types import MemoryType, PROVIDER_MAPPING
            memory_type = MemoryType(default_provider)
            if memory_type in PROVIDER_MAPPING:
                _, module_name = PROVIDER_MAPPING[memory_type]
                provider_path = Path(f"EvolveLab/providers/{module_name}.py")
            else:
                # Fallback: try with default_provider name directly
                provider_path = Path(f"EvolveLab/providers/{default_provider}_provider.py")
            
            if provider_path.exists():
                with open(provider_path, 'r', encoding='utf-8') as f:
                    provider_template = f"""### Provider Template Reference

Below is an example provider for reference. You should implement your own innovative approach:

```python
{f.read()}
```
"""
            else:
                print(f"  Warning: Provider file not found: {provider_path}")
        except Exception as e:
            print(f"  Warning: Failed to load provider template: {e}")
            pass
        
        # Find memory database files for the current provider
        memory_files = self._find_memory_database_files(default_provider)
        if memory_files:
            memory_files_info = "\n".join([f"  - {f}" for f in memory_files])
        else:
            memory_files_info = f"  (No memory files found in storage/{default_provider}/)"
        
        # Load prompt template from YAML file
        template = self._load_prompt_template()
        if not template:
            raise ValueError(f"Failed to load analysis prompt template from prompts/analysis_prompt.yaml")
        
        # Use template from YAML file
        prompt = template.format(
            default_provider=default_provider,
            memory_files_info=memory_files_info,
            provider_template=provider_template,
            total_tasks=stats['total_tasks'],
            overview=overview
        )
        return prompt
