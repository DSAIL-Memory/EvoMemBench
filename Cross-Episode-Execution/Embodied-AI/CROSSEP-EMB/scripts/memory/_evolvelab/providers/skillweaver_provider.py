"""
SkillWeaver provider for unified memory system (vendored from Flash-Searcher EvolveLab).

ToolWrapper dependency is replaced with a no-op stub — in the AgentGym ALFWorld
context skills are injected as text guidance, not as callable tools.
"""

import os
import importlib.util
import uuid
import re
import ast
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable

from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest,
    MemoryResponse,
    TrajectoryData,
    MemoryType,
    MemoryItem,
    MemoryItemType,
    MemoryStatus,
)


class _ToolWrapperStub:
    """No-op replacement for Flash-Searcher's storage.tools.tool_wrapper.ToolWrapper."""

    def __init__(self, model=None, logger=None):
        pass

    def wrap_function(self, tool_func, tool_name):
        return None

    def clear_cache(self):
        pass


class SkillWeaverProvider(BaseMemoryProvider):
    """
    SkillWeaver memory provider that manages generated skills.

    On update: generates a reusable Python function from a successful trajectory
    and appends it to a .py file on disk.
    On inject: retrieves relevant skills by keyword and returns their description
    as text guidance injected into the system prompt.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.SKILLWEAVER, config)

        self.skills_file_path = self.config.get(
            "skills_file_path",
            "./storage/skillweaver/skillweaver_generated_skills.py",
        )
        self.skills_dir = self.config.get("skills_dir", "./storage/skillweaver")
        self.model = self.config.get("model")

        self.skills_registry: Dict[str, Callable] = {}
        self.skills_metadata: Dict[str, Dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [SkillWeaver] [%(levelname)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.tool_wrapper = _ToolWrapperStub(model=self.model, logger=self.logger)

    def initialize(self) -> bool:
        try:
            if self.skills_dir:
                os.makedirs(self.skills_dir, exist_ok=True)
            parent_dir = os.path.dirname(self.skills_file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            if os.path.isdir(self.skills_dir):
                self._load_skills_from_dir(self.skills_dir)
            elif os.path.exists(self.skills_file_path):
                self._load_skills_from_file(self.skills_file_path)
            return True
        except Exception as e:
            self.logger.error(f"Error initializing SkillWeaver provider: {e}")
            return False

    def _load_skills_from_file(self, file_path: str):
        try:
            spec = importlib.util.spec_from_file_location("skillweaver_skills", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._populate_registry_from_module(module)
        except Exception as e:
            self.logger.warning(f"Error loading skills from file {file_path}: {e}")

    def _load_skills_from_dir(self, dir_path: str):
        try:
            for filename in os.listdir(dir_path):
                if not filename.endswith(".py") or filename.startswith("__"):
                    continue
                file_path = os.path.join(dir_path, filename)
                try:
                    spec = importlib.util.spec_from_file_location(filename[:-3], file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._populate_registry_from_module(module)
                except Exception as inner_e:
                    self.logger.warning(f"Error loading skills from {file_path}: {inner_e}")
        except Exception as e:
            self.logger.warning(f"Error scanning skills directory {dir_path}: {e}")

    def _populate_registry_from_module(self, module):
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj):
                self.skills_registry[name] = obj
                docstring = getattr(obj, "__doc__", "") or ""
                self.skills_metadata[name] = {
                    "description": (docstring.split("\n")[0] if docstring else name),
                    "full_docstring": docstring,
                    "module": getattr(module, "__name__", "skillweaver_skills"),
                }

    def _reload_skills(self):
        self.skills_registry.clear()
        self.skills_metadata.clear()
        self.tool_wrapper.clear_cache()
        if os.path.isdir(self.skills_dir):
            self._load_skills_from_dir(self.skills_dir)
        elif os.path.exists(self.skills_file_path):
            self._load_skills_from_file(self.skills_file_path)

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        try:
            relevant_skills = []
            query_lower = request.query.lower()

            for skill_name, metadata in self.skills_metadata.items():
                description = metadata.get("description", "").lower()
                docstring = metadata.get("full_docstring", "").lower()

                score = 0.0
                for word in query_lower.split():
                    if word in skill_name.lower():
                        score += 2.0
                    elif word in description:
                        score += 1.5
                    elif word in docstring:
                        score += 1.0

                if score > 0:
                    relevant_skills.append(
                        {"skill_name": skill_name, "metadata": metadata, "score": score}
                    )

            relevant_skills.sort(key=lambda x: x["score"], reverse=True)
            top_skills = relevant_skills[:3]

            memories: List[MemoryItem] = []
            for skill_info in top_skills:
                skill_name = skill_info["skill_name"]
                content = self._format_skill_content(
                    skill_name, skill_info["metadata"], request.status
                )
                if content is None:
                    continue
                memory_item = MemoryItem(
                    id=f"skill_{skill_name}",
                    content=content,
                    metadata={
                        "skill_name": skill_name,
                        "description": skill_info["metadata"].get("description", ""),
                        "score": skill_info["score"],
                        "status": request.status.value,
                    },
                    score=skill_info["score"],
                    type=MemoryItemType.TEXT,
                )
                memories.append(memory_item)

            return MemoryResponse(
                memories=memories,
                memory_type=self.memory_type,
                total_count=len(memories),
                request_id=str(uuid.uuid4()),
            )
        except Exception as e:
            self.logger.error(f"Error providing SkillWeaver memory: {e}")
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

    def _format_skill_content(
        self, skill_name: str, metadata: Dict, status: MemoryStatus
    ) -> Optional[str]:
        if status == MemoryStatus.IN:
            return None  # SkillWeaver only provides memory in BEGIN phase
        desc = metadata.get("description", "")
        full_doc = metadata.get("full_docstring", "").strip()
        if full_doc and full_doc != desc:
            return f"Skill: {skill_name}\n{full_doc}"
        return f"Skill: {skill_name}\n{desc}"

    def _extract_function_from_code(self, code: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ast.parse(code)
            func_defs = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]
            if not func_defs:
                return None
            func = func_defs[0]
            return {"name": func.name, "code": code}
        except Exception:
            return None

    def _is_dangerous_code(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in {
                        "exec", "eval", "compile", "__import__"
                    }:
                        return True
                    if isinstance(node.func, ast.Name) and node.func.id == "open":
                        return True
                if isinstance(node, ast.Attribute):
                    if node.attr in {"system", "popen", "spawn", "remove", "rmdir"}:
                        return True
            return False
        except Exception:
            return True

    def _append_skill_to_file(self, function_name: str, code: str) -> bool:
        try:
            os.makedirs(os.path.dirname(self.skills_file_path) or ".", exist_ok=True)
            existing = ""
            if os.path.exists(self.skills_file_path):
                with open(self.skills_file_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            else:
                existing = (
                    '"""\nSkillWeaver Generated Skills\n'
                    "Auto-generated and continuously updated by SkillWeaverProvider.\n"
                    '"""\n\n'
                )
            if f"def {function_name}(" in existing:
                return True
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_content = existing + f"\n# Generated on {timestamp}\n{code}\n\n"
            with open(self.skills_file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        except Exception as e:
            self.logger.error(f"Error saving generated skill: {e}")
            return False

    def _generate_skill_from_trajectory(
        self, trajectory_data: TrajectoryData
    ) -> Optional[Dict[str, str]]:
        if self.model is None:
            return None
        try:
            import json as _json
            try:
                trajectory_json = _json.dumps(
                    trajectory_data.trajectory, indent=2, ensure_ascii=False
                )
            except Exception:
                trajectory_json = str(trajectory_data.trajectory)

            prompt = f"""You are an expert Python programmer specializing in creating reusable, generic functions. \
Your task is to analyze a successful task execution and extract a GENERAL, PARAMETERIZED skill \
that can be reused for similar problems.

CRITICAL REQUIREMENTS:
- Create a GENERIC function that accepts parameters, NOT a function that returns hardcoded values
- The function must be REUSABLE for different inputs of the same type of problem
- Focus on the METHODOLOGY and APPROACH, not the specific data from this execution
- Make the function PARAMETERIZED so it can handle various inputs

Original Task:
{trajectory_data.query}

Agent's Successful Trajectory:
```json
{trajectory_json}
```

FUNCTION REQUIREMENTS:
1. Write a single, self-contained Python function that is GENERIC and PARAMETERIZED
2. Use descriptive parameter names and include type hints
3. Include comprehensive docstring with Args and Returns sections
4. Add proper error handling and input validation
5. The function should work for DIFFERENT inputs of the same problem type
6. DO NOT hardcode specific values from this execution - make them parameters instead

Output ONLY the Python code for this generic function inside a single markdown code block:"""

            messages = [{"role": "user", "content": prompt}]
            response = self.model(messages)
            content = getattr(response, "content", str(response))
            m = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            code = m.group(1).strip() if m else content.strip()
            if self._is_dangerous_code(code):
                return None
            func_info = self._extract_function_from_code(code)
            if not func_info:
                return None
            return {"name": func_info["name"], "code": code}
        except Exception as e:
            self.logger.error(f"Skill generation error: {e}")
            return None

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        try:
            metadata = trajectory_data.metadata or {}
            is_correct = metadata.get("is_correct", False)
            task_success = metadata.get("task_success", False)

            if not (is_correct and task_success):
                msg = (
                    f"SkillWeaverProvider: skipping skill extraction "
                    f"(is_correct={is_correct}, task_success={task_success})"
                )
                self.logger.info(msg)
                return True, msg

            skill = self._generate_skill_from_trajectory(trajectory_data)
            if not skill:
                msg = "SkillWeaverProvider: generation skipped (no model or validation failed)"
                self.logger.info(msg)
                return True, msg

            saved = self._append_skill_to_file(skill["name"], skill["code"])
            if saved:
                self._reload_skills()
                msg = f"SkillWeaverProvider: saved skill '{skill['name']}'"
                self.logger.info(msg)
                return True, msg
            else:
                return False, f"Failed to save skill: {skill['name']}"
        except Exception as e:
            error_msg = f"Error taking in SkillWeaver memory: {e}"
            self.logger.error(error_msg)
            return False, error_msg
