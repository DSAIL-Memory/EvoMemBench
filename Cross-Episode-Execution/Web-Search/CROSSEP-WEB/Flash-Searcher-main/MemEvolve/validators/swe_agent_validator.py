#!/usr/bin/env python
# coding=utf-8

"""
Memory System Auto-Fix with SWE-Agent

Uses mini-swe-agent to automatically fix memory system issues
when simulation testing fails.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


class SWEAgentValidator:
    """Auto-fixes memory system issues using mini-swe-agent"""
    
    def __init__(
        self,
        work_dir: Path,
        isolated_env_dir: Optional[Path] = None,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.isolated_env_dir = isolated_env_dir
        
        self._agent = None
        self._agent_initialized = False
    
    def _init_mini_swe_agent(self):
        """Initialize mini-swe-agent using LitellmModel from environment variables"""
        if self._agent_initialized:
            return self._agent is not None
        
        self._agent_initialized = True
        
        try:
            mini_swe_path = Path(__file__).parent.parent.parent / 'mini-swe-agent' / 'src'
            if str(mini_swe_path) not in sys.path:
                sys.path.insert(0, str(mini_swe_path))
            
            from minisweagent.agents.interactive import InteractiveAgent
            from minisweagent.environments.local import LocalEnvironment
            from minisweagent.models.litellm_model import LitellmModel
            
            # Check API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("[SWE-Agent] OPENAI_API_KEY not set, auto-fix disabled")
                return False
            
            # Get model configuration from environment variables
            model_name = os.getenv("MSWEA_MODEL_NAME", os.getenv("DEFAULT_MODEL", "gpt-4"))
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
            cost_tracking = os.getenv("MSWEA_COST_TRACKING", "ignore_errors")
            
            # Ensure isolated_cwd is an absolute path
            if self.isolated_env_dir:
                isolated_cwd = str(self.isolated_env_dir.resolve())  # Convert to absolute path
            else:
                isolated_cwd = os.getcwd()
            env = LocalEnvironment(cwd=isolated_cwd)
            
            # Create LitellmModel with optional base_url
            model_kwargs = {}
            if base_url:
                model_kwargs = {
                    "custom_llm_provider": "openai",
                    "api_base": base_url
                }
            
            if model_kwargs:
                model = LitellmModel(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    cost_tracking=cost_tracking
                )
            else:
                model = LitellmModel(
                    model_name=model_name,
                    cost_tracking=cost_tracking
                )
            
            step_limit = int(os.getenv("MSWEA_STEP_LIMIT", "30"))
            cost_limit = float(os.getenv("MSWEA_COST_LIMIT", "10.0"))
            
            self._agent = InteractiveAgent(
                model,
                env,
                mode="yolo",
                confirm_exit=False,
                cost_limit=cost_limit,
                step_limit=step_limit
            )
            
            print(f"[SWE-Agent] Initialized with LitellmModel ({model_name}) in directory: {isolated_cwd}")
            return True
            
        except Exception as e:
            print(f"[SWE-Agent] Initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def fix_memory_system(self, memory_system_name: str, error_report: Dict[str, Any]) -> bool:
        """
        Fix memory system using mini-swe-agent based on error report
        
        Args:
            memory_system_name: Name of the memory system to fix
            error_report: Error report from simulation testing
            
        Returns:
            True if fix was applied successfully
        """
        if not self._init_mini_swe_agent():
            print("[SWE-Agent] Cannot initialize agent, skipping fix")
            return False
        
        if not self.isolated_env_dir or not self.isolated_env_dir.exists():
            print(f"[SWE-Agent] Isolated environment not found: {self.isolated_env_dir}")
            return False
        
        provider_file = self.isolated_env_dir / "EvolveLab" / "providers" / f"{memory_system_name}_provider.py"
        if not provider_file.exists():
            print(f"[SWE-Agent] Provider file not found: {provider_file}")
            return False
        
        task = self._build_fix_task(memory_system_name, provider_file, error_report)
        
        try:
            print(f"[SWE-Agent] Executing fix task in isolated environment...")
            print(f"[SWE-Agent] Working directory: {self.isolated_env_dir.resolve()}")
            
            exit_status, result = self._agent.run(task)
            
            # Save trajectory
            self._save_trajectory(memory_system_name, exit_status, result)
            
            if exit_status == "Submitted":
                print(f"[SWE-Agent] Fix completed successfully")
                return True
            else:
                print(f"[SWE-Agent] Fix not completed: {exit_status}")
                return False
                
        except Exception as e:
            print(f"[SWE-Agent] Fix execution failed: {e}")
            traceback.print_exc()
            return False
    
    def _save_trajectory(self, memory_system_name: str, exit_status: str, result: str):
        """Save agent trajectory to file"""
        try:
            from minisweagent.run.utils.save import save_traj
            
            trajectory_file = self.work_dir / f"{memory_system_name}_fix_trajectory.json"
            save_traj(
                agent=self._agent,
                path=trajectory_file,
                exit_status=exit_status,
                result=result,
                print_path=True
            )
        except Exception as e:
            print(f"[SWE-Agent] Failed to save trajectory: {e}")

    def _build_fix_task(self, memory_system_name: str, provider_file: Path, error_report: Dict[str, Any]) -> str:
        """Build fix task description for mini-swe-agent based on error report"""
        
        provider_rel_path = provider_file.relative_to(self.isolated_env_dir) if self.isolated_env_dir else provider_file
        
        errors = error_report.get("errors", [])
        error_logs = error_report.get("error_logs", [])
        tests = error_report.get("tests", {})
        memory_details = error_report.get("memory_details", {})
        captured_logs = error_report.get("captured_logs", [])
        
        error_summary = "\n".join([f"- {err}" for err in errors[:5]])
        error_details = "\n\n".join([f"```\n{log}\n```" for log in error_logs[:3]])
        
        # Extract ERROR logs - include more lines and ensure complete error messages
        error_log_lines = [log for log in captured_logs if '[ERROR]' in log]
        if error_log_lines:
            # For errors containing "Raw:", try to include the full context
            error_logs_str = "\n".join(error_log_lines[:20])  # Increased from 10 to 20
            # If error mentions "Raw:" but seems truncated, try to find continuation
            if "Raw:" in error_logs_str and error_logs_str.count("```") < 2:
                # Look for continuation in subsequent log lines
                for i, log in enumerate(captured_logs):
                    if '[ERROR]' in log and i < len(captured_logs) - 1:
                        # Check if next few lines might be continuation
                        continuation = "\n".join(captured_logs[i+1:i+15])
                        if continuation.strip() and not any('[ERROR]' in l or '[INFO]' in l or '[WARNING]' in l for l in captured_logs[i+1:i+15]):
                            error_logs_str += "\n" + continuation
                            break
        else:
            error_logs_str = "No ERROR logs captured"
        
        # Extract code context from traceback - find all relevant error locations
        code_context = ""
        traceback_text = "\n".join(captured_logs)
        import re
        
        # Find all traceback locations (there might be multiple frames)
        traceback_matches = list(re.finditer(r'File "([^"]+)", line (\d+), in (\w+)', traceback_text))
        
        print(f"[SWE-Agent] Code context extraction: Found {len(traceback_matches)} traceback matches")
        if traceback_matches:
            for i, match in enumerate(traceback_matches):
                print(f"[SWE-Agent] Traceback match {i}: {match.group(1)}:{match.group(2)} in {match.group(3)}")
        
        if traceback_matches and self.isolated_env_dir:
            # Focus on the most relevant traceback (usually the first one in user code)
            # Filter to only show files in the isolated environment (user code, not stdlib)
            relevant_matches = []
            for match in traceback_matches:
                file_path = match.group(1)
                # Skip standard library and third-party paths
                if 'site-packages' not in file_path and '/lib/python' not in file_path:
                    relevant_matches.append(match)
            
            # Use the first relevant match, or fall back to the last match (where error occurred)
            traceback_match = relevant_matches[0] if relevant_matches else traceback_matches[-1] if traceback_matches else None
            
            print(f"[SWE-Agent] Code context: Found {len(relevant_matches)} relevant matches, using: {traceback_match.group(1) if traceback_match else None}")
            
            if traceback_match:
                error_file_path = traceback_match.group(1)
                error_line_num = int(traceback_match.group(2))
                function_name = traceback_match.group(3)
                
                # Try to read the actual file to show code context
                try:
                    # Ensure isolated_env_dir is an absolute path for comparison
                    isolated_env_abs = Path(self.isolated_env_dir).resolve() if self.isolated_env_dir else None
                    
                    # The error_file_path might be absolute or relative to isolated_env_dir
                    original_file_path = error_file_path
                    if not Path(error_file_path).is_absolute():
                        error_file_path = isolated_env_abs / error_file_path if isolated_env_abs else Path(error_file_path)
                    else:
                        error_file_path = Path(error_file_path).resolve()
                    
                    # Check if file is in isolated_env_dir or a subdirectory
                    try:
                        if isolated_env_abs:
                            error_file_path.relative_to(isolated_env_abs)
                    except ValueError:
                        # File is outside isolated_env_dir, skip it
                        print(f"[SWE-Agent] Code context: File {error_file_path} is outside isolated_env_dir {isolated_env_abs}, skipping")
                        pass
                    else:
                        if error_file_path.exists() and error_file_path.is_file():
                            with open(error_file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                            
                            # Show context around error line (7 lines before and after for better context)
                            error_line_idx = error_line_num - 1  # Convert 1-based to 0-based
                            if error_line_idx >= len(lines):
                                print(f"[SWE-Agent] Code context: Error line {error_line_num} is beyond file length {len(lines)}")
                            else:
                                start_line_idx = max(0, error_line_idx - 7)
                                end_line_idx = min(len(lines), error_line_idx + 8)
                                
                                code_context = "\n### Code Context (Error Location Highlighted)\n"
                                code_context += f"**File**: `{error_file_path.name}`  \n"
                                code_context += f"**Line**: {error_line_num}  \n"
                                code_context += f"**Function**: `{function_name}`\n\n"
                                code_context += "```python\n"
                                for i in range(start_line_idx, end_line_idx):
                                    line_num = i + 1
                                    prefix = ">>> " if line_num == error_line_num else "    "
                                    # Preserve existing line content, including trailing newline
                                    line_content = lines[i].rstrip('\n') + '\n'
                                    code_context += f"{prefix}{line_num:4d} | {line_content}"
                                code_context += "```\n\n"
                                code_context += "*Use the code above to understand the exact error location and variable names.*\n"
                                print(f"[SWE-Agent] Code context: Successfully generated context for {error_file_path} line {error_line_num}")
                        else:
                            print(f"[SWE-Agent] Code context: File {error_file_path} does not exist or is not a file")
                except Exception as e:
                    # If we can't read the file, log the error for debugging
                    print(f"[SWE-Agent] Code context: Failed to generate context: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if not traceback_matches:
                print(f"[SWE-Agent] Code context: No traceback matches found in captured_logs")
            if not self.isolated_env_dir:
                print(f"[SWE-Agent] Code context: isolated_env_dir is not set")
        
        failed_tests = []
        for test_name, test_info in tests.items():
            if isinstance(test_info, dict) and test_info.get("status") in ["failed", "warning"]:
                failed_tests.append(f"- {test_name}: {test_info.get('error', 'Unknown error')}")
        
        failed_tests_str = "\n".join(failed_tests) if failed_tests else "No specific test failures"
        
        # Memory operation details
        memory_info = ""
        if memory_details:
            if "provide_memory" in memory_details:
                pm = memory_details["provide_memory"]
                memory_info += f"\n### provide_memory() Details\n"
                memory_info += f"- Returned {pm.get('count', 0)} memories\n"
                if pm.get('memories'):
                    memory_info += f"- First memory preview: {pm['memories'][0].get('content_preview', 'N/A')}\n"
            
            if "take_in_memory" in memory_details:
                tim = memory_details["take_in_memory"]
                memory_info += f"\n### take_in_memory() Details\n"
                memory_info += f"- Success: {tim.get('success')}\n"
                memory_info += f"- Description: {tim.get('description', 'N/A')}\n"
                
                storage_changes = tim.get("storage_changes", {})
                if storage_changes:
                    memory_info += f"- Storage changes: {list(storage_changes.keys())}\n"
                    for filename, change in list(storage_changes.items())[:2]:
                        memory_info += f"  - {filename}: size changed by {change.get('size_change', 0)} bytes\n"
                else:
                    memory_info += f"- ⚠️ WARNING: No storage changes detected (memory might not be persisted)\n"
        
        task = f"""Fix the Python implementation error in `{provider_rel_path}`

## Error
{error_logs_str}
{code_context}
## Failed Tests
{failed_tests_str}

## Requirements
1. **Read and understand the error**: Carefully review the error message and traceback above
2. **Examine the code context**: The Code Context section shows the exact location where the error occurred. Pay attention to:
   - The variable names actually used in the code (use `grep -n` if needed to verify)
   - The surrounding code logic to understand what should happen
   - The error type and message to understand what went wrong
3. **Identify the root cause**: Analyze why the error occurs and what needs to be fixed
4. **Make minimal changes**: Fix only the specific issue without changing working code or logic
5. **Verify the fix**: Run `python -m py_compile {provider_rel_path}` to ensure the code compiles
6. **For JSON parsing errors with markdown code blocks**: If the error shows markdown code block markers (```json ... ```), you need to extract the JSON content first. You can use regex or string manipulation. If you need `re` module, check if it's imported at the top of the file first.

## CRITICAL: Command Format
Your response MUST use this exact format:
```bash
your_command_here
```

Note: Use ```bash (not just ```), otherwise your command will be rejected.

## Important
- Only modify `{provider_rel_path}`
- **PREFER sed for simple line replacements** - it preserves indentation automatically
- Use `sed` for most fixes (especially single-line replacements). Example: `sed -i '' 's/old_code/new_code/' file.py`
- Only use `python -c` or other tools if you need complex multi-line logic that sed cannot handle
- **CRITICAL**: Preserve original indentation exactly. sed does this automatically, but if using other tools, extract and maintain the original line's indentation.
- Make the smallest possible change that correctly fixes the issue
- Do not change the logic or behavior of working code
- For sed on macOS, use `sed -i ''` (note the empty string after -i)
- Example with sed (recommended for most cases):
  ```bash
  sed -i '' 's/json.loads(extracted_raw)/json.loads(extracted_raw.strip().replace("```json", "").replace("```", "").strip())/' file.py
  ```
"""
        return task
