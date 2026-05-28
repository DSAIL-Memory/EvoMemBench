#!/usr/bin/env python
# coding=utf-8

"""
Phase 2: Generation phase for memory evolution
Generates new memory system configurations based on analysis
"""

import json
import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

from ..config import (
    CREATIVITY_INDEX,
    CREATIVITY_TEMP_BASE,
    CREATIVITY_TEMP_SPAN,
)


class PhaseGenerator:
    """
    Handles memory system generation phase
    
    Uses LLM to generate new memory provider configurations
    based on analysis findings
    """
    
    def __init__(self, openai_client: OpenAI, model_id: str, work_dir: Path, default_provider: Optional[str] = "agent_kb"):
        """
        Initialize generator
        
        Args:
            openai_client: OpenAI client for generation
            model_id: Model ID to use
            work_dir: Working directory for outputs
            default_provider: Default provider name for template reference (required)
        """
        if not default_provider or default_provider.strip() == "":
            raise ValueError("default_provider is required and cannot be empty")
        
        self.openai_client = openai_client
        self.model_id = model_id
        self.work_dir = work_dir
        self.default_provider = default_provider
    
    def run_generation(self, analysis_data: Dict, creativity_index: float = CREATIVITY_INDEX) -> Dict:
        """
        Generate a single memory system configuration
        
        Args:
            analysis_data: Analysis result from phase 1
            creativity_index: Innovation level (0-1). 0=conservative, 1=highly creative
            
        Returns:
            Generation result with single config
        """
        # Validate creativity_index
        creativity_index = max(0.0, min(1.0, creativity_index))
        
        print(f"[Generate] Generating memory system")
        print(f"  Creativity index: {creativity_index:.2f} ({'conservative' if creativity_index < 0.3 else 'moderate' if creativity_index < 0.7 else 'highly creative'})")
        
        # Extract analysis text from agent result
        # Only extract the actual analysis report, not the entire trajectory
        agent_result = analysis_data.get("agent_result", "")
        if isinstance(agent_result, dict):
            # If agent_result is a dict, extract only the analysis report
            analysis_text = agent_result.get("agent_result", "")
        else:
            # If agent_result is already a string, use it directly
            analysis_text = agent_result
        
        # Get type system reference
        memory_types_ref = self._get_memory_types_reference()
        
        # Generate single system
        prompt = self._build_generation_prompt(
            analysis_text, memory_types_ref, creativity_index
        )
        
        config = self._generate_single_system(prompt, creativity_index)
        if not config:
            print(f"  Warning: Failed to generate system")
            return {
                "success": False,
                "config": None
            }
        
        print(f"[Generate] Complete. Generated 1 system")
        
        return {
            "success": True,
            "config": config
        }
    
    def _get_memory_types_reference(self) -> str:
        """
        Get memory types definition for reference
        
        Returns:
            Content of memory_types.py file
        """
        try:
            # Calculate absolute path to EvolveLab directory
            # phase_generator.py is in MemEvolve/phases/, so go up 2 levels to reach Flash-Searcher-main/
            types_file = Path(__file__).parent.parent.parent / "EvolveLab" / "memory_types.py"
            if types_file.exists():
                with open(types_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"  [OK] Loaded memory types reference from: {types_file}")
                    return content
            else:
                print(f"  Warning: Memory types file not found: {types_file}")
                return ""
        except Exception as e:
            print(f"  Warning: Could not load memory_types.py: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _load_prompt_template(self) -> str:
        """
        Load generation prompt template from YAML file
        
        Returns:
            Prompt template string
        """
        # Load prompt template
        # prompts directory is at MemEvolve/prompts/, not MemEvolve/phases/prompts/
        prompt_file = Path(__file__).parent.parent / "prompts" / "generation_prompt.yaml"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get("prompt_template", "")
        except Exception as e:
            print(f"  Error: Failed to load prompt template from {prompt_file}: {e}")
            raise
    
    def _extract_existing_systems(self) -> list:
        """Extract all existing memory system names from MemoryType enum"""
        try:
            from EvolveLab.memory_types import MemoryType
            return [mem_type.value for mem_type in MemoryType]
        except Exception as e:
            print(f"  Warning: Could not extract existing systems: {e}")
            return []
    
    def _build_generation_prompt(self, analysis_text: str, memory_types_ref: str, 
                                 creativity_index: float) -> str:
        """
        Build prompt for generating a single system focused on three core operations
        
        Args:
            analysis_text: Analysis result text
            memory_types_ref: Memory types reference code
            creativity_index: Creativity level (0-1)
            
        Returns:
            Generation prompt string
        """
        # Extract existing system names
        existing_systems = self._extract_existing_systems()
        existing_systems_section = ""
        if existing_systems:
            existing_systems_section = f"""
### CRITICAL: Existing System Names (DO NOT USE THESE)
The following system names ALREADY EXIST. You MUST create a completely different, unique name:
{', '.join(existing_systems)}

Your new system name must be different from ALL of the above names.
"""
        
        # Format analysis section
        if analysis_text.strip():
            analysis_section = f"""### Analysis Insights

{analysis_text}"""
        else:
            analysis_section = ""
        
        # Get provider template for reference
        provider_template = ""
        try:
            # Get the correct module name from PROVIDER_MAPPING
            from EvolveLab.memory_types import MemoryType, PROVIDER_MAPPING
            memory_type = MemoryType(self.default_provider)
            if memory_type in PROVIDER_MAPPING:
                _, module_name = PROVIDER_MAPPING[memory_type]
                provider_path = f"EvolveLab/providers/{module_name}.py"
            else:
                # Fallback: try with default_provider name directly
                provider_path = f"EvolveLab/providers/{self.default_provider}_provider.py"
            
            if os.path.exists(provider_path):
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
        
        # Format memory types definition
        memory_types_definition = f"""```python
{memory_types_ref}
```"""
        
        # Load prompt template from YAML file
        template = self._load_prompt_template()
        if not template:
            raise ValueError(f"Failed to load generation prompt template from prompts/generation_prompt.yaml")
        
        # Use template from YAML file
        prompt = template.format(
            default_provider=self.default_provider,
            provider_template=provider_template,
            analysis_section=analysis_section,
            memory_types_definition=memory_types_definition
        )
        
        # Insert existing systems warning at the beginning of the prompt
        if existing_systems_section:
            prompt = existing_systems_section + "\n" + prompt
        
        return prompt
    
    def _generate_single_system(self, prompt: str, 
                               creativity_index: float) -> Optional[Dict]:
        """
        Generate a single memory system using LLM with controlled creativity
        
        Args:
            prompt: Generation prompt
            creativity_index: Creativity level (mapped to temperature)
            
        Returns:
            System configuration dict or None if failed
        """
        try:
            temperature = CREATIVITY_TEMP_BASE + (creativity_index * CREATIVITY_TEMP_SPAN)
            
            # Print the prompt being sent to LLM
            print(f"\n{'='*80}")
            print(f"[LLM Prompt] (Temperature: {temperature:.2f})")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
            
            response = self.openai_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60000,
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            
            # Always save the raw LLM output for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_output_path = self.work_dir / f"llm_raw_output_{timestamp}.txt"
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Temperature: {temperature:.2f}\n")
                f.write(f"Model: {self.model_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"{'='*80}\n\n")
                f.write(response_text)
            print(f"  [Debug] Raw LLM output saved to: {raw_output_path}")
            
            # Parse the response
            config = self._parse_system_config(response_text)
            
            if config:
                # Add the raw output to the config for reference
                config["_raw_llm_output"] = response_text
                config["_generation_metadata"] = {
                    "temperature": temperature,
                    "model_id": self.model_id,
                    "timestamp": timestamp
                }
                print(f"  Successfully parsed system (temperature={temperature:.2f})")
                return config
            else:
                print(f"  Failed to parse system")
                print(f"  Raw response saved to: {raw_output_path}")
                return None
                
        except Exception as e:
            print(f"  Error generating system: {e}")
            return None
    
    def _parse_system_config(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response into system configuration with robust format handling
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed configuration dict or None if parsing failed
        """
        try:
            # Extract class name and module name
            class_match = re.search(r'\*\*Class Name\*\*:\s*(\w+)', response_text)
            module_match = re.search(r'\*\*Module Name\*\*:\s*(\w+)', response_text)
            
            if not class_match or not module_match:
                print(f"  Parse error: Could not find Class Name or Module Name")
                return None
            
            class_name = class_match.group(1)
            module_name = module_match.group(1)
            
            # Extract Python code
            code_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
            if not code_match:
                print(f"  Parse error: Could not find Python code block")
                return None
            
            code = code_match.group(1).strip()
            
            # Extract enum info
            enum_name_match = re.search(r'\*\*Enum Name\*\*:\s*(\w+)', response_text)
            enum_value_match = re.search(r'\*\*Enum Value\*\*:\s*(\w+)', response_text)
            
            if not enum_name_match or not enum_value_match:
                print(f"  Parse error: Could not find Enum Name or Enum Value")
                return None
            
            enum_name = enum_name_match.group(1)
            enum_value = enum_value_match.group(1)
            
            # Extract configuration (everything with **key**: value pattern)
            # Keep values as-is, memory_creator.py will handle cleaning and type conversion
            config_updates = {}
            config_pattern = r'\*\*([^*]+)\*\*:\s*([^\n]+)'
            for match in re.finditer(config_pattern, response_text):
                key, value = match.groups()
                key = key.strip()
                value = value.strip()
                
                # Skip metadata fields
                if key in ['Class Name', 'Module Name', 'Enum Name', 'Enum Value']:
                    continue
                
                # Skip empty values
                if not value:
                    continue
                
                # Store key and value as-is
                # memory_creator.py will handle normalization, cleaning, and type conversion
                config_updates[key] = value
            
            print(f"  Successfully parsed: {len(config_updates)} config parameters")
            
            return {
                "provider_code": {
                    "class_name": class_name,
                    "module_name": module_name,
                    "code": code
                },
                "config_updates": config_updates,
                "memory_type_info": {
                    "enum_name": enum_name,
                    "enum_value": enum_value
                }
            }
            
        except Exception as e:
            print(f"  Parse error: {e}")
            import traceback
            traceback.print_exc()
            return None
