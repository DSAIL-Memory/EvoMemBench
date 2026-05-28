#!/usr/bin/env python3
"""
Memory System Creator

Tool class for safely creating new memory systems
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple


class MemorySystemCreator:
    """Tool class for safely creating new memory systems"""

    @staticmethod
    def _normalize_config_key(key: str) -> str:
        """
        Normalize configuration key to a valid Python dictionary key format.
        Replaces special characters (colons, commas, spaces) with underscores.
        
        Args:
            key: Original key name (may contain special characters)
            
        Returns:
            Normalized key name safe for Python dictionaries
        """
        # Replace colons and commas with underscores
        normalized = key.replace(':', '_').replace(',', '_')
        # Replace spaces with underscores
        normalized = normalized.replace(' ', '_')
        # Remove consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        return normalized

    @staticmethod
    def _validate_config_updates(config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration updates before saving."""
        import ast
        
        errors, warnings = [], []
        
        for key, value in config_updates.items():
            try:
                normalized_key = MemorySystemCreator._normalize_config_key(key)
                if normalized_key != key:
                    warnings.append(f"Key normalized: '{key}' -> '{normalized_key}'")
                
                _, python_repr = MemorySystemCreator._parse_config_value(value)
                ast.literal_eval(python_repr)
            except Exception as e:
                errors.append(f"Key '{key}': {str(e)}")
        
        return {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    @staticmethod
    def _parse_config_value(value: Any) -> Tuple[Any, str]:
        """Parse configuration value to Python literal."""
        import ast
        
        if not isinstance(value, str):
            return value, repr(value)
        
        # Clean LLM output: remove backticks and parenthetical notes
        value = value.replace('`', '')  # Remove all backticks
        value = re.sub(r'\s*\([^)]*\)', '', value).strip()  # Remove parenthetical notes
        
        if not value:
            return "", '""'
        
        # Handle boolean strings (case-insensitive)
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true', value.capitalize()
        
        # Try ast.literal_eval for numbers, None, dicts, lists, etc.
        try:
            parsed = ast.literal_eval(value)
            return parsed, repr(parsed)
        except (ValueError, SyntaxError):
            return value, repr(value)

    @staticmethod
    def create_memory_system(memory_system_config: Dict[str, Any], base_dir: str = ".") -> Dict[str, Any]:
        """Create new memory system"""
        try:
            # Validate configuration format
            required_keys = ["provider_code", "config_updates", "memory_type_info"]
            for key in required_keys:
                if key not in memory_system_config:
                    return {"success": False, "error": f"Missing required key: {key}"}

            provider_info = memory_system_config["provider_code"]
            config_updates = memory_system_config["config_updates"]
            memory_type_info = memory_system_config["memory_type_info"]

            # Validate provider_code format
            if not all(k in provider_info for k in ["class_name", "module_name", "code"]):
                return {"success": False, "error": "provider_code must contain class_name, module_name, and code"}

            # Validate memory_type_info format
            if not all(k in memory_type_info for k in ["enum_name", "enum_value"]):
                return {"success": False, "error": "memory_type_info must contain enum_name and enum_value"}

            results = {}
            base_path = Path(base_dir)

            # 1. Create provider file
            provider_path = base_path / f"EvolveLab/providers/{provider_info['module_name']}.py"
            if provider_path.exists():
                return {"success": False, "error": f"Provider file {provider_path} already exists"}

            # Ensure directory exists
            provider_path.parent.mkdir(parents=True, exist_ok=True)

            with open(provider_path, 'w', encoding='utf-8') as f:
                f.write(provider_info["code"])
            results["provider_created"] = str(provider_path)

            # 2. Update memory_types.py - using simplified comment marker logic
            memory_types_path = base_path / "EvolveLab/memory_types.py"
            with open(memory_types_path, 'r', encoding='utf-8') as f:
                content = f.read()

            enum_name = memory_type_info["enum_name"]
            enum_value = memory_type_info["enum_value"]
            class_name = provider_info["class_name"]
            module_name = provider_info["module_name"]

            # Check if enum already exists
            if f'{enum_name} = "{enum_value}"' not in content and f'{enum_name}=' not in content:
                print(f"Adding new enum: {enum_name}")

                # Add enum using comment marker
                enum_marker_text = "add new memory type upside this line(Enum)"
                lines = content.split('\n')
                enum_marker_line_index = -1

                for i, line in enumerate(lines):
                    if enum_marker_text in line.strip():
                        enum_marker_line_index = i
                        break

                if enum_marker_line_index >= 0:
                    # Add new enum above marker line
                    new_enum_line = f'    {enum_name} = "{enum_value}"'
                    lines.insert(enum_marker_line_index, new_enum_line)
                    content = '\n'.join(lines)
                    print(f"Successfully added enum above marker line: {enum_name}")
                else:
                    print(f"Enum marker line not found: {enum_marker_text}")
                    results["enum_update_warning"] = "Could not find enum marker line"
            else:
                print(f"Enum {enum_name} already exists, skipping addition")

            # Check if mapping already exists
            if f'MemoryType.{enum_name}:' not in content:
                print(f"Adding new mapping: {enum_name}")

                # Add mapping using comment marker
                mapping_marker_text = "add new memory type upside this line(PROVIDER_MAPPING)"
                lines = content.split('\n')
                mapping_marker_line_index = -1

                for i, line in enumerate(lines):
                    if mapping_marker_text in line.strip():
                        mapping_marker_line_index = i
                        break

                if mapping_marker_line_index >= 0:
                    # Add new mapping above marker line
                    new_mapping_line = f'    MemoryType.{enum_name}: ("{class_name}", "{module_name}"),'
                    lines.insert(mapping_marker_line_index, new_mapping_line)
                    content = '\n'.join(lines)
                    print(f"Successfully added mapping above marker line: {enum_name}")
                else:
                    print(f"Mapping marker line not found: {mapping_marker_text}")
                    results["mapping_update_warning"] = "Could not find mapping marker line"
            else:
                print(f"Mapping {enum_name} already exists, skipping addition")

            with open(memory_types_path, 'w', encoding='utf-8') as f:
                f.write(content)
            results["memory_types_updated"] = True

            # 3. Validate config before saving
            print(f"\n[Validation] Validating configuration before saving...")
            validation_result = MemorySystemCreator._validate_config_updates(config_updates)
            
            # Display warnings
            if validation_result["warnings"]:
                print(f"[WARNING] Configuration Warnings:")
                for warning in validation_result["warnings"]:
                    print(f"  - {warning}")
            
            # Check for errors
            if not validation_result["success"]:
                print(f"[ERROR] Configuration Validation Failed:")
                for error in validation_result["errors"]:
                    print(f"  - {error}")
                return {
                    "success": False,
                    "error": "Configuration validation failed. See errors above.",
                    "validation_errors": validation_result["errors"]
                }
            
            print(f"[OK] Configuration validation passed!")

            # 4. Update config.py
            config_path = base_path / "EvolveLab/config.py"
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add new configuration - using correct format
            if f'MemoryType.{memory_type_info["enum_name"]}:' not in content:
                # Build new configuration block, format consistent with existing providers
                config_lines = []
                config_lines.append(f'        MemoryType.{memory_type_info["enum_name"]}: {{')

                # Add configuration items with normalization
                for key, value in config_updates.items():
                    # Normalize key name (handle special characters)
                    normalized_key = MemorySystemCreator._normalize_config_key(key)
                    
                    # Warn if key was changed
                    if normalized_key != key:
                        print(f"  [WARNING] Key normalized: '{key}' -> '{normalized_key}'")
                    
                    # Parse and format value
                    parsed_value, python_repr = MemorySystemCreator._parse_config_value(value)
                    
                    # Add configuration line
                    config_lines.append(f'            "{normalized_key}": {python_repr},')

                # Remove last comma and add closing brace
                if config_lines[-1].endswith(','):
                    config_lines[-1] = config_lines[-1][:-1]
                config_lines.append('        },')

                config_block = '\n'.join(config_lines)

                # Simplified config update logic: directly match comment text
                marker_text = "add new memory type upside this line"

                # Find line containing marker text
                lines = content.split('\n')
                marker_line_index = -1
                actual_marker_line = ""

                for i, line in enumerate(lines):
                    if marker_text in line.strip():
                        marker_line_index = i
                        actual_marker_line = line
                        break

                if marker_line_index >= 0:
                    # Add new configuration above marker line
                    lines.insert(marker_line_index, config_block)
                    content = '\n'.join(lines)
                    print(f"Successfully added configuration above marker line: {memory_type_info['enum_name']}")
                    print(f"Found marker line {marker_line_index + 1}: '{actual_marker_line.strip()}'")
                else:
                    print("Config marker line not found, cannot add configuration")
                    print(f"Searched marker text: '{marker_text}'")
                    # Display relevant parts of config file for debugging
                    for i, line in enumerate(lines):
                        if "add new memory type" in line.lower():
                            print(f"Found similar line {i+1}: '{line.strip()}'")
                    results["config_update_warning"] = "Could not find config marker line"

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            results["config_updated"] = True

            return {
                "success": True,
                "results": results,
                "memory_type": memory_type_info["enum_value"]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def delete_memory_system(memory_type_enum_name: str = None, base_dir: str = ".") -> Dict[str, Any]:
        """
        Delete a memory system
        
        Args:
            memory_type_enum_name: The enum name to delete (e.g., "CEREBRA_FUSION_MEMORY"). 
                                  If None, deletes the last one based on memory_types ordering.
            base_dir: Base directory path
            
        Returns:
            Dict with success status and details
        """
        try:
            base_path = Path(base_dir)
            memory_types_path = base_path / "EvolveLab/memory_types.py"
            
            # Read memory_types.py
            with open(memory_types_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse enum definitions and mappings
            enum_lines = []
            mapping_lines = []
            lines = content.split('\n')
            
            in_enum = False
            in_mapping = False
            
            for i, line in enumerate(lines):
                if 'class MemoryType(Enum):' in line:
                    in_enum = True
                    continue
                elif in_enum and line.strip().startswith('#') and 'add new memory type upside this line(Enum)' in line:
                    in_enum = False
                elif in_enum and '=' in line and not line.strip().startswith('#'):
                    enum_lines.append((i, line))
                
                if 'PROVIDER_MAPPING = {' in line:
                    in_mapping = True
                    continue
                elif in_mapping and line.strip().startswith('#') and 'add new memory type upside this line(PROVIDER_MAPPING)' in line:
                    in_mapping = False
                elif in_mapping and 'MemoryType.' in line and ':' in line:
                    mapping_lines.append((i, line))
            
            if not enum_lines:
                return {"success": False, "error": "No memory types found to delete"}
            
            # Determine which enum to delete
            if memory_type_enum_name is None:
                # Delete the last one
                target_line_num, target_line = enum_lines[-1]
                # Extract enum name from line
                enum_name = target_line.split('=')[0].strip()
            else:
                # Find specified enum
                target_line_num = None
                enum_name = memory_type_enum_name.strip()
                for line_num, line in enum_lines:
                    if line.strip().startswith(enum_name + ' '):
                        target_line_num = line_num
                        target_line = line
                        break
                
                if target_line_num is None:
                    return {"success": False, "error": f"Memory type {enum_name} not found"}
            
            print(f"Deleting memory system: {enum_name}")
            
            # Extract enum_value and find corresponding mapping
            enum_value = target_line.split('=')[1].strip().strip('"').strip("'")
            
            # Find provider info from mapping
            provider_class_name = None
            module_name = None
            mapping_line_to_delete = None
            
            for line_num, line in mapping_lines:
                if f'MemoryType.{enum_name}:' in line:
                    mapping_line_to_delete = line_num
                    # Extract class name and module name
                    # Format: MemoryType.XXX: ("ClassName", "module_name"),
                    import re
                    match = re.search(r'\("([^"]+)",\s*"([^"]+)"\)', line)
                    if match:
                        provider_class_name = match.group(1)
                        module_name = match.group(2)
                    break
            
            if not provider_class_name or not module_name:
                return {"success": False, "error": f"Could not find provider mapping for {enum_name}"}
            
            results = {}
            
            # 1. Delete provider file
            provider_path = base_path / f"EvolveLab/providers/{module_name}.py"
            if provider_path.exists():
                provider_path.unlink()
                results["provider_deleted"] = str(provider_path)
                print(f"Deleted provider file: {provider_path}")
            else:
                results["provider_warning"] = f"Provider file not found: {provider_path}"
                print(f"Provider file not found: {provider_path}")
            
            # 2. Update memory_types.py - remove enum and mapping
            lines = content.split('\n')
            
            # Remove enum line
            if target_line_num < len(lines):
                del lines[target_line_num]
                print(f"Removed enum definition: {enum_name}")
            
            # Adjust mapping line number (since we deleted a line above)
            if mapping_line_to_delete is not None:
                if mapping_line_to_delete > target_line_num:
                    mapping_line_to_delete -= 1
                
                if mapping_line_to_delete < len(lines):
                    del lines[mapping_line_to_delete]
                    print(f"Removed provider mapping: {enum_name}")
            
            content = '\n'.join(lines)
            with open(memory_types_path, 'w', encoding='utf-8') as f:
                f.write(content)
            results["memory_types_updated"] = True
            
            # 3. Delete config entry from config.py
            config_path = base_path / "EvolveLab/config.py"
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            config_lines = config_content.split('\n')
            
            # Find and remove the config block for this memory type
            start_idx = None
            end_idx = None
            brace_count = 0
            
            for i, line in enumerate(config_lines):
                if f'MemoryType.{enum_name}:' in line:
                    start_idx = i
                    # Count opening braces on this line
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                    continue
                
                if start_idx is not None:
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                    
                    # Check if we've closed all braces and reached the end
                    if brace_count <= 0 and (line.strip().endswith('},') or line.strip().endswith('}')):
                        end_idx = i
                        break
            
            if start_idx is not None and end_idx is not None:
                # Remove lines from start_idx to end_idx (inclusive)
                del config_lines[start_idx:end_idx + 1]
                config_content = '\n'.join(config_lines)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(config_content)
                results["config_deleted"] = True
                print(f"Removed config entry: {enum_name}")
            else:
                results["config_warning"] = f"Could not find config entry for {enum_name}"
                print(f" Could not find config entry for {enum_name}")
            
            print(f"\nSuccessfully deleted memory system: {enum_name}")
            
            return {
                "success": True,
                "deleted_system": enum_name,
                "enum_value": enum_value,
                "provider_class": provider_class_name,
                "module_name": module_name,
                "results": results
            }
            
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
