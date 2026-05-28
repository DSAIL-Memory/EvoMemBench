#!/usr/bin/env python
# coding=utf-8

"""
MemEvolve Command-Line Interface
Provides commands for memory system evolution workflow
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


def cmd_analyze(args):
    """Analyze task trajectories"""
    from MemEvolve import MemoryEvolver
    import os
    
    print(f"=== Phase 1: Analyze ===")
    print(f"Task logs: {args.task_logs_dir}")
    print(f"Work directory: {args.work_dir}")
    
    analysis_model_id = args.model or os.getenv("ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    print(f"Analysis Model: {analysis_model_id}")
    
    generation_model_id = os.getenv("GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    
    evolver = MemoryEvolver(
        work_dir=args.work_dir,
        analysis_model_id=analysis_model_id,
        gen_model_id=generation_model_id
    )
    
    # Run analysis
    result = evolver.analyze(
        task_logs_dir=args.task_logs_dir,
        default_provider=args.provider
    )
    
    if result["success"]:
        print(f"\nAnalysis complete!")
        print(f"Report: {result['report_path']}")
        print(f"Statistics:")
        stats = result["stats"]
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Correct: {stats['correct_tasks']} ({stats['correct_tasks']/stats['total_tasks']*100:.1f}%)")
    else:
        print(f"\nAnalysis failed")
        sys.exit(1)


def cmd_generate(args):
    """Generate a single new memory system"""
    from MemEvolve import MemoryEvolver
    from pathlib import Path
    import os
    
    print(f"=== Phase 2: Generate ===")
    print(f"Work directory: {args.work_dir}")
    print(f"Creativity index: {args.creativity}")
    print(f"\nNote: This command generates ONE memory system.")
    print(f"To generate multiple systems, use 'auto' mode with --num-systems parameter.")
    
    generation_model_id = args.model or os.getenv("GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    print(f"Generation Model: {generation_model_id}")
    
    evolver = MemoryEvolver(
        work_dir=args.work_dir,
        gen_model_id=generation_model_id
    )
    
    # If provider is specified, update state with it
    if hasattr(args, 'provider') and args.provider:
        if evolver.state["phases"]["analyze"]["completed"]:
            evolver.state["phases"]["analyze"]["default_provider"] = args.provider
            evolver._save_state()
            print(f"Template provider: {args.provider}")
    
    # Generate single system
    result = evolver.generate(
        creativity_index=args.creativity
    )
    
    if result["success"]:
        print(f"\nGeneration complete!")
        print(f"Config saved: {result['config_path']}")
        
        # Display generated system info
        config = result["config"]
        provider_info = config.get('provider_code', {})
        memory_info = config.get('memory_type_info', {})
        
        enum_name = memory_info.get('enum_name', 'Unknown')
        enum_value = memory_info.get('enum_value', 'Unknown')
        
        print(f"\n--- Generated System Information ---")
        print(f"Class Name: {provider_info.get('class_name', 'Unknown')}")
        print(f"Module Name: {provider_info.get('module_name', 'Unknown')}")
        print(f"Enum Name: {enum_name}")
        print(f"Enum Value: {enum_value}")
        print(f"Memory Type: MemoryType.{enum_name} = \"{enum_value}\"")
        
        # Show config params count
        config_updates = config.get('config_updates', {})
        print(f"Config Parameters: {len(config_updates)} items")
        
        # Ask if user wants to create the system
        print(f"\n" + "="*50)
        create_now = input(f"Do you want to create this memory system now? (yes/no): ")
        
        if create_now.lower() in ['yes', 'y']:
            print(f"\n=== Phase 3: Create ===")
            print(f"Creating generated system...")
            
            create_result = evolver.create()
            if create_result["success"]:
                created_systems = create_result.get("created", [])
                failed_systems = create_result.get("failed", [])
                
                print(f"\nCreation complete!")
                print(f"Created systems: {', '.join(created_systems)}")
                if failed_systems:
                    print(f"Failed systems: {len(failed_systems)}")
                    for fail in failed_systems:
                        if isinstance(fail, dict):
                            print(f"  - {fail['system']}: {fail['error']}")
                        else:
                            print(f"  - {fail}")
            else:
                print(f"\nCreation failed")
                sys.exit(1)
        else:
            print(f"\nSkipping creation. You can create later using: python evolve_cli.py create")
    else:
        print(f"\nGeneration failed")
        sys.exit(1)


def cmd_create(args):
    """Create memory system files"""
    from MemEvolve import MemoryEvolver
    
    print(f"=== Phase 3: Create ===")
    print(f"Work directory: {args.work_dir}")
    
    evolver = MemoryEvolver(
        work_dir=args.work_dir
    )
    
    # Run creation
    result = evolver.create()
    
    if result["success"]:
        print(f"\nCreation complete!")
        print(f"Created systems: {', '.join(result['created'])}")
        if result['failed']:
            print(f"Failed systems: {len(result['failed'])}")
            for fail in result['failed']:
                print(f"  - {fail['system']}: {fail['error']}")
    else:
        print(f"\nCreation failed")
        sys.exit(1)


def cmd_validate(args):
    """Validate created systems with static checks and tests"""
    from MemEvolve import MemoryEvolver

    print(f"=== Phase 4: Validate ===")
    print(f"Work directory: {args.work_dir}")
    print(f"Sanity tests: 5 test cases")

    evolver = MemoryEvolver(
        work_dir=args.work_dir
    )

    result = evolver.validate()

    if result["success"]:
        print(f"\nValidation complete!")
        print(f"Passed: {', '.join(result['validated']) if result['validated'] else 'None'}")
        print(f"Failed: {', '.join(result['failed']) if result['failed'] else 'None'}")

        if result['failed']:
            print(f"\nFailed systems need to be regenerated.")
            print(f"Check validation reports in: {args.work_dir}/validation_reports/")
    else:
        print(f"\nValidation failed")
        sys.exit(1)


def cmd_delete(args):
    """Delete a memory system"""
    from MemEvolve.phases.memory_creator import MemorySystemCreator
    
    print(f"=== Delete Memory System ===")
    
    if args.memory_type:
        print(f"Target: {args.memory_type}")
    else:
        print(f"Target: Last memory system (auto-detected)")
    
    # Confirm deletion
    if not args.yes:
        confirm = input(f"\nAre you sure you want to delete this memory system? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Deletion cancelled.")
            return
    
    # Delete the system
    result = MemorySystemCreator.delete_memory_system(
        memory_type_enum_name=args.memory_type,
        base_dir="."
    )
    
    if result["success"]:
        print(f"\nSuccessfully deleted memory system!")
        print(f"System: {result['deleted_system']}")
        print(f"Enum value: {result['enum_value']}")
        print(f"Provider class: {result['provider_class']}")
        print(f"Module: {result['module_name']}")
        
        if 'results' in result:
            res = result['results']
            if 'provider_deleted' in res:
                print(f"Provider file deleted: {res['provider_deleted']}")
            if 'memory_types_updated' in res:
                print(f"memory_types.py updated")
            if 'config_deleted' in res:
                print(f"config.py updated")
    else:
        print(f"\nDeletion failed: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(f"\nTraceback:\n{result['traceback']}")
        sys.exit(1)


def cmd_run_all(args):
    """Run complete evolution workflow (Analyze -> Generate -> Create -> Validate)"""
    from MemEvolve import MemoryEvolver
    import os

    print(f"=== Running Complete Evolution Workflow ===")
    print(f"Task logs: {args.task_logs_dir}")
    print(f"Work directory: {args.work_dir}")
    print(f"Creativity index: {args.creativity}")
    print(f"Sanity tests: 5 test cases")
    print(f"\nNote: This workflow generates ONE memory system.")
    print(f"To generate multiple systems, use 'auto-evolve' mode with --num-systems parameter.")
    
    analysis_model_id = args.model or os.getenv("ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    print(f"Analysis Model: {analysis_model_id}")
    
    generation_model_id = os.getenv("GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    print(f"Generation Model: {generation_model_id}")
    
    evolver = MemoryEvolver(
        work_dir=args.work_dir,
        analysis_model_id=analysis_model_id,
        gen_model_id=generation_model_id
    )
    
    # Phase 1: Analyze
    print("\n--- Phase 1: Analyze ---")
    result = evolver.analyze(
        task_logs_dir=args.task_logs_dir,
        default_provider=args.provider
    )
    if not result["success"]:
        print("Analysis failed")
        sys.exit(1)
    print("Analysis complete")
    
    # Phase 2: Generate (single system)
    print("\n--- Phase 2: Generate ---")
    result = evolver.generate(
        creativity_index=args.creativity
    )
    if not result["success"]:
        print("Generation failed")
        sys.exit(1)
    print("Generation complete")
    
    # Phase 3: Create
    print("\n--- Phase 3: Create ---")
    result = evolver.create()
    if not result["success"]:
        print("Creation failed")
        sys.exit(1)
    print(f"Created {len(result['created'])} system(s)")
    
    # Phase 4: Validate
    print("\n--- Phase 4: Validate ---")
    result = evolver.validate()
    if not result["success"]:
        print("Validation failed")
        sys.exit(1)
    
    print(f"\n=== Evolution Complete ===")
    print(f"Validated systems: {', '.join(result['validated']) if result['validated'] else 'None'}")


def cmd_status(args):
    """Show current status"""
    from MemEvolve import MemoryEvolver
    
    work_dir = Path(args.work_dir)
    state_file = work_dir / "state.json"
    
    if not state_file.exists():
        print(f"No evolution state found in {work_dir}")
        sys.exit(1)
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    print(f"=== MemEvolve Status ===")
    print(f"Work directory: {work_dir}")
    print(f"Model: {state.get('model_id', 'unknown')}")
    print(f"Created: {state.get('created_at', 'unknown')}")
    print()
    
    phases = state.get("phases", {})
    
    # Analyze
    analyze = phases.get("analyze", {})
    status = "COMPLETE" if analyze.get("completed") else "PENDING"
    print(f"1. Analyze: {status}")
    if analyze.get("completed"):
        print(f"   Output: {analyze.get('output')}")
        print(f"   Task logs: {analyze.get('task_logs_dir')}")
    
    # Generate
    generate = phases.get("generate", {})
    status = "COMPLETE" if generate.get("completed") else "PENDING"
    print(f"2. Generate: {status}")
    if generate.get("completed"):
        print(f"   Output: {generate.get('output')}")
        creativity = generate.get('creativity_index', 0.5)
        print(f"   Creativity: {creativity:.2f}")
    
    # Create
    create = phases.get("create", {})
    status = "COMPLETE" if create.get("completed") else "PENDING"
    print(f"3. Create: {status}")
    if create.get("completed"):
        print(f"   Output: {create.get('output')}")
        created = create.get('created_systems', [])
        failed = create.get('failed_systems', [])
        print(f"   Created: {len(created)} systems")
        if created:
            for sys in created:
                print(f"     - {sys}")
        if failed:
            print(f"   Failed: {len(failed)} systems")
    
    # Validate
    validate = phases.get("validate", {})
    status = "COMPLETE" if validate.get("completed") else "PENDING"
    print(f"4. Validate: {status}")
    if validate.get("completed"):
        print(f"   Output: {validate.get('output')}")
        validated = validate.get('validated_systems', [])
        failed = validate.get('failed_systems', [])
        print(f"   Validated: {len(validated)} systems")
        if validated:
            for sys in validated:
                print(f"     - {sys}")
        if failed:
            print(f"   Failed: {len(failed)} systems")
            for sys in failed:
                print(f"     - {sys}")


def cmd_auto_evolve(args):
    """Run multi-round automatic evolution"""
    from MemEvolve import AutoEvolver
    from MemEvolve.utils.run_provider import run_provider
    from MemEvolve.config import DEFAULT_DATASETS
    import os
    
    print(f"=== Auto Evolution: Multi-Round ===")
    print(f"Dataset: {args.dataset}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Work directory: {args.work_dir}")
    print(f"Base provider: {args.provider}")
    print(f"Systems per round: {args.num_systems}")
    print(f"Task batch size (x): {args.task_batch_x}")
    print(f"Top candidates (t): {args.top_t}")
    print(f"Extra sample size (y): {args.extra_sample_y}")
    print(f"Creativity index: {args.creativity}")
    print(f"Use Pareto selection: {args.use_pareto_selection}")
    print(f"Clear storage per round: {args.clear_storage_per_round}")
    
    # Get model IDs
    analysis_model_id = args.model or os.getenv("ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    generation_model_id = os.getenv("GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
    print(f"Analysis Model: {analysis_model_id}")
    print(f"Generation Model: {generation_model_id}")
    
    # Create AutoEvolver
    auto_evolver = AutoEvolver(
        analysis_model_id=analysis_model_id,
        gen_model_id=generation_model_id,
        work_root=args.work_dir,
        dataset_name=args.dataset,
        run_provider=run_provider,
        default_provider=args.provider,
        num_systems=args.num_systems,
        creativity_index=args.creativity,
        task_batch_x=args.task_batch_x,
        top_t=args.top_t,
        extra_sample_y=args.extra_sample_y,
        datasets_config=DEFAULT_DATASETS,
        use_pareto_selection=args.use_pareto_selection,
        clear_storage_per_round=args.clear_storage_per_round,
    )
    
    # Confirm before starting
    if not args.yes:
        print(f"\n" + "="*50)
        print(f"This will run {args.num_rounds} rounds of evolution.")
        print(f"Each round will:")
        print(f"  1. Run base provider on {args.task_batch_x} tasks")
        print(f"  2. Generate {args.num_systems} new systems")
        print(f"  3. Evaluate all {args.num_systems+1} systems")
        print(f"  4. Select top {args.top_t} systems")
        print(f"  5. Run finalists on {args.extra_sample_y} + {args.task_batch_x} = {args.extra_sample_y + args.task_batch_x} tasks")
        print(f"  6. Select winner as next base provider")
        print(f"="*50)
        confirm = input(f"Continue? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Run evolution
    result = auto_evolver.run(num_rounds=args.num_rounds)
    
    print(f"\n=== Auto Evolution Complete ===")
    print(f"Total rounds: {result['rounds']}")
    print(f"History saved to: {result['history_path']}")
    
    # Show round summaries
    for round_info in result['history']:
        round_num = round_info['round']
        winner = round_info['winner']
        print(f"\nRound {round_num}: Winner = {winner}")


def cmd_list(args):
    """List all memory systems"""
    from pathlib import Path
    
    print(f"=== Memory Systems ===")
    
    # Read memory_types.py
    memory_types_path = Path("EvolveLab/memory_types.py")
    if not memory_types_path.exists():
        print(f"Error: memory_types.py not found")
        sys.exit(1)
    
    with open(memory_types_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse enum definitions
    enum_items = []
    lines = content.split('\n')
    in_enum = False
    
    for line in lines:
        if 'class MemoryType(Enum):' in line:
            in_enum = True
            continue
        elif in_enum and line.strip().startswith('#') and 'add new memory type upside this line(Enum)' in line:
            break
        elif in_enum and '=' in line and not line.strip().startswith('#'):
            # Extract enum name and value
            parts = line.split('=')
            if len(parts) == 2:
                enum_name = parts[0].strip()
                enum_value = parts[1].strip().strip('"').strip("'")
                enum_items.append((enum_name, enum_value))
    
    if not enum_items:
        print("No memory systems found")
        return
    
    print(f"Total: {len(enum_items)} memory systems\n")
    for i, (name, value) in enumerate(enum_items, 1):
        marker = " [LAST]" if i == len(enum_items) else ""
        print(f"{i:2d}. {name:40s} = {value}{marker}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="memevolve",
        description="Memory System Evolution Tool"
    )
    
    # Global arguments
    parser.add_argument(
        "--work-dir",
        default="./memevolve_work",
        help="Working directory for evolution state (default: ./memevolve_work)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID for LLM operations (default: from ANALYSIS_MODEL or GENERATION_MODEL env var, fallback to DEFAULT_MODEL)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze task trajectories"
    )
    p_analyze.add_argument(
        "task_logs_dir",
        help="Directory containing task execution logs"
    )
    p_analyze.add_argument(
        "--provider",
        default="agent_kb",
        help="Provider to use as template (default: agent_kb)"
    )
    p_analyze.set_defaults(func=cmd_analyze)
    
    # Generate command
    p_generate = subparsers.add_parser(
        "generate",
        help="Generate a single new memory system configuration"
    )
    p_generate.add_argument(
        "--creativity",
        type=float,
        default=0.5,
        help="Creativity index 0-1, controls innovation level (default: 0.5)"
    )
    p_generate.add_argument(
        "--provider",
        default=None,
        help="Provider to use as template (optional, uses value from analyze phase if not specified)"
    )
    p_generate.set_defaults(func=cmd_generate)
    
    # Create command
    p_create = subparsers.add_parser(
        "create",
        help="Create memory system files"
    )
    p_create.set_defaults(func=cmd_create)
    
    # Validate command
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate created systems with static checks and tests"
    )
    p_validate.set_defaults(func=cmd_validate)
    
    # Delete command
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete a memory system"
    )
    p_delete.add_argument(
        "--memory-type",
        dest="memory_type",
        default=None,
        help="Memory type enum name to delete (e.g., CEREBRA_FUSION_MEMORY). If not specified, deletes the last one."
    )
    p_delete.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    p_delete.set_defaults(func=cmd_delete)
    
    # Run-all command
    p_run_all = subparsers.add_parser(
        "run-all",
        help="Run complete evolution workflow (Analyze -> Generate -> Create -> Validate)"
    )
    p_run_all.add_argument(
        "task_logs_dir",
        help="Directory containing task execution logs"
    )
    p_run_all.add_argument(
        "--provider",
        default="agent_kb",
        help="Provider to use as template (default: agent_kb)"
    )
    p_run_all.add_argument(
        "--creativity",
        type=float,
        default=0.5,
        help="Creativity index 0-1, controls innovation level (default: 0.5)"
    )
    p_run_all.set_defaults(func=cmd_run_all)
    
    # Status command
    p_status = subparsers.add_parser(
        "status",
        help="Show current evolution status"
    )
    p_status.set_defaults(func=cmd_status)
    
    # List command to show available memory types
    p_list = subparsers.add_parser(
        "list",
        help="List all memory systems"
    )
    p_list.set_defaults(func=cmd_list)
    
    # Auto-evolve command - multi-round evolution
    p_auto = subparsers.add_parser(
        "auto-evolve",
        help="Run multi-round automatic evolution"
    )
    p_auto.add_argument(
        "dataset",
        choices=["gaia", "webwalkerqa", "xbench", "taskcraft"],
        help="Dataset to use for evolution"
    )
    p_auto.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of evolution rounds to run (default: 1)"
    )
    p_auto.add_argument(
        "--provider",
        default="agent_kb",
        help="Initial base provider (default: agent_kb)"
    )
    p_auto.add_argument(
        "--num-systems",
        type=int,
        default=3,
        help="Number of systems to generate per round (default: 3)"
    )
    p_auto.add_argument(
        "--task-batch-x",
        type=int,
        default=20,
        help="Task batch size for each evaluation (default: 20)"
    )
    p_auto.add_argument(
        "--top-t",
        type=int,
        default=2,
        help="Number of top systems to select for finals (default: 2)"
    )
    p_auto.add_argument(
        "--extra-sample-y",
        type=int,
        default=5,
        help="Number of tasks to resample for finals (default: 5)"
    )
    p_auto.add_argument(
        "--creativity",
        type=float,
        default=0.5,
        help="Creativity index 0-1 (default: 0.5)"
    )
    p_auto.add_argument(
        "--use-pareto-selection",
        action="store_true",
        default=False,
        help="Use Pareto optimality-based selection for choosing best memory systems (default: False, uses traditional accuracy+token sorting)"
    )
    p_auto.add_argument(
        "--clear-storage-per-round",
        action="store_true",
        default=True,
        help="Clear base provider storage at the start of each round for fair evolution (default: True). Set to False to test continual learning."
    )
    p_auto.add_argument(
        "--no-clear-storage",
        dest="clear_storage_per_round",
        action="store_false",
        help="Disable storage clearing (providers keep knowledge from previous rounds)"
    )
    p_auto.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    p_auto.set_defaults(func=cmd_auto_evolve)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

