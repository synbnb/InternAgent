import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from internagent.mas.memory import TaskMemoryLayer

dotenv.load_dotenv()

DEFAULT_CONFIG = {
    "task_memory": {
        "memory_dir": "./config/mem_store",
        "label_threshold_percent": 0.0,
        "embedding_mode": "description",
        "embedding": {
            "model_type": "local",
            "model_name": "BAAI/bge-base-en-v1.5",
        },
    },
    "agents": {
        "exp_analyze": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "",
            "temperature": 0.7,
            "max_tokens": 4096,
            "timeout": 600,
            "use_llm_for_metric_direction": True,
            "use_llm_for_primary_metric": True,
        },
    },
}

TASK_SPECIFIC_CONFIGS = {}


def create_memory_from_config(
    task_name: Optional[str] = None,
    memory_dir: Optional[str] = None,
    label_threshold: Optional[float] = None,
    embedding_model_path: Optional[str] = None
) -> TaskMemoryLayer:
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)

    if task_name and task_name in TASK_SPECIFIC_CONFIGS:
        task_config = TASK_SPECIFIC_CONFIGS[task_name]
        for key, value in task_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key] = {**config[key], **copy.deepcopy(value)}
            else:
                config[key] = copy.deepcopy(value)

    if memory_dir:
        config.setdefault("task_memory", {})["memory_dir"] = memory_dir

    if label_threshold is not None:
        config.setdefault("task_memory", {})["label_threshold_percent"] = label_threshold

    if embedding_model_path:
        config.setdefault("task_memory", {}).setdefault("embedding", {})["model_name"] = embedding_model_path

    return TaskMemoryLayer.from_config(config)


def find_run_directories(idea_dir: Path) -> list:
    return sorted([d for d in idea_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])


def batch_save_from_traj(
    traj_dir: Path,
    task_name: str,
    memory_dir: str = "./config/mem_store",
    session_id: Optional[str] = None,
    aggregation: str = "best",
    skip_existing: bool = True,
    label_threshold_percent: float = 0.0,
    embedding_model_path: Optional[str] = None
) -> tuple:
    if not traj_dir.exists():
        print(f"✗ Trajectory directory not found: {traj_dir}")
        return (0, 0)

    print(f"\nInitializing Task Memory: {task_name}")
    print(f"Memory directory: {memory_dir}")

    memory = create_memory_from_config(
        task_name=task_name,
        memory_dir=memory_dir,
        label_threshold=label_threshold_percent,
        embedding_model_path=embedding_model_path
    )

    print(f"Current records: {len(memory.records)}")
    if memory.analyze_agent:
        print("LLM analysis: Enabled")
    else:
        print("LLM analysis: Disabled")

    # Support both new session-based structure and old structure
    all_items = sorted(traj_dir.iterdir())
    experiment_dirs = []
    traj_files = []

    # Check for new session-based structure (session_*/traj.json)
    session_dirs = [item for item in all_items if item.is_dir() and item.name.startswith('session_')]

    if session_dirs:
        print(f"Detected new session-based structure with {len(session_dirs)} sessions")
        # New structure: results/task/session_*/
        for session_dir in session_dirs:
            # Find traj.json in session directory
            traj_file = session_dir / "traj.json"
            if traj_file.exists():
                traj_files.append(traj_file)

            # Find experiment directories in session directory
            for item in sorted(session_dir.iterdir()):
                if item.is_dir() and find_run_directories(item):
                    experiment_dirs.append(item)
    else:
        print(f"Detected old flat structure")
        # Old structure: results/task/ (flat)
        for item in all_items:
            if item.is_dir() and not item.name.startswith('traj_session'):
                if find_run_directories(item):
                    experiment_dirs.append(item)

        # Find old-style traj files
        traj_files = sorted(traj_dir.glob("traj_session_*.json"))

    print(f"Found {len(experiment_dirs)} experiment directories")
    print(f"Found {len(traj_files)} trajectory files")

    if not experiment_dirs:
        print(f"✗ No experiment directories found")
        return (0, 0)

    traj_data_map = {}
    for traj_file in traj_files:
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                traj_data = json.load(f)

            top_idea_ids = traj_data.get('top_ideas', [])
            all_ideas = traj_data.get('ideas', [])
            ideas_map = {idea.get('id'): idea for idea in all_ideas if 'id' in idea}

            ideas_by_name = {}
            for idea in all_ideas:
                refined_details = idea.get('refined_method_details', {})
                if isinstance(refined_details, dict) and 'name' in refined_details:
                    idea_name = refined_details['name'].strip()
                    if idea_name:
                        ideas_by_name[idea_name.lower()] = idea

            # Extract timestamp from file path
            # New structure: session_123456/traj.json -> use session_id as timestamp
            # Old structure: traj_session_123456.json -> use 123456 as timestamp
            if traj_file.name == "traj.json":
                # New structure: extract session_id from parent directory
                session_id_match = traj_file.parent.name
                if session_id_match.startswith('session_'):
                    traj_unix_ts = float(session_id_match.split('_')[1])
                else:
                    traj_unix_ts = 0
            else:
                # Old structure
                traj_unix_ts = float(traj_file.stem.split('_')[-1])

            traj_data_map[traj_file] = {
                'top_idea_ids': top_idea_ids,
                'ideas_map': ideas_map,
                'ideas_by_name': ideas_by_name,
                'all_ideas': all_ideas,
                'timestamp': traj_unix_ts
            }
        except Exception as e:
            print(f"Warning: Error reading {traj_file.name}: {e}")
            continue

    print(f"Loaded {len(traj_data_map)} trajectory files\n")

    success_count = 0
    failure_count = 0
    skipped_count = 0
    failed_ideas = []
    processed_ideas = []

    for exp_dir in experiment_dirs:
        try:
            parts = exp_dir.name.split('_', 2)
            idea_name = parts[2] if len(parts) >= 3 else exp_dir.name
        except Exception:
            idea_name = exp_dir.name

        matched_traj_file = None
        matched_idea_data = None
        idea_name_lower = idea_name.lower()

        for traj_file, traj_info in traj_data_map.items():
            if idea_name_lower in traj_info['ideas_by_name']:
                matched_idea_data = traj_info['ideas_by_name'][idea_name_lower]
                matched_traj_file = traj_file
                break

            idea_name_normalized = idea_name_lower.replace('_', ' ').replace('-', ' ').replace('  ', ' ')
            for traj_name, idea_dict in traj_info['ideas_by_name'].items():
                traj_name_normalized = traj_name.replace('_', ' ').replace('-', ' ').replace('  ', ' ')
                if idea_name_normalized == traj_name_normalized:
                    matched_idea_data = idea_dict
                    matched_traj_file = traj_file
                    break

            if matched_idea_data:
                break

        if not matched_idea_data:
            idea_keywords = [w for w in idea_name_lower.split() if len(w) > 3]
            if idea_keywords:
                for traj_file, traj_info in traj_data_map.items():
                    for traj_name, idea_dict in traj_info['ideas_by_name'].items():
                        matches = sum(1 for kw in idea_keywords if kw in traj_name)
                        if matches >= len(idea_keywords) * 0.7:
                            matched_idea_data = idea_dict
                            matched_traj_file = traj_file
                            break
                    if matched_idea_data:
                        break

        if not matched_idea_data:
            continue

        refined_details = matched_idea_data.get('refined_method_details', {})
        if isinstance(refined_details, dict) and refined_details:
            idea_info = {
                "name": idea_name,
                "title": refined_details.get('title', matched_idea_data.get('text', idea_name)),
                "description": refined_details.get('description', matched_idea_data.get('text', idea_name)),
                "statement": refined_details.get('statement', matched_idea_data.get('rationale', '')),
                "method": refined_details.get('method', matched_idea_data.get('text', idea_name)),
                "score": matched_idea_data.get('score', 0.0),
                "rationale": matched_idea_data.get('rationale', ''),
                "baseline_summary": matched_idea_data.get('baseline_summary', ''),
                "id": matched_idea_data.get('id', ''),
                "critiques": matched_idea_data.get('critiques', []),
                "evidence": matched_idea_data.get('evidence', []),
                "references": matched_idea_data.get('references', [])
            }
        else:
            idea_info = {
                "name": idea_name,
                "title": matched_idea_data.get('text', idea_name),
                "description": matched_idea_data.get('text', idea_name),
                "statement": matched_idea_data.get('rationale', ''),
                "method": matched_idea_data.get('text', idea_name),
                "score": matched_idea_data.get('score', 0.0),
                "rationale": matched_idea_data.get('rationale', ''),
                "baseline_summary": matched_idea_data.get('baseline_summary', ''),
                "id": matched_idea_data.get('id', ''),
                "critiques": matched_idea_data.get('critiques', []),
                "evidence": matched_idea_data.get('evidence', []),
                "references": matched_idea_data.get('references', [])
            }

        processed_ideas.append((idea_info, exp_dir, matched_traj_file))

    print(f"Prepared {len(processed_ideas)} ideas for processing\n")

    for i, (idea_info, exp_dir, traj_file) in enumerate(processed_ideas, 1):
        idea_name = idea_info['name']
        print(f"\n[{i}/{len(processed_ideas)}] {idea_name}")

        try:
            if skip_existing:
                existing = [r for r in memory.records if r.name == idea_name and r.task == task_name]
                if existing:
                    print(f"  ⊘ Skipped (already in memory)")
                    skipped_count += 1
                    continue

            run_dirs = find_run_directories(exp_dir)
            if not run_dirs:
                print(f"  ✗ No run directories")
                failure_count += 1
                failed_ideas.append((idea_name, "No run directories"))
                continue

            baseline_dir = exp_dir / "run_0"
            if not baseline_dir.exists():
                print(f"  ✗ No baseline (run_0)")
                failure_count += 1
                failed_ideas.append((idea_name, "No baseline"))
                continue

            record = memory.save_experiment_result(
                idea=idea_info,
                results_dir=exp_dir,
                task_name=task_name,
                session_id=session_id or "offline_import",
                aggregation=aggregation,
                traj_path=traj_file
            )

            if record:
                success_count += 1
                print(f"  ✓ Saved")
            else:
                failure_count += 1
                failed_ideas.append((idea_name, "Failed to save"))
                print(f"  ✗ Failed to save")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failure_count += 1
            failed_ideas.append((idea_name, str(e)))
            continue

    print(f"\n{'='*50}")
    print(f"Results: {success_count} saved, {failure_count} failed, {skipped_count} skipped")
    print(f"Total records: {len(memory.records)}")

    if failed_ideas:
        print(f"\nFailed ({len(failed_ideas)}):")
        for idea_name, error in failed_ideas:
            error_display = error if len(error) <= 60 else error[:57] + "..."
            print(f"  - {idea_name}: {error_display}")

    stats = memory.get_statistics()
    if stats.get('total_records', 0) > 0:
        dist = stats.get('label_distribution', {})
        print(f"\nLabels: +{dist.get('positive', 0)} / 0:{dist.get('neutral', 0)} / -{dist.get('negative', 0)}")
        print(f"Success rate: {stats.get('success_rate', 'N/A')}")

    return (success_count, failure_count)


def auto_import_all_tasks(
    traj_base: Path,
    memory_dir: Path,
    aggregation: str = "best",
    label_threshold: float = 0.0,
    embedding_model_path: Optional[str] = None
) -> int:
    print(f"\nBatch Task Memory Import")
    print(f"Trajectory: {traj_base}")
    print(f"Memory base: {memory_dir}")
    print(f"Aggregation: {aggregation}, Threshold: {label_threshold}%\n")

    if not traj_base.exists():
        print(f"✗ Trajectory directory not found")
        return 1

    tasks = [d.name for d in traj_base.iterdir() if d.is_dir()]
    if not tasks:
        print(f"✗ No task directories found")
        return 1

    print(f"Found {len(tasks)} tasks: {', '.join(sorted(tasks))}\n")

    success_count = 0
    failure_count = 0
    results = {}

    for task in sorted(tasks):
        print(f"{'='*50}")
        print(f"Task: {task}")
        print(f"{'='*50}")

        try:
            success, failures = batch_save_from_traj(
                traj_dir=traj_base / task,
                task_name=task,
                memory_dir=str(memory_dir / task),
                aggregation=aggregation,
                skip_existing=True,
                label_threshold_percent=label_threshold,
                embedding_model_path=embedding_model_path
            )

            if failures == 0:
                success_count += 1
                results[task] = "success"
            else:
                failure_count += 1
                results[task] = "partial"

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failure_count += 1
            results[task] = "error"

    print(f"\n{'='*50}")
    print(f"Summary: {success_count} success, {failure_count} failed\n")

    for task in sorted(tasks):
        task_memory_dir = memory_dir / task
        if task_memory_dir.exists():
            try:
                records_file = task_memory_dir / "records.json"
                if records_file.exists():
                    with open(records_file, 'r') as f:
                        records_data = json.load(f)
                    record_count = len(records_data)
                else:
                    record_count = 0
                status = results.get(task, "unknown")
                print(f"  {task}: {record_count} records [{status}]")
            except Exception:
                print(f"  {task}: error loading [{results.get(task, 'unknown')}]")

    return 0 if failure_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Import experiment results to TaskMemoryLayer",
        epilog="""
        Examples:
        Single task:  python add_offline_mem.py --traj_dir ./IA15_traj/AutoPower --task AutoPower
        Batch import: python add_offline_mem.py --auto-import --traj_dir ./IA15_traj
        """
    )
    parser.add_argument("--auto-import", action="store_true",
                       help="Auto-import all tasks under traj_dir")
    parser.add_argument("--traj_dir", type=str,
                       help="Single task: trajectory dir (e.g., ./IA15_traj/AutoPower). Batch: base dir (e.g., ./IA15_traj)")
    parser.add_argument("--task", type=str,
                       help="Task name (required for single task import)")
    parser.add_argument("--memory_dir", type=str, default="./config/mem_store",
                       help="Single task: task memory dir. Batch: base dir (task subdirs auto-created)")
    parser.add_argument("--session_id", type=str, default=None,
                       help="Session ID (default: offline_import)")
    parser.add_argument("--aggregation", type=str, choices=["best", "avg", "last"], default="best",
                       help="Result aggregation method")
    parser.add_argument("--no-skip-existing", action="store_true",
                       help="Don't skip ideas already in memory")
    parser.add_argument("--label-threshold", type=float, default=0.0,
                       help="Label threshold percent (0.0 for binary, 5.0 for neutral zone)")
    parser.add_argument("--embedding-model", type=str, default=None,
                       help="Override embedding model path")

    args = parser.parse_args()

    try:
        if args.auto_import:
            if not args.traj_dir:
                print("✗ Error: --traj_dir required for --auto-import")
                sys.exit(1)
            exit_code = auto_import_all_tasks(
                traj_base=Path(args.traj_dir),
                memory_dir=Path(args.memory_dir),
                aggregation=args.aggregation,
                label_threshold=args.label_threshold,
                embedding_model_path=args.embedding_model
            )
            sys.exit(exit_code)

        elif args.traj_dir and args.task:
            success_count, failure_count = batch_save_from_traj(
                traj_dir=Path(args.traj_dir),
                task_name=args.task,
                memory_dir=args.memory_dir,
                session_id=args.session_id,
                aggregation=args.aggregation,
                skip_existing=not args.no_skip_existing,
                label_threshold_percent=args.label_threshold,
                embedding_model_path=args.embedding_model
            )
            sys.exit(0 if failure_count == 0 else 1)

        else:
            parser.print_help()
            print("\n✗ Error: Use --auto-import --traj_dir OR provide --traj_dir and --task")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
