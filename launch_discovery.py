"""
Launch InternAgent 
"""
import os
import os.path as osp
import sys
import json
import argparse
import asyncio
import logging
import glob
import shutil
import torch
import yaml
from datetime import datetime
from dotenv import load_dotenv

# Import MAS components
from internagent.stage import IdeaGenerator, ExperimentRunner
from typing import List, Dict, Any, Optional

# Long memory imports (optional - only if long_memory is available)
try:
    from internagent.mas.memory.long_memory import MemoryModule, ExperienceGenerator
    LONG_MEMORY_AVAILABLE = True
except ImportError:
    LONG_MEMORY_AVAILABLE = False
    MemoryModule = None
    ExperienceGenerator = None

load_dotenv()


# ============================================================================
# Task Type Detection & Normalization
# ============================================================================
def detect_task_type(task_dir: str) -> str:
    """
    Detect whether a task directory is an 'auto' task (has prompt.json) or
    a 'sci' task (has task_info.json for paper reproduction).

    Returns 'sci' or 'auto'.
    """
    if osp.exists(osp.join(task_dir, "task_info.json")):
        return "sci"
    return "auto"


def normalize_sci_task(task_dir: str, output_path: str) -> dict:
    """
    Read task_info.json + checklist.json from a sci_task directory and produce
    a synthetic prompt.json compatible with InternAgent's MAS pipeline.

    Args:
        task_dir: Path to the sci_task directory (e.g., tasks/sci_tasks/Chemistry_000)
        output_path: Where to write the synthesized prompt.json

    Returns:
        The synthesized prompt dict (also written to output_path)
    """
    # Load task_info.json
    task_info_path = osp.join(task_dir, "task_info.json")
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)

    # Load checklist.json
    checklist_path = osp.join(task_dir, "target_study", "checklist.json")
    checklist = []
    if osp.exists(checklist_path):
        with open(checklist_path, 'r') as f:
            checklist = json.load(f)

    # Extract domain from directory name (e.g., "Chemistry_000" → "Chemistry")
    dir_name = osp.basename(task_dir.rstrip('/\\'))
    domain = dir_name.split('_')[0] if '_' in dir_name else dir_name

    # Build data manifest
    data_items = task_info.get('data', [])
    if data_items:
        data_lines = [f"- {d['name']}: {d.get('description', '')}" for d in data_items]
        data_manifest = "\n".join(data_lines)
    else:
        data_manifest = "No data files specified."

    # Build checklist summary for constraints
    constraints = []
    for i, item in enumerate(checklist):
        w = item.get('weight', 0)
        t = item.get('type', 'text')
        preview = item.get('content', '')[:200]
        constraints.append(f"Item {i} (type={t}, weight={w:.2f}): {preview}")

    # Build composite task description
    task_description = (
        f"Reproduce the findings from a scientific paper in the {domain} domain.\n\n"
        f"## Research Task\n{task_info.get('task', '')}\n\n"
        f"## Available Data\n{data_manifest}\n\n"
        f"## Evaluation Criteria ({len(checklist)} checklist items)\n"
        + "\n".join(constraints) +
        "\n\n## Workspace Layout\n"
        "- Write analysis code in `code/`\n"
        "- Save intermediate outputs in `outputs/`\n"
        "- Write final report as `report/report.md`\n"
        "- Save generated figures in `report/images/`\n"
        "- Reference papers are in `related_work/`, raw data in `data/`\n"
    )

    prompt_data = {
        "system": f"You are a scientific researcher reproducing findings from a {domain} paper.",
        "task_description": task_description,
        "domain": domain,
        "background": f"Data files available:\n{data_manifest}",
        "constraints": constraints,
        "task_type": "sci",
    }

    with open(output_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)

    return prompt_data


# ============================================================================
# Helper Functions
# ============================================================================
def _find_best_experiment_result(results: List[Dict[str, Any]], logger) -> Optional[Dict[str, Any]]:
    """
    Find the best experiment result based on overall improvement rate.

    Args:
        results: List of experiment result dictionaries
        logger: Logger instance

    Returns:
        Best experiment result dict, or None if no successful experiments
    """
    successful_results = [r for r in results if r.get('success', False)]

    if not successful_results:
        return None

    # Find the result with highest overall_improvement_rate
    best_result = None
    best_performance = float('-inf')

    for result in successful_results:
        perf_data = result.get('performance', {})
        improvement_rate = perf_data.get('overall_improvement_rate', 0)

        if improvement_rate > best_performance:
            best_performance = improvement_rate
            best_result = result

    return best_result


def _update_baseline_for_incremental(best_code_path: str, logger, task_type: str = 'auto') -> bool:
    """
    Update baseline (code + final_info) with the best run's results for incremental mode.

    For 'auto' tasks updates:
      1. code/ - main code directory
      2. run_0/code/ - baseline code backup
      3. run_0/final_info.json - baseline metrics

    For 'sci' tasks additionally updates:
      4. outputs/ - intermediate outputs
      5. report/ - report and figures
    """
    run_dirs = sorted(glob.glob(osp.join(best_code_path, "run_[1-9]*")))

    if not run_dirs:
        logger.warning(f"No run directories found in {best_code_path}")
        return False

    # Find the last valid run (assumed to be the best after iterations)
    best_run_dir = None
    best_final_info = None

    for run_dir in run_dirs:
        final_info_path = osp.join(run_dir, "final_info.json")
        if osp.exists(final_info_path):
            try:
                with open(final_info_path, 'r') as f:
                    best_final_info = json.load(f)
                best_run_dir = run_dir
            except Exception as e:
                logger.warning(f"Failed to load {final_info_path}: {e}")

    if not best_run_dir or not best_final_info:
        logger.warning(f"No valid final_info.json found in run directories")
        return False

    # Update run_0/final_info.json
    run0_dir = osp.join(best_code_path, "run_0")
    os.makedirs(run0_dir, exist_ok=True)
    run0_final_info = osp.join(run0_dir, "final_info.json")

    try:
        with open(run0_final_info, 'w') as f:
            json.dump(best_final_info, f, indent=2)
        logger.info(f"Updated baseline metrics: {osp.join(best_run_dir, 'final_info.json')} -> {run0_final_info}")
    except Exception as e:
        logger.error(f"Failed to update baseline metrics: {e}")
        return False

    # Update code/ directory with best run's code
    best_run_code_dir = osp.join(best_run_dir, "code")
    main_code_dir = osp.join(best_code_path, "code")

    if osp.exists(best_run_code_dir) and osp.isdir(best_run_code_dir):
        try:
            if osp.exists(main_code_dir):
                shutil.rmtree(main_code_dir)
            shutil.copytree(best_run_code_dir, main_code_dir)
            logger.info(f"Updated main code: {best_run_code_dir} -> {main_code_dir}")
        except Exception as e:
            logger.error(f"Failed to update main code: {e}")
            return False

        # Also update run_0/code/ to keep baseline backup in sync
        run0_code_dir = osp.join(run0_dir, "code")
        try:
            if osp.exists(run0_code_dir):
                shutil.rmtree(run0_code_dir)
            shutil.copytree(best_run_code_dir, run0_code_dir)
            logger.info(f"Updated baseline code backup: {best_run_code_dir} -> {run0_code_dir}")
        except Exception as e:
            logger.warning(f"Failed to update baseline code backup: {e}")
            # Non-fatal, continue

    # For sci tasks: also propagate outputs/ and report/ from the best run
    if task_type == 'sci':
        for dir_name in ['outputs', 'report']:
            best_run_dir_src = osp.join(best_run_dir, dir_name)
            main_dir_dst = osp.join(best_code_path, dir_name)
            if osp.exists(best_run_dir_src) and osp.isdir(best_run_dir_src):
                try:
                    if osp.exists(main_dir_dst):
                        shutil.rmtree(main_dir_dst)
                    shutil.copytree(best_run_dir_src, main_dir_dst)
                    logger.info(f"Updated {dir_name}/: {best_run_dir_src} -> {main_dir_dst}")
                except Exception as e:
                    logger.warning(f"Failed to update {dir_name}/: {e}")

    return True


def _generate_experiences_for_round(args, memory, session_id, logger) -> bool:
    """
    Generate experiences from a single round's experiments.

    Args:
        args: Command line arguments
        memory: MemoryModule instance
        session_id: Current session ID
        logger: Logger instance

    Returns:
        True if experiences were generated successfully, False otherwise
    """
    if memory is None:
        return False

    try:
        from internagent.mas.memory.long_memory import ExperienceGenerator
    except ImportError:
        logger.warning("Long memory not available, skipping experience generation")
        return False

    # Load domain from prompt.json (use args.prompt_path which points to launch directory)
    prompt_path = getattr(args, 'prompt_path', None) or osp.join(args.task_dir, "prompt.json")
    domain = "machine learning"  # default
    if osp.exists(prompt_path):
        try:
            with open(prompt_path, 'r') as f:
                prompt_data = json.load(f)
                domain = prompt_data.get("domain", domain)
        except Exception as e:
            logger.warning(f"Failed to load domain from prompt.json: {e}, using default")

    # Load ideas and notes from the current session
    session_dir = osp.join(args.output_dir, session_id)
    if osp.exists(session_dir):
        # Load ideas from this session
        ideas_path = osp.join(session_dir, "ideas.json")
        if osp.exists(ideas_path):
            memory.load_idea_generation_output(ideas_path)

        # Load experiment notes from this session
        memory.load_all_notes_from_directory(session_dir, args.task_name)

    summary = memory.get_memory_summary()
    logger.info(f"Loaded {summary['total_ideas']} ideas and {summary['total_experiments']} experiments")

    if summary['total_experiments'] > 0:
        experience_generator = ExperienceGenerator(logger=logger, config_path=args.config)

        result = asyncio.run(
            experience_generator.generate_experiences_from_memory(
                memory=memory,
                task_domain=domain,
                output_dir=args.base_output_dir
            )
        )

        new_experiences = result.get("new_experiences", [])
        updated_library = result.get("updated_library", [])

        logger.info(f"Generated {len(new_experiences)} new experiences")
        logger.info(f"Experience library now has {len(updated_library)} total experiences")
        return True
    else:
        logger.warning("No experiments found in this round, skipping experience generation")
        return False


# ============================================================================
# Logging Configuration
# ============================================================================
def setup_logging():
    """Setup logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_internagent.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger("InternAgent")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Integrated InternAgent Pipeline: Idea Generation + Experiment Execution"
    )
    
    # ========================================
    # Task Configuration
    # ========================================
    task_group = parser.add_argument_group('Task Configuration')
    task_group.add_argument(
        "--task",
        type=str,
        default="AutoSeg",
        help="Task name or path to task directory. If it's a name, will use tasks/{task}; if it's a path, will use it directly"
    )
    task_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results output directory (defaults to results/{task_name})"
    )
    task_group.add_argument(
        "--config",
        type=str,
        default='config/default_config.yaml',
        help="Path to configuration file"
    )
    
    # ========================================
    # Idea Generation Phase
    # ========================================
    idea_group = parser.add_argument_group('Idea Generation Phase')
    idea_group.add_argument(
        "--skip_idea_generation",
        action="store_true",
        help="Skip idea generation and use existing ideas from idea_path"
    )
    idea_group.add_argument(
        "--idea_path",
        type=str,
        default=None,
        help="Path to existing ideas JSON (used when skip_idea_generation=True)"
    )
    idea_group.add_argument(
        "--ref_code_path",
        type=str,
        default=None,
        help="Baseline reference code path (defaults to {task_dir}/experiment.py)"
    )
    idea_group.add_argument(
        "--offline_feedback",
        type=str,
        default='config/feedback_global.json',
        help="Offline feedback file for idea generation"
    )
    
    # ========================================
    # Experiment Execution Config
    # ========================================
    exp_group = parser.add_argument_group('Experiment Execution Phase')
    exp_group.add_argument(
        "--mode",
        type=str,
        default="experiment",
        choices=["experiment", "report"],
        help="Execution mode: 'experiment' for running experiments, 'report' for generating reports only"
    )
    exp_group.add_argument(
        "--exp_backend",
        type=str,
        required=True,
        default="claudecode",
        choices=["openhands", "claudecode", "iflow"],
        help="Experiment backend to use (required for experiment mode)"
    )
    # Note: Model configuration is handled through config file (experiment.model)
    # Note: GPU configuration is handled through CUDA_VISIBLE_DEVICES environment variable or auto-detection
    # Note: Parallel execution is configured in config file (experiment.max_parallel_experiments and experiment.gpu_per_experiment)
    # Note: OpenHands-specific configuration (mount_paths, uri_prefix) is handled through config file

    # ========================================
    # Resume Configuration
    # ========================================
    resume_group = parser.add_argument_group('Resume Configuration')
    resume_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing launch folder (e.g., results/TaskName/20260112_101127_launch) to resume from last completed loop"
    )

    return parser.parse_args()


# ============================================================================
# Resume State Detection
# ============================================================================
def load_resume_state(resume_path: str, logger) -> dict:
    """
    Load resume state from an existing launch folder.

    Args:
        resume_path: Path to the launch folder (e.g., results/TaskName/20260112_101127_launch)
        logger: Logger instance

    Returns:
        Dictionary containing:
        - completed_rounds: Number of completed rounds
        - all_round_results: Results from completed rounds
        - all_session_ids: Session IDs from completed rounds
        - best_code_path: Best code path from incremental mode (if applicable)
        - best_overall_performance: Best performance from incremental mode (if applicable)
        - launch_id: The launch ID from the folder name
        - config_overrides: Any config values that should be preserved
    """
    resume_state = {
        'completed_rounds': 0,
        'all_round_results': [],
        'all_session_ids': [],
        'best_code_path': None,
        'best_overall_performance': None,
        'launch_id': None,
        'loop_mode': 'fresh',
        'loop_rounds': 1,
        'original_task_dir': None,
        'base_output_dir': None,
        'prompt_path': None
    }

    if not osp.exists(resume_path):
        logger.error(f"Resume path does not exist: {resume_path}")
        return resume_state

    # Extract launch_id from folder name
    resume_state['launch_id'] = osp.basename(resume_path)
    resume_state['base_output_dir'] = osp.dirname(resume_path)

    # Try to load discovery_summary.json
    summary_path = osp.join(resume_path, "discovery_summary.json")
    if osp.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            resume_state['completed_rounds'] = summary.get('total_rounds', 0)
            resume_state['all_round_results'] = summary.get('rounds', [])
            resume_state['all_session_ids'] = summary.get('sessions', [])
            resume_state['loop_mode'] = summary.get('loop_mode', 'fresh')
            resume_state['loop_rounds'] = summary.get('loop_rounds', 1)
            resume_state['original_task_dir'] = summary.get('original_task_dir', summary.get('task_dir'))

            # Load incremental mode state if available
            if 'incremental_mode' in summary:
                inc_state = summary['incremental_mode']
                resume_state['best_code_path'] = inc_state.get('final_best_code_path')
                resume_state['best_overall_performance'] = inc_state.get('final_best_performance')

            logger.info(f"Loaded resume state from discovery_summary.json")
            logger.info(f"  Completed rounds: {resume_state['completed_rounds']}/{resume_state['loop_rounds']}")
            logger.info(f"  Sessions: {resume_state['all_session_ids']}")

        except Exception as e:
            logger.warning(f"Failed to load discovery_summary.json: {e}")
            # Fall back to scanning directories
            resume_state = _scan_completed_rounds(resume_path, resume_state, logger)
    else:
        # No summary file, scan directories to detect completed rounds
        logger.info("No discovery_summary.json found, scanning directories...")
        resume_state = _scan_completed_rounds(resume_path, resume_state, logger)

    # Check if prompt.json exists in launch folder (for evolved prompt)
    prompt_path = osp.join(resume_path, "prompt.json")
    if osp.exists(prompt_path):
        resume_state['prompt_path'] = prompt_path
        logger.info(f"Found evolved prompt at: {prompt_path}")

    return resume_state


def _scan_completed_rounds(resume_path: str, resume_state: dict, logger) -> dict:
    """
    Scan directories to detect completed rounds when discovery_summary.json is not available.

    Args:
        resume_path: Path to the launch folder
        resume_state: Current resume state dictionary
        logger: Logger instance

    Returns:
        Updated resume state dictionary
    """
    # Find all session directories
    session_dirs = glob.glob(osp.join(resume_path, "session_*"))
    session_dirs.sort()  # Sort by name (which includes timestamp)

    completed_rounds = 0
    for session_dir in session_dirs:
        session_id = osp.basename(session_dir)

        # Check if this session has completed experiments (has experiment folders with final_info.json)
        experiment_folders = [d for d in os.listdir(session_dir)
                            if osp.isdir(osp.join(session_dir, d)) and not d.startswith('session_')]

        has_completed_experiments = False
        for exp_folder in experiment_folders:
            # Check for final_info.json in any run folder
            run_folders = glob.glob(osp.join(session_dir, exp_folder, "run_*", "final_info.json"))
            if run_folders:
                has_completed_experiments = True
                break

        if has_completed_experiments:
            completed_rounds += 1
            resume_state['all_session_ids'].append(session_id)
            logger.info(f"  Found completed session: {session_id}")

    resume_state['completed_rounds'] = completed_rounds
    logger.info(f"Detected {completed_rounds} completed rounds from directory scan")

    return resume_state


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    logger = setup_logging()
    args = parse_arguments()

    # ========================================
    # Resume Mode Handling
    # ========================================
    resume_state = None
    start_round = 1  # Default: start from round 1

    if args.resume:
        logger.info("=" * 80)
        logger.info("RESUME MODE ENABLED")
        logger.info(f"Resuming from: {args.resume}")
        logger.info("=" * 80)

        resume_state = load_resume_state(args.resume, logger)

        if resume_state['completed_rounds'] == 0:
            logger.warning("No completed rounds found, starting fresh from round 1")
        else:
            start_round = resume_state['completed_rounds'] + 1
            logger.info(f"Will resume from round {start_round}")

    # ========================================
    # Setup Task Directory
    # ========================================
    # If task is a path (contains / or \), use it directly; otherwise use tasks/{task}
    if '/' in args.task or '\\' in args.task or osp.isdir(args.task):
        args.task_dir = args.task
        args.task_name = osp.basename(args.task.rstrip('/\\'))
    else:
        args.task_dir = osp.join("tasks", args.task)
        args.task_name = args.task

    if not osp.exists(args.task_dir):
        raise FileNotFoundError(f"Task directory not found: {args.task_dir}")

    # Detect task type: 'sci' (has task_info.json) or 'auto' (has prompt.json)
    args.task_type = detect_task_type(args.task_dir)

    # Setup reference code path
    if args.ref_code_path is None:
        if args.task_type == 'sci':
            args.ref_code_path = None  # No reference code for sci tasks
        else:
            args.ref_code_path = osp.join(args.task_dir, "code")
    
    # ========================================
    # Setup Output Directory
    # ========================================
    if args.resume and resume_state and resume_state['launch_id']:
        # Resume mode: use existing launch folder
        launch_id = resume_state['launch_id']
        base_output_dir = resume_state['base_output_dir']
        args.output_dir = args.resume
        args.base_output_dir = base_output_dir

        # Use existing prompt.json from launch folder if available
        if resume_state['prompt_path'] and osp.exists(resume_state['prompt_path']):
            args.prompt_path = resume_state['prompt_path']
        else:
            # Fall back: regenerate for sci tasks, use original for auto tasks
            if args.task_type == 'sci':
                fallback_prompt_path = osp.join(args.output_dir, "prompt.json")
                normalize_sci_task(args.task_dir, fallback_prompt_path)
                args.prompt_path = fallback_prompt_path
                logger.info(f"Regenerated sci_task prompt.json for resume: {fallback_prompt_path}")
            else:
                args.prompt_path = osp.join(args.task_dir, "prompt.json")

        logger.info(f"Resuming with existing launch folder: {launch_id}")
    else:
        # Fresh start: create new launch folder
        # Structure: results/{task_name}/{launch_id}/session_xxx/...
        launch_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        launch_id = f"{launch_time}_launch"

        if args.output_dir is None:
            base_output_dir = osp.join("results", args.task_name)
        else:
            base_output_dir = osp.join("results", args.output_dir)

        # Create base output directory (for shared resources like experience_library)
        os.makedirs(base_output_dir, exist_ok=True)

        # Create launch folder under base output directory
        args.output_dir = osp.join(base_output_dir, launch_id)
        os.makedirs(args.output_dir, exist_ok=True)

        # Store base_output_dir for shared resources
        args.base_output_dir = base_output_dir

        # Copy or generate prompt.json in launch directory
        launch_prompt_path = osp.join(args.output_dir, "prompt.json")
        if args.task_type == 'sci':
            # Generate synthetic prompt.json from task_info.json + checklist.json
            normalize_sci_task(args.task_dir, launch_prompt_path)
            args.prompt_path = launch_prompt_path
            logger.info(f"Generated synthetic prompt.json for sci_task: {launch_prompt_path}")
        else:
            original_prompt_path = osp.join(args.task_dir, "prompt.json")
            if osp.exists(original_prompt_path):
                shutil.copy2(original_prompt_path, launch_prompt_path)
                args.prompt_path = launch_prompt_path
            else:
                raise FileNotFoundError(f"prompt.json not found in task directory: {original_prompt_path}")

    # ========================================
    # Load Configuration
    # ========================================
    config = {}
    if args.config and osp.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Get loop_rounds and loop_mode from config
    # Note: Always use config for loop_rounds/loop_mode, resume_state only provides completed round info
    # This allows users to change loop_rounds when resuming (e.g., extend from 5 to 10 rounds)
    loop_rounds = config.get('workflow', {}).get('loop_rounds', 1)
    loop_mode = config.get('workflow', {}).get('loop_mode', 'fresh')

    # If resume_state has valid loop settings from discovery_summary.json, use them as reference
    # but still allow config to override (for extending runs)
    if args.resume and resume_state:
        resume_loop_rounds = resume_state.get('loop_rounds', 0)
        resume_loop_mode = resume_state.get('loop_mode')

        # If config doesn't specify loop_rounds (using default), use resume state's value
        if resume_loop_rounds > 0 and loop_rounds == 1:
            loop_rounds = resume_loop_rounds
            logger.info(f"Using loop_rounds={loop_rounds} from resume state")

        # Use resume state's loop_mode if config doesn't override
        if resume_loop_mode and resume_loop_mode != 'fresh':
            loop_mode = resume_loop_mode
            logger.info(f"Using loop_mode={loop_mode} from resume state")

    # If skip_idea_generation is True, only run one round (no iterative discovery)
    if args.skip_idea_generation:
        loop_rounds = 1
        logger.info("Skip idea generation is enabled, running single round only")

    # Validate loop_mode
    if loop_mode not in ['fresh', 'incremental']:
        logger.warning(f"Invalid loop_mode '{loop_mode}', defaulting to 'fresh'")
        loop_mode = 'fresh'

    # Check if all rounds are already completed
    if start_round > loop_rounds:
        logger.info("=" * 80)
        logger.info("All rounds already completed! Nothing to resume.")
        logger.info(f"Completed: {start_round - 1}/{loop_rounds} rounds")
        logger.info("=" * 80)
        return

    logger.info("=" * 80)
    logger.info("InternAgent Pipeline Started" + (" (RESUMED)" if args.resume else ""))
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Task Type: {args.task_type.upper()}")
    logger.info(f"Task Directory: {args.task_dir}")
    logger.info(f"Experiment Backend: {args.exp_backend}")
    logger.info(f"Launch ID: {launch_id}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Prompt Path: {args.prompt_path}")
    logger.info(f"Shared Resources: {args.base_output_dir}")
    logger.info(f"Loop Rounds: {loop_rounds}")
    logger.info(f"Loop Mode: {loop_mode.upper()}")
    if args.resume:
        logger.info(f"Resume: Starting from round {start_round}/{loop_rounds}")
    if loop_mode == 'incremental':
        logger.info("  → Each round starts from the best result of previous rounds")
    else:
        logger.info("  → Each round starts fresh from the original baseline")
    logger.info("=" * 80)

    # Initialize Long Memory Module (for IdeaGraph and experience tracking)
    memory = None
    if LONG_MEMORY_AVAILABLE:
        try:
            long_memory_config = config.get("memory", {}).get("long_memory", {})
            if long_memory_config.get("enabled", True):
                memory = MemoryModule(logger=logger)
                logger.info("Long memory module initialized")

                # Load historical data from previous launches (for experience continuity)
                logger.info("Loading historical ideas and experiment results...")

                # Load all ideas from previous sessions in base_output_dir
                ideas_files = glob.glob(osp.join(base_output_dir, "*_launch", "session_*", "ideas.json"))
                ideas_files.extend(glob.glob(osp.join(base_output_dir, "session_*", "ideas.json")))  # Legacy format
                for ideas_file in ideas_files:
                    memory.load_idea_generation_output(ideas_file)

                # Load all experiment notes from previous sessions in base_output_dir
                memory.load_all_notes_from_directory(base_output_dir, args.task_name)

                # Log summary
                summary = memory.get_memory_summary()
                logger.info(f"Historical data loaded: {summary['total_ideas']} ideas, {summary['total_experiments']} experiments")
        except Exception as e:
            logger.warning(f"Failed to initialize long memory module: {e}")

    # Store results from all rounds (initialize from resume state if available)
    if args.resume and resume_state:
        all_round_results = resume_state.get('all_round_results', [])
        all_session_ids = resume_state.get('all_session_ids', [])
        logger.info(f"Restored {len(all_round_results)} completed rounds from resume state")
    else:
        all_round_results = []
        all_session_ids = []

    # Track the best code path for incremental mode
    original_task_dir = args.task_dir
    if args.resume and resume_state and resume_state.get('best_code_path'):
        best_code_path = resume_state['best_code_path']
        best_overall_performance = resume_state.get('best_overall_performance')
        logger.info(f"Restored best code path from resume: {best_code_path}")
    else:
        best_code_path = original_task_dir
        best_overall_performance = None  # Will store (improvement_rate, code_path)

    # Multi-round discovery loop (start from start_round for resume)
    base_code_dir = None  # Track the code directory for incremental mode
    for round_num in range(start_round, loop_rounds + 1):
        # In incremental mode, use code from best result, but keep original task_dir for prompt.json
        if loop_mode == 'incremental' and round_num > 1 and best_code_path != original_task_dir:
            logger.info(f"Incremental Mode: Using best result from previous rounds as baseline")
            logger.info(f"  Previous best code: {best_code_path}")
            base_code_dir = best_code_path
        else:
            base_code_dir = args.task_dir

        # Debug: Ensure base_code_dir is valid
        if not base_code_dir:
            logger.error(f"ERROR: base_code_dir is empty! args.task_dir={args.task_dir}, best_code_path={best_code_path}")
            raise ValueError("base_code_dir cannot be empty")

        logger.info(f"Base code directory: {base_code_dir}")
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STARTING DISCOVERY ROUND {round_num}/{loop_rounds}")
        logger.info("=" * 80)

        # Step 1: Idea Generation
        session_id = None  # Initialize session_id
        if args.skip_idea_generation and round_num == 1:
            logger.info("Skipping idea generation, loading existing ideas...")
            if not args.idea_path or not osp.exists(args.idea_path):
                raise FileNotFoundError(f"Idea path not found: {args.idea_path}")

            with open(args.idea_path, 'r') as f:
                ideas_data = json.load(f)

            # Extract top hypotheses
            if 'hypotheses' in ideas_data and 'top_hypotheses' in ideas_data:
                top_ideas = [
                    item for item in ideas_data['hypotheses']
                    if item['id'] in ideas_data['top_hypotheses']
                ]
            else:
                # Assume it's already a list of ideas
                top_ideas = ideas_data

            logger.info(f"Loaded {len(top_ideas)} ideas from {args.idea_path}")
            session_json = args.idea_path

            # Try to extract session_id from path (if it contains session_)
            import re
            match = re.search(r'session_(\d+)', args.idea_path)
            if match:
                session_id = match.group(1)

        else:
            logger.info(f"Starting idea generation with MAS (Round {round_num})...")
            idea_generator = IdeaGenerator(args, logger, round_num=round_num, config=config)

            try:
                top_ideas, session_json = asyncio.run(idea_generator.generate_ideas())
            except Exception as e:
                logger.error(f"Idea generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            # Store session_id for later use (already contains 'session_' prefix)
            session_id = idea_generator.session_id

            # Save ideas in standard format (also in session directory)
            session_dir = osp.join(args.output_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            ideas_output = osp.join(session_dir, "ideas.json")

            aligned_ideas = [idea['refined_method_details'] for idea in top_ideas]
            with open(ideas_output, 'w') as f:
                json.dump(aligned_ideas, f, indent=4)

            logger.info(f"Ideas saved to {ideas_output}")

            # Clear memory cache after idea generation to free GPU memory
            try:
                from internagent.mas.tools.memory_retrieval import clear_memory_cache
                clear_memory_cache()
            except Exception as e:
                logger.warning(f"Failed to clear memory cache: {e}")

        # Step 2: Experiment Execution or Report Generation
        logger.info("=" * 80)

        if args.mode == "report":
            logger.info("Starting report generation")
            logger.info(f"Number of ideas to process: {len(top_ideas)}")
            logger.info(f"Reports will be saved to: {args.output_dir}")
            logger.info("=" * 80)

            from internagent.stage import ReportWriter

            report_writer = ReportWriter(args, logger, config)

            try:
                results = report_writer.generate_reports(
                    results_dir=args.output_dir,
                    ideas=top_ideas
                )
            except Exception as e:
                logger.error(f"Report generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        else:  # experiment mode
            if not args.exp_backend:
                logger.error("--exp_backend is required for experiment mode")
                sys.exit(1)

            logger.info(f"Starting experiment execution with {args.exp_backend} backend")
            logger.info(f"Number of ideas to test: {len(top_ideas)}")
            logger.info("=" * 80)

            # Validate backend-specific requirements
            if args.exp_backend == "openhands":
                openhands_config = config.get("experiment", {}).get("openhands", {})
                mount_paths = openhands_config.get("mount_paths", [])
                uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")

                if not mount_paths:
                    logger.warning("No mount paths specified in config for OpenHands backend")
                else:
                    logger.info(f"OpenHands mount paths: {mount_paths}")
                logger.info(f"OpenHands URI prefix: {uri_prefix}")

            # Run experiments
            experiment_runner = ExperimentRunner(args, logger, config, session_id=session_id, base_code_dir=base_code_dir)

            # Use session_dir if session_id exists, otherwise use args.output_dir
            if session_id:
                # Session directory was already created during idea generation
                # session_id already contains 'session_' prefix
                experiment_results_dir = osp.join(args.output_dir, session_id)
            else:
                experiment_results_dir = args.output_dir

            try:
                results = experiment_runner.run_experiments(
                    base_dir=base_code_dir,
                    results_dir=experiment_results_dir,
                    ideas=top_ideas
                )
            except Exception as e:
                logger.error(f"Experiment execution failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        # Store round results
        round_result = {
            'round': round_num,
            'session_id': session_id,
            'results': results,
            'successful': sum(1 for r in results if r['success']),
            'failed': len(results) - sum(1 for r in results if r['success'])
        }
        all_round_results.append(round_result)
        all_session_ids.append(session_id)

        # Log round summary
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num}/{loop_rounds} COMPLETED")
        logger.info(f"Session: {session_id}")
        logger.info(f"Successful: {round_result['successful']}/{len(results)}")
        logger.info(f"Failed: {round_result['failed']}/{len(results)}")
        logger.info("=" * 80)

        # Generate experiences after each round (for prompt evolution in subsequent rounds)
        if LONG_MEMORY_AVAILABLE and memory is not None:
            logger.info(f"Generating experiences from Round {round_num}...")
            _generate_experiences_for_round(args, memory, session_id, logger)

        # If not the last round, update for next round
        if round_num < loop_rounds:
            logger.info(f"Preparing for Round {round_num + 1}...")

            # In incremental mode, find the best result and update baseline for next round
            if loop_mode == 'incremental':
                logger.info(f"Incremental Mode: Finding best result from Round {round_num}...")
                round_best = _find_best_experiment_result(results, logger)

                if round_best:
                    round_best_perf = round_best.get('performance', {}).get('overall_improvement_rate', 0)
                    round_best_path = round_best.get('code_path', '')

                    logger.info(f"  Round {round_num} best: {round_best['idea_name']} "
                              f"(improvement: {round_best_perf:+.2f}%)")

                    if best_overall_performance is None or round_best_perf > best_overall_performance:
                        best_overall_performance = round_best_perf
                        best_code_path = round_best_path
                        logger.info(f"  New best found! Updating baseline for next round...")
                        _update_baseline_for_incremental(best_code_path, logger, task_type=args.task_type)
                    else:
                        logger.info(f"  Current best remains: {best_code_path} "
                                  f"(improvement: {best_overall_performance:+.2f}%)")
                else:
                    logger.warning(f"  No successful experiments in Round {round_num}")

            logger.info(f"Starting Round {round_num + 1} in next iteration...")

    # Note: Experience generation now happens after each round (see loop above)
    # This ensures experience_library is up-to-date for prompt evolution in subsequent rounds

    # Step 3: Final Summary (After all rounds)
    logger.info("")
    logger.info("=" * 80)
    if args.mode == "report":
        logger.info("ALL REPORT GENERATION ROUNDS COMPLETED")
    else:
        logger.info("ALL DISCOVERY ROUNDS COMPLETED")
    logger.info("=" * 80)

    # Aggregate statistics across all rounds
    total_successful = sum(round_result['successful'] for round_result in all_round_results)
    total_ideas = sum(len(round_result['results']) for round_result in all_round_results)
    total_failed = total_ideas - total_successful

    logger.info(f"Total Rounds: {len(all_round_results)}")
    logger.info(f"Loop Mode: {loop_mode.upper()}")
    if loop_mode == 'incremental' and best_code_path != original_task_dir:
        logger.info(f"Final Best Code Path: {best_code_path}")
        logger.info(f"Final Best Performance: {best_overall_performance:+.2f}%")
    logger.info(f"Sessions: {', '.join(all_session_ids)}")

    if args.mode == "report":
        logger.info(f"Total Reports Generated: {total_ideas}")
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Failed: {total_failed}")
    else:
        logger.info(f"Total Ideas Tested: {total_ideas}")
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Failed: {total_failed}")

    # Print detailed results per round
    logger.info("\nDetailed Results by Round:")
    for round_result in all_round_results:
        logger.info(f"\n  Round {round_result['round']} (Session: {round_result['session_id']}):")
        for i, result in enumerate(round_result['results'], 1):
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            logger.info(f"    {i}. {result['idea_name']}: {status}")
            if 'error' in result:
                logger.info(f"       Error: {result['error']}")
            elif 'report_path' in result:
                logger.info(f"       Report: {result['report_path']}")

    # Save comprehensive summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'launch_id': launch_id,
        'task': args.task_name,
        'task_dir': args.task_dir,
        'task_type': args.task_type,
        'original_task_dir': original_task_dir,
        'mode': args.mode,
        'output_dir': args.output_dir,
        'base_output_dir': args.base_output_dir,
        'skip_idea_generation': args.skip_idea_generation,
        'total_rounds': len(all_round_results),
        'loop_rounds': loop_rounds,
        'loop_mode': loop_mode,
        'sessions': all_session_ids,
        'total_ideas': total_ideas,
        'total_successful': total_successful,
        'total_failed': total_failed,
        'rounds': all_round_results
    }

    # Add incremental mode specific info
    if loop_mode == 'incremental':
        summary['incremental_mode'] = {
            'final_best_code_path': best_code_path,
            'final_best_performance': best_overall_performance
        }

    if args.mode == "experiment":
        summary['exp_backend'] = args.exp_backend
        summary['model'] = (
            config.get("experiment", {}).get("model") or
            "anthropic/claude-3-7-sonnet-20250219"
        )

    summary_path = osp.join(args.output_dir, "discovery_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("=" * 80)
    logger.info("All done!")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiscovery pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
