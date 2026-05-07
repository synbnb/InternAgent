import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from internagent.prompts import (
    CODER_PROMPT_OPENHANDS,
    NEXT_EXPERIMENT_PROMPT,
    CODER_PROMPT_SCI_TASK,
    NEXT_EXPERIMENT_PROMPT_SCI,
)

import threading

MAX_ITERS = 5
MAX_RUNS = 5  # Default value, can be overridden by config
MAX_STDERR_OUTPUT = 30000

SCI_SYMLINK_DIRS = {'data', 'related_work', 'target_study'}

def extract_idea_info(idea):
    """Extract idea information from different formats"""
    # Try refined_method_details first (from full MAS pipeline)
    if 'refined_method_details' in idea and idea['refined_method_details']:
        details = idea['refined_method_details']
        return {
            'name': details.get('name', 'unnamed_idea'),
            'title': details.get('title', 'Untitled'),
            'description': details.get('description', ''),
            'method': details.get('method', '')
        }

    # Fall back to method_details (from method development only)
    elif 'method_details' in idea and idea['method_details']:
        details = idea['method_details']
        return {
            'name': details.get('name', 'unnamed_idea'),
            'title': details.get('title', 'Untitled'),
            'description': details.get('description', ''),
            'method': details.get('method', '')
        }

    # Fall back to basic idea structure (from JSON files)
    else:
        # Handle different possible field names
        name = idea.get('name') or idea.get('title') or 'unnamed_idea'
        title = idea.get('title') or idea.get('name') or 'Untitled'
        description = idea.get('description') or idea.get('content') or ''
        method = idea.get('method') or ''

        return {
            'name': name[:50] if name else 'unnamed_idea',  # Limit name length
            'title': title,
            'description': description,
            'method': method
        }


def info_traceback(stderr):
    pattern = r'File "(.*)", line (\d+), in (.+)\n (.*)'
    matches = re.findall(pattern, stderr)
    match = re.search(rf'\w*Error\w*(.*)', stderr, re.DOTALL)
    message = match.group(1).strip() if match else "Unknown error"
    externel = []
    for match in matches:
        if match[0].split('/')[-1] == 'experiment.py':
            continue
        else:
            externel.append(match)
    for e in externel:
        matches.remove(e)

    return matches, message


class ClaudeCodeRunner:
    """Claude Code Runner class to handle interactions with Claude CLI"""

    def __init__(self, proxy_settings=None, model='claude-sonnet-4-5-20250929'):
        """
        Initialize the Claude Code Runner

        Args:
            proxy_settings: Optional dictionary with HTTP_PROXY and HTTPS_PROXY settings
            model: Model name to use (default: claude-sonnet-4-5-20250929)
        """
        self.proxy_settings = proxy_settings or {}
        self.model = model
        
    def run(self, prompt, cwd=None):
        """
        Run Claude Code with the given prompt
        
        Args:
            prompt: The prompt to send to Claude
            cwd: The working directory for Claude to operate in
            
        Returns:
            The stdout output from Claude
        """
        # Set proxy environment variables
        env = os.environ.copy()
        for key, value in self.proxy_settings.items():
            env[key] = value
        
        # Enhanced logging - Log start time and command
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Running Claude CLI with prompt: {prompt}..."
        logger.info(log_message)
            
        # Run Claude with acceptEdits permission mode
        # Capture both stdout and stderr so we can log them properly
        result = subprocess.run(
            ['claude', '--permission-mode', 'acceptEdits', '--model', self.model, prompt],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env
        )

        # Enhanced logging - Log completion and output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Claude command completed with return code: {result.returncode}"
        logger.info(log_message)

        # Log output summary
        if result.stdout:
            output_summary = f"Claude output: {result.stdout}..."
            logger.info(output_summary)

        # If there was an error, log that too
        if result.returncode != 0 or result.stderr:
            error_message = f"Claude CLI error (return code {result.returncode}): {result.stderr}"
            logger.error(error_message)
        
        # Return the stdout result
        return result.stdout

def run_experiment(folder_name, run_num, timeout=None, gpu_ids=None, log_file=None, task_type='auto'):
    """
    Run experiment with the code in the folder

    Args:
        folder_name: The folder containing the code
        run_num: The run number
        timeout: Maximum execution time in seconds
        gpu_ids: GPU IDs to use (string like "0,1" or None for CPU)
        log_file: Optional file object to write logs to

    Returns:
        Tuple of (return_code, next_prompt, traceback, message)
    """
    def log_message(msg):
        """Write message to both stdout and log file"""
        print(msg)
        sys.stdout.flush()
        if log_file:
            try:
                log_file.write(msg + "\n")
                log_file.flush()
            except (ValueError, OSError):
                pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] Starting experiment run {run_num} in {folder_name}")

    cwd = osp.abspath(folder_name)
    # Create run directory
    run_dir = osp.join(cwd, f"run_{run_num}")
    if not osp.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    # Copy all files from the main folder to the run directory
    logger.info(f"Copying files to run directory: {run_dir}")
    for item in os.listdir(cwd):
        if item.startswith("run_") or item == ".git":
            continue
        src = osp.join(cwd, item)
        dst = osp.join(run_dir, item)
        if osp.isdir(src):
            if task_type == 'sci' and item in SCI_SYMLINK_DIRS:
                if osp.exists(dst) or osp.islink(dst):
                    os.remove(dst) if osp.islink(dst) else shutil.rmtree(dst)
                os.symlink(osp.abspath(src), dst)
            else:
                if osp.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # LAUNCH COMMAND
    command = ["bash", f"launcher.sh"]
    logger.info(f"Running command: {command} in {run_dir}")

    # Prepare environment variables (thread-safe copy)
    env = os.environ.copy()
    if gpu_ids:
        env['CUDA_VISIBLE_DEVICES'] = gpu_ids
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_ids} for run {run_num}")

    log_message(f"\n{'='*80}")
    log_message(f"Running experiment: run_{run_num}")
    log_message(f"{'='*80}\n")

    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env
        )

        # Stream output in a background thread so timeout can work
        def _drain_stdout():
            try:
                for line in process.stdout:
                    log_message(line.rstrip('\n'))
            except Exception as e:
                log_message(f"Error reading output: {e}")
        reader = threading.Thread(target=_drain_stdout, daemon=True)
        reader.start()

        # Wait for process to complete with real timeout
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(command, timeout)
        reader.join(timeout=5)  # give reader a moment to flush

        returncode = process.returncode

        log_message(f"\n{'='*80}")
        log_message(f"Experiment run_{run_num} completed with return code: {returncode}")
        log_message(f"{'='*80}\n")

        logger.info(f"Command completed with return code: {returncode}")

        # Check for errors in output
        traceback_file = osp.join(cwd, f"run_{run_num}", "traceback.log")
        if osp.exists(traceback_file):
            try:
                with open(traceback_file, "r") as file:
                    tb = file.read()
                traceback, message = info_traceback(tb)
                logger.info(f"Parsed traceback: {traceback}")
                logger.info(f"Error message: {message}")
            except Exception as e:
                logger.info(f"Error reading traceback.log: {e}")
                traceback = tb = None
                message = "Unknown error"
        elif returncode != 0:
            traceback = tb = None
            message = f"Experiment failed with return code {returncode}"
        else:
            traceback, message, tb = None, None, None

        if returncode != 0:
            log_message(f"Run {run_num} failed with return code {returncode}")
            logger.info(f"Run failed with the following error: {message[:200] if message else 'Unknown error'}...")
            if tb:
                stderr_output = tb
            else:
                stderr_output = message or f"Experiment failed with return code {returncode}"
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"After running the code, the following error occurs:\n {stderr_output}. You need to modify the code to fix this error.\n Please fix this error by editing the error code in {cwd}/code (your current working directory). I'll copy it into my workspace later to run."
            logger.info(f"Generated error prompt for failed run")
        else:
            results = {}
            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if osp.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    results["baseline"] = json.load(f)
            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if osp.exists(run_path):
                    with open(run_path, "r") as f:
                        results[f"improve_{run_idx}"] = json.load(f)

            if task_type == 'sci':
                next_prompt = NEXT_EXPERIMENT_PROMPT_SCI.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)
            else:
                next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1, code_server_path=folder_name)

        return returncode, next_prompt, traceback, message
    except subprocess.TimeoutExpired:
        log_message(f"Run {run_num} timed out after {timeout} seconds")
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


def perform_experiments(
    idea,
    folder_name,
    proxy_settings=None,
    model='claude-sonnet-4-5-20250929',
    gpu_ids=None,
    max_runs=None,
    log_file=None,
    task_type='auto',
    task_info=None,
    checklist=None,
    sci_scorer_model='gpt-5.1',
    run_timeout=None,
) -> bool:
    """
    Perform multi-round experiments using Claude Code.

    Args:
        idea: The idea to implement
        folder_name: The folder to work in
        proxy_settings: Optional proxy settings for Claude
        model: Model name to use
        gpu_ids: GPU IDs to use (string like "0,1" or None for CPU)
        max_runs: Maximum number of runs (default: uses MAX_RUNS constant)
        log_file: Optional file object to write logs to
        task_type: 'auto' for standard tasks, 'sci' for paper reproduction tasks
        task_info: For sci tasks — parsed task_info.json dict (task description, data)
        checklist: For sci tasks — parsed checklist.json list
        sci_scorer_model: Model to use for LLM-as-judge scoring of sci tasks

    Returns:
        True if experiments completed successfully, False otherwise
    """
    def log_message(msg):
        """Write message to both stdout and log file"""
        print(msg)
        sys.stdout.flush()
        if log_file:
            try:
                log_file.write(msg + "\n")
                log_file.flush()
            except (ValueError, OSError):
                pass

    # Use provided max_runs or fall back to default
    if max_runs is None:
        max_runs = MAX_RUNS

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] Starting {task_type} experiments for idea in {folder_name}")
    if gpu_ids:
        logger.info(f"[{timestamp}] Using GPUs: {gpu_ids}")

    current_iter = 0
    run = 1

    # Initialize Claude Code runner
    claude_runner = ClaudeCodeRunner(proxy_settings, model=model)

    # Extract idea information
    idea_info = extract_idea_info(idea)

    # Prepare initial prompt based on task type
    if task_type == 'sci':
        next_prompt = _build_sci_initial_prompt(idea_info, task_info, checklist, max_runs, folder_name)
        # Save as INSTRUCTIONS.md for the evaluation judge (official scorer uses it as context)
        with open(osp.join(folder_name, "INSTRUCTIONS.md"), 'w') as f:
            f.write(next_prompt)
    else:
        next_prompt = CODER_PROMPT_OPENHANDS.format(
            idea_description=idea_info["description"],
            code_server_path=folder_name,
            method=idea_info["method"],
            max_runs=max_runs,
        )

    logger.info(f"Starting experiments for idea: {idea_info['title']}")
    log_message(f"Starting experiments for idea: {idea_info['title']}")
    logger.info(f"Method description: {str(idea_info['method'])[:300]}...")

    while run < max_runs + 1:
        if current_iter >= MAX_ITERS:
            log_message(f"Max iterations reached for run {run}. Moving to next run.")
            run += 1
            current_iter = 0
            continue

        log_message(f"Running Claude Code iteration {current_iter+1} for run {run}")
        claude_output = claude_runner.run(next_prompt, cwd=folder_name)
        log_message(f"Claude output received (length: {len(claude_output)})")
        log_message(claude_output)

        if "litellm.BadRequestError" in claude_output:
            log_message("Error: litellm.BadRequestError detected in Claude output")
            return False
        if "ALL_COMPLETED" in claude_output:
            log_message("All experiments completed successfully")
            break

        # Run experiment with specified GPU IDs
        log_message(f"Running experiment for run {run}")
        return_code, next_prompt, traceback, message = run_experiment(folder_name, run, timeout=run_timeout, gpu_ids=gpu_ids, log_file=log_file, task_type=task_type)
        logger.info(f"Experiment run {run} completed with return code {return_code}")

        # For sci tasks: if report exists but no final_info.json, score the run
        if task_type == 'sci':
            return_code, next_prompt = _handle_sci_run_scoring(
                folder_name, run, return_code, next_prompt,
                checklist, sci_scorer_model, log_message,
            )

        if return_code == 0:
            run += 1
            current_iter = 0
            logger.info(f"Run {run-1} succeeded, moving to run {run}")
        else:
            current_iter += 1
            logger.info(f"Run {run} failed, iteration {current_iter}/{MAX_ITERS}")

    if (run <= max_runs) and (current_iter >= MAX_ITERS):
        logger.info("Not all experiments completed.")
        _generate_report_with_claude(claude_runner, folder_name, run, current_iter, completed=False)
        return False

    logger.info("Experiments completed successfully")
    _generate_report_with_claude(claude_runner, folder_name, run, current_iter, completed=True)
    return True


def _build_sci_initial_prompt(idea_info, task_info, checklist, max_runs, folder_name):
    """Build the initial coder prompt for sci_task paper reproduction."""
    task_description = ""
    data_manifest = "No data files specified."

    if task_info:
        task_description = task_info.get('task', '')
        data_items = task_info.get('data', [])
        if data_items:
            lines = []
            for d in data_items:
                lines.append(f"- {d['name']}: {d.get('description', '')}")
            data_manifest = "\n".join(lines)

    checklist_summary = ""
    checklist_count = 0
    if checklist:
        checklist_count = len(checklist)
        lines = []
        for i, item in enumerate(checklist):
            w = item.get('weight', 0)
            t = item.get('type', 'text')
            content_preview = item.get('content', '')[:200]
            lines.append(f"  Item {i} (type={t}, weight={w:.2f}): {content_preview}")
        checklist_summary = "\n".join(lines)

    return CODER_PROMPT_SCI_TASK.format(
        idea_description=idea_info["description"],
        method=idea_info["method"],
        task_description=task_description,
        data_manifest=data_manifest,
        checklist_count=checklist_count,
        checklist_summary=checklist_summary,
        max_runs=max_runs,
    )


def _handle_sci_run_scoring(folder_name, run_num, return_code, next_prompt,
                             checklist, sci_scorer_model, log_message):
    """
    After a sci_task run: if report exists but no final_info.json, score it.

    Returns updated (return_code, next_prompt).
    """
    cwd = osp.abspath(folder_name)
    run_dir = osp.join(cwd, f"run_{run_num}")
    final_info_path = osp.join(run_dir, "final_info.json")
    report_path = osp.join(run_dir, "report", "report.md")

    if osp.exists(final_info_path):
        return return_code, next_prompt  # Already scored

    if not osp.exists(report_path):
        logger.info(f"No report.md found in {run_dir}, treating as failed run")
        return return_code, next_prompt

    # Report exists — score it
    log_message(f"Scoring sci_task run {run_num} with LLM judge (model={sci_scorer_model})...")
    checklist_path = osp.join(run_dir, "target_study", "checklist.json")

    # Fall back to checklist object if file not found in run_dir
    if not osp.exists(checklist_path) and checklist:
        # Write a temporary copy so score_run can read it
        os.makedirs(osp.join(run_dir, "target_study"), exist_ok=True)
        with open(checklist_path, 'w') as f:
            json.dump(checklist, f)

    # Ensure run dir has INSTRUCTIONS.md for the judge
    instructions_src = osp.join(cwd, "INSTRUCTIONS.md")
    instructions_dst = osp.join(run_dir, "INSTRUCTIONS.md")
    if osp.exists(instructions_src) and not osp.exists(instructions_dst):
        shutil.copy2(instructions_src, instructions_dst)

    try:
        from internagent.sci_eval import score_run, write_final_info
        scores = score_run(run_dir, checklist_path, model=sci_scorer_model)
        write_final_info(run_dir, scores)
        total = scores.get('total_score', 0)
        log_message(f"Sci task score for run {run_num}: {total:.1f}/100")

        # Build next_prompt reflecting the score
        results = {}
        baseline_path = osp.join(cwd, "run_0", "final_info.json")
        if osp.exists(baseline_path):
            with open(baseline_path) as f:
                results["baseline"] = json.load(f)
        results[f"improve_{run_num}"] = {"sci_task": {"means": scores}}

        next_prompt = NEXT_EXPERIMENT_PROMPT_SCI.format(
            RUN_NUM=run_num,
            RESULTS=json.dumps(results, indent=2),
            NEXT_RUN_NUM=run_num + 1,
        )
        return 0, next_prompt  # Override to success

    except Exception as e:
        log_message(f"Failed to score sci_task run {run_num}: {e}")
        logger.error(f"Sci scoring error: {e}", exc_info=True)
        return return_code, next_prompt


def _generate_report_with_claude(claude_runner, folder_name, final_run, final_iter, completed=True):
    """
    Use Claude Code to generate a comprehensive experiment report

    Args:
        claude_runner: The ClaudeCodeRunner instance
        folder_name: The folder where experiments were run
        final_run: The final run number reached
        final_iter: The final iteration number
        completed: Whether all experiments completed successfully
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] Requesting Claude Code to generate experiment report for {folder_name}")

    # Prepare prompt for Claude to generate the report
    status = "successfully completed" if completed else "partially completed or encountered issues"

    report_prompt = f"""Based on all the experiments we just conducted, please generate a concise experiment report and save it as 'experiment_report.txt' in the current directory.

The experiments have {status}. Total runs attempted: {final_run}.

Please check the experimental runs (run_1, run_2, ..., run_{final_run} directories) and create a brief, objective report:

**For each run (run_1 to run_{final_run})**, state:
- What was implemented/modified in this run
- The results obtained (from final_info.json if successful, or error summary from traceback.log if failed)

Keep descriptions factual and concise. No comparisons needed, just objectively document what each run did and what results it produced. Save as 'experiment_report.txt' in the root directory.
"""

    logger.info("Sending report generation request to Claude Code")
    claude_output = claude_runner.run(report_prompt, cwd=folder_name)
    logger.info(f"Claude Code report generation completed (output length: {len(claude_output)})")

    # Check if report was created
    report_path = osp.join(folder_name, "experiment_report.txt")
    if osp.exists(report_path):
        logger.info(f"Experiment report successfully generated at: {report_path}")
    else:
        logger.warning(f"Report generation completed but experiment_report.txt not found at expected path: {report_path}")
