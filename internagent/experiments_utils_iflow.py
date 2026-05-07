import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os
from datetime import datetime

from internagent.prompts import (
    CODER_PROMPT_OPENHANDS, 
    NEXT_EXPERIMENT_PROMPT
)

import filecmp

MAX_ITERS = 5
MAX_RUNS = 5  # Default value, can be overridden by config
MAX_STDERR_OUTPUT = 30000

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


# return (file, line, function, content), message
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


class IFlowCodeRunner:
    """iFlow Code Runner class to handle interactions with iFlow CLI"""

    def __init__(self, proxy_settings=None, model='iflow-default'):
        """
        Initialize the iFlow Code Runner

        Args:
            proxy_settings: Optional dictionary with HTTP_PROXY and HTTPS_PROXY settings
            model: Model name to use (default: iflow-default)
        """
        self.proxy_settings = proxy_settings or {
            'HTTP_PROXY': 'http://127.0.0.1:7890',
            'HTTPS_PROXY': 'http://127.0.0.1:7890'
        }
        self.model = model
        
    def run(self, prompt, cwd=None):
        """
        Run iFlow Code with the given prompt
        
        Args:
            prompt: The prompt to send to iFlow
            cwd: The working directory for iFlow to operate in
            
        Returns:
            The stdout output from iFlow
        """
        # Set proxy environment variables
        env = os.environ.copy()
        for key, value in self.proxy_settings.items():
            env[key] = value
        
        # Enhanced logging - Log start time and command
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Running iFlow CLI with prompt: {prompt}..."
        print(log_message)
            
        # Run iFlow with appropriate parameters
        # Capture both stdout and stderr so we can log them properly
        result = subprocess.run(
            ['iflow',  '--prompt', prompt], # '--model', self.model,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env
        )

        # Enhanced logging - Log completion and output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] iFlow command completed with return code: {result.returncode}"
        print(log_message)

        # Log output summary
        if result.stdout:
            output_summary = f"iFlow output : {result.stdout}..."
            print(output_summary)

        # If there was an error, log that too
        if result.returncode != 0 or result.stderr:
            error_message = f"iFlow CLI error (return code {result.returncode}): {result.stderr}"
            print(error_message)
        
        # Return the stdout result
        return result.stdout


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=180000, log_file=None):
    """
    Run experiment with the code in the folder

    Args:
        folder_name: The folder containing the code
        run_num: The run number
        timeout: Maximum execution time in seconds
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
    log_message(f"[{timestamp}] Starting experiment run {run_num} in {folder_name}")

    cwd = osp.abspath(folder_name)
    # Create run directory
    run_dir = osp.join(cwd, f"run_{run_num}")
    if not osp.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    # Copy all files from the main folder to the run directory
    log_message(f"Copying files to run directory: {run_dir}")
    for item in os.listdir(cwd):
        if item.startswith("run_") or item == ".git":
            continue
        src = osp.join(cwd, item)
        dst = osp.join(run_dir, item)
        if osp.isdir(src):
            if osp.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # LAUNCH COMMAND
    command = ["bash", f"launcher.sh", f"run_{run_num}"]
    log_message(f"Running command: {command} in {cwd}")

    log_message(f"\n{'='*80}")
    log_message(f"Running experiment: run_{run_num}")
    log_message(f"{'='*80}\n")

    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )

        # Stream output line by line
        try:
            for line in process.stdout:
                line = line.rstrip('\n')
                log_message(line)
        except Exception as e:
            log_message(f"Error reading output: {e}")

        # Wait for process to complete
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(command, timeout)

        returncode = process.returncode

        log_message(f"\n{'='*80}")
        log_message(f"Experiment run_{run_num} completed with return code: {returncode}")
        log_message(f"{'='*80}\n")

        log_message(f"Command completed with return code: {returncode}")

        if os.path.exists(osp.join(cwd, f"run_{run_num}", "final_info.json")):
            log_message(f"Run {run_num} completed successfully with final_info.json")
            results = {}

            # Read baseline (run_0) results
            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                baseline_results = {k: v["means"] for k, v in baseline_data.items()}
                results["baseline"] = baseline_results
                log_message(f"Loaded baseline results: {baseline_results}")

            # Read results from run_1 to run_num
            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if os.path.exists(run_path):
                    with open(run_path, "r") as f:
                        run_data = json.load(f)
                    run_results = {k: v["means"] for k, v in run_data.items()}
                    results[f"improve_{run_idx}"] = run_results
                    log_message(f"Loaded run_{run_idx} results: {run_results}")

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1, code_server_path=folder_name)
            log_message(f"Generated next prompt for successful run")
            traceback, message, tb = None, None, None
            return returncode, next_prompt, traceback, message

        # Check for errors in output
        traceback_file = osp.join(cwd, f"run_{run_num}", "traceback.log")
        if osp.exists(traceback_file):
            try:
                with open(traceback_file, "r") as file:
                    tb = file.read()
                traceback, message = info_traceback(tb)
                log_message(f"Parsed traceback: {traceback}")
                log_message(f"Error message: {message}")
            except Exception as e:
                log_message(f"Error reading traceback.log: {e}")
                traceback = tb = None
                message = "Unknown error"
        elif returncode != 0:
            # No traceback.log found
            traceback = tb = None
            message = f"Experiment failed with return code {returncode}"
            log_message(f"No traceback.log found, using generic error message")
        else:
            traceback, message, tb = None, None, None

        if returncode != 0:
            log_message(f"Run {run_num} failed with return code {returncode}")
            log_message(f"Run failed with the following error: {message[:200] if message else 'Unknown error'}...")
            if tb:
                stderr_output = tb
            else:
                stderr_output = message or f"Experiment failed with return code {returncode}"
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"After running the code, the following error occurs:\n {stderr_output}. You need to modify the code to fix this error."
            log_message(f"Generated error prompt for failed run")
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1, code_server_path=folder_name)
            log_message(f"Generated next prompt for successful run")

        return returncode, next_prompt, traceback, message
    except subprocess.TimeoutExpired:
        log_message(f"Run {run_num} timed out after {timeout} seconds")
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


# PERFORM EXPERIMENTS with iFlow Code for multiple rounds
def perform_experiments(idea, folder_name, proxy_settings=None, model='iflow-default', gpu_ids=None, max_runs=None, log_file=None) -> bool:
    """
    Perform multi-round experiments using iFlow Code

    Args:
        idea: The idea to implement
        folder_name: The folder to work in
        proxy_settings: Optional proxy settings for iFlow
        model: Model name to use (default: iflow-default)
        gpu_ids: GPU IDs to use (string like "0,1" or None for CPU)
        max_runs: Maximum number of runs (default: uses MAX_RUNS constant)
        log_file: Optional file object to write logs to

    Returns:
        True if all experiments completed successfully, False otherwise
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
    log_message(f"[{timestamp}] Starting experiments for idea in {folder_name}")
    if gpu_ids:
        log_message(f"[{timestamp}] Using GPUs: {gpu_ids}")

    current_iter = 0
    run = 1

    # Initialize iFlow Code runner
    iflow_runner = IFlowCodeRunner(proxy_settings, model=model)

    # Extract idea information
    idea_info = extract_idea_info(idea)

    # Prepare initial prompt
    next_prompt = CODER_PROMPT_OPENHANDS.format(
        idea_description=idea_info["description"],
        code_server_path=folder_name,
        method=idea_info["method"],
        max_runs=max_runs,
    )

    log_message(f"Starting experiments for idea: {idea_info['title']}")
    log_message(f"Method description: {idea_info['method'][:300]}...")

    while run < max_runs + 1:
        if current_iter >= MAX_ITERS:
            log_message(f"Max iterations reached for run {run}. Moving to next run.")
            run += 1
            current_iter = 0
            continue

        log_message(f"Running iFlow Code iteration {current_iter+1} for run {run}")
        iflow_output = iflow_runner.run(next_prompt, cwd=folder_name)
        log_message(f"iFlow output received (length: {len(iflow_output)})")
        log_message(iflow_output)

        if "litellm.BadRequestError" in iflow_output:
            log_message("Error: litellm.BadRequestError detected in iFlow output")
            return False
        if "ALL_COMPLETED" in iflow_output:
            log_message("All experiments completed successfully")
            break

        # Run experiment
        log_message(f"Running experiment for run {run}")
        return_code, next_prompt, traceback, message = run_experiment(folder_name, run, log_file=log_file)
        log_message(f"Experiment run {run} completed with return code {return_code}")

        if return_code == 0:
            run += 1
            current_iter = 0
            log_message(f"Run {run-1} succeeded, moving to run {run}")
        else:
            current_iter += 1
            log_message(f"Run {run} failed, iteration {current_iter}/{MAX_ITERS}")

    if (run <= MAX_RUNS) and (current_iter >= MAX_ITERS):
        log_message("Not all experiments completed.")
        return False

    print("Experiments completed successfully")
    return True