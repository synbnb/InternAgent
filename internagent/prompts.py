CODER_PROMPT_MCTS_DRAFT = """Your goal is to implement the following idea: {idea}
The proposed method is as follows: {method}.

You are given a single run to complete the necessary experiments.
**Note**: It is highly recommended that you implement the core functionality of your code within this run, choose important designs in the proposed method. 

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

Then, you need to implement code. After you complete the implementation, we will run the command `bash launcher.sh .` to execute the experiment. 
Do NOT modify launcher.sh under any circumstances.
Do NOT change the schema of final_info.json; only write to the existing means fields and keep the structure identical to the baseline.

**IMPORTANT CONSTRAINTS:**
1. YOU MUST NOT MODIFY `launcher.sh` - it must remain as is. We always run this command.
2. DO NOT create shell scripts (.sh files) or run_xxx folder, any files will be saved to the current directory (`.`)
3. Treat the current working directory as the project root; results will be saved to the current directory (`.`)
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically: Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
"""

CODER_PROMPT_OPENHANDS = """Your goal is to implement the following idea: {idea_description} in the codes {code_server_path}. 
Please read the code of all the files of {code_server_path} first (important), each time before modifying the file you need to determine the location of the insertion again, and after modification to confirm that the content and location of the modification is correct through observation. After that, I will give you a method and you need to adapt that method appropriately based on the existing baseline code.

You are given a total of up to {max_runs} runs to complete the necessary experiments. Keep this in mind so that you don't miss the ultimate goal of performance improvement in your attempt.

## Requirements:
    1. Integrate the core concepts of my improved method into the baseline code
    2. Make necessary adaptations to ensure compatibility with the existing codebase
    3. When conflicts arise between the improved method and baseline implementation: 
        1) Prioritize maintaining the stability of the baseline code 
        2) Adapt the improved method's concepts rather than forcing exact implementation 
        3) Preserve the overall architecture of the baseline while enhancing its functionality
    4. Ensure that the final file to be executed is {code_server_path}/launcher.sh
    5. DO NOT make changes to the original content in the {code_server_path}/launcher.sh, such as the GPU ID, data_root, etc. However, it is allowed to add or modify model-related parameters.
    6. DO NOT attempt to install the environment in the {code_server_path}/launcher.sh
    7. When checking the correctness of the code, ignore the runtime environment issues.

The proposed method is as follows: {method}.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.

"""

CODE_STRUCTURE_PROMPT = """You are an expert code analyst specializing in error detection, debugging, and error handling patterns. Your task is to thoroughly analyze the provided code with a focus on potential errors below:

{error_messages}

You need to focus on error-related aspects of code and analyze their relations. The following functions and codes may highly related to the error which is extracted from the traceback.

{function_code}

Note that you do not need to modify the code in this step and just need to give the error-related code structure.
"""

DEBUG_PROMPT_WITH_STRUCTURE = """You are an expert code debugger specializing in structural analysis and error diagnosis. Your task is to debug the code based on the following error message:

{error_messages}

Previously, you have analyzed the error-related code structure as follows:

{code_structure}

You need to first analyze the error message and list all the possible reasons and code modification plan of the error. Then, modify the code based on the plan. You can refer to the code structure obtained from the previous analysis. 

**IMPORTANT CONSTRAINTS:**
1. DO NOT modify or create any shell scripts (.sh files), including `launcher.sh` or any other .sh files
2. Only modify Python code file

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
"""

NEXT_EXPERIMENT_PROMPT = """Run {RUN_NUM} completed. Here are the results:
{RESULTS}

Based on these results:
1. Analyze what worked and what didn't work in your approach.
2. Compare the current run with previous runs and baseline.
3. Decide if you need to re-plan your experiments or continue with your current strategy.
4. If continuing, implement the next improvement on your list.
5. If re-planning, explain why and outline your new approach.

We will run the command `bash launcher.sh {NEXT_RUN_NUM}` to execute your next experiment.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

If you believe you have completed all necessary experiments and found the optimal solution, respond with 'ALL_COMPLETED'.

You don't have to follow the initial approach exactly, feel free to suggest additional optimizations if you think there are better improvements. Our ultimate goal is to improve the performance of the optimization algorithm, not to limit our vision to the initial method, but to analyze, optimize, and improve according to the phenomenon.
**It's important to make sure that all of your proposed changes are already directly performed in {code_server_path}, don't just say it and leave it there. And do not care about files in {code_server_path}/run_xxx/, because they are only versions of a certain time.**
Make sure that your toggle Settings, such as True /False, are aligned with what you've modified by default, and that you don't need to do any extra toggle control on launcher.sh.
"""

CODER_PROMPT_SCI_TASK = """Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
{idea_description}

## Proposed Method
{method}

## Research Task
{task_description}

## Available Data
{data_manifest}

## Evaluation Criteria (Checklist)
Your work will be scored on {checklist_count} criteria:
{checklist_summary}

## Workspace Layout
- Write analysis code in `code/experiment.py` (and helper modules in `code/`)
- Save intermediate outputs (data files, CSV, etc.) in `outputs/`
- Write your final report as `report/report.md` — this is REQUIRED and will be scored
- Save ALL generated figures in `report/images/`
- Reference papers are in `related_work/`
- Raw data is in `data/`

## Execution
You have up to {max_runs} runs. Each run executes `bash launcher.sh` from the workspace directory.
Do NOT modify `launcher.sh`.

## Requirements
1. Implement a complete, runnable `code/experiment.py` that reproduces the findings
2. Paths should be relative to the workspace root (e.g., `data/filename.dat`, `outputs/result.csv`)
3. Your `report/report.md` MUST describe all results quantitatively and reference generated figures
4. Each generated figure must be saved to `report/images/` and referenced in the report
5. Focus on the checklist criteria — they specify exactly what must be reproduced

Any modifications to argparse parameters **must set improved implementations as defaults**.
"""

NEXT_EXPERIMENT_PROMPT_SCI = """Run {RUN_NUM} completed. Here are the results:
{RESULTS}

Based on these results:
1. Check what the current score is (total_score out of 100) and which checklist items are weakest.
2. Analyze what worked and what did not in your implementation.
3. Decide what to improve: fix errors, improve accuracy, add missing checklist items, or enhance the report.
4. Implement the improvements in `code/` and update `report/report.md` and `report/images/`.

Run {NEXT_RUN_NUM} will execute next via `bash launcher.sh`.
Do NOT modify `launcher.sh`.

If you believe your reproduction is complete and optimal, respond with 'ALL_COMPLETED'.

Focus on the checklist criteria with the highest weights — they contribute most to the final score.
Make sure `report/report.md` exists and describes ALL results with actual numbers and figure references.
"""

MCTS_IMPROVE_PROMPT = """Previous implementation completed. Here are the results:
{RESULTS}

Based on these results, implement an improved desigin to improve metric performance. Focus on:
1. Analyzing what worked and what didn't work in the previous approach
2. Comparing with baseline results  
3. Implementing the next improvement, for example focused on performance tuning and hyperparameter optimization based on observed results
4. Propose a single actionable and specific improvement. This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change
5. When proposing the design, take the Memory section into account
6. Ensure the improvement is distinct from previous attempts

We will run the command `bash launcher.sh .` to execute your next experiment. 
Do NOT modify launcher.sh under any circumstances.
Do NOT change the schema of final_info.json; only write to the existing means fields and keep the structure identical to the baseline.

**IMPORTANT CONSTRAINTS:**
1. DO NOT modify `launcher.sh` - it must remain as is. We always run this command.
2. DO NOT create shell scripts (.sh files) or run_xxx folder, any files will be saved to the current directory (`.`)
3. Treat the current working directory as the project root; results will be saved to the current directory (`.`)
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

If you believe you have completed all necessary improvements and found the optimal solution, respond with 'ALL_COMPLETED'.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically: Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
"""