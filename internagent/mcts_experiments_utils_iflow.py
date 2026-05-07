"""
MCTS experiments for iFlow backend
Using MCTS tree structure (node_X) + iFlow native run_experiment mechanism
"""

import os
import shutil
import json
from typing import Optional, Dict, Any

from internagent.mcts_node import AiderMCTSNode, MetricValue, WorstMetricValue
from internagent.prompts import (
    CODER_PROMPT_MCTS_DRAFT,
    MCTS_IMPROVE_PROMPT
)
from internagent.experiments_utils_iflow import (
    IFlowCodeRunner,
    run_experiment,
    extract_idea_info
)

# MCTS Configuration
MCTS_MAX_ITERATIONS = 30
MCTS_EXPLORATION_CONSTANT = 1.414
NUM_DRAFTS = 2
NUM_IMPROVES = 3
MAX_ITERS = 5
METRIC_IMPROVEMENT_THRESHOLD = 0.0001
MAX_IMPROVE_FAILURE = 2
USE_BASELINE_AS_ROOT = True


def _get_metric_config(task_key: str) -> Optional[tuple]:
    """Get metric name and maximize value for specified task_key from config file"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "tasks", "metric_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            if task_key in config:
                task_config = config[task_key]
                for metric_name, maximize in task_config.items():
                    return (metric_name, maximize)
        except Exception as e:
            print(f"[MCTS] Warning: Failed to read metric_config.json: {e}")
    return None


class IFlowMCTSSearch:
    """iFlow MCTS Searcher - MCTS tree structure + iFlow native execution"""

    def __init__(
        self,
        folder_name: str,
        baseline_results: Dict[str, Any],
        idea_info: Dict[str, str],
        proxy_settings=None,
        model='iflow-default'
    ):
        self.folder_name = folder_name
        self.baseline_results = baseline_results
        self.idea_info = idea_info

        # Create iFlow runner
        self.iflow_runner = IFlowCodeRunner(proxy_settings, model)

        # Create root node
        if USE_BASELINE_AS_ROOT:
            root_metric = self._extract_baseline_metric()
        else:
            root_metric = WorstMetricValue()

        self.root = AiderMCTSNode(
            run_number=0,
            stage="root",
            metric=root_metric
        )
        self.root.method_info = idea_info
        self.root.baseline_results = baseline_results
        self.root.is_virtual = True
        self.root.local_best_node = self.root

        # Search state
        if USE_BASELINE_AS_ROOT and self.root.metric and self.root.metric.value is not None:
            self.best_metric = self.root.metric.value
            self.best_node = self.root
        else:
            self.best_metric = None
            self.best_node = None

        self.successful_nodes = []
        self.branch_run_numbers = {}

        # MCTS logging
        import logging
        self.mcts_log_path = os.path.join(self.folder_name, "mcts.log")
        self.mcts_logger = logging.getLogger(f"mcts-iflow-{id(self)}")
        self.mcts_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.mcts_log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        if not self.mcts_logger.handlers:
            self.mcts_logger.addHandler(file_handler)

        self.scfg = type('SearchConfig', (), {
            'num_drafts': NUM_DRAFTS,
            'num_improves': NUM_IMPROVES
        })()

    def _log(self, message: str):
        """Log message"""
        try:
            self.mcts_logger.info(message)
        except Exception:
            print(f"[MCTS-LOG] {message}")

    def _print_and_log(self, message: str):
        """Print and log"""
        print(message)
        self._log(message)

    def is_root(self, node: AiderMCTSNode) -> bool:
        """Check if node is root"""
        return node == self.root

    def get_branch_run_number(self, branch_id: int) -> str:
        """Get run number for branch"""
        if branch_id not in self.branch_run_numbers:
            self.branch_run_numbers[branch_id] = 1
        run = self.branch_run_numbers[branch_id]
        self.branch_run_numbers[branch_id] += 1
        return f"run_{branch_id}_{run}"

    def get_current_branch_id(self, node: AiderMCTSNode) -> int:
        """Get branch ID for node"""
        if self.is_root(node):
            return 0
        current = node
        while current.parent and not self.is_root(current.parent):
            if hasattr(current, 'branch_id') and current.branch_id is not None:
                return current.branch_id
            current = current.parent
        if hasattr(current, 'branch_id') and current.branch_id is not None:
            return current.branch_id
        return 1

    def _extract_baseline_metric(self) -> MetricValue:
        """Extract baseline metric"""
        try:
            run_0_path = os.path.join(self.folder_name, "run_0", "final_info.json")
            with open(run_0_path, "r") as f:
                data = json.load(f)

            for task_key, value in data.items():
                if isinstance(value, dict) and "means" in value:
                    means = value["means"]
                    if isinstance(means, dict):
                        metric_config = _get_metric_config(task_key)
                        if metric_config:
                            metric_name, maximize = metric_config
                            if metric_name in means and isinstance(means[metric_name], (int, float)):
                                baseline_value = float(means[metric_name])
                                print(f"[MCTS] Using baseline metric: {metric_name}={baseline_value:.6f} (maximize={maximize})")
                                return MetricValue(value=baseline_value, maximize=maximize)
        except Exception as e:
            print(f"[MCTS] Warning: Failed to extract baseline metric: {e}")

        print("[MCTS] Using WorstMetricValue as fallback")
        return WorstMetricValue()

    def _extract_metric_from_node(self, node_folder: str) -> Optional[MetricValue]:
        """Extract metric from node's run_1"""
        final_info_path = os.path.join(node_folder, "run_1", "final_info.json")
        if not os.path.exists(final_info_path):
            return None

        try:
            with open(final_info_path, "r") as f:
                data = json.load(f)

            for task_key, value in data.items():
                if isinstance(value, dict) and "means" in value:
                    means = value["means"]
                    if isinstance(means, dict):
                        metric_config = _get_metric_config(task_key)
                        if metric_config:
                            metric_name, maximize = metric_config
                            if metric_name in means and isinstance(means[metric_name], (int, float)):
                                return MetricValue(value=float(means[metric_name]), maximize=maximize)
                        else:
                            for metric_name, metric_val in means.items():
                                if isinstance(metric_val, (int, float)) and metric_name != "epoch":
                                    metric_lower = metric_name.lower()
                                    default_maximize = not any(keyword in metric_lower for keyword in ["loss", "rmse", "error", "mse", "mae"])
                                    return MetricValue(value=float(metric_val), maximize=default_maximize)
        except Exception as e:
            print(f"[MCTS] Error reading final_info.json from {node_folder}: {e}")

        return None

    def _setup_node_workspace(self, node: AiderMCTSNode, parent_node: Optional[AiderMCTSNode]):
        """Setup node workspace - MCTS tree structure"""
        node.workspace_folder = os.path.join(self.folder_name, f"node_{node.id}")
        os.makedirs(node.workspace_folder, exist_ok=True)

        # Determine source directory
        if parent_node and parent_node.workspace_folder:
            source_dir = parent_node.workspace_folder
        else:
            source_dir = self.folder_name

        # Copy files
        print(f"[MCTS] Copying files to node workspace: {node.workspace_folder}")
        skip_names = {"mcts.log", ".DS_Store"}
        skip_dir_names = {".git"}
        skip_prefixes = ("node_", "run_")

        for item in os.listdir(source_dir):
            if any(item.startswith(p) for p in skip_prefixes):
                continue
            if item in skip_dir_names or item in skip_names:
                continue

            src = os.path.join(source_dir, item)
            dst = os.path.join(node.workspace_folder, item)

            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # Create node metadata
        meta_data = {
            "node_id": node.id,
            "run_number": node.run_number,
            "stage": node.stage,
            "branch_id": getattr(node, 'branch_id', None),
            "parent_id": parent_node.id if parent_node else None,
            "created_time": node.created_time
        }
        with open(os.path.join(node.workspace_folder, "node_meta.json"), 'w') as f:
            json.dump(meta_data, f, indent=2)

        print(f"[MCTS] Setup workspace for node {node.id}: {node.workspace_folder}")

    def _draft(self) -> AiderMCTSNode:
        """Generate draft node"""
        print(f"[MCTS] Generating draft node (attempt {len(self.root.children) + 1})")

        self.root.add_expected_child_count()

        try:
            draft_attempt = len(self.root.children) + 1
            branch_id = draft_attempt
            run_number = self.get_branch_run_number(branch_id)

            node = AiderMCTSNode(
                run_number=run_number,
                parent=self.root,
                stage="draft"
            )
            node.draft_attempt = draft_attempt
            node.branch_id = branch_id
            node.local_best_node = self.root

            # Setup node workspace (MCTS structure)
            self._setup_node_workspace(node, None)

            # Step 1: Generate initial code (execute only once)
            try:
                prompt = CODER_PROMPT_MCTS_DRAFT.format(
                    idea=self.idea_info["description"],
                    method=self.idea_info["method"],
                    baseline_results=self.baseline_results,
                )
                self._log(f"[draft] node={node.id} generating initial code")

                iflow_output = self.iflow_runner.run(prompt, cwd=node.workspace_folder)
                if iflow_output:
                    print(f"iFlow output: {iflow_output}")

                # Autodebug loop: run → check → fix
                return_code = 1
                for current_iter in range(MAX_ITERS):
                    self._log(f"[draft] node={node.id} iteration {current_iter+1}/{MAX_ITERS} running experiment")

                    # Run experiment (iFlow native way, creates node_X/run_1)
                    return_code, next_prompt, _, _ = run_experiment(
                        node.workspace_folder, 1
                    )

                    if return_code == 0:
                        self._log(f"[draft] node={node.id} run SUCCESS")
                        node.is_buggy = False
                        node.metric = self._extract_metric_from_node(node.workspace_folder)
                        break
                    else:
                        self._log(f"[draft] node={node.id} run FAILED, attempting to fix")
                        # If not last iteration, try to fix
                        if current_iter < MAX_ITERS - 1 and next_prompt:
                            fix_output = self.iflow_runner.run(next_prompt, cwd=node.workspace_folder)
                            if fix_output:
                                print(f"iFlow debug output: {fix_output}")
                else:
                    # Max iterations reached but still failed
                    print(f"[MCTS] Max iterations reached for draft node_{node.id}")
                    node.is_buggy = True
                    node.is_terminal = True

            except Exception as e:
                print(f"[MCTS] Error in draft: {e}")
                node.is_buggy = True
                node.is_terminal = True

            print(f"[MCTS] Generated draft node {node.id} with run_number {run_number}")
            return node

        except Exception as e:
            self.root.sub_expected_child_count()
            print(f"Error in _draft: {e}")
            raise e

    def _improve(self, parent_node: AiderMCTSNode) -> AiderMCTSNode:
        """Generate improve node"""
        print(f"[MCTS] Generating improve node from parent {parent_node.id}")

        parent_node.add_expected_child_count()

        try:
            branch_id = self.get_current_branch_id(parent_node)
            run_number = self.get_branch_run_number(branch_id)

            node = AiderMCTSNode(
                run_number=run_number,
                parent=parent_node,
                stage="improve"
            )
            node.branch_id = branch_id
            node.local_best_node = parent_node.local_best_node

            # Copy code from parent node to new node (MCTS structure)
            self._setup_node_workspace(node, parent_node)

            # Collect ancestor results
            ancestor_results = self._collect_ancestor_results(parent_node)

            # Step 1: Improve code (execute only once)
            try:
                prompt = MCTS_IMPROVE_PROMPT.format(RESULTS=ancestor_results)
                self._log(f"[improve] parent={parent_node.id} node={node.id} improving code")

                improve_output = self.iflow_runner.run(prompt, cwd=node.workspace_folder)
                if improve_output:
                    print(f"iFlow improve output: {improve_output}")

                # Autodebug loop: run → check → fix
                return_code = 1
                for current_iter in range(MAX_ITERS):
                    self._log(f"[improve] node={node.id} iteration {current_iter+1}/{MAX_ITERS} running experiment")

                    # Run experiment (iFlow native way, creates node_X/run_1)
                    return_code, next_prompt, _, _ = run_experiment(
                        node.workspace_folder, 1
                    )

                    if return_code == 0:
                        self._log(f"[improve] node={node.id} run SUCCESS")
                        node.is_buggy = False
                        node.metric = self._extract_metric_from_node(node.workspace_folder)
                        break
                    else:
                        self._log(f"[improve] node={node.id} run FAILED, attempting to fix")
                        # If not last iteration, try to fix
                        if current_iter < MAX_ITERS - 1 and next_prompt:
                            fix_output = self.iflow_runner.run(next_prompt, cwd=node.workspace_folder)
                            if fix_output:
                                print(f"iFlow debug output: {fix_output}")
                else:
                    # Max iterations reached but still failed
                    print(f"[MCTS] Max iterations reached for improve node_{node.id}")
                    node.is_buggy = True
                    node.is_terminal = True

            except Exception as e:
                print(f"[MCTS] Error in improve: {e}")
                node.is_buggy = True
                node.is_terminal = True

            print(f"[MCTS] Generated improve node {node.id} with run_number {run_number}")
            return node

        except Exception as e:
            parent_node.sub_expected_child_count()
            print(f"Error in _improve: {e}")
            raise e

    def _collect_ancestor_results(self, node: AiderMCTSNode) -> Dict[str, Any]:
        """Collect ancestor node results"""
        results = {"baseline": self.baseline_results}

        path = []
        current = node
        while current and current != self.root:
            path.insert(0, current)
            current = current.parent

        for i, ancestor in enumerate(path):
            if ancestor.metric is not None and ancestor.workspace_folder:
                final_info_path = os.path.join(ancestor.workspace_folder, "run_1", "final_info.json")
                if os.path.exists(final_info_path):
                    try:
                        with open(final_info_path, "r") as f:
                            data = json.load(f)
                        results[f"improve_{i+1}"] = data
                    except Exception:
                        pass

        return results

    def check_improvement(self, cur_node: AiderMCTSNode) -> bool:
        """Check improvement and decide whether to backpropagate"""
        should_backpropagate = False
        local_best_node = cur_node.local_best_node

        if cur_node.is_buggy is False and cur_node.metric:
            new_metric = cur_node.metric.value
            local_best_metric = local_best_node.metric.value if local_best_node and local_best_node.metric else None

            if new_metric is not None and local_best_metric is not None:
                self._print_and_log(
                    f"Comparing Node {cur_node.id} (metric: {new_metric:.6f}) "
                    f"with local best Node {local_best_node.id} (metric: {local_best_metric:.6f})"
                )

                if cur_node.metric.maximize:
                    improvement = new_metric - local_best_metric
                else:
                    improvement = local_best_metric - new_metric

                if improvement < METRIC_IMPROVEMENT_THRESHOLD and local_best_node.improve_failure_depth < MAX_IMPROVE_FAILURE:
                    local_best_node.improve_failure_depth += 1
                    self._print_and_log(
                        f"Improvement ({improvement:.6f}) below threshold, "
                        f"try again ({local_best_node.improve_failure_depth}/{MAX_IMPROVE_FAILURE})"
                    )
                    cur_node.continue_improve = True
                    cur_node.local_best_node = local_best_node
                    should_backpropagate = False
                elif improvement < METRIC_IMPROVEMENT_THRESHOLD:
                    self._print_and_log(f"Max improvement attempts reached")
                    cur_node.continue_improve = False
                    cur_node.is_terminal = True
                    should_backpropagate = True
                else:
                    self._print_and_log(f"Improvement sufficient, continue")
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
            elif new_metric is not None:
                self._print_and_log(f"Node {cur_node.id} is first success, set as local best")
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
                should_backpropagate = False
            else:
                should_backpropagate = True
        else:
            should_backpropagate = True

        if cur_node.is_terminal:
            should_backpropagate = True

        if should_backpropagate:
            reward = self._calculate_reward(cur_node)
            self._backpropagate(cur_node, reward)

        return should_backpropagate

    def _calculate_reward(self, node: AiderMCTSNode) -> float:
        """Calculate reward"""
        if node.is_buggy or node.metric is None or node.metric.value is None:
            return -1

        reward = 1
        if self.best_metric is not None and self.best_node is not None:
            if node.metric.maximize:
                improvement = node.metric.value - self.best_metric
            else:
                improvement = self.best_metric - node.metric.value

            if improvement > 0:
                self._print_and_log(f"Node {node.id} is new best!")
                reward += 1

        if node not in self.successful_nodes:
            self.successful_nodes.append(node)

        return reward

    def _backpropagate(self, node: AiderMCTSNode, reward: float):
        """Backpropagate"""
        current = node
        while current is not None:
            if current.parent and current.parent.stage != "root":
                current.parent.continue_improve = current.continue_improve

            if current.improve_failure_depth > 0:
                current.improve_failure_depth = 0

            # Unlock draft node to allow re-exploration
            if current.stage == "draft" and current.lock:
                current.lock = False

            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def select(self, node: AiderMCTSNode) -> AiderMCTSNode:
        """Select node"""
        self._log(f"[select] Starting from node {node.id}, stage={node.stage}, is_terminal={node.is_terminal}")

        iteration_count = 0
        max_iterations = 100  # Prevent true infinite loops

        while node and not node.is_terminal:
            iteration_count += 1
            if iteration_count > max_iterations:
                self._log(f"[select] ERROR: Max iterations reached! Node {node.id} stuck in loop. Forcing terminal.")
                node.is_terminal = True
                break

            is_fully_expanded = node.is_fully_expanded_with_expected(scfg=self.scfg)
            self._log(f"[select] iteration={iteration_count}, node={node.id}, is_terminal={node.is_terminal}, fully_expanded={is_fully_expanded}, continue_improve={node.continue_improve}, children={len(node.children)}")

            if not is_fully_expanded:
                if node.continue_improve and len(node.children) > 0:
                    self._log(f"[select] node={node.id} has continue_improve and children, selecting child via UCT")
                    node = self._uct_select(node)
                    self._log(f"[select] After UCT select: node={node.id}, is_terminal={node.is_terminal}")
                else:
                    self._log(f"[select] node={node.id} not fully expanded, returning for expansion")
                    return node
            else:
                self._log(f"[select] node={node.id} is fully expanded, selecting child via UCT")
                node = self._uct_select(node)
                self._log(f"[select] After UCT select: node={node.id}, is_terminal={node.is_terminal}")

        self._log(f"[select] Exited loop at iteration {iteration_count}. Terminal or null node: {node.id if node else 'None'}, is_terminal={node.is_terminal if node else 'N/A'}")
        return node

    def _uct_select(self, node: AiderMCTSNode) -> AiderMCTSNode:
        """UCT selection"""
        if self.is_root(node):
            filtered_children = [c for c in node.children if not c.lock and not c.is_terminal]
            self._log(f"[_uct_select] root node, filtered_children={len(filtered_children)}")
            if len(filtered_children) > 0:
                selected = max(filtered_children, key=lambda c: c.uct_value(exploration_constant=MCTS_EXPLORATION_CONSTANT))
                if selected.stage == "draft":
                    selected.lock = True
                self._log(f"[_uct_select] Selected child {selected.id}")
                return selected
            else:
                # When no children can be selected:
                #
                # Reasons for empty filtered_children:
                # 1. All drafts are terminal (each draft fully expanded and all subtrees explored)
                # 2. All drafts are locked (theoretically shouldn't happen, unlock during backpropagate)
                #
                # Case 1 is normal search completion:
                # - Each draft creates NUM_IMPROVES improve branches
                # - Each improve chain explores to terminal
                # - Draft marked as terminal in _uct_select's else branch
                # - Root has no selectable child, should mark as terminal
                #
                # Case 2 is anomaly protection (unlock logic failed or concurrency issues)
                if len(node.children) > 0:
                    all_terminal_count = len([c for c in node.children if c.is_terminal])
                    locked_count = len([c for c in node.children if c.lock])
                    self._log(f"[_uct_select] No filtered children. total={len(node.children)}, terminal={all_terminal_count}, locked={locked_count}")
                    self._log(f"[_uct_select] Marking root as terminal (no available children to explore)")
                    node.is_terminal = True
                self._log(f"[_uct_select] Returning root (is_terminal={node.is_terminal})")
                return node
        else:
            non_terminal = [c for c in node.children if not c.is_terminal]
            terminal_count = len([c for c in node.children if c.is_terminal])
            self._log(f"[_uct_select] non-root node {node.id}, non_terminal={len(non_terminal)}, terminal={terminal_count}, total={len(node.children)}")
            if len(non_terminal) > 0:
                selected = max(non_terminal, key=lambda c: c.uct_value(exploration_constant=MCTS_EXPLORATION_CONSTANT))
                self._log(f"[_uct_select] Selected non-terminal child {selected.id}")
                return selected
            else:
                # Fix root cause: if no non-terminal children, mark parent as terminal
                self._log(f"[_uct_select] All children of node {node.id} are terminal, marking as terminal")
                node.is_terminal = True
                self._log(f"[_uct_select] Returning parent node {node.id} (is_terminal={node.is_terminal})")
                return node

    def step(self, start_node: AiderMCTSNode = None) -> AiderMCTSNode:
        """Execute one MCTS step"""
        if start_node is None:
            start_node = self.root
            self._log(f"[step] Starting from root")
        else:
            self._log(f"[step] Starting from node {start_node.id}")

        selected = self.select(start_node)
        self._log(f"[step] Selected node: {selected.id}, stage={selected.stage}, is_terminal={selected.is_terminal}")

        if not selected.is_terminal:
            try:
                if selected == self.root:
                    self._log(f"[step] Creating draft node from root")
                    result_node = self._draft()
                else:
                    self._log(f"[step] Creating improve node from parent {selected.id}")
                    result_node = self._improve(selected)

                should_backpropagate = self.check_improvement(result_node)
                self._log(f"[step] should_backpropagate={should_backpropagate}")

                # Update global best
                if result_node.is_buggy is False and result_node.metric and result_node.metric.value is not None:
                    if result_node not in self.successful_nodes:
                        self.successful_nodes.append(result_node)

                    if self.best_node is None or self.best_node.metric < result_node.metric:
                        self._print_and_log(f"Node {result_node.id} is the best node so far (metric: {result_node.metric.value:.6f})")
                        self.best_node = result_node
                        self.best_metric = result_node.metric.value

                return_node = self.root if should_backpropagate else result_node
                self._log(f"[step] Returning {'root' if return_node == self.root else result_node.id}")
                return return_node
            except Exception as e:
                print(f"[MCTS] Error in step: {e}")
                self._log(f"[step] Exception occurred: {e}")
                self._backpropagate(selected, 0)
                selected.is_terminal = True
                raise e
        else:
            self._log(f"[step] Selected node {selected.id} is terminal, backpropagating")
            self._backpropagate(selected, 0)
            return self.root

    def run_mcts_search(self, iterations: int = MCTS_MAX_ITERATIONS) -> bool:
        """Run MCTS search"""
        print(f"[MCTS] Starting MCTS search with {iterations} iterations")

        current_node = None
        for i in range(iterations):
            if len(self.root.children) > 0 and all(c.is_terminal for c in self.root.children if c.stage == "draft"):
                print(f"[MCTS] All draft nodes terminal, stopping")
                break

            try:
                current_node = self.step(current_node)  # Pass current_node!
            except Exception as e:
                print(f"[MCTS] Step {i+1} failed: {e}")
                current_node = None  # Reset after error, restart from root

            if current_node == self.root or current_node is None:
                if self.root.is_terminal:
                    print(f"[MCTS] Root is terminal, stopping")
                    break
                current_node = None  # Reset to None, next time start from root

            print(f"MCTS progress: {i+1}/{iterations}, {len(self.successful_nodes)} successful nodes")

        if len(self.successful_nodes) == 0:
            print("No experiments completed successfully")
            return False

        print(f"[MCTS] Search completed. Found {len(self.successful_nodes)} successful nodes")
        return True


def perform_experiments_mcts(
    idea,
    folder_name: str,
    proxy_settings=None,
    model='iflow-default'
) -> bool:
    """
    Execute iFlow experiments using MCTS

    Args:
        idea: Experiment idea
        folder_name: Experiment folder
        proxy_settings: Proxy settings
        model: Model name

    Returns:
        bool: Whether experiment succeeded
    """
    print("=" * 50)
    print("IFLOW MCTS EXPERIMENTS STARTED")
    print("=" * 50)

    idea_info = extract_idea_info(idea)
    print(f"Experiment idea: {idea_info.get('title', 'Unknown')}")
    print(f"Experiment folder: {folder_name}")

    # Load baseline
    baseline_path = os.path.join(folder_name, "run_0", "final_info.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        baseline_results = baseline_data
    else:
        baseline_results = {}

    # Run MCTS
    try:
        mcts_search = IFlowMCTSSearch(
            folder_name=folder_name,
            baseline_results=baseline_results,
            idea_info=idea_info,
            proxy_settings=proxy_settings,
            model=model
        )
        success = mcts_search.run_mcts_search()
    except Exception as e:
        print(f"MCTS search failed: {e}")
        return False

    print(f"MCTS search completed. Success: {success}")
    print("=" * 50)

    return success
