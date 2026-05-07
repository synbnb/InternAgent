import os.path as osp
import os
import glob
import sys
import json
import shutil
import yaml
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock, Thread, Event
import math

from internagent.mas.interface import InternAgentInterface
from internagent.experiments_utils_claude import perform_experiments as perform_experiments_claudecode
from internagent.experiments_utils_iflow import perform_experiments as perform_experiments_iflow
from internagent.mcts_experiments_utils_claude import perform_experiments_mcts as perform_experiments_mcts_claude
from internagent.mcts_experiments_utils_iflow import perform_experiments_mcts as perform_experiments_mcts_iflow
from internagent.vis_tree import vis_tree



# Optional long memory imports
LONG_MEMORY_AVAILABLE = False
PromptEvolver = None
IdeaGraph = None

try:
    from internagent.mas.memory.long_memory import PromptEvolver, IdeaGraph
    LONG_MEMORY_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Long memory module not available: {e}. Long memory features will be disabled.")


class Tee:
    """A class that writes to multiple streams simultaneously"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()  # Ensure immediate output
            except (ValueError, OSError):
                # Skip closed or invalid streams
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except (ValueError, OSError):
                # Skip closed or invalid streams
                pass

    def isatty(self):
        # Return True if any stream is a TTY
        return any(hasattr(s, 'isatty') and s.isatty() for s in self.streams)


# ============================================================================
# Idea Generation Config
# ============================================================================
class IdeaGenerator:
    """Handles idea generation using MAS"""

    def __init__(self, args, logger, round_num=1, config=None):
        self.args = args
        self.logger = logger
        self.round_num = round_num  # Current loop round number (1-indexed)
        self.config = config or {}  # Config dict for accessing evolution_interval
        # Pass exp_backend to the interface so it can be accessed by agents
        self.interface = InternAgentInterface(
            args.config,
            work_dir=args.task_dir,
            task_name=args.task_name,
            exp_backend=args.exp_backend
        )
        self.session_id = None
        self.status = None

        # Initialize IdeaGraph (if long memory is available)
        self.idea_graph = None
        if LONG_MEMORY_AVAILABLE and IdeaGraph is not None:
            # Use base_output_dir for IdeaGraph (task-level, shared across launches)
            output_dir = getattr(self.args, 'base_output_dir', None) or osp.join("results", self.args.task_name)
            os.makedirs(output_dir, exist_ok=True)
            try:
                self.idea_graph = IdeaGraph(
                    working_dir=output_dir,
                    namespace=self.args.task_name,
                    similarity_threshold=0.7
                )
                self.logger.info("IdeaGraph initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize IdeaGraph: {e}")
                self.idea_graph = None
        else:
            self.logger.info("Long memory not available, IdeaGraph disabled")
    
    def _load_historical_ideas_to_graph(self):
        """
        Load all historical ideas from previous sessions into the IdeaGraph.

        This allows the graph to maintain a complete history of all generated ideas
        across multiple iterations, enabling better exploration score calculation.
        """
        if self.idea_graph is None:
            return

        # Use base_output_dir for consistency with other components
        output_dir = getattr(self.args, 'base_output_dir', None) or osp.join("results", self.args.task_name)
        if not osp.exists(output_dir):
            self.logger.info("No historical ideas found (output directory doesn't exist)")
            return

        # Find all ideas.json files in session directories (new structure: *_launch/session_*/ideas.json)
        ideas_files = glob.glob(osp.join(output_dir, "*_launch", "session_*", "ideas.json"))
        # Also check legacy structure (session_*/ideas.json)
        ideas_files.extend(glob.glob(osp.join(output_dir, "session_*", "ideas.json")))
        # Also check old format (ideas_*.json)
        ideas_files.extend(glob.glob(osp.join(output_dir, "ideas_*.json")))

        if not ideas_files:
            self.logger.info("No historical ideas files found")
            return

        self.logger.info(f"Found {len(ideas_files)} historical ideas files")

        loaded_count = 0
        for ideas_file in ideas_files:
            try:
                with open(ideas_file, 'r', encoding='utf-8') as f:
                    ideas_data = json.load(f)

                # Handle both list format and dict format
                if isinstance(ideas_data, list):
                    ideas = ideas_data
                elif isinstance(ideas_data, dict):
                    # Try to extract ideas from different formats
                    ideas = ideas_data.get('hypotheses', ideas_data.get('ideas', []))
                else:
                    self.logger.warning(f"Unknown format in {ideas_file}, skipping")
                    continue

                if not ideas:
                    continue

                # Add each idea to the graph
                for idea in ideas:
                    # Ensure idea has required fields
                    if not isinstance(idea, dict):
                        continue

                    # Some ideas might not have 'id', generate one if needed
                    if 'id' not in idea and 'name' in idea:
                        idea['id'] = idea['name']
                    elif 'id' not in idea:
                        idea['id'] = f"idea_{loaded_count}"

                    self.idea_graph.add_idea_node(idea)
                    loaded_count += 1

                self.logger.info(f"Loaded {len(ideas)} ideas from {osp.basename(ideas_file)}")

            except Exception as e:
                self.logger.warning(f"Failed to load ideas from {ideas_file}: {e}")
                continue

        if loaded_count > 0:
            stats = self.idea_graph.get_graph_stats()
            self.logger.info(
                f"Successfully loaded {loaded_count} historical ideas into graph. "
                f"Graph now has {stats['num_nodes']} nodes, {stats['num_edges']} edges"
            )

    async def load_task(self):
        """Load task and create MAS session"""
        self.logger.info(f"Creating research session for: {self.args.task_dir}")

        await self.interface.startup()

        # Use args.prompt_path if available (points to launch directory copy), otherwise fall back to task_dir
        task_desc_path = getattr(self.args, 'prompt_path', None) or osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")
        self.logger.info(f"Using prompt file: {task_desc_path}")

        # Check if experience library exists and evolve prompt if it does
        # Use base_output_dir if available (set by launch_discovery.py), otherwise use default
        base_output_dir = getattr(self.args, 'base_output_dir', None) or osp.join("results", self.args.task_name)
        library_path = osp.join(base_output_dir, "experience_library.json")

        # Get evolution_interval from config (default: 1, meaning evolve every round)
        evolution_interval = self.config.get('memory', {}).get('long_memory', {}).get('prompt_evolver', {}).get('evolution_interval', 1)

        # Only evolve prompt when:
        # 1. round_num > 1 (first round has no experience yet)
        # 2. round_num % evolution_interval == 0 (respects the configured interval)
        # Note: With interval=1, evolves at rounds 2, 3, 4, ...
        # With interval=2, evolves at rounds 2, 4, 6, ...
        should_evolve = (self.round_num > 1) and ((self.round_num - 1) % evolution_interval == 0)

        if should_evolve:
            self.logger.info(f"Round {self.round_num}: Prompt evolution scheduled (interval={evolution_interval})")
        else:
            self.logger.info(f"Round {self.round_num}: Skipping prompt evolution (interval={evolution_interval})")

        if should_evolve and osp.exists(library_path) and LONG_MEMORY_AVAILABLE and PromptEvolver is not None:
            self.logger.info("Experience library found, evolving prompt...")
            try:
                # Create a backup of the prompt before evolution (in the same directory as prompt file)
                prompt_dir = osp.dirname(task_desc_path)
                backup_path = osp.join(prompt_dir, f"prompt_backup_round{self.round_num}.json")
                import shutil
                shutil.copy(task_desc_path, backup_path)
                self.logger.info(f"Backed up prompt to: {backup_path}")

                # Load config for PromptEvolver
                config = {}
                if self.args.config and osp.exists(self.args.config):
                    try:
                        with open(self.args.config, 'r') as f:
                            if self.args.config.endswith(('.yaml', '.yml')):
                                config = yaml.safe_load(f)
                            else:
                                config = json.load(f)
                    except Exception as e:
                        self.logger.warning(f"Failed to load config: {e}")

                # Initialize PromptEvolver with IdeaGraph
                prompt_evolver = PromptEvolver(self.args, self.logger, config, idea_graph=self.idea_graph)

                # Get domain from current prompt
                with open(task_desc_path, 'r') as f:
                    current_prompt = json.load(f)
                task_domain = current_prompt.get('domain', 'machine learning')
                fix_direction =  current_prompt.get('fix_direction', "")
                # Evolve the prompt (async call)
                evolved_prompt = await prompt_evolver.evolve_prompt(
                    library_path=library_path,
                    current_prompt_path=task_desc_path,
                    output_path=task_desc_path,  # Overwrite the original
                    task_domain=task_domain,
                    fix_direction = fix_direction
                )

                self.logger.info("Prompt successfully evolved based on experience library")

            except Exception as e:
                self.logger.warning(f"Failed to evolve prompt: {str(e)}")
                self.logger.warning("Continuing with original prompt...")
        else:
            self.logger.info("No experience library found, using original prompt")

        # Load the (potentially evolved) prompt
        with open(task_desc_path, 'r') as f:
            params = json.load(f)

        goal = params.get('task_description')
        domain = params.get('domain')
        background = params.get('background', "")
        constraints = params.get('constraints', [])

        if not goal or not domain:
            raise ValueError("Task description and domain are required")

        # For sci tasks, ref_code_path may be None (no reference code)
        ref_code_path = getattr(self.args, 'ref_code_path', None)

        self.session_id = await self.interface.create_session(
            goal_description=goal,
            domain=domain,
            background=background,
            ref_code_path=ref_code_path,
            constraints=constraints
        )

        self.logger.info(f"Session created: {self.session_id}")
    
    
    async def generate_ideas(self):
        """Run MAS to generate ideas"""
        if self.session_id is None:
            await self.load_task()

        async def status_callback(session_id, old_state, new_state):
            # Agent transition logging is handled by OrchestrationAgent
            pass
        
        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']
                
                if self.status == "awaiting_feedback":
                    if self.args.offline_feedback:
                        with open(self.args.offline_feedback, "r") as f:
                            feedback = json.load(f)
                        await self.interface.add_feedback(self.session_id, feedback)
                        self.logger.info(f"Feedback added: {feedback}")
                
                elif self.status == "completed":
                    self.logger.info("Idea generation completed")
                    break
                
                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")
                
                self.logger.info(f"Running session {self.session_id}, iteration {iterations}")
                self.status = await self.interface.run_session(
                    self.session_id,
                    status_callback=status_callback
                )
                
            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise
        
        top_ideas = await self.interface.get_top_ideas(self.session_id)
        self.logger.info(f"Generated {len(top_ideas)} top ideas")

        # Add ideas to IdeaGraph (Long Memory)
        if self.idea_graph is not None:
            try:
                # First, load all historical ideas to build the complete graph
                self.logger.info("Loading historical ideas into IdeaGraph...")
                self._load_historical_ideas_to_graph()
                # Get graph statistics
                stats = self.idea_graph.get_graph_stats()
                self.logger.info(f"IdeaGraph stats: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

                # Perform clustering if enough ideas
                if stats['num_nodes'] >= 3:
                    self.logger.info("Clustering ideas in graph...")
                    self.idea_graph.cluster_ideas(method="louvain")

                    # Log cluster summary
                    clusters = self.idea_graph.get_cluster_summary()
                    self.logger.info(f"Created {len(clusters)} clusters: {dict((k, len(v)) for k, v in clusters.items())}")

            except Exception as e:
                self.logger.warning(f"Failed to add ideas to graph: {e}")

        # Save and visualize ideas
        # Create session-specific directory under output_dir (launch directory)
        # Use output_dir if available (set by launch_discovery.py), otherwise use default
        output_dir = getattr(self.args, 'output_dir', None) or osp.join("results", self.args.task_name)
        session_dir = osp.join(output_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Memory manager saves trajectory as traj_{session_id}.json in base_output_dir
        # Copy it to session directory as traj.json for the new structure
        base_output_dir = getattr(self.args, 'base_output_dir', None) or osp.join("results", self.args.task_name)
        original_traj_path = osp.join(base_output_dir, f"traj_{self.session_id}.json")
        session_traj_path = osp.join(session_dir, "traj.json")

        # Wait a bit for memory manager to save the file
        import time
        for _ in range(10):  # Wait up to 1 second
            if osp.exists(original_traj_path):
                break
            time.sleep(0.1)

        if osp.exists(original_traj_path):
            import shutil
            shutil.copy2(original_traj_path, session_traj_path)
            session_json = session_traj_path
            self.logger.info(f"Trajectory copied to: {session_traj_path}")
        else:
            self.logger.warning(f"Trajectory file not found: {original_traj_path}")
            session_json = None

        try:
            if session_json and osp.exists(session_json):
                vis_output = osp.join(session_dir, "ideas_visualization.pdf")
                vis_tree(session_json, vis_output)
                self.logger.info(f"Visualization saved: {vis_output}")
        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")

        return top_ideas, session_json


# ============================================================================
# GPU Resource Allocator
# ============================================================================
class GPUAllocator:
    """Manages GPU allocation for parallel experiments"""

    def __init__(self, available_gpus, gpu_per_experiment):
        """
        Initialize GPU allocator

        Args:
            available_gpus: List of available GPU IDs (e.g., [0, 1, 2, 3])
            gpu_per_experiment: GPU allocation per experiment (can be fractional, e.g., 0.25)
        """
        self.available_gpus = available_gpus if available_gpus else []
        self.gpu_per_experiment = gpu_per_experiment
        self.total_gpus = len(self.available_gpus)

        # Calculate how many experiments can run in parallel
        if self.total_gpus > 0 and gpu_per_experiment > 0:
            self.max_parallel = int(self.total_gpus / gpu_per_experiment)
        else:
            self.max_parallel = 1

        # Semaphore to control concurrent access
        self.semaphore = Semaphore(self.max_parallel)

        # GPU assignment strategy
        self.next_gpu_idx = 0

        # Lock to ensure thread-safe GPU allocation
        self.allocation_lock = Lock()

    def allocate_gpus(self):
        """
        Allocate GPUs for an experiment (thread-safe)

        Returns:
            List of GPU IDs to use, or empty list if no GPU available
        """
        if not self.available_gpus:
            return []

        # Use lock to ensure thread-safe allocation
        with self.allocation_lock:
            # For fractional GPU allocation (e.g., 0.25), multiple experiments share GPUs
            if self.gpu_per_experiment < 1.0:
                # Round-robin assignment for fractional GPU
                gpu_id = self.available_gpus[self.next_gpu_idx % self.total_gpus]
                self.next_gpu_idx += 1
                return [gpu_id]

            # For >= 1 GPU per experiment
            num_gpus = int(self.gpu_per_experiment)
            start_idx = self.next_gpu_idx % self.total_gpus
            allocated = []

            for i in range(num_gpus):
                gpu_idx = (start_idx + i) % self.total_gpus
                allocated.append(self.available_gpus[gpu_idx])

            self.next_gpu_idx += num_gpus
            return allocated

    def get_gpu_env(self):
        """
        Get GPU IDs for current experiment

        Returns:
            String of comma-separated GPU IDs (e.g., "0,1")
        """
        gpus = self.allocate_gpus()
        return ",".join(map(str, gpus)) if gpus else ""


# ============================================================================
# Experiment Execution Module
# ============================================================================
class ExperimentRunner:
    """Handles experiment execution with different backends"""

    def __init__(self, args, logger, config=None, session_id=None, base_code_dir=None):
        self.args = args
        self.logger = logger
        self.backend = args.exp_backend
        self.config = config or {}
        self.session_id = session_id  # Session ID for organizing results
        self.base_code_dir = base_code_dir or args.task_dir  # Code directory (for incremental mode)

        # Initialize GPU allocator for parallel execution
        self._init_gpu_allocator()

        # Initialize online memory saver if enabled
        self.memory_saver = None
        if config:
            online_memory_config = config.get("memory", {}).get("online_memory", {})
            if online_memory_config.get("enabled", False):
                try:
                    from internagent.mas.memory import OnlineMemorySaver
                    task_name = getattr(args, 'task_name', None)
                    if task_name:
                        self.memory_saver = OnlineMemorySaver(config, task_name)
                        self.logger.info("Online memory saver initialized")
                    else:
                        self.logger.warning("Online memory enabled but task_name not available")
                except Exception as e:
                    self.logger.error(f"Failed to initialize online memory saver: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())

    def _init_gpu_allocator(self):
        """Initialize GPU allocator based on configuration"""
        # Get parallel execution config
        exp_config = self.config.get("experiment", {})
        max_parallel = exp_config.get("max_parallel_experiments", 1)
        gpu_per_experiment = exp_config.get("gpu_per_experiment", 1.0)

        # Detect available GPUs from environment or auto-detection
        available_gpus = []

        # First check CUDA_VISIBLE_DEVICES environment variable
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            try:
                available_gpus = [int(gid.strip()) for gid in cuda_visible.split(',') if gid.strip()]
                self.logger.info(f"Using GPUs from CUDA_VISIBLE_DEVICES: {available_gpus}")
            except ValueError:
                self.logger.warning(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible}")
                available_gpus = []

        # If not set, try to auto-detect using torch
        if not available_gpus:
            try:
                import torch
                if torch.cuda.is_available():
                    available_gpus = list(range(torch.cuda.device_count()))
                    self.logger.info(f"Auto-detected GPUs: {available_gpus}")
                else:
                    self.logger.info("No CUDA devices available, will run on CPU")
            except ImportError:
                self.logger.info("PyTorch not available for GPU detection, will run on CPU")

        # Initialize GPU allocator
        self.gpu_allocator = GPUAllocator(available_gpus, gpu_per_experiment)

        # Calculate actual max parallel experiments
        if available_gpus and gpu_per_experiment > 0:
            gpu_based_max = int(len(available_gpus) / gpu_per_experiment)
            self.max_parallel_experiments = min(max_parallel, gpu_based_max)
        else:
            self.max_parallel_experiments = max_parallel

        self.logger.info(f"GPU allocation initialized:")
        self.logger.info(f"  Available GPUs: {available_gpus}")
        self.logger.info(f"  GPU per experiment: {gpu_per_experiment}")
        self.logger.info(f"  Configured max parallel: {max_parallel}")
        self.logger.info(f"  Actual max parallel: {self.max_parallel_experiments}")

    def _extract_idea_info(self, idea):
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

    def _calculate_experiment_performance(self, folder_name, base_dir):
        """Calculate experiment performance by comparing with baseline."""
        performance = {}

        # Load baseline metrics
        baseline_path = osp.join(base_dir, "run_0", "final_info.json")
        baseline_metrics = self._extract_metrics_from_final_info(baseline_path)
        if not baseline_metrics:
            self.logger.warning("No baseline metrics found")
            return performance

        # Load current metrics from the latest run
        current_metrics = {}
        run_dirs = sorted(glob.glob(osp.join(folder_name, "run_[1-9]*")))
        for run_dir in reversed(run_dirs):
            run_final_info = osp.join(run_dir, "final_info.json")
            current_metrics = self._extract_metrics_from_final_info(run_final_info)
            if current_metrics:
                break

        if not current_metrics:
            self.logger.warning("No current metrics found")
            return performance

        # Calculate improvement rates
        improvement_rates = {}
        for metric, baseline_val in baseline_metrics.items():
            if metric in current_metrics:
                try:
                    baseline_val = float(baseline_val) if isinstance(baseline_val, str) else baseline_val
                    current_val = float(current_metrics[metric]) if isinstance(current_metrics[metric], str) else current_metrics[metric]
                    if baseline_val != 0:
                        improvement_rates[metric] = (current_val - baseline_val) / abs(baseline_val) * 100
                except (ValueError, TypeError):
                    pass

        overall_improvement_rate = sum(improvement_rates.values()) / len(improvement_rates) if improvement_rates else 0.0

        performance = {
            'baseline_metrics': baseline_metrics,
            'current_metrics': current_metrics,
            'improvement_rates': improvement_rates,
            'overall_improvement_rate': overall_improvement_rate
        }
        self.logger.info(f"Performance: improvement={overall_improvement_rate:+.2f}%")
        return performance

    def _extract_metrics_from_final_info(self, path):
        """Extract metrics from final_info.json, handling nested structures."""
        if not osp.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for task_name, task_data in data.items():
                if isinstance(task_data, dict):
                    if 'means' in task_data:
                        return task_data['means']
                    return task_data
        except Exception:
            pass
        return {}

    def setup_experiment_folder(self, base_dir, results_dir, idea):
        """Create experiment folder and setup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"

        # results_dir already includes session directory if session_id exists
        # (created by launch_discovery.py), so no need to create it again
        folder_name = osp.join(results_dir, idea_name)

        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")

        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

        # Ensure experiment.py exists in the experiment folder
        experiment_src = osp.join(base_dir, "experiment.py")
        experiment_dst = osp.join(folder_name, "experiment.py")
        if not osp.exists(experiment_dst):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, experiment_dst)
                self.logger.info(f"Copied experiment.py from {experiment_src} to {experiment_dst}")
            else:
                raise FileNotFoundError(f"experiment.py not found in base directory: {experiment_src}")

        # Backup baseline code to run_0/ for comparison
        # This preserves the current baseline code before any modifications
        # In incremental mode, base_dir contains the updated best code from previous rounds
        run0_dir = osp.join(folder_name, "run_0")
        os.makedirs(run0_dir, exist_ok=True)

        # Always copy experiment.py to run_0/ (overwrite if exists for incremental mode)
        run0_experiment = osp.join(run0_dir, "experiment.py")
        if osp.exists(experiment_src):
            shutil.copy2(experiment_src, run0_experiment)
            self.logger.info(f"Backed up baseline experiment.py to: {run0_experiment}")

        # Always copy final_info.json to run_0/ (overwrite for incremental mode)
        base_run0_final_info = osp.join(base_dir, "run_0", "final_info.json")
        run0_final_info = osp.join(run0_dir, "final_info.json")
        if osp.exists(base_run0_final_info):
            shutil.copy2(base_run0_final_info, run0_final_info)

        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write(f"## Run 0: Baseline\n")

        return folder_name, idea_name
    
    def setup_sci_experiment_folder(self, base_dir, results_dir, idea):
        """
        Create experiment folder for sci_tasks (paper reproduction).

        Symlinks data/, related_work/, target_study/ from the original task_dir.
        In incremental mode (base_dir != task_dir), overlays code/, outputs/, report/
        from the best previous run.
        Creates a default launcher.sh and run_0/final_info.json baseline.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"

        # Atomic folder creation to avoid race conditions in parallel experiments
        folder_name = osp.join(results_dir, idea_name)
        suffix = 0
        while True:
            try:
                os.makedirs(folder_name)  # atomic: raises if exists
                break
            except FileExistsError:
                suffix += 1
                idea_name = f"{timestamp}_{idea_info['name']}_{suffix}"
                folder_name = osp.join(results_dir, idea_name)

        # Always use args.task_dir as the source of scientific assets
        original_task_dir = self.args.task_dir

        # Symlink scientific assets from original task dir (avoid copying large datasets)
        for dir_name in ['data', 'related_work', 'target_study']:
            src = osp.join(original_task_dir, dir_name)
            dst = osp.join(folder_name, dir_name)
            if osp.exists(src):
                os.symlink(osp.abspath(src), dst)

        # Create empty workspace directories
        os.makedirs(osp.join(folder_name, "code"), exist_ok=True)
        os.makedirs(osp.join(folder_name, "outputs"), exist_ok=True)
        os.makedirs(osp.join(folder_name, "report", "images"), exist_ok=True)

        # In incremental mode, overlay best code/outputs/report from base_dir
        if osp.abspath(base_dir) != osp.abspath(original_task_dir):
            for dir_name in ['code', 'outputs', 'report']:
                src = osp.join(base_dir, dir_name)
                if osp.exists(src) and osp.isdir(src):
                    dst = osp.join(folder_name, dir_name)
                    if osp.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    self.logger.info(f"Incremental: overlaid {dir_name}/ from {src}")

        # Generate default launcher.sh
        launcher_path = osp.join(folder_name, "launcher.sh")
        with open(launcher_path, 'w') as f:
            launcher_cmd = self.config.get('sci_task', {}).get(
                'default_launcher', 'python code/experiment.py'
            )
            f.write(f"#!/bin/bash\nset -e\nmkdir -p outputs report/images\n{launcher_cmd}\n")
        os.chmod(launcher_path, 0o755)

        # Set up run_0/final_info.json (baseline = previous best score or 0)
        run0_dir = osp.join(folder_name, "run_0")
        os.makedirs(run0_dir, exist_ok=True)
        run0_final_info = osp.join(run0_dir, "final_info.json")

        base_run0_final_info = osp.join(base_dir, "run_0", "final_info.json")
        if osp.exists(base_run0_final_info):
            shutil.copy2(base_run0_final_info, run0_final_info)
        else:
            baseline = {"sci_task": {"means": {"total_score": 0}}}
            with open(run0_final_info, 'w') as f:
                json.dump(baseline, f, indent=2)

        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, 'w') as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write("## Run 0: Baseline\n")

        self.logger.info(f"Sci experiment folder created: {folder_name}")
        return folder_name, idea_name

    def setup_repo_experiment_folder(self, base_dir, results_dir, idea):
        """Create experiment folder and setup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"

        # results_dir already includes session directory if session_id exists
        # (created by launch_discovery.py), so no need to create it again
        folder_name = osp.join(results_dir, idea_name)

        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")

        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

        # Backup baseline code to run_0/code/ for comparison
        # This preserves the current baseline code before any modifications
        # In incremental mode, base_dir/code/ contains the updated best code from previous rounds
        run0_dir = osp.join(folder_name, "run_0")
        run0_code_dir = osp.join(run0_dir, "code")
        base_code_dir = osp.join(base_dir, "code")

        if osp.exists(base_code_dir):
            os.makedirs(run0_dir, exist_ok=True)
            # Always update run_0/code/ with current base_dir/code/
            # This ensures incremental mode uses the updated baseline from previous rounds
            if osp.exists(run0_code_dir):
                shutil.rmtree(run0_code_dir)
            shutil.copytree(base_code_dir, run0_code_dir)
            self.logger.info(f"Backed up baseline code to: {run0_code_dir}")

        # Also ensure run_0/final_info.json has current baseline metrics
        # In incremental mode, base_dir/run_0/final_info.json contains updated metrics
        base_run0_final_info = osp.join(base_dir, "run_0", "final_info.json")
        run0_final_info = osp.join(run0_dir, "final_info.json")
        if osp.exists(base_run0_final_info):
            os.makedirs(run0_dir, exist_ok=True)
            shutil.copy2(base_run0_final_info, run0_final_info)

        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write(f"## Run 0: Baseline\n")

        return folder_name, idea_name

    def setup_experiment_log(self, folder_name):
        """Create a log file for experiment output (thread-safe, no global redirect)"""
        log_path = osp.join(folder_name, "log.txt")
        log_file = open(log_path, "a", buffering=1)  # Line buffered for real-time output
        return log_file

    def close_experiment_log(self, log_file):
        """Close the experiment log file"""
        if log_file:
            try:
                log_file.close()
            except (ValueError, OSError):
                pass

    def _start_progress_monitor(self, folder_name, idea_name):
        """Start a background thread to monitor experiment progress"""
        stop_event = Event()

        def monitor_progress():
            import time
            start_time = time.time()
            log_path = osp.join(folder_name, "log.txt")

            while not stop_event.is_set():
                elapsed = time.time() - start_time
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)

                # Check log file size to show activity
                log_size = 0
                if osp.exists(log_path):
                    log_size = osp.getsize(log_path) / 1024  # KB

                self.logger.info(f"  [{idea_name}] Running... {elapsed_min}m{elapsed_sec:02d}s | Log: {log_size:.1f}KB")

                # Wait 30 seconds before next update
                stop_event.wait(30)

        monitor_thread = Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        return stop_event, monitor_thread

    def _stop_progress_monitor(self, stop_event, monitor_thread):
        """Stop the progress monitor thread"""
        stop_event.set()
        monitor_thread.join(timeout=1)
    

    def run_claude_experiment(self, base_dir, results_dir, idea, gpu_ids=""):
        """
        Run experiment using Claude Code backend.
        Branches on task_type ('sci' vs 'auto') stored in self.args.task_type.

        Args:
            base_dir: Base directory for the experiment
            results_dir: Results directory
            idea: The idea to experiment with
            gpu_ids: Comma-separated GPU IDs to use (e.g., "0,1")
        """
        task_type = getattr(self.args, 'task_type', 'auto')

        if task_type == 'sci':
            folder_name, idea_name = self.setup_sci_experiment_folder(base_dir, results_dir, idea)
        else:
            folder_name, idea_name = self.setup_repo_experiment_folder(base_dir, results_dir, idea)

        # Create experiment log file (thread-safe, no global redirect)
        log_file = self.setup_experiment_log(folder_name)

        # Start progress monitor
        stop_event, monitor_thread = self._start_progress_monitor(folder_name, idea_name)

        cwd = osp.abspath(folder_name)
        try:
            # Check if MCTS mode is enabled
            use_mcts = self.config.get("experiment", {}).get("use_mcts", False)

            if use_mcts:
                self.logger.info(f"Starting Claude Code MCTS experiment: {idea_name}")
            else:
                self.logger.info(f"Starting Claude Code experiment: {idea_name}")

            if gpu_ids:
                self.logger.info(f"Claude experiment using GPUs: {gpu_ids}")

            experiment_model = (
                self.config.get("experiment", {}).get("model") or
                "claude-sonnet-4-5-20250929"
            )

            if use_mcts:
                success = perform_experiments_mcts_claude(
                    idea,
                    cwd,
                    model=experiment_model,
                    gpu_ids=gpu_ids,
                    log_file=log_file
                )
            else:
                # Get max_runs from config
                max_runs = self.config.get("experiment", {}).get("max_runs", 5)

                # Load sci_task context if needed
                task_info = None
                checklist = None
                if task_type == 'sci':
                    task_info_path = osp.join(self.args.task_dir, "task_info.json")
                    if osp.exists(task_info_path):
                        with open(task_info_path) as _f:
                            task_info = json.load(_f)
                    checklist_path = osp.join(self.args.task_dir, "target_study", "checklist.json")
                    if osp.exists(checklist_path):
                        with open(checklist_path) as _f:
                            checklist = json.load(_f)

                sci_scorer_model = self.config.get('sci_task', {}).get('scorer_model', 'gpt-5.1')
                run_timeout = self.config.get('experiment', {}).get('run_timeout', None)

                success = perform_experiments_claudecode(
                    idea,
                    cwd,
                    model=experiment_model,
                    gpu_ids=gpu_ids,
                    max_runs=max_runs,
                    log_file=log_file,
                    task_type=task_type,
                    task_info=task_info,
                    checklist=checklist,
                    sci_scorer_model=sci_scorer_model,
                    run_timeout=run_timeout,
                )

            self.logger.info(f"Claude Code experiment {'succeeded' if success else 'failed'}: {idea_name}")
            return success, folder_name

        except Exception as e:
            self.logger.error(f"Claude Code experiment error: {str(e)}")
            return False, folder_name
        finally:
            self._stop_progress_monitor(stop_event, monitor_thread)
            self.close_experiment_log(log_file)

    def run_iflow_experiment(self, base_dir, results_dir, idea, gpu_ids=""):
        """
        Run experiment using iFlow backend

        Args:
            base_dir: Base directory for the experiment
            results_dir: Results directory
            idea: The idea to experiment with
            gpu_ids: Comma-separated GPU IDs to use (e.g., "0,1")
        """
        if gpu_ids:
            self.logger.info(f"iFlow experiment using GPUs: {gpu_ids}")

        folder_name, idea_name = self.setup_repo_experiment_folder(base_dir, results_dir, idea)

        # Create experiment log file (thread-safe, no global redirect)
        log_file = self.setup_experiment_log(folder_name)

        # Start progress monitor
        stop_event, monitor_thread = self._start_progress_monitor(folder_name, idea_name)

        cwd = osp.abspath(folder_name)
        try:
            # Check if MCTS mode is enabled
            use_mcts = self.config.get("experiment", {}).get("use_mcts", False)

            if use_mcts:
                self.logger.info(f"Starting iFlow MCTS experiment: {idea_name}")
            else:
                self.logger.info(f"Starting iFlow experiment: {idea_name}")

            experiment_model = (
                self.config.get("experiment", {}).get("model") or
                "claude-sonnet-4-5-20250929"
            )

            if use_mcts:
                success = perform_experiments_mcts_iflow(
                    idea,
                    cwd,
                    model=experiment_model,
                    gpu_ids=gpu_ids,
                    log_file=log_file
                )
            else:
                # Get max_runs from config
                max_runs = self.config.get("experiment", {}).get("max_runs", 5)
                success = perform_experiments_iflow(
                    idea,
                    cwd,
                    model=experiment_model,
                    gpu_ids=gpu_ids,
                    max_runs=max_runs,
                    log_file=log_file
                )

            self.logger.info(f"iFlow experiment {'succeeded' if success else 'failed'}: {idea_name}")
            return success, folder_name

        except Exception as e:
            self.logger.error(f"iFlow experiment error: {str(e)}")
            return False, folder_name
        finally:
            self._stop_progress_monitor(stop_event, monitor_thread)
            self.close_experiment_log(log_file)

    def _run_single_experiment(self, idx, idea, base_dir, results_dir, total_ideas):
        """
        Run a single experiment with GPU allocation

        Args:
            idx: Index of the idea (1-based)
            idea: The idea to experiment with
            base_dir: Base directory for experiments
            results_dir: Results directory
            total_ideas: Total number of ideas

        Returns:
            Dictionary with experiment result
        """
        idea_info = self._extract_idea_info(idea)
        idea_name = idea_info['name']

        # Acquire semaphore to control parallelism
        with self.gpu_allocator.semaphore:
            # Allocate GPUs for this experiment
            gpu_ids = self.gpu_allocator.get_gpu_env()

            self.logger.info(f"[{idx}/{total_ideas}] Starting experiment: {idea_name} (GPUs: {gpu_ids or 'CPU'})")

            try:
                if self.backend == "openhands":
                    success, folder_name = self.run_openhands_experiment(base_dir, results_dir, idea, gpu_ids)
                elif self.backend == "claudecode":
                    success, folder_name = self.run_claude_experiment(base_dir, results_dir, idea, gpu_ids)
                elif self.backend == "iflow":
                    success, folder_name = self.run_iflow_experiment(base_dir, results_dir, idea, gpu_ids)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")

                self.logger.info(f"[{idx}/{total_ideas}] Experiment {'succeeded' if success else 'failed'}: {idea_name}")

                # Load performance metrics and calculate improvement rate
                performance = {}
                if success:
                    performance = self._calculate_experiment_performance(folder_name, base_dir)

                # Save to online memory if enabled and experiment succeeded
                if success and self.memory_saver:
                    try:
                        self.logger.info(f"Saving experiment result to online memory for {idea_name}")
                        # Extract idea info for memory saving
                        idea_info = self._extract_idea_info(idea)
                        # Add additional fields from the full idea dict if available
                        memory_idea = {
                            'name': idea_info['name'],
                            'title': idea_info['title'],
                            'description': idea_info['description'],
                            'method': idea_info['method'],
                            'statement': idea.get('rationale', ''),
                            'score': idea.get('score', 0.0),
                            'rationale': idea.get('rationale', ''),
                            'baseline_summary': idea.get('baseline_summary', ''),
                            'id': idea.get('id', ''),
                            'critiques': idea.get('critiques', []),
                            'evidence': idea.get('evidence', []),
                            'references': idea.get('references', [])
                        }

                        # Get traj path if session_id is available
                        traj_path = None
                        if self.session_id:
                            from pathlib import Path
                            # results_dir already includes session directory
                            traj_file = Path(results_dir) / "traj.json"
                            if traj_file.exists():
                                traj_path = traj_file
                                self.logger.info(f"Using trajectory file: {traj_path}")
                            else:
                                self.logger.warning(f"Trajectory file not found: {traj_file}")

                        self.logger.info(f"Calling memory_saver.save_idea_result for {folder_name}")
                        result = self.memory_saver.save_idea_result(
                            idea=memory_idea,
                            results_dir=Path(folder_name),
                            session_id=self.session_id,
                            traj_path=traj_path
                        )
                        if result:
                            self.logger.info(f"Successfully saved to online memory for {idea_name}")
                        else:
                            self.logger.warning(f"Failed to save to online memory (returned False) for {idea_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save to online memory: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())

                return {
                    'idea_name': idea_name,
                    'success': success,
                    'gpu_ids': gpu_ids,
                    'folder_name': folder_name,
                    'code_path': folder_name,  # For incremental mode
                    'performance': performance  # For finding best result
                }

            except Exception as e:
                self.logger.error(f"[{idx}/{total_ideas}] Failed to run experiment for {idea_name}: {str(e)}")
                return {
                    'idea_name': idea_name,
                    'success': False,
                    'error': str(e),
                    'gpu_ids': gpu_ids,
                    'code_path': '',  # Empty for failed experiments
                    'performance': {}  # Empty for failed experiments
                }

    def run_experiments(self, base_dir, results_dir, ideas):
        """
        Run experiments for all ideas with parallel execution support

        Args:
            base_dir: Base directory for experiments (if None, uses self.base_code_dir)
            results_dir: Results directory
            ideas: List of ideas to experiment with

        Returns:
            List of experiment results
        """
        # Use self.base_code_dir if base_dir is not provided or is empty
        if not base_dir:
            base_dir = self.base_code_dir
            self.logger.info(f"Using base_code_dir from init: {base_dir}")

        results = []
        total_ideas = len(ideas)

        self.logger.info(f"Starting experiments for {total_ideas} ideas")
        self.logger.info(f"Parallel execution mode: {self.max_parallel_experiments} experiments in parallel")

        # Check if we should use sequential execution (backward compatible)
        exp_config = self.config.get("experiment", {})
        max_parallel = exp_config.get("max_parallel_experiments", 1)
        gpu_per_experiment = exp_config.get("gpu_per_experiment", 1.0)

        if max_parallel == 1 and gpu_per_experiment == 1.0:
            # Sequential execution (original behavior)
            self.logger.info("Using sequential execution mode (backward compatible)")

            for idx, idea in enumerate(ideas, 1):
                result = self._run_single_experiment(idx, idea, base_dir, results_dir, total_ideas)
                results.append(result)

        else:
            # Parallel execution
            self.logger.info(f"Using parallel execution mode (max {self.max_parallel_experiments} concurrent)")

            with ThreadPoolExecutor(max_workers=self.max_parallel_experiments) as executor:
                # Submit all experiments
                future_to_idea = {
                    executor.submit(self._run_single_experiment, idx, idea, base_dir, results_dir, total_ideas): (idx, idea)
                    for idx, idea in enumerate(ideas, 1)
                }

                # Collect results as they complete
                for future in as_completed(future_to_idea):
                    idx, idea = future_to_idea[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        idea_info = self._extract_idea_info(idea)
                        self.logger.error(f"Unexpected error for idea {idx}: {str(e)}")
                        results.append({
                            'idea_name': idea_info['name'],
                            'success': False,
                            'error': f"Unexpected error: {str(e)}"
                        })

        # Log summary
        successful = sum(1 for r in results if r.get('success', False))
        self.logger.info(f"Experiments completed: {successful}/{total_ideas} successful")

        return results

# ============================================================================
# Report Writing Module
# ============================================================================
class ReportWriter:
    """Handles report generation without experiments"""
    
    def __init__(self, args, logger, config=None):
        self.args = args
        self.logger = logger
        self.config = config or {}
    
    def _extract_idea_info(self, idea):
        """Extract idea information from different formats"""
        # Try refined_method_details first (from full MAS pipeline)
        if 'refined_method_details' in idea and idea['refined_method_details']:
            details = idea['refined_method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', ''),
                'expected_outcomes': details.get('expected_outcomes', ''),
                'limitations': details.get('limitations', '')
            }
        elif 'method_details' in idea and idea['method_details']:
            details = idea['method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', ''),
                'expected_outcomes': details.get('expected_outcomes', ''),
                'limitations': details.get('limitations', '')
            }
        else:
            name = idea.get('name') or idea.get('title') or 'unnamed_idea'
            title = idea.get('title') or idea.get('name') or 'Untitled'
            description = idea.get('description') or idea.get('content') or ''
            method = idea.get('method') or ''
            expected_outcomes = idea.get('expected_outcomes') or ''
            limitations = idea.get('limitations') or ''
            
            return {
                'name': name[:50] if name else 'unnamed_idea',
                'title': title,
                'description': description,
                'method': method,
                'expected_outcomes': expected_outcomes,
                'limitations': limitations
            }
    
    def setup_report_folder(self, results_dir, idea):
        """Create report folder under results/{task_name}/"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"
        folder_name = osp.join(results_dir, idea_name)
        
        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")
        
        os.makedirs(folder_name, exist_ok=True)
        
        return folder_name, idea_name, idea_info
    
    def generate_markdown_report(self, folder_name, idea_info):
        """Generate markdown report from idea information"""
        report_path = osp.join(folder_name, "report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Title
            f.write(f"# {idea_info['title']}\n\n")
            
            # Metadata
            f.write(f"**Idea Name:** {idea_info['name']}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Description
            f.write("## Description\n\n")
            f.write(f"{idea_info['description']}\n\n")
            
            # Method
            if idea_info['method']:
                f.write("## Method\n\n")
                f.write(f"{idea_info['method']}\n\n")
            
            # Expected Outcomes
            if idea_info['expected_outcomes']:
                f.write("## Expected Outcomes\n\n")
                f.write(f"{idea_info['expected_outcomes']}\n\n")
            
            # Limitations
            if idea_info['limitations']:
                f.write("## Limitations\n\n")
                f.write(f"{idea_info['limitations']}\n\n")
        
        self.logger.info(f"Report generated: {report_path}")
        return report_path
    
    def generate_reports(self, results_dir, ideas):
        """Generate reports for all ideas in results_dir (results/{task_name}/)"""
        results = []
        
        for idx, idea in enumerate(ideas, 1):
            idea_info = self._extract_idea_info(idea)
            idea_name = idea_info['name']
            self.logger.info(f"Processing idea {idx}/{len(ideas)}: {idea_name}")
            
            try:
                folder_name, full_idea_name, idea_info = self.setup_report_folder(
                    results_dir, idea
                )
                
                report_path = self.generate_markdown_report(folder_name, idea_info)
                
                results.append({
                    'idea_name': idea_name,
                    'success': True,
                    'report_path': report_path,
                    'folder': folder_name
                })
                
                self.logger.info(f"Successfully generated report for {idea_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate report for {idea_name}: {str(e)}")
                results.append({
                    'idea_name': idea_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results

