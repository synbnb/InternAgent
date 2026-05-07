"""
Task Memory Layer

Provides unified interface for storing and retrieving experiment records.
"""

# Disable TensorFlow backend before any imports
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from internagent.mas.models.embedding_models import EmbeddingModel
from internagent.mas.models.model_factory import ModelFactory
from internagent.mas.memory.retriever import HybridRetriever
from internagent.mas.agents.exp_analyze_agent import ExpAnalyzeAgent, get_metric_direction_by_pattern as get_metric_direction
import asyncio


@dataclass
class TaskMemRecord:
    """Data structure for storing a memory record with baseline and improved results"""
    record_id: str  # Unique identifier for this record
    name: str
    title: str
    description: str
    statement: str
    method: str

    # Experimental results
    baseline_results: Dict[str, float]  # run_0 results
    improved_results: Dict[str, float]  # average or best of run_1~run_N
    all_run_results: List[Dict[str, float]] = field(default_factory=list)  # All runs including baseline

    # Label: 1 (positive), 0 (neutral), -1 (negative)
    label: int = 0

    # Improvement details
    improvement_rates: Dict[str, float] = field(default_factory=dict)  # per-metric improvement
    overall_improvement_rate: float = 0.0
    primary_metric: Optional[str] = None  # The primary metric used for overall_improvement_rate

    success: bool = False

    # Optional metadata
    task: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None

    def get_label_description(self) -> str:
        """Get human-readable label description"""
        if self.label == 1:
            return f"Positive (↑{self.overall_improvement_rate:.2f}%)"
        elif self.label == -1:
            return f"Negative (↓{abs(self.overall_improvement_rate):.2f}%)"
        else:
            return f"Neutral (~{self.overall_improvement_rate:.2f}%)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskMemRecord':
        """Create from dictionary"""
        return cls(**data)

class TaskMemoryLayer:
    """
    Main memory layer API

    Provides unified interface for:
    - Initializing memory store
    - Retrieving similar ideas
    - Saving experiment results
    - Generating guidance prompts
    """

    def __init__(
        self,
        memory_dir: str = "./config/mem_store",
        embedding_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        label_threshold_percent: float = 5.0,
        embedding_mode: str = "description",
        custom_metric_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialize memory layer

        Args:
            memory_dir: Directory to store memory files
            embedding_config: Configuration for EmbeddingModel
                {
                    "model_type": "local" or "openai" or "azure",
                    "model_name": "",
                    "api_key": "...",
                    "base_url": "...",
                }
            llm_config: Configuration for AnalyzeAgent (follows InternAgent model config pattern)
                {
                    "provider": "openai",  # Model provider (openai/azure/custom)
                    "model_name": "gpt-4",  # Model name
                    "api_key": "...",  # API key
                    "temperature": 0.7,  # Optional: sampling temperature
                    "max_tokens": 4096,  # Optional: max tokens
                    "timeout": 600,  # Optional: timeout in seconds

                    # Agent-specific configuration
                    "custom_metric_config": {...},  # Optional: custom metric directions
                    "use_llm_for_metric_direction": True,  # Optional: use LLM for unknown metrics
                    "primary_metric": "val/PQ_Vm_rmse",  # Optional: specify primary metric
                    "use_llm_for_primary_metric": True,  # Optional: use LLM to auto-select primary metric
                }
            label_threshold_percent: Threshold for labeling ideas
            embedding_mode: Mode for text extraction (title/description/method/full)
            custom_metric_config: Custom metric direction configuration (can also be in llm_config)
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.label_threshold_percent = label_threshold_percent
        self.embedding_mode = embedding_mode

        # Initialize embedding model
        embedding_config = embedding_config or {"model_type": "local"}
        self.embedding_model = EmbeddingModel(**embedding_config)

        # Initialize LLM for analysis (optional)
        self.analyze_agent = None
        if llm_config:
            # Allow custom_metric_config at both levels
            if custom_metric_config and "custom_metric_config" not in llm_config:
                llm_config["custom_metric_config"] = custom_metric_config

            # Create model from config
            try:
                model = ModelFactory.create_model(llm_config)
                self.analyze_agent = ExpAnalyzeAgent(model, llm_config)
            except Exception as e:
                print(f"Warning: Failed to initialize ExpAnalyzeAgent: {e}")
                print(f"  Memory will use simple comparison without LLM")
                self.analyze_agent = None

        # Initialize hybrid retriever
        self.retriever = HybridRetriever(self.embedding_model)

        # Load existing memory if available
        self.records: List[TaskMemRecord] = []
        self._load_memory()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TaskMemoryLayer':
        """
        Create TaskMemoryLayer from configuration dictionary (InternAgent config pattern)

        This factory method enables consistent initialization from InternAgent's
        configuration system.

        Args:
            config: Configuration dictionary following InternAgent pattern:
                {
                    "task_memory": {
                        "memory_dir": "./memory_store",
                        "embedding": {
                            "model_type": "local",
                            "model_name": "..."
                        },
                        "label_threshold_percent": 0.0,
                        "embedding_mode": "description"
                    },
                    "exp_analyze": {  # Set to None to disable
                        "provider": "openai",
                        "model_name": "gpt-4",
                        "api_key": "...",
                        "primary_metric": "val/rmse",
                        "use_llm_for_primary_metric": True,
                        "custom_metric_config": {...}
                    }
                }

        Returns:
            Configured TaskMemoryLayer instance
        """
        # Extract task_memory config
        if "task_memory" not in config:
            raise ValueError("Configuration must contain 'task_memory' section")

        task_memory_config = config["task_memory"]

        # Extract parameters from task_memory section
        memory_dir = task_memory_config.get("memory_dir", "./config/mem_store")
        label_threshold = task_memory_config.get("label_threshold_percent", 5.0)
        embedding_mode = task_memory_config.get("embedding_mode", "description")

        # Build embedding config
        embedding_config = None
        if "embedding" in task_memory_config:
            embedding_config = task_memory_config["embedding"].copy()

        # Build LLM config for ExpAnalyzeAgent from agents.exp_analyze
        llm_config = None
        if "agents" in config and isinstance(config["agents"], dict):
            agent_config = config["agents"].get("exp_analyze", None)
            if agent_config is not None and isinstance(agent_config, dict):
                llm_config = agent_config.copy()

        # Extract custom_metric_config if at top level
        custom_metric_config = config.get("custom_metric_config", None)

        return cls(
            memory_dir=memory_dir,
            embedding_config=embedding_config,
            llm_config=llm_config,
            label_threshold_percent=label_threshold,
            embedding_mode=embedding_mode,
            custom_metric_config=custom_metric_config
        )

    def _save_record_source(
        self,
        record_id: str,
        idea_name: str,
        experiment_path: Path,
        traj_path: Optional[Path] = None
    ):
        """
        Save or update the source information for a memory record
        Saves to CSV format with newest records at the top

        Args:
            record_id: Unique record identifier
            idea_name: Name of the idea
            experiment_path: Path to the experiment results directory
            traj_path: Optional path to the trajectory session file
        """
        import csv

        sources_file = self.memory_dir / "source.csv"

        # Prepare new record
        new_record = {
            "record_id": record_id,
            "idea_name": idea_name,
            "experiment_path": str(experiment_path.absolute()),
            "traj_path": str(traj_path.absolute()) if traj_path else "",
            "last_updated": datetime.now().isoformat()
        }

        # Load existing records
        existing_records = []
        if sources_file.exists():
            try:
                with open(sources_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    existing_records = [row for row in reader if row['record_id'] != record_id]
            except Exception as e:
                print(f"Warning: Failed to load source.csv: {e}")
                existing_records = []

        # Insert new record at the beginning (newest first)
        all_records = [new_record] + existing_records

        # Save back to CSV
        try:
            with open(sources_file, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['record_id', 'idea_name', 'experiment_path', 'traj_path', 'last_updated']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)
        except Exception as e:
            print(f"Warning: Failed to save source.csv: {e}")

    def _load_memory(self):
        """Load existing memory from disk"""
        records_file = self.memory_dir / "records.json"
        # Also support legacy "ideas.json" filename
        if not records_file.exists():
            records_file = self.memory_dir / "ideas.json"

        if records_file.exists():
            with open(records_file, 'r', encoding='utf-8') as f:
                records_data = json.load(f)

            self.records = [TaskMemRecord(**item) for item in records_data]

            # Deduplicate by record_id
            seen_ids = set()
            unique_records = []
            duplicates_removed = 0

            for record in self.records:
                if record.record_id not in seen_ids:
                    seen_ids.add(record.record_id)
                    unique_records.append(record)
                else:
                    duplicates_removed += 1
                    print(f"  Warning: Duplicate record_id found and removed: {record.record_id}")

            if duplicates_removed > 0:
                print(f"  Removed {duplicates_removed} duplicate records (total: {len(self.records)} → {len(unique_records)})")
                self.records = unique_records
                # Save cleaned records back to disk
                self._save_memory()
                print(f"  Saved cleaned records to disk")

            # Try to load persisted embeddings and index
            embeddings_file = self.memory_dir / "embeddings.npy"
            index_file = self.memory_dir / "faiss.index"

            if self.records:
                # Check if persisted embeddings exist and are valid
                if embeddings_file.exists() and index_file.exists():
                    try:
                        # Load embeddings and index from disk
                        texts = [self._extract_text(record) for record in self.records]
                        self.retriever.load_index(
                            self.records,
                            texts,
                            str(embeddings_file),
                            str(index_file)
                        )
                        print(f"Loaded {len(self.records)} records from memory (with cached embeddings)")
                    except Exception as e:
                        print(f"Warning: Failed to load cached embeddings: {e}")
                        print("Rebuilding embeddings from scratch...")
                        # Delete corrupted cache files
                        if embeddings_file.exists():
                            embeddings_file.unlink()
                        if index_file.exists():
                            index_file.unlink()
                        # Rebuild
                        texts = [self._extract_text(record) for record in self.records]
                        self.retriever.build_index(self.records, texts)
                        # Save new cache
                        try:
                            self.retriever.save_index(str(embeddings_file), str(index_file))
                            print("Saved rebuilt embeddings cache")
                        except:
                            pass
                else:
                    # No cached embeddings, rebuild from scratch
                    print(f"No cached embeddings found, building from scratch...")
                    texts = [self._extract_text(record) for record in self.records]
                    self.retriever.build_index(self.records, texts)
                    # Save cache for next time
                    try:
                        self.retriever.save_index(str(embeddings_file), str(index_file))
                        print("Saved embeddings cache")
                    except:
                        pass
                    print(f"Loaded {len(self.records)} records from memory")

    def _clean_nan_values(self, data):
        """
        Recursively clean NaN and Inf values from data structure
        Replaces them with None for valid JSON serialization
        """
        if isinstance(data, dict):
            return {k: self._clean_nan_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_nan_values(item) for item in data]
        elif isinstance(data, float):
            if np.isnan(data) or np.isinf(data):
                return None
            return data
        else:
            return data

    def _save_memory(self):
        """Save memory to disk"""
        # Save records
        records_file = self.memory_dir / "records.json"
        with open(records_file, 'w', encoding='utf-8') as f:
            records_data = [asdict(record) for record in self.records]
            # Clean NaN/Inf values before saving to JSON
            records_data = self._clean_nan_values(records_data)
            json.dump(records_data, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            "embedding_model_type": self.embedding_model.model_type,
            "embedding_model_name": self.embedding_model.model_name,
            "dimension": self.embedding_model.dimension,
            "embedding_mode": self.embedding_mode,
            "label_threshold_percent": self.label_threshold_percent,
        }
        metadata_file = self.memory_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings and FAISS index
        embeddings_file = self.memory_dir / "embeddings.npy"
        index_file = self.memory_dir / "faiss.index"
        try:
            self.retriever.save_index(str(embeddings_file), str(index_file))
            print(f"Saved {len(self.records)} records to {self.memory_dir} (with embeddings cache)")
        except Exception as e:
            print(f"Warning: Failed to save embeddings cache: {e}")
            print(f"Saved {len(self.records)} records to {self.memory_dir}")

    def _extract_text(self, record: TaskMemRecord) -> str:
        """Extract text from record based on embedding_mode"""
        if self.embedding_mode == "title":
            return record.title
        elif self.embedding_mode == "description":
            return record.description
        elif self.embedding_mode == "method":
            return record.method
        elif self.embedding_mode == "full":
            return f"{record.title}\n{record.description}\n{record.statement}\n{record.method}"
        else:
            return record.description

    def retrieve_similar_records(
        self,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5,
        label_filter: Optional[int] = None,
        min_score: float = 0.0
    ) -> List[Tuple[TaskMemRecord, float]]:
        """
        Retrieve similar records using hybrid search

        Args:
            query_text: Query text (record description)
            top_k: Number of results to return
            alpha: Weight for BM25 (1-alpha for vector). Range [0, 1]
            label_filter: Only return records with this label (1/0/-1)
            min_score: Minimum similarity score threshold

        Returns:
            List of (TaskMemRecord, score) tuples
        """
        if not self.records:
            return []

        # Search
        results = self.retriever.search(query_text, top_k=top_k * 2, alpha=alpha)

        # Apply filters
        filtered_results = []
        for record, score in results:
            if label_filter is not None and record.label != label_filter:
                continue
            if score < min_score:
                continue
            filtered_results.append((record, score))

        return filtered_results[:top_k]

    def generate_guidance_prompt(
        self,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> str:
        """
        Generate guidance prompt based on similar records

        Args:
            query_text: Current idea description
            top_k: Number of similar records to retrieve
            alpha: Hybrid search weight

        Returns:
            Guidance prompt string
        """
        similar_records = self.retrieve_similar_records(query_text, top_k=top_k, alpha=alpha)

        if not similar_records:
            return "No similar records found in memory. This is a novel direction."

        # Count labels
        positive_count = sum(1 for record, _ in similar_records if record.label == 1)
        negative_count = sum(1 for record, _ in similar_records if record.label == -1)
        neutral_count = sum(1 for record, _ in similar_records if record.label == 0)

        positive_ratio = positive_count / len(similar_records)
        negative_ratio = negative_count / len(similar_records)

        # Determine recommendation
        if positive_ratio >= 0.5:
            recommendation = "accept"
            message = "This direction has shown strong positive results in similar ideas."
        elif negative_ratio >= 0.5:
            recommendation = "reject"
            message = "This direction has shown poor results in similar ideas."
        elif positive_count > negative_count:
            recommendation = "cautious_accept"
            message = "This direction has mixed results, but leans positive."
        elif negative_count > positive_count:
            recommendation = "cautious_reject"
            message = "This direction has mixed results, but leans negative."
        else:
            recommendation = "uncertain"
            message = "This direction has highly uncertain outcomes."

        # Build prompt
        prompt_parts = [
            f"## Guidance from Memory",
            f"",
            f"**Recommendation**: {recommendation.upper()}",
            f"**Reasoning**: {message}",
            f"",
            f"**Similar Records Analysis** (found {len(similar_records)} similar records):",
            f"- Positive outcomes: {positive_count}",
            f"- Neutral outcomes: {neutral_count}",
            f"- Negative outcomes: {negative_count}",
            f"",
        ]

        # Add examples
        if positive_count > 0:
            prompt_parts.append("**Successful Similar Records:**")
            for record, score in similar_records:
                if record.label == 1:
                    prompt_parts.append(f"- {record.name}: {record.title}")
                    prompt_parts.append(f"  Improvement: {record.overall_improvement_rate:.2f}%")
                    prompt_parts.append(f"  Description: {record.description[:200]}...")
            prompt_parts.append("")

        if negative_count > 0:
            prompt_parts.append("**Failed Similar Records:**")
            for record, score in similar_records:
                if record.label == -1:
                    prompt_parts.append(f"- {record.name}: {record.title}")
                    prompt_parts.append(f"  Degradation: {record.overall_improvement_rate:.2f}%")
                    prompt_parts.append(f"  Description: {record.description[:200]}...")
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def save_experiment_result(
        self,
        idea: Dict[str, Any],
        results_dir: Path,
        task_name: str,
        session_id: Optional[str] = None,
        aggregation: str = "best",
        traj_path: Optional[Path] = None
    ) -> TaskMemRecord:
        """
        Save experiment result to memory

        Args:
            idea: Idea dictionary with name, title, description, statement, method
            results_dir: Path to experiment results directory (contains run_0, run_1, ...)
            task_name: Task name
            session_id: Session ID
            aggregation: How to aggregate improved runs (best/avg/last)
            traj_path: Optional path to the trajectory session file

        Returns:
            Created TaskMemRecord
        """
        idea_name = idea["name"]

        # Load baseline (run_0)
        run_0_dir = results_dir / "run_0"
        baseline_results = None
        if run_0_dir.exists() and self.analyze_agent:
            baseline_results = asyncio.run(
                self.analyze_agent.analyze_result_file(run_0_dir / "final_info.json")
            )
        elif run_0_dir.exists():
            # Fallback to simple loading
            baseline_results = self._load_result_simple(run_0_dir / "final_info.json")

        if not baseline_results:
            print(f"Warning: No baseline results found for {idea_name}")
            return None

        # Load improved runs (run_1, run_2, ...)
        improved_runs = []
        all_run_results = [baseline_results.copy()]

        for run_dir in sorted(results_dir.glob("run_*")):
            # Skip non-directories (e.g., run_analysis.txt)
            if not run_dir.is_dir():
                continue

            # Try to parse run number, skip if not a valid format
            try:
                run_num = int(run_dir.name.split("_")[1])
            except (ValueError, IndexError):
                # Skip files like run_analysis.txt or invalid formats
                continue

            if run_num == 0:
                continue

            if self.analyze_agent:
                result = asyncio.run(
                    self.analyze_agent.analyze_result_file(run_dir / "final_info.json")
                )
            else:
                result = self._load_result_simple(run_dir / "final_info.json")

            if result:
                improved_runs.append(result)
                all_run_results.append(result.copy())

        # Aggregate improved results
        improved_results = self._aggregate_results(improved_runs, aggregation)

        # Check if we have improved results
        if not improved_results:
            print(f"Warning: No improved runs found for {idea_name}, only baseline exists")
            print(f"  → Skipping save. Ideas without improved runs cannot be evaluated.")
            return None

        # Compute label
        primary_metric_used = None
        if self.analyze_agent and improved_results:
            improvement_rates, overall_rate, label, primary_metric_used = asyncio.run(
                self.analyze_agent.compare_results(
                    baseline_results, improved_results, self.label_threshold_percent, task_name=task_name
                )
            )
        else:
            # Simple fallback: compute improvement without LLM
            improvement_rates, overall_rate, label = self._simple_compare_results(
                baseline_results, improved_results, self.label_threshold_percent
            )
            # For simple comparison, we use average, so no single primary metric
            primary_metric_used = "average_of_all_metrics"

        # Create record
        record = TaskMemRecord(
            record_id=f"{task_name}_{idea_name}",
            name=idea_name,
            title=idea.get("title", ""),
            description=idea.get("description", ""),
            statement=idea.get("statement", ""),
            method=idea.get("method", ""),
            baseline_results=baseline_results,
            improved_results=improved_results or {},
            all_run_results=all_run_results,
            label=label,
            improvement_rates=improvement_rates,
            overall_improvement_rate=overall_rate,
            primary_metric=primary_metric_used,
            success=(label == 1),
            task=task_name,
            timestamp=str(results_dir.name.split("_")[0]),
            session_id=session_id or "unknown"
        )

        # Check for duplicate record_id before adding
        existing_ids = {r.record_id for r in self.records}
        if record.record_id in existing_ids:
            print(f"  ⚠ Warning: Record with ID '{record.record_id}' already exists in memory!")
            print(f"  → Skipping to prevent duplicate. This might indicate a bug in the import logic.")
            return None

        # Add to memory
        self.records.append(record)

        # Incremental update: only compute embedding for the new record
        new_text = self._extract_text(record)

        # Check if we have an existing index to update incrementally
        if self.retriever.vector_index is not None and len(self.records) > 1:
            # Incremental update
            self.retriever.add_to_index([record], [new_text])
        else:
            # Build from scratch (first record or no existing index)
            texts = [self._extract_text(r) for r in self.records]
            self.retriever.build_index(self.records, texts)

        # Save to disk (including embeddings cache)
        self._save_memory()

        # Save source tracking information
        self._save_record_source(
            record_id=record.record_id,
            idea_name=idea_name,
            experiment_path=results_dir,
            traj_path=traj_path
        )

        print(f"✓ Saved: {idea_name} (label={label}, improvement={overall_rate:.2f}%, primary_metric={primary_metric_used})")

        return record

    def _load_result_simple(self, result_file: Path) -> Optional[Dict[str, float]]:
        """Simple result loading without LLM"""
        if not result_file.exists():
            return None

        with open(result_file, 'r') as f:
            data = json.load(f)
            if data:
                first_key = list(data.keys())[0]

                # Format 1: {"dataset": {"means": {...}}}
                if "means" in data[first_key]:
                    results = data[first_key]["means"]
                    # Filter out epoch and try to convert values to float
                    converted_results = {}
                    for k, v in results.items():
                        if k == "epoch":
                            continue
                        # Try to convert to float, skip NaN/Inf values
                        try:
                            float_val = float(v)
                            # Skip NaN and Inf values
                            if np.isnan(float_val) or np.isinf(float_val):
                                continue
                            converted_results[k] = float_val
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            continue
                    return converted_results

                # Format 2: {"dataset": {"metric1": {"mean": ..., "std": ...}, ...}}
                else:
                    # Try to extract 'mean' values from nested dicts
                    converted_results = {}
                    for metric_name, metric_data in data[first_key].items():
                        if isinstance(metric_data, dict) and 'mean' in metric_data:
                            # Extract mean value, skip NaN/Inf
                            try:
                                float_val = float(metric_data['mean'])
                                if not (np.isnan(float_val) or np.isinf(float_val)):
                                    converted_results[metric_name] = float_val
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                continue
                        elif isinstance(metric_data, (int, float, str)):
                            # Direct value, skip NaN/Inf
                            try:
                                float_val = float(metric_data)
                                if not (np.isnan(float_val) or np.isinf(float_val)):
                                    converted_results[metric_name] = float_val
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                continue

                    if converted_results:
                        return converted_results

        return None

    def _aggregate_results(
        self,
        improved_runs: List[Dict[str, float]],
        aggregation: str
    ) -> Optional[Dict[str, float]]:
        """Aggregate improved run results"""
        if not improved_runs:
            return None

        if aggregation == "avg":
            improved_results = {}
            all_metrics = set()
            for run in improved_runs:
                all_metrics.update(run.keys())

            for metric in all_metrics:
                values = [run[metric] for run in improved_runs if metric in run]
                if values:
                    # Try to convert to numeric values, skip NaN/Inf
                    numeric_values = []
                    for v in values:
                        try:
                            float_val = float(v)
                            # Skip NaN and Inf values
                            if not (np.isnan(float_val) or np.isinf(float_val)):
                                numeric_values.append(float_val)
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            pass

                    if numeric_values:
                        improved_results[metric] = float(np.mean(numeric_values))
                    # else: skip this metric if all values are non-numeric or NaN

        elif aggregation == "best":
            improved_results = {}
            all_metrics = set()
            for run in improved_runs:
                all_metrics.update(run.keys())

            for metric in all_metrics:
                values = [run[metric] for run in improved_runs if metric in run]
                if values:
                    # Try to convert to numeric values
                    numeric_values = []
                    for v in values:
                        try:
                            numeric_values.append(float(v))
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            pass

                    if numeric_values:
                        direction = get_metric_direction(metric)
                        if direction == "lower":
                            improved_results[metric] = float(np.min(numeric_values))
                        elif direction == "higher":
                            improved_results[metric] = float(np.max(numeric_values))
                        else:
                            improved_results[metric] = float(np.mean(numeric_values))
                    else:
                        # If all values are non-numeric, use the first value
                        improved_results[metric] = values[0]

        elif aggregation == "last":
            improved_results = improved_runs[-1].copy()

        else:
            improved_results = improved_runs[-1].copy()

        return improved_results

    def _simple_compare_results(
        self,
        baseline_results: Dict[str, float],
        improved_results: Dict[str, float],
        threshold_percent: float = 5.0
    ) -> Tuple[Dict[str, float], float, int]:
        """
        Simple comparison without LLM - uses pattern matching for metric direction

        Args:
            baseline_results: Baseline metrics
            improved_results: Improved metrics
            threshold_percent: Threshold for labeling (set to 0 for binary labels)

        Returns:
            (improvement_rates, overall_improvement_rate, label)
        """
        if not improved_results:
            return {}, 0.0, 0

        improvement_rates = {}

        for metric in baseline_results:
            if metric not in improved_results:
                continue

            baseline_val = baseline_results[metric]
            improved_val = improved_results[metric]

            # Skip if baseline is zero or if values are non-numeric
            try:
                baseline_val = float(baseline_val)
                improved_val = float(improved_val)
            except (ValueError, TypeError):
                continue

            if baseline_val == 0:
                continue

            # Use pattern matching to determine direction
            direction = get_metric_direction(metric)

            if direction == "lower":
                # Lower is better (e.g., loss, error)
                improvement_rate = (baseline_val - improved_val) / abs(baseline_val) * 100
            elif direction == "higher":
                # Higher is better (e.g., accuracy)
                improvement_rate = (improved_val - baseline_val) / abs(baseline_val) * 100
            else:
                # Unknown direction, assume higher is better
                improvement_rate = (improved_val - baseline_val) / abs(baseline_val) * 100

            improvement_rates[metric] = improvement_rate

        # Overall improvement rate
        if improvement_rates:
            overall_improvement_rate = float(np.mean(list(improvement_rates.values())))
        else:
            # No valid metrics to compare - this shouldn't happen if we have valid results
            # But if it does, treat as failed comparison
            print("Warning: No valid metrics found for comparison (all metrics skipped)")
            overall_improvement_rate = 0.0

        # Assign label
        if overall_improvement_rate > threshold_percent:
            label = 1
        elif overall_improvement_rate < -threshold_percent:
            label = -1
        else:
            # When threshold is 0, this means exactly 0% improvement
            # Or when threshold > 0, this means improvement is within [-threshold, +threshold]
            label = 0

        return improvement_rates, overall_improvement_rate, label

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.records:
            return {"total_records": 0}

        positive_count = sum(1 for record in self.records if record.label == 1)
        negative_count = sum(1 for record in self.records if record.label == -1)
        neutral_count = sum(1 for record in self.records if record.label == 0)

        success_count = sum(1 for record in self.records if record.success)

        tasks = set(record.task for record in self.records)

        stats = {
            "total_records": len(self.records),
            "label_distribution": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
            },
            "success_rate": f"{success_count / len(self.records) * 100:.1f}%",
            "tasks": sorted(list(tasks)),
        }

        # Positive records stats
        positive_records = [record for record in self.records if record.label == 1]
        if positive_records:
            improvements = [record.overall_improvement_rate for record in positive_records]
            stats["positive_records_stats"] = {
                "avg_improvement": float(np.mean(improvements)),
                "min_improvement": float(np.min(improvements)),
                "max_improvement": float(np.max(improvements)),
            }

        return stats
