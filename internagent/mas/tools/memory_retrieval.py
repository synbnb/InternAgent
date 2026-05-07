"""
Memory Retrieval Tool

Provides task memory retrieval functionality for LLM agents to query historical
experiment results and learn from past experiences. Supports semantic search with
hybrid BM25+vector retrieval and provides guidance based on historical success/failure patterns.
"""

# Disable TensorFlow backend before any imports
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..memory import TaskMemoryLayer

logger = logging.getLogger(__name__)

# Global memory layer cache
_memory_cache: Dict[str, TaskMemoryLayer] = {}


def _generate_recommendation(stats: Dict[str, Any]) -> tuple:
    """Generate recommendation based on statistics"""
    positive_ratio = stats['positive_ratio']
    negative_ratio = stats['negative_ratio']
    positive_count = stats['positive']
    negative_count = stats['negative']

    if positive_ratio >= 0.5:
        recommendation = "accept"
        reasoning = "This direction has shown strong positive results in similar ideas."
    elif negative_ratio >= 0.5:
        recommendation = "reject"
        reasoning = "This direction has shown poor results in similar ideas. Consider alternative approaches."
    elif positive_count > negative_count:
        recommendation = "cautious_accept"
        reasoning = "This direction has mixed results, but leans positive. Proceed with careful validation."
    elif negative_count > positive_count:
        recommendation = "cautious_reject"
        reasoning = "This direction has mixed results, but leans negative. Consider improvements."
    else:
        recommendation = "uncertain"
        reasoning = "This direction has highly uncertain outcomes. Requires careful experimentation."

    return recommendation, reasoning


def _format_guidance_prompt(records: List[Dict], recommendation: str, reasoning: str, stats: Dict) -> str:
    """Format guidance prompt for LLM"""
    prompt = "## Guidance from Memory\n\n"
    prompt += f"**Recommendation**: {recommendation.upper()}\n"
    prompt += f"**Reasoning**: {reasoning}\n\n"

    prompt += f"**Similar Records Analysis** (found {stats['total']} similar records):\n"
    prompt += f"- Positive outcomes: {stats['positive']}\n"
    prompt += f"- Negative outcomes: {stats['negative']}\n"
    prompt += f"- Neutral outcomes: {stats['neutral']}\n\n"

    if records:
        prompt += "**Top Similar Records**:\n"
        for i, record in enumerate(records[:5], 1):
            label_symbol = {1: "[+]", 0: "[0]", -1: "[-]"}[record['label']]
            prompt += f"{i}. {label_symbol} {record['name']} (similarity: {record['similarity_score']:.2f})\n"
            prompt += f"   Description: {record['description'][:100]}...\n"
            if record.get('improvement'):
                prompt += f"   Outcome: {record['improvement']}\n"
            prompt += "\n"

    return prompt


class TaskMemoryRetriever:
    """
    Task memory retriever with pre-configured parameters.

    Initialize once with all configuration, then retrieve with only query parameter.
    """

    def __init__(
        self,
        task_name: str,
        memory_dir: str = "./config/mem_store",
        top_k: int = 5,
        alpha: float = 0.5,
        label_filter: Optional[int] = None,
        min_score: float = 0.0,
        include_details: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize memory retriever with configuration

        Args:
            task_name: Name of the task (e.g., "AutoCls2D")
            memory_dir: Base memory directory
            top_k: Number of results to retrieve (max: 20)
            alpha: Weight for BM25 vs vector (0-1)
            label_filter: Filter by label (1/0/-1)
            min_score: Minimum similarity threshold
            include_details: Include detailed experiment info
            config: Optional full config dict (for embedding model configuration)
        """
        self.task_name = task_name
        self.memory_dir = memory_dir
        self.top_k = min(max(1, top_k), 20)
        self.alpha = max(0.0, min(1.0, alpha))
        self.label_filter = label_filter
        self.min_score = min_score
        self.include_details = include_details
        self.config = config

        logger.info(f"TaskMemoryRetriever initialized: task={task_name}, "
                   f"dir={memory_dir}, top_k={top_k}, alpha={alpha}")

    async def retrieve(self, query) -> Dict[str, Any]:
        """
        Retrieve similar experiment results with only query parameter

        Args:
            query: Research idea query (str or dict with goal info)

        Returns:
            Dictionary containing retrieval results
        """
        # Convert query to string if it's a dict (goal object)
        if isinstance(query, dict):
            query_parts = []
            if query.get("description"):
                query_parts.append(query["description"])
            if query.get("domain"):
                query_parts.append(f"Domain: {query['domain']}")
            if query.get("constraints"):
                constraints = query["constraints"]
                if isinstance(constraints, list):
                    query_parts.append(f"Constraints: {', '.join(str(c) for c in constraints)}")
                else:
                    query_parts.append(f"Constraints: {constraints}")
            if query.get("background"):
                query_parts.append(f"Background: {query['background'][:200]}")

            query_text = "\n".join(query_parts) if query_parts else str(query)
            logger.info(f"Converted goal dict to query text ({len(query_text)} chars)")
        elif isinstance(query, str):
            query_text = query
        else:
            return {
                "success": False,
                "error": f"Invalid query type: must be string or dict, got {type(query)}"
            }

        # Validate query text
        if not query_text or not query_text.strip():
            return {
                "success": False,
                "error": "Invalid query: empty after processing"
            }

        logger.info(f"Memory retrieval for task '{self.task_name}': {query_text[:100]}...")

        try:
            # Load task memory with config for embedding model
            memory = _load_task_memory(self.task_name, self.memory_dir, self.config)

            if len(memory.records) == 0:
                return {
                    "success": True,
                    "query": query_text,
                    "task_name": self.task_name,
                    "recommendation": "novel",
                    "reasoning": "No historical records found. This is a novel direction.",
                    "statistics": {"total": 0, "positive": 0, "neutral": 0, "negative": 0},
                    "similar_records": [],
                    "guidance_prompt": "No similar records found in memory. This is a novel direction."
                }

            # Retrieve similar records
            logger.info(f"Searching {len(memory.records)} records with top_k={self.top_k}, alpha={self.alpha}")
            similar_records = memory.retrieve_similar_records(
                query_text=query_text,
                top_k=self.top_k,
                alpha=self.alpha,
                label_filter=self.label_filter,
                min_score=self.min_score
            )

            if not similar_records:
                return {
                    "success": True,
                    "query": query_text,
                    "task_name": self.task_name,
                    "recommendation": "novel",
                    "reasoning": "No sufficiently similar records found.",
                    "statistics": {"total": 0, "positive": 0, "neutral": 0, "negative": 0},
                    "similar_records": [],
                    "guidance_prompt": "No sufficiently similar records found. This appears to be a novel direction."
                }

            # Calculate statistics
            # similar_records is a list of (record, score) tuples
            labels = [record.label for record, score in similar_records]
            stats = {
                'total': len(labels),
                'positive': sum(1 for l in labels if l == 1),
                'neutral': sum(1 for l in labels if l == 0),
                'negative': sum(1 for l in labels if l == -1)
            }
            stats['positive_ratio'] = stats['positive'] / stats['total'] if stats['total'] > 0 else 0
            stats['negative_ratio'] = stats['negative'] / stats['total'] if stats['total'] > 0 else 0

            # Generate recommendation
            recommendation, reasoning = _generate_recommendation(stats)

            # Format records with details if requested
            from dataclasses import asdict

            formatted_records = []
            for record, score in similar_records:
                if self.include_details:
                    # Include all fields from dataclass
                    formatted = asdict(record)
                else:
                    # Only include essential fields
                    formatted = {
                        'name': record.name,
                        'title': record.title,
                        'description': record.description,
                        'label': record.label,
                        'method': record.method
                    }

                # Add computed fields
                formatted['label_text'] = {1: 'successful', 0: 'neutral', -1: 'failed'}[record.label]
                formatted['similarity_score'] = round(score, 4)

                formatted_records.append(formatted)

            # Generate guidance prompt
            guidance = _format_guidance_prompt(formatted_records, recommendation, reasoning, stats)

            return {
                "success": True,
                "query": query_text,
                "task_name": self.task_name,
                "recommendation": recommendation,
                "reasoning": reasoning,
                "statistics": stats,
                "similar_records": formatted_records,
                "guidance_prompt": guidance
            }

        except FileNotFoundError as e:
            error_msg = f"Task memory not found: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "query": query_text,
                "task_name": self.task_name,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Memory retrieval failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "query": query_text,
                "task_name": self.task_name,
                "error": error_msg
            }


MEMORY_RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_task_memory",
        "description": (
            "Retrieve historical experiment results and learnings from task memory. "
            "Search for similar ideas that have been tried before and get guidance "
            "on whether to pursue or avoid certain research directions based on past outcomes. "
            "Helpful for avoiding repeated failures and building on successful approaches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Description of the research idea or direction to query. "
                        "Should be a clear, detailed description of the method or approach. "
                        "Example: 'Using graph neural networks for molecular property prediction', "
                        "'Implementing attention mechanism in sequence modeling'"
                    )
                }
            },
            "required": ["query"]
        }
    }
}


def _find_task_directory(task_name: str, memory_dir: str) -> Optional[Path]:
    """
    Find task directory with exact matching only

    Args:
        task_name: Task name to search for
        memory_dir: Base memory directory

    Returns:
        Path to task directory if found, None otherwise
    """
    base_path = Path(memory_dir)
    if not base_path.exists():
        return None

    # Only exact match
    exact_path = base_path / task_name
    if exact_path.exists():
        return exact_path

    return None


def _load_task_memory(task_name: str, memory_dir: str, config: Optional[Dict[str, Any]] = None) -> TaskMemoryLayer:
    """
    Load or retrieve cached task memory layer

    Args:
        task_name: Name of the task
        memory_dir: Base memory directory
        config: Optional configuration for TaskMemoryLayer

    Returns:
        TaskMemoryLayer instance
    """
    cache_key = f"{task_name}:{memory_dir}"

    if cache_key in _memory_cache:
        logger.debug(f"Using cached memory for task: {task_name}")
        return _memory_cache[cache_key]

    # Find task directory with fuzzy matching
    task_memory_path = _find_task_directory(task_name, memory_dir)

    if not task_memory_path:
        available_tasks = [d.name for d in Path(memory_dir).iterdir() if d.is_dir()] if Path(memory_dir).exists() else []
        logger.warning(f"Memory directory not found for task: {task_name}")
        if available_tasks:
            logger.warning(f"Available tasks: {', '.join(available_tasks)}")
        raise FileNotFoundError(f"Task memory not found for '{task_name}' in {memory_dir}")

    # Create config for loading
    # Handle both config structures:
    # 1. Flat: {"task_memory": {...}}
    # 2. Nested: {"memory": {"task_memory": {...}}}
    if config:
        if "memory" in config and "task_memory" in config["memory"]:
            # Nested structure - extract task_memory from memory section
            load_config = {
                "task_memory": config["memory"]["task_memory"].copy(),
                "agents": config.get("agents", {})
            }
        elif "task_memory" in config:
            # Flat structure - use as is
            load_config = config.copy()
        else:
            # No task_memory config found - create minimal config
            load_config = {"task_memory": {}}
    else:
        load_config = {"task_memory": {}}

    # Set the specific memory directory for this task
    load_config["task_memory"]["memory_dir"] = str(task_memory_path)

    # Load memory
    logger.info(f"Loading task memory: {task_name} from {task_memory_path}")
    memory = TaskMemoryLayer.from_config(load_config)

    # Cache for reuse
    _memory_cache[cache_key] = memory
    logger.info(f"Loaded {len(memory.records)} records for task: {task_name}")

    return memory


async def retrieve_task_memory(
    query,
    config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Retrieve similar experiment results from task memory

    All configuration parameters are read from the config dict, including:
    - task_name: Name of the task (from context)
    - memory_dir: Base memory directory
    - memory_top_k: Number of results to retrieve
    - memory_alpha: Weight for BM25 vs vector search
    - Other optional parameters

    Args:
        query: Research idea query (str or dict with goal info)
        config: Configuration dict containing all hyperparameters and context
        **kwargs: Additional parameters (for backward compatibility)

    Returns:
        Dictionary containing:
            - success: Boolean indicating success
            - query: Original query
            - recommendation: accept/reject/uncertain based on history
            - reasoning: Explanation of recommendation
            - statistics: Label distribution in results
            - similar_records: List of similar experiment records
            - guidance_prompt: Formatted guidance text
    """
    # Extract parameters from config
    task_name = config.get("task_name")
    memory_dir = config.get("memory_dir", "./config/mem_store")
    top_k = config.get("memory_top_k", 5)
    alpha = config.get("memory_alpha", 0.5)
    label_filter = config.get("memory_label_filter")
    min_score = config.get("memory_min_score", 0.0)
    include_details = config.get("memory_include_details", True)
    # Convert query to string if it's a dict (goal object)
    if isinstance(query, dict):
        query_parts = []
        if query.get("description"):
            query_parts.append(query["description"])
        if query.get("domain"):
            query_parts.append(f"Domain: {query['domain']}")
        if query.get("constraints"):
            constraints = query["constraints"]
            if isinstance(constraints, list):
                query_parts.append(f"Constraints: {', '.join(str(c) for c in constraints)}")
            else:
                query_parts.append(f"Constraints: {constraints}")
        if query.get("background"):
            query_parts.append(f"Background: {query['background'][:200]}")

        query_text = "\n".join(query_parts) if query_parts else str(query)
        logger.info(f"Converted goal dict to query text ({len(query_text)} chars)")
    elif isinstance(query, str):
        query_text = query
    else:
        return {
            "success": False,
            "error": f"Invalid query type: must be string or dict, got {type(query)}"
        }

    logger.info(f"Memory retrieval for query: {query_text[:100]}...")

    # Validate query text
    if not query_text or not query_text.strip():
        return {
            "success": False,
            "error": "Invalid query: empty after processing"
        }

    if not task_name:
        return {
            "success": False,
            "error": "task_name is required"
        }

    # Limit top_k
    top_k = min(max(1, top_k), 20)

    # Validate alpha
    alpha = max(0.0, min(1.0, alpha))

    try:
        # Load task memory
        memory = _load_task_memory(task_name, memory_dir, config)

        if len(memory.records) == 0:
            return {
                "success": True,
                "query": query_text,
                "task_name": task_name,
                "recommendation": "novel",
                "reasoning": "No historical records found. This is a novel direction.",
                "statistics": {"total": 0, "positive": 0, "neutral": 0, "negative": 0},
                "similar_records": [],
                "guidance_prompt": "No similar records found in memory. This is a novel direction."
            }

        # Retrieve similar records
        logger.info(f"Searching {len(memory.records)} records with top_k={top_k}, alpha={alpha}")
        similar_records = memory.retrieve_similar_records(
            query_text=query_text,
            top_k=top_k,
            alpha=alpha,
            label_filter=label_filter,
            min_score=min_score
        )

        if not similar_records:
            return {
                "success": True,
                "query": query_text,
                "task_name": task_name,
                "recommendation": "novel",
                "reasoning": "No similar records found matching criteria. This is a novel direction.",
                "statistics": {"total": 0, "positive": 0, "neutral": 0, "negative": 0},
                "similar_records": [],
                "guidance_prompt": "No similar records found in memory. This is a novel direction."
            }

        # Analyze results
        positive_count = sum(1 for record, _ in similar_records if record.label == 1)
        negative_count = sum(1 for record, _ in similar_records if record.label == -1)
        neutral_count = sum(1 for record, _ in similar_records if record.label == 0)

        positive_ratio = positive_count / len(similar_records)
        negative_ratio = negative_count / len(similar_records)

        # Determine recommendation
        if positive_ratio >= 0.5:
            recommendation = "accept"
            reasoning = "This direction has shown strong positive results in similar ideas."
        elif negative_ratio >= 0.5:
            recommendation = "reject"
            reasoning = "This direction has shown poor results in similar ideas. Consider alternative approaches."
        elif positive_count > negative_count:
            recommendation = "cautious_accept"
            reasoning = "This direction has mixed results, but leans positive. Proceed with careful validation."
        elif negative_count > positive_count:
            recommendation = "cautious_reject"
            reasoning = "This direction has mixed results, but leans negative. Consider improvements."
        else:
            recommendation = "uncertain"
            reasoning = "This direction has highly uncertain outcomes. Requires careful experimentation."

        # Format records for response
        formatted_records = []
        for record, score in similar_records:
            record_info = {
                "name": record.name,
                "title": record.title,
                "description": record.description,
                "label": record.label,
                "label_text": {1: "successful", 0: "neutral", -1: "failed"}[record.label],
                "similarity_score": round(score, 4)
            }

            if include_details:
                record_info.update({
                    "statement": record.statement,
                    "method": record.method,
                    "baseline_metrics": record.baseline_metrics,
                    "experiment_metrics": record.experiment_metrics,
                    "improvement": record.improvement,
                    "analysis": record.analysis
                })

            formatted_records.append(record_info)

        # Generate guidance prompt
        guidance_prompt = memory.generate_guidance_prompt(
            query_text=query_text,
            top_k=top_k,
            alpha=alpha
        )

        # Prepare response
        result = {
            "success": True,
            "query": query_text,
            "task_name": task_name,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "statistics": {
                "total": len(similar_records),
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
                "positive_ratio": round(positive_ratio, 2),
                "negative_ratio": round(negative_ratio, 2)
            },
            "similar_records": formatted_records,
            "guidance_prompt": guidance_prompt,
            "message": f"Found {len(similar_records)} similar records. Recommendation: {recommendation.upper()}"
        }

        logger.info(f"Memory retrieval completed: {len(similar_records)} records, recommendation: {recommendation}")
        return result

    except FileNotFoundError as e:
        error_msg = f"Task memory not found: {str(e)}"
        logger.warning(error_msg)
        return {
            "success": False,
            "query": query_text if 'query_text' in locals() else str(query),
            "task_name": task_name,
            "error": error_msg,
            "recommendation": "novel",
            "reasoning": "No historical data available for this task."
        }

    except Exception as e:
        error_msg = f"Memory retrieval failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "query": query_text if 'query_text' in locals() else str(query),
            "task_name": task_name,
            "error": error_msg
        }


def clear_memory_cache():
    """Clear cached memory instances and free GPU memory"""
    global _memory_cache

    num_entries = len(_memory_cache)
    _memory_cache.clear()

    # Clear PyTorch GPU cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Cleared memory cache ({num_entries} entries) and GPU cache")
        else:
            logger.info(f"Cleared memory cache ({num_entries} entries)")
    except ImportError:
        logger.info(f"Cleared memory cache ({num_entries} entries)")
