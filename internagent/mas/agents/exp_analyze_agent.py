"""
Experiment Analysis Agent

This module provides an agent for analyzing experiment results and extracting metrics.
Follows the InternAgent architecture pattern with BaseAgent and BaseModel.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from .base_agent import BaseAgent
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


# Default metric direction configuration
# This is used as a fallback when LLM cannot determine the direction
DEFAULT_METRIC_CONFIG = {
    # RMSE metrics: lower is better
    "val/PQ_Vm_rmse": "lower",
    "val/PQ_Va_rmse": "lower",
    "val/PV_Va_rmse": "lower",
    "PQ_Vm_rmse": "lower",
    "PQ_Va_rmse": "lower",
    "PV_Va_rmse": "lower",

    # Delta metrics: lower is better (typically errors)
    "delta_p": "lower",
    "delta_q": "lower",

    # Accuracy/F1 metrics: higher is better
    "accuracy": "higher",
    "f1": "higher",
    "precision": "higher",
    "recall": "higher",
    "auc": "higher",
    "mIoU": "higher",

    # Loss metrics: lower is better
    "loss": "lower",
    "train_loss": "lower",
    "val_loss": "lower",
    "test_loss": "lower",
}


def get_metric_direction_by_pattern(metric_name: str) -> str:
    """
    Get metric direction using simple pattern matching (fallback method)

    Args:
        metric_name: Name of the metric

    Returns:
        "higher", "lower", or "unknown"
    """
    # Check exact match first
    if metric_name in DEFAULT_METRIC_CONFIG:
        return DEFAULT_METRIC_CONFIG[metric_name]

    # Pattern matching
    metric_lower = metric_name.lower()

    # Lower is better patterns
    if any(x in metric_lower for x in ["rmse", "mse", "mae", "error", "loss", "delta", "cost"]):
        return "lower"

    # Higher is better patterns
    if any(x in metric_lower for x in ["accuracy", "acc", "f1", "precision", "recall", "auc", "score"]):
        return "higher"

    return "unknown"


class ExpAnalyzeAgent(BaseAgent):
    """
    Agent for analyzing experiment results and extracting metrics.

    Analyzes final_info.json and extracts metrics with LLM assistance.
    Supports automatic primary metric selection and metric direction detection.

    Attributes:
        custom_metric_config (Dict[str, str]): User-provided metric direction config
        use_llm_for_metric_direction (bool): Whether to use LLM for unknown metrics
        primary_metric (Optional[str]): Manually specified primary metric name
        use_llm_for_primary_metric (bool): Whether to use LLM to auto-select primary metric
    """

    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        """
        Initialize the experiment analysis agent.

        Args:
            model (BaseModel): Language model for analysis
            config (Dict[str, Any]): Configuration with keys:
                - custom_metric_config (Dict[str, str]): Custom metric directions
                - use_llm_for_metric_direction (bool): Use LLM for direction detection (default: True)
                - primary_metric (str): Manually specified primary metric (optional)
                - use_llm_for_primary_metric (bool): Use LLM to auto-select primary metric (default: True)
        """
        super().__init__(model, config)

        # Load agent-specific configuration
        self.custom_metric_config = config.get("custom_metric_config", {})
        self.use_llm_for_metric_direction = config.get("use_llm_for_metric_direction", True)
        self.primary_metric = config.get("primary_metric", None)
        self.use_llm_for_primary_metric = config.get("use_llm_for_primary_metric", True)

        # Merge custom config with defaults
        self.metric_config = DEFAULT_METRIC_CONFIG.copy()
        if self.custom_metric_config:
            self.metric_config.update(self.custom_metric_config)

        # Cached primary metric selection
        self._selected_primary_metric = None

        logger.info(f"ExpAnalyzeAgent initialized with primary_metric={self.primary_metric}, use_llm={self.use_llm_for_primary_metric}")

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute experiment analysis.

        This is the standard agent interface method. For direct usage, prefer the
        specific methods like analyze_result_file, compare_results, etc.

        Args:
            context: Execution context
            params: Runtime parameters

        Returns:
            Analysis results
        """
        # This agent is typically used directly via its methods rather than execute()
        # But we provide a basic implementation for consistency
        result_file = context.get("result_file")
        if result_file:
            metrics = await self.analyze_result_file(Path(result_file))
            return {"metrics": metrics}
        else:
            return {"error": "No result_file provided in context"}

    async def analyze_result_file(self, result_file: Path) -> Optional[Dict[str, float]]:
        """
        Analyze a final_info.json file and extract metrics

        Args:
            result_file: Path to final_info.json

        Returns:
            Dictionary of metric name -> value, or None if analysis fails
        """
        if not result_file.exists():
            return None

        # Read file
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Try simple extraction first (no LLM needed if structure is clear)
        try:
            if isinstance(data, dict) and data:
                first_key = list(data.keys())[0]
                if "means" in data[first_key]:
                    results = data[first_key]["means"]
                    results = {k: v for k, v in results.items() if k != "epoch"}
                    return results
        except Exception as e:
            pass

        # If simple extraction fails, use LLM
        return await self._analyze_with_llm(data)

    async def _analyze_with_llm(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Use LLM to analyze complex result structure

        Args:
            data: Raw data from final_info.json

        Returns:
            Extracted metrics dictionary
        """
        prompt = f"""
You are analyzing experimental results from a machine learning experiment.
The result file contains the following data structure:

{json.dumps(data, indent=2)}

Please extract the key performance metrics from this data.
Return a JSON object with metric names as keys and their values as numbers.
Focus on metrics like: rmse, accuracy, f1, precision, recall, loss, etc.
Ignore metadata like epoch numbers or timestamps.

Return ONLY a valid JSON object, no other text.
Example format: {{"val_rmse": 0.123, "accuracy": 0.95}}
"""

        try:
            result = await self.model.generate_json(
                prompt=prompt,
                schema={
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                system_prompt="You are a helpful assistant that extracts metrics from experimental results.",
                temperature=0.0
            )
            return result

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None

    async def get_metric_direction(self, metric_name: str, metric_values: Optional[Dict[str, float]] = None) -> str:
        """
        Determine metric direction using config + LLM fallback

        Args:
            metric_name: Name of the metric
            metric_values: Optional dict of all metrics for context

        Returns:
            "higher", "lower", or "unknown"
        """
        # 1. Check user-provided config
        if metric_name in self.metric_config:
            return self.metric_config[metric_name]

        # 2. Try pattern matching
        direction = get_metric_direction_by_pattern(metric_name)
        if direction != "unknown":
            return direction

        # 3. Use LLM if enabled
        if self.use_llm_for_metric_direction:
            try:
                direction = await self._get_metric_direction_from_llm(metric_name, metric_values)
                if direction != "unknown":
                    # Cache the result
                    self.metric_config[metric_name] = direction
                    return direction
            except Exception as e:
                logger.warning(f"LLM metric direction detection failed for '{metric_name}': {e}")

        return "unknown"

    async def _get_metric_direction_from_llm(self, metric_name: str, metric_values: Optional[Dict[str, float]] = None) -> str:
        """
        Use LLM to determine metric direction

        Args:
            metric_name: Name of the metric
            metric_values: Optional dict of all metrics for context

        Returns:
            "higher", "lower", or "unknown"
        """
        context = ""
        if metric_values:
            context = f"\n\nFor context, here are all the metrics in this experiment:\n{json.dumps(metric_values, indent=2)}"

        prompt = f"""You are analyzing machine learning experiment metrics.
Given the metric name: "{metric_name}"{context}

Determine whether this metric is better when HIGHER or LOWER.

Examples:
- RMSE, MSE, MAE, loss, error → LOWER is better
- Accuracy, F1, precision, recall, AUC → HIGHER is better
- Throughput, speed (ops/sec) → HIGHER is better
- Latency, time (seconds) → LOWER is better

Return ONLY one word: "higher", "lower", or "unknown"
If you cannot determine with confidence, return "unknown".
"""

        try:
            result = await self.model.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that analyzes ML metrics.",
                temperature=0.0,
                max_tokens=10
            )

            result = result.strip().lower()

            if result in ["higher", "lower"]:
                return result
            else:
                return "unknown"

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "unknown"

    async def select_primary_metric(
        self,
        baseline_results: Dict[str, float],
        task_name: Optional[str] = None
    ) -> str:
        """
        Select the primary metric for comparing results

        Priority:
        1. Use manually specified primary_metric if provided
        2. Use LLM to auto-select if enabled
        3. Fallback to first available metric

        Args:
            baseline_results: Baseline metrics dictionary
            task_name: Optional task name for context

        Returns:
            Name of the selected primary metric
        """
        # 1. Use manually specified primary metric
        if self.primary_metric:
            if self.primary_metric in baseline_results:
                return self.primary_metric
            else:
                logger.warning(f"Specified primary metric '{self.primary_metric}' not found in results")
                logger.warning(f"  Available metrics: {list(baseline_results.keys())}")
                logger.warning(f"  Will try auto-selection...")

        # 2. Use cached auto-selected metric if available
        if self._selected_primary_metric and self._selected_primary_metric in baseline_results:
            return self._selected_primary_metric

        # 3. Use LLM to auto-select
        if self.use_llm_for_primary_metric:
            try:
                selected = await self._select_primary_metric_with_llm(baseline_results, task_name)
                if selected and selected in baseline_results:
                    self._selected_primary_metric = selected
                    logger.info(f"✓ LLM selected primary metric: {selected}")
                    return selected
            except Exception as e:
                logger.warning(f"LLM primary metric selection failed: {e}")

        # 4. Fallback: use first metric (or first non-epoch metric)
        available_metrics = [m for m in baseline_results.keys() if m != "epoch"]
        if available_metrics:
            fallback_metric = available_metrics[0]
            logger.warning(f"⚠ Using fallback primary metric: {fallback_metric}")
            return fallback_metric
        else:
            # No metrics available at all
            return list(baseline_results.keys())[0]

    async def _select_primary_metric_with_llm(
        self,
        metrics: Dict[str, float],
        task_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Use LLM to select the most important metric for comparison

        Args:
            metrics: Dictionary of available metrics
            task_name: Optional task name for context

        Returns:
            Name of the selected primary metric, or None if selection fails
        """
        metric_list = list(metrics.keys())
        task_context = f" for task '{task_name}'" if task_name else ""

        prompt = f"""You are analyzing machine learning experiment results{task_context}.

Available metrics:
{json.dumps(metric_list, indent=2)}

Sample values:
{json.dumps(metrics, indent=2)}

Select the SINGLE MOST IMPORTANT metric that should be used to determine whether the experiment was successful or not.

Guidelines:
- Choose the metric that best represents the main objective of the experiment
- Prefer final evaluation metrics over training metrics
- Prefer task-specific metrics (e.g., RMSE for regression, accuracy for classification)
- Avoid auxiliary metrics like epoch numbers or intermediate losses

Return ONLY the exact metric name from the list above, nothing else.
Example: "val/PQ_Vm_rmse"
"""

        try:
            response = await self.model.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that analyzes ML experiments.",
                temperature=0.0,
                max_tokens=50
            )

            selected_metric = response.strip()

            # Validate that the returned metric is in the list
            if selected_metric in metric_list:
                return selected_metric
            else:
                # Try to find a fuzzy match
                for metric in metric_list:
                    if metric.lower() == selected_metric.lower():
                        return metric
                logger.warning(f"LLM returned invalid metric '{selected_metric}'")
                return None

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    async def compare_results(
        self,
        baseline_results: Dict[str, float],
        improved_results: Dict[str, float],
        threshold_percent: float = 5.0,
        task_name: Optional[str] = None
    ) -> Tuple[Dict[str, float], float, int, str]:
        """
        Compare baseline and improved results using primary metric

        Args:
            baseline_results: Baseline metrics
            improved_results: Improved metrics
            threshold_percent: Threshold for labeling
            task_name: Optional task name for context

        Returns:
            (improvement_rates, overall_improvement_rate, label, primary_metric)

        Note:
            overall_improvement_rate is now based on the primary metric,
            not the average of all metrics
        """
        improvement_rates = {}

        # Calculate improvement rate for all metrics
        for metric in baseline_results:
            if metric not in improved_results:
                continue

            # Convert to float to handle string values from JSON
            try:
                baseline_val = float(baseline_results[metric])
                improved_val = float(improved_results[metric])

                # Skip NaN and Inf values
                import numpy as np
                if np.isnan(baseline_val) or np.isinf(baseline_val) or \
                   np.isnan(improved_val) or np.isinf(improved_val):
                    print(f"  ⚠ Warning: Metric '{metric}' has NaN or Inf value, skipping")
                    continue

            except (ValueError, TypeError):
                print(f"  ⚠ Warning: Cannot convert metric '{metric}' to float, skipping")
                continue

            if baseline_val == 0:
                continue

            direction = await self.get_metric_direction(metric, baseline_results)

            if direction == "lower":
                improvement_rate = (baseline_val - improved_val) / abs(baseline_val) * 100
            elif direction == "higher":
                improvement_rate = (improved_val - baseline_val) / abs(baseline_val) * 100
            else:
                improvement_rate = 0.0

            improvement_rates[metric] = improvement_rate

        # Select primary metric
        primary_metric = await self.select_primary_metric(baseline_results, task_name)

        # Use primary metric's improvement rate as the overall rate
        if primary_metric in improvement_rates:
            overall_improvement_rate = improvement_rates[primary_metric]
            print(f"  Primary metric: {primary_metric} → {overall_improvement_rate:+.2f}%")
        elif improvement_rates:
            # Fallback to average if primary metric not available
            overall_improvement_rate = float(np.mean(list(improvement_rates.values())))
            print(f"  ⚠ Primary metric not available, using average: {overall_improvement_rate:+.2f}%")
        else:
            overall_improvement_rate = 0.0
            print(f"  ⚠ No valid metrics for comparison")

        # Assign label based on primary metric improvement
        if overall_improvement_rate > threshold_percent:
            label = 1
        elif overall_improvement_rate < -threshold_percent:
            label = -1
        else:
            label = 0

        return improvement_rates, overall_improvement_rate, label, primary_metric
