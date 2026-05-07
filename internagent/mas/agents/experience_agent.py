"""
Experience Agent for InternAgent

This agent generates valuable experiences from experimental results through:
1. Evaluation: Assess whether runs show improvement over baseline
2. Contrastive Learning: Pairwise comparison of ideas to understand differences
3. Experience Synthesis: Extract generalizable insights from comparisons

The experiences capture what worked, what didn't, and why, providing
actionable guidance for future research.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class ExperienceAgent(BaseAgent):
    """
    Agent that generates experiences from experimental results using contrastive learning.

    This agent performs a three-step process:
    1. Evaluate run improvements for each idea
    2. Perform pairwise contrastive analysis between ideas
    3. Synthesize experiences from all comparisons

    The generated experiences capture comparative insights about what makes
    methods succeed or fail, with associated confidence levels.

    Attributes:
        comparison_dimensions (List[str]): Aspects to compare between ideas
    """

    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the experience agent with model and configuration.

        Args:
            model (BaseModel): Language model for analysis and synthesis
            config (Dict[str, Any]): Configuration with keys:
                - comparison_dimensions (List[str]): Aspects to compare
        """
        super().__init__(model, config)

        # Load agent-specific configuration

        self.temperature = config.get("temperature", None)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate experiences from multiple ideas using contrastive learning.

        This is the main entry point that orchestrates the three-step process:
        1. Evaluate run improvements for all ideas
        2. Perform pairwise contrastive analysis
        3. Synthesize experiences from comparisons

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - ideas_data (List[Dict]): List of idea/hypothesis information
                - notes_data_list (List[Dict]): List of experiment notes for each idea
                - task_domain (str): Domain of the research task
            params (Dict[str, Any]): Runtime parameters:
                - include_comparisons (bool): Return comparisons in output (default: False)
                - include_evaluations (bool): Return evaluations in output (default: False)

        Returns:
            Dict[str, Any]: Experience generation results containing:
                - experiences (List[Dict]): Synthesized experiences with:
                    - name (str): Short name for the experience
                    - type (str): Type of experience
                    - content (str): The experience content
                    - confidence (str): "high", "medium", or "low"
                - evaluations (List[Dict]): Run improvement evaluations (optional)
                - comparisons (List[Dict]): Pairwise comparison results (optional)
                - metadata (Dict): Generation context

        Raises:
            AgentExecutionError: If required data is missing or generation fails
        """
        # Extract required data
        ideas_data = context.get("ideas_data", [])
        notes_data_list = context.get("notes_data_list", [])
        task_domain = context.get("task_domain", "machine learning")

        if not ideas_data:
            raise AgentExecutionError("Ideas data is required for experience generation")

        if not notes_data_list:
            raise AgentExecutionError("Notes data is required for experience generation")

        if len(ideas_data) != len(notes_data_list):
            raise AgentExecutionError("Number of ideas and notes must match")

        # Extract parameters
        include_comparisons = params.get("include_comparisons", False)
        include_evaluations = params.get("include_evaluations", False)

        try:
            # Step 1: Evaluate run improvements for all ideas
            logger.info(f"Evaluating run improvements for {len(ideas_data)} ideas")
            evaluations = []
            for idea_data, notes_data in zip(ideas_data, notes_data_list):
                notes_content = notes_data.get('raw_content', '')
                evaluation = await self._evaluate_run_improvements(notes_content)
                evaluations.append({
                    "idea_name": idea_data.get("name", "unknown"),
                    "evaluation": evaluation
                })

            # Step 2: Perform pairwise contrastive analysis
            logger.info("Performing pairwise contrastive analysis")
            comparisons = await self._contrastive_learning(
                ideas_data, notes_data_list, evaluations, task_domain
            )

            # Step 3: Synthesize experiences from comparisons
            logger.info("Synthesizing experiences from comparisons")
            experiences = await self._synthesize_experiences(
                comparisons, task_domain
            )

            # Prepare result
            result = {
                "experiences": experiences,
                "metadata": {
                    "task_domain": task_domain,
                    "num_ideas": len(ideas_data),
                    "num_comparisons": len(comparisons),
                    "num_experiences": len(experiences)
                }
            }

            if include_comparisons:
                result["comparisons"] = comparisons

            if include_evaluations:
                result["evaluations"] = evaluations

            return result

        except Exception as e:
            logger.error(f"Experience generation failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate experiences: {str(e)}")

    async def _contrastive_learning(
        self,
        ideas_data: List[Dict[str, Any]],
        notes_data_list: List[Dict[str, Any]],
        evaluations: List[Dict[str, Any]],
        task_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Perform pairwise contrastive analysis between all ideas.

        Args:
            ideas_data: List of idea dictionaries
            notes_data_list: List of notes dictionaries
            evaluations: List of evaluation results
            task_domain: Research domain

        Returns:
            List of comparison dictionaries, each containing:
                - idea_a: Name of first idea
                - idea_b: Name of second idea
                - analysis: Detailed comparison analysis
        """
        comparisons = []

        # Create all pairs of ideas
        pairs = list(combinations(range(len(ideas_data)), 2))

        logger.info(f"Comparing {len(pairs)} pairs of ideas")

        for i, j in pairs:
            idea_a = ideas_data[i]
            idea_b = ideas_data[j]
            notes_a = notes_data_list[i]
            notes_b = notes_data_list[j]
            # Perform pairwise comparison
            comparison = await self._compare_two_ideas(
                idea_a, notes_a,
                idea_b, notes_b,
                task_domain
            )
            comparison['analysis_result'] = f"The comparative analysis report between Method {idea_a.get('name')} and {idea_b.get('name')}" + comparison['analysis_result'] 
            comparisons.append(comparison)

        return comparisons

    async def _compare_two_ideas(
        self,
        idea_a: Dict[str, Any],
        notes_a: Dict[str, Any],
        idea_b: Dict[str, Any],
        notes_b: Dict[str, Any],
        task_domain: str
    ) -> Dict[str, Any]:
        """
        Compare two ideas in detail from multiple dimensions.

        Args:
            idea_a: First idea data
            notes_a: First idea's experiment notes
            eval_a: First idea's evaluation
            idea_b: Second idea data
            notes_b: Second idea's experiment notes
            eval_b: Second idea's evaluation
            task_domain: Research domain

        Returns:
            Dictionary with comparison results
        """
        idea_a_name = idea_a.get("name", "Idea A")
        idea_b_name = idea_b.get("name", "Idea B")

        logger.info(f"Comparing: {idea_a_name} vs {idea_b_name}")

        # Process notes for idea A: identify metric, direction, and extract best run
        notes_a_content = notes_a.get('raw_content', 'N/A')
        notes_b_content = notes_b.get('raw_content', 'N/A')

        # Process idea A
        if notes_a_content != 'N/A':
            logger.info(f"Processing notes for {idea_a_name}")
            metric_a = await self._identify_primary_metric(notes_a_content)
            logger.info(f"Identified primary metric for {idea_a_name}: {metric_a}")

            direction_a = await self._determine_metric_direction(metric_a, notes_a_content)
            logger.info(f"Metric direction for {idea_a_name}: {direction_a}")

            best_run_a = await self._extract_best_run(notes_a_content, metric_a, direction_a)

            # Append best run summary to notes
            if best_run_a.get("best_run_number", 0) > 0:
                notes_a_content = f"{notes_a_content}\n\n--- BEST RUN SUMMARY ---\n{best_run_a['summary']}"
                logger.info(f"Best run for {idea_a_name}: Run {best_run_a['best_run_number']}")

        # Process idea B
        if notes_b_content != 'N/A':
            logger.info(f"Processing notes for {idea_b_name}")
            metric_b = await self._identify_primary_metric(notes_b_content)
            logger.info(f"Identified primary metric for {idea_b_name}: {metric_b}")

            direction_b = await self._determine_metric_direction(metric_b, notes_b_content)
            logger.info(f"Metric direction for {idea_b_name}: {direction_b}")

            best_run_b = await self._extract_best_run(notes_b_content, metric_b, direction_b)

            # Append best run summary to notes
            if best_run_b.get("best_run_number", 0) > 0:
                notes_b_content = f"{notes_b_content}\n\n--- BEST RUN SUMMARY ---\n{best_run_b['summary']}"
                logger.info(f"Best run for {idea_b_name}: Run {best_run_b['best_run_number']}")

        # Prepare comparison prompt
        comparison_prompt = f"""# Task: Contrastive Analysis of Two Research Methods

## Domain
{task_domain}

## Method A: {idea_a_name}

### Description
{idea_a.get('description', 'N/A')}

### Key Details
{idea_a.get('key_details', idea_a.get('refined_method_details', 'N/A'))}

### Experiment Results
{notes_a_content}


## Method B: {idea_b_name}

### Description
{idea_b.get('description', 'N/A')}

### Key Details
{idea_b.get('key_details', idea_b.get('refined_method_details', 'N/A'))}

### Experiment Results
{notes_b_content}


## Your Task

Perform a detailed contrastive analysis of these two methods. Provide a focused comparison, emphasizing the advantages and disadvantages of each method.

**CRITICAL: Base your comparison on the BEST PERFORMANCE achieved by each method** (shown in the "BEST RUN SUMMARY" section at the end of each method's experiment results). Compare the best-performing run of Method A against the best-performing run of Method B, not just the first runs.

Analyze from the following dimensions:

1. **Technical Approach**: How do the core technical ideas differ? What are the main strengths of the better-performing method, and what weaknesses are evident in the method that performed worse? Focus on the algorithmic or architectural choices that made a difference.

2. **Implementation Complexity**: Which method was more challenging to implement? Identify any specific challenges or hurdles that may have contributed to poor performance in one of the methods.

3. **Performance Comparison**: Compare the BEST performance achieved by both methods. Which method's best run performed better, and by how much? Use the concrete metrics from the "BEST RUN SUMMARY" sections to illustrate the performance gap, and discuss why one method's best performance exceeded the other's.

4. **Reason for Success or Failure**: Analyze the causal factors behind the success of the better-performing method and the failure of the worse-performing method. What specific design choices or mistakes led to these results? What aspects of the method's design made it either more effective or less effective in achieving optimal performance?

## Important Guidelines

- **USE THE BEST RUN SUMMARY**: Always refer to the "BEST RUN SUMMARY" section for each method when making performance comparisons. This represents the peak capability of each method.
- Focus on identifying the key factors that differentiate the success or failure of the methods, based on their best experimental results.
- Provide clear, causal explanations for why one method's best performance exceeded the other's best performance.
- Be specific in your analysis, especially when it comes to the practical and theoretical aspects of each method's design.
- Consider both the strengths and weaknesses of each method, identifying areas of improvement for the method that performed worse even at its best.

Generate a comprehensive comparison analysis that helps explain the strengths and weaknesses of these methods in the context of this domain.
"""



        # Call LLM for comparison
        analysis_result = await self._call_model(
            prompt=comparison_prompt,
            system_prompt="You are an expert at analyzing and comparing machine learning methods. You excel at identifying the key factors that differentiate successful from unsuccessful approaches, and you provide detailed, evidence-based comparative analysis.",
            temperature=self.temperature
        )

        return {
            "idea_a": idea_a_name,
            "idea_b": idea_b_name,
            "analysis_result":analysis_result
        }

    async def _synthesize_experiences(
        self,
        comparisons: List[Dict[str, Any]],
        task_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Synthesize generalizable experiences from all pairwise comparisons.

        Args:
            comparisons: List of comparison results
            task_domain: Research domain

        Returns:
            List of experience dictionaries with:
                - name: Short name
                - type: Experience type
                - content: Experience content
                - confidence: "high", "medium", or "low"
        """
        logger.info(f"Synthesizing experiences from {len(comparisons)} comparisons")

        # Prepare synthesis prompt
        comparison_summaries = []
        for comp in comparisons:
            summary = comp['analysis_result']
            comparison_summaries.append(summary)

        synthesis_prompt = f"""# Task: Synthesize Generalizable Experiences from Multiple Comparative Analysis

## Domain
{task_domain}

## All Comparative Analysis Results

{chr(10).join(comparison_summaries)}

## Your Task

Synthesize **generalizable insights** from the above comparative analyses. Each experience should meet the following criteria:

1. **Name**: A concise and memorable title (2-5 words) that captures the essence of the insight.
2. **Type**: Specify whether the experience is related to **methodological insights** (e.g., general approach or framework) or **practical insights** (e.g., parameter tuning, implementation strategies).
3. **Content**: A concise, directional insight (2-3 sentences) that can guide future research or method development. The insight should focus on high-level principles, strategies, or patterns that emerged from the analysis. These should be **generalizable** and **actionable**, offering clear guidance for improving future work.
4. **Confidence**: Rate the confidence of the insight on a scale from 0 to 10, where:
   - 0-3: Low confidence, based on limited or inconsistent evidence.
   - 4-6: Medium confidence, supported by some evidence but not fully consistent.
   - 7-10: High confidence, supported by multiple comparisons with consistent evidence.

## Important Guidelines
- Each experience may describe: **What tends to work well** (positive, recommended patterns), or **What tends to fail or should be avoided** (negative, cautionary patterns)
- Focus on **generalizable** and **strategic** insights that will inform future research or development. Do not make the insights overly specific to a single comparison.
- Look for **patterns** that appear consistently across multiple analyses and extract the **big-picture lessons**.
- Prioritize **causal insights**: explain why certain outcomes occurred and identify the strategies that contributed to success or failure.
- Ensure that insights are **referenceable** and **concrete**, providing practical guidance for future work.
- You can generate **5-10 insights** that will help inform future research or method development in this domain.

Generate the insights that will guide future research and method development, highlighting both **positive** and **negative** experiences based on the comparative analyses.
"""

        # Define output schema
        output_schema = {
    "type": "object",
    "properties": {
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name (2-5 words)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["methodological", "practical"],
                        "description": "Indicates whether the experience is related to methodology (approach, strategy) or practical implementation (parameters, fine-tuning, etc.)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Clear, actionable experience statement (2-3 sentences)"
                    },
                    "confidence": {
                        "type": "integer",
                        "description": "Confidence level on a scale from 0 to 10",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "supporting_evidence": {
                        "type": "string",
                        "description": "Brief note on which comparisons support this experience"
                    }
                },
                "required": ["name", "type", "content", "confidence"]
            },
            "minItems": 5,
            "maxItems": 15
        }
    },
    "required": ["experiences"]
}

        # Call LLM for synthesis
        synthesis_result = await self._call_model(
            prompt=synthesis_prompt,
            system_prompt="You are an expert at synthesizing generalizable insights from comparative analyses. You excel at identifying patterns across multiple comparisons and formulating actionable, high-confidence experiences that guide future research.",
            schema=output_schema,
            temperature=self.temperature
        )

        experiences = synthesis_result.get("experiences", [])

        logger.info(f"Synthesized {len(experiences)} experiences")

        return experiences

    async def _evaluate_run_improvements(self, notes_content: str) -> Dict[str, Any]:
        """
        Evaluate whether runs 1-5 show improvement over baseline using LLM.

        This method is kept from the original implementation.

        Args:
            notes_content: Raw content of the notes.txt file

        Returns:
            Dict containing:
                - has_improvement (bool): Whether any run improved over baseline
                - best_run (int): Run number with best improvement (0 if none)
                - best_run_name (str): Name of the best run
                - improvement_summary (str): Summary of the improvement
                - best_run_metrics (Dict): Metrics of best run
                - baseline_metrics (Dict): Metrics of baseline
        """
        evaluation_prompt = f"""# Task: Evaluate Experimental Run Improvements

You are analyzing experimental results from a machine learning research project. Your task is to determine if any experimental runs (Run 1-5) showed improvement over the baseline (Run 0).

## Notes Content:
{notes_content}

## Your Task:
1. **Did any of runs 1-5 achieve improvement over the baseline (Run 0)?**
   - Improvement means BETTER performance (lower is better for MAE/MSE/loss, higher is better for accuracy).
   - Look for explicit comparisons like "MAE: 0.400 (-8.7%)" where negative % means improvement for loss metrics.

2. **If yes, which run achieved the BEST improvement?**
   - Identify the run number (1-5) with the maximum improvement over Run 0.
   - If no improvement is found, return 0 for the best run.

Provide your response in the following format:
- **has_improvement**: Yes/No
- **best_run**: Run number (0 if no improvement is found, or 1-5 for the best run)
"""

        output_schema = {
            "type": "object",
            "properties": {
                "has_improvement": {
                    "type": "boolean",
                    "description": "True if any run 1-5 improved over baseline Run 0"
                },
                "best_run": {
                    "type": "integer",
                    "description": "Run number (1-5) with best improvement, or 0 if no improvement"
                }
            },
            "required": ["has_improvement", "best_run"]
        }

        result = await self._call_model(
            prompt=evaluation_prompt,
            system_prompt="You are an expert at analyzing machine learning experimental results and determining whether improvements were achieved over baselines. You carefully examine metrics and provide accurate assessments.",
            schema=output_schema,
            temperature=self.temperature
        )

        return result

    async def _identify_primary_metric(self, notes_content: str) -> str:
        """
        Identify the primary metric used in the experiment results.

        Args:
            notes_content: Raw content of the notes.txt file containing experiment results

        Returns:
            str: Name of the primary metric (e.g., 'MAE', 'MSE', 'accuracy')
        """
        prompt = f"""# Task: Identify Primary Evaluation Metric

You are analyzing experimental results to identify the primary metric used for evaluation.

## Experiment Results:
{notes_content}

## Your Task:
Identify the PRIMARY metric that is most important for evaluating this experiment.
This is typically the metric that appears most prominently or is used as the main comparison point.

Common metrics include: MAE, MSE, RMSE, accuracy, F1-score, loss, etc.

Return ONLY the metric name (e.g., "MAE", "MSE", "accuracy"), without any additional explanation.
"""

        output_schema = {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "Name of the primary metric (e.g., 'MAE', 'MSE', 'accuracy')"
                }
            },
            "required": ["metric_name"]
        }

        result = await self._call_model(
            prompt=prompt,
            system_prompt="You are an expert at analyzing experimental results and identifying key evaluation metrics.",
            schema=output_schema,
            temperature=self.temperature
        )

        return result.get("metric_name", "")

    async def _determine_metric_direction(self, metric_name: str, notes_content: str) -> str:
        """
        Determine the optimization direction for a given metric.

        Args:
            metric_name: Name of the metric
            notes_content: Raw content of the notes.txt file

        Returns:
            str: Either 'lower' (lower is better) or 'higher' (higher is better)
        """
        prompt = f"""# Task: Determine Metric Optimization Direction

You are analyzing a metric to determine whether lower or higher values are better.

## Metric: {metric_name}

## Experiment Results Context:
{notes_content}  

## Your Task:
Determine whether for this metric, LOWER values are better or HIGHER values are better.

- For metrics like MAE, MSE, RMSE, loss, error rate: lower is better
- For metrics like accuracy, F1-score, precision, recall, R²: higher is better

Return ONLY "lower" or "higher".
"""

        output_schema = {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["lower", "higher"],
                    "description": "Whether lower or higher values are better for this metric"
                }
            },
            "required": ["direction"]
        }

        result = await self._call_model(
            prompt=prompt,
            system_prompt="You are an expert at understanding evaluation metrics and their optimization directions.",
            schema=output_schema,
            temperature=self.temperature
        )

        return result.get("direction", "lower")

    async def _extract_best_run(self, notes_content: str, metric_name: str, direction: str) -> Dict[str, Any]:
        """
        Extract the best run results from notes content using LLM.

        Args:
            notes_content: Raw content of the notes.txt file
            metric_name: Name of the primary metric to optimize
            direction: 'lower' or 'higher' - optimization direction

        Returns:
            Dict containing:
                - best_run_number: int (1-5)
                - best_metric_value: float
                - summary: str describing the best result
        """
        prompt = f"""# Task: Extract Best Run from Experiment Results

You are analyzing experiment results to identify the best performing run.

## Experiment Results:
{notes_content}

## Your Task:
Identify the BEST run (Run 1-5) based on the primary metric: {metric_name}

## Optimization Direction:
- For this metric, {"LOWER" if direction == "lower" else "HIGHER"} values are BETTER

## Instructions:
1. Look through all Run 1-5 results (ignore Run 0 as it's the baseline)
2. Find the run with the best {metric_name} value ({"minimum" if direction == "lower" else "maximum"})
3. Extract the exact {metric_name} value for that run
4. Create a concise summary describing:
   - Which run performed best
   - The {metric_name} value achieved
   - How it compares to other runs

## Important:
- Only consider Run 1-5 (not Run 0/baseline)
- If no valid runs found, return best_run_number as 0
- Focus on the {metric_name} metric specifically
- Extract the exact numeric value from the results
"""

        output_schema = {
            "type": "object",
            "properties": {
                "best_run_number": {
                    "type": "integer",
                    "description": "Run number (1-5) with best performance, or 0 if no valid runs found"
                },
                "best_metric_value": {
                    "type": ["number", "null"],
                    "description": f"The {metric_name} value achieved by the best run"
                },
                "summary": {
                    "type": "string",
                    "description": "Concise summary describing the best run and its performance"
                }
            },
            "required": ["best_run_number", "best_metric_value", "summary"]
        }

        result = await self._call_model(
            prompt=prompt,
            system_prompt="You are an expert at analyzing experimental results and identifying the best performing runs based on evaluation metrics.",
            schema=output_schema,
            temperature=self.temperature
        )

        best_run_number = result.get("best_run_number", 0)
        best_metric_value = result.get("best_metric_value", None)
        summary = result.get("summary", "Could not extract best run from experiment results")

        logger.info(f"Extracted best run: Run {best_run_number} with {metric_name}={best_metric_value}")

        return {
            "best_run_number": best_run_number,
            "best_metric_value": best_metric_value,
            "summary": summary
        }

    async def update_experience_library(
        self,
        existing_experiences: List[Dict[str, Any]],
        new_experiences: List[Dict[str, Any]],
        task_domain: str,
        learning_objective: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update experience library by integrating new experiences with existing ones.

        This method uses LLM to decide how to update the experience library with
        new experiences through four operations: ADD, UPDATE, DELETE, or NONE.

        Args:
            existing_experiences: List of existing experiences in the library, each with:
                - id (str): Unique identifier
                - name (str): Short name
                - content (str): Experience content
                - confidence (int): Confidence level
            new_experiences: List of newly generated experiences to be integrated, each with:
                - name (str): Short name
                - content (str): Experience content
                - confidence (int): Confidence level
            task_domain: Research domain
            learning_objective: Optional objective describing what the agent should learn

        Returns:
            Dict containing:
                - operations (List[Dict]): List of operations to perform
                - updated_library (List[Dict]): Updated experience library after applying operations
                - metadata (Dict): Update statistics
        """
        logger.info(f"Updating experience library: {len(existing_experiences)} existing, {len(new_experiences)} new")

        if not learning_objective:
            learning_objective = f"Improve research method development and experimental design in {task_domain}"

        # Prepare existing experiences for the prompt
        existing_exp_list = []
        for exp in existing_experiences:
            exp_id = exp.get("id", f"exp_{existing_experiences.index(exp)}")
            exp_name = exp.get("name", "")
            exp_content = exp.get("content", "")
            existing_exp_list.append(f"- ID: {exp_id}\n  Content: {exp_name}: {exp_content}")

        # Prepare new experiences for the prompt
        new_exp_list = []
        for exp in new_experiences:
            exp_name = exp.get("name", "")
            exp_content = exp.get("content", "")
            new_exp_list.append(f"{exp_name}: {exp_content}")

        # Create update prompt
        update_prompt = f"""# Task: Experience Library Optimization

## Context

You are helping to maintain an experience library for a research agent working in {task_domain}.

**Agent Objective**: Discover and validate effective research methods through systematic experimentation.

**Learning Objective**: {learning_objective}

## Your Task

Perform experience optimization by deciding how to integrate new experiences into the existing experience library. For each new experience, compare it with all existing experiences and choose one of four operations.

## Input

### Existing Experiences
{chr(10).join(existing_exp_list) if existing_exp_list else "No existing experiences in the library."}

### New Experiences
{chr(10).join([f"{i+1}. {exp}" for i, exp in enumerate(new_exp_list)])}

## Valid Operations

For each new experience, you must choose ONE of these operations:

1. **ADD**: The new experience contains entirely new information not present in any existing experience.
   - Use when the insight is genuinely novel
   - System will auto-generate an ID
   - Set `id` field to `null`

2. **UPDATE**: The new experience refines, expands, or improves an existing experience.
   - Use when new experience adds meaningful details to an existing one
   - Use when new experience adds nuance or slightly conflicts with existing
   - Synthesize to create a more comprehensive high-level version
   - Must specify the `id` of the existing experience to update
   - Focus on keeping the key insight while adding value

3. **DELETE**: The new experience directly contradicts or invalidates an existing experience.
   - Use when new experience shows previous guidance was wrong
   - Use when existing experience is now outdated or harmful
   - Must specify the `id` of the existing experience to delete

4. **NONE**: The new experience is redundant or not valuable.
   - Use when fully covered by existing experiences
   - Use when too similar without meaningful additions
   - Use when irrelevant to learning objective
   - Set `id` field to `null`

## Important Guidelines

1. **Focus on Learning Objective**: Every decision should support the learning objective
2. **Avoid Redundancy**: Don't add experiences that don't add new value
3. **Prefer UPDATE over ADD**: If an existing experience can be improved, update it rather than adding a duplicate
4. **Be Conservative with DELETE**: Only delete if truly contradicted or harmful
5. **Maintain Quality**: The library should contain high-quality, actionable experiences

## Output Format

You MUST output a JSON array with EXACTLY {len(new_experiences)} objects (one per new experience).

Each object must have this structure:
```json
{{
  "operation": "ADD | UPDATE | DELETE | NONE",
  "id": "existing_exp_id or null",
  "content": "Experience name: Brief description.",
  "reasoning": "Brief explanation of why this operation was chosen"
}}
```

**Important**:
- For ADD and NONE: `id` must be `null`
- For UPDATE and DELETE: `id` must be a valid existing experience ID
- `content` should be the final experience text (for ADD/UPDATE) or the new experience text (for DELETE/NONE)
- For UPDATE: synthesize existing and new into an improved version

Generate your response now.
"""

        # Define output schema
        output_schema = {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["ADD", "UPDATE", "DELETE", "NONE"]
                            },
                            "id": {
                                "type": ["string", "null"],
                                "description": "ID of existing experience (for UPDATE/DELETE) or null (for ADD/NONE)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Final experience content (name: description format)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the decision"
                            }
                        },
                        "required": ["operation", "id", "content", "reasoning"]
                    },
                    "minItems": len(new_experiences),
                    "maxItems": len(new_experiences)
                }
            },
            "required": ["decisions"]
        }

        # Call LLM for update decisions
        result = await self._call_model(
            prompt=update_prompt,
            system_prompt="You are an expert at maintaining and optimizing experience libraries for AI agents. You excel at identifying redundancy, synthesizing information, and ensuring the library contains high-quality, actionable experiences that support the learning objective.",
            schema=output_schema,
            temperature=self.temperature
        )

        decisions = result.get("decisions", [])

        # Apply operations to create updated library
        updated_library = [exp.copy() for exp in existing_experiences]

        # Track statistics
        stats = {
            "add": 0,
            "update": 0,
            "delete": 0,
            "none": 0
        }

        # Generate ID counter for new experiences
        max_id = 0
        for exp in existing_experiences:
            try:
                exp_id = exp.get("id", "")
                if exp_id.startswith("exp_"):
                    num = int(exp_id.split("_")[1])
                    max_id = max(max_id, num)
            except:
                pass

        for i, decision in enumerate(decisions):
            logger.info(f"Decision {i}: {decision}")
            operation = decision.get("operation")
            exp_id = decision.get("id")
            content = decision.get("content")

            # Parse content into name and content
            if ": " in content:
                parts = content.split(": ", 1)
                exp_name = parts[0].strip()
                exp_content = parts[1].strip()
            else:
                exp_name = new_experiences[i].get("name", "")
                exp_content = content

            if operation == "ADD":
                # Add new experience with auto-generated ID
                max_id += 1
                new_exp = {
                    "id": f"exp_{max_id}",
                    "name": exp_name,
                    "content": exp_content,
                    "confidence": new_experiences[i].get("confidence", 5),
                    "type": new_experiences[i].get("type", "general")  # Add type field
                }
                updated_library.append(new_exp)
                stats["add"] += 1
                logger.info(f"ADD: {exp_name}")

            elif operation == "UPDATE":
                # Update existing experience
                for j, exp in enumerate(updated_library):
                    if exp.get("id") == exp_id:
                        updated_library[j] = {
                            "id": exp_id,
                            "name": exp_name,
                            "content": exp_content,
                            "confidence": max(exp.get("confidence", 5), new_experiences[i].get("confidence", 5)),
                            "type": new_experiences[i].get("type", exp.get("type", "general")),  # Preserve or update type
                            "updated": True
                        }
                        stats["update"] += 1
                        logger.info(f"UPDATE: {exp_id} -> {exp_name}")
                        break

            elif operation == "DELETE":
                # Delete existing experience
                updated_library = [exp for exp in updated_library if exp.get("id") != exp_id]
                stats["delete"] += 1
                logger.info(f"DELETE: {exp_id}")

            elif operation == "NONE":
                stats["none"] += 1
                logger.info(f"NONE: {exp_name} (redundant/not valuable)")

        logger.info(f"Library update complete: +{stats['add']} ~{stats['update']} -{stats['delete']} ={stats['none']}")

        return {
            "operations": decisions,
            "updated_library": updated_library,
            "metadata": {
                "original_count": len(existing_experiences),
                "final_count": len(updated_library),
                "stats": stats
            }
        }
