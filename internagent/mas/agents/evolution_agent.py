"""
Evolution Agent for InternAgent

Implements the hypothesis refinement agent that evolves and improves research hypotheses
by addressing critiques, incorporating evidence, and responding to feedback. Generates
multiple improved versions with configurable creativity levels for iterative refinement.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Agent that refines hypotheses through iterative evolution.

    Generates improved hypothesis versions by systematically addressing critiques,
    incorporating supporting evidence, and responding to scientist feedback. Produces
    multiple evolution candidates with documented changes and improvements. Supports
    creativity control from conservative refinement to bold restructuring.

    Attributes:
        evolution_count (int): Number of evolved versions per hypothesis
        min_improvement_threshold (float): Minimum improvement required
        creativity_level (float): Evolution creativity 0-1
        temperature (float): Model sampling temperature
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the evolution agent with model and configuration.

        Args:
            model (BaseModel): Language model for hypothesis evolution
            config (Dict[str, Any]): Configuration with keys:
                - evolution_count (int): Evolved versions count (default: 2)
                - min_improvement_threshold (float): Min improvement (default: 0.3)
                - creativity_level (float): Creativity 0-1 (default: 0.6)
                - temperature (float): Sampling temperature (optional)
                - use_memory (bool): Enable memory for this agent (default: True)
                - memory (Dict): Global memory configuration with:
                    - task_memory (Dict): Task memory settings:
                        - enabled (bool): Enable task memory globally (default: True)
                        - memory_dir (str): Base directory (default: ./config/mem_store)
                        - top_k (int): Number of records to retrieve (default: 5)
                        - alpha (float): Hybrid search weight (default: 0.5)
                        - include_details (bool): Include detailed info (default: True)
        """
        super().__init__(model, config)

        # Load agent-specific configuration
        self.model = model
        self.config = config  # Save config for later use (e.g., embedding configuration)

        self.evolution_count = config.get("evolution_count", 2)  # Number of evolutions per hypothesis
        self.min_improvement_threshold = config.get("min_improvement_threshold", 0.3)
        self.creativity_level = config.get("creativity_level", 0.6)
        self.temperature = config.get("temperature", None)

        # Memory configuration
        # Agent-level: whether this agent uses memory system
        self.use_memory = config.get("use_memory", True)

        # Task memory configuration from global memory config
        memory_config = config.get("memory", {})
        task_memory_config = memory_config.get("task_memory", {})

        # Only enable if both global task_memory and agent use_memory are enabled
        task_memory_enabled = task_memory_config.get("enabled", True)
        self.use_memory = self.use_memory and task_memory_enabled

        # Read task memory parameters from global config
        self.memory_dir = task_memory_config.get("memory_dir", "./config/mem_store")
        self.memory_top_k = task_memory_config.get("top_k", 5)
        self.memory_alpha = task_memory_config.get("alpha", 0.5)
        self.memory_include_details = task_memory_config.get("include_details", True)

        # Hypothesis filtering configuration
        self.filter_failed_ideas = config.get("filter_failed_ideas", True)
        self.failed_similarity_threshold = config.get("failed_similarity_threshold", 0.7)
        self.max_regeneration_attempts = config.get("max_regeneration_attempts", 2)

        if self.use_memory:
            logger.info(f"Task memory enabled for EvolutionAgent: dir={self.memory_dir}, top_k={self.memory_top_k}")

        # Memory retriever instance (will be initialized in execute with task_name)
        self.memory_retriever = None

    async def _check_evolved_hypothesis_against_failed_records(self, evolved_hypothesis: Dict[str, Any]) -> tuple:
        """
        Check if an evolved hypothesis is similar to failed attempts in memory.

        Args:
            evolved_hypothesis: Evolved hypothesis dict with 'text' and 'rationale'

        Returns:
            Tuple of (should_filter, similar_failed_records):
            - should_filter: True if hypothesis should be filtered/regenerated
            - similar_failed_records: List of similar failed records with similarity scores
        """
        if not self.use_memory or not self.memory_retriever or not self.filter_failed_ideas:
            return False, []

        try:
            # Query memory with evolved hypothesis text
            query = evolved_hypothesis.get("text", "")
            if not query:
                return False, []

            # Retrieve similar records
            memory_result = await self.memory_retriever.retrieve(query=query)

            if not memory_result.get("success") or not memory_result.get("similar_records"):
                return False, []

            # Filter for failed records above similarity threshold
            records = memory_result.get("similar_records", [])
            similar_failed = []

            for record in records:
                # Check if it's a failed attempt (label == -1)
                if record.get('label') == -1:
                    similarity = record.get('similarity_score', 0)
                    # Check if similarity exceeds threshold
                    if similarity >= self.failed_similarity_threshold:
                        similar_failed.append({
                            'name': record.get('name', ''),
                            'description': record.get('description', ''),
                            'similarity_score': similarity,
                            'overall_improvement_rate': record.get('overall_improvement_rate', 0)
                        })

            should_filter = len(similar_failed) > 0

            if should_filter:
                logger.warning(f"Evolved hypothesis similar to {len(similar_failed)} failed attempt(s): '{query[:100]}...'")
                for failed in similar_failed:
                    logger.warning(f"  - {failed['name']} (similarity: {failed['similarity_score']:.2f}, "
                                 f"improvement: {failed['overall_improvement_rate']:.1%})")

            return should_filter, similar_failed

        except Exception as e:
            logger.error(f"Error checking evolved hypothesis against failed records: {e}")
            return False, []

    async def _filter_and_regenerate_evolved_hypotheses(
        self,
        evolved_hypotheses: List[Dict[str, Any]],
        goal: Dict[str, Any],
        original_hypothesis: Dict[str, Any],
        critiques: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]],
        iteration: int,
        system_prompt: str,
        output_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter evolved hypotheses similar to failed attempts and regenerate them.

        This method iteratively:
        1. Checks each evolved hypothesis against failed memory records
        2. Identifies hypotheses that are too similar to failed attempts
        3. Regenerates those hypotheses with explicit avoidance instructions
        4. Repeats until no more filtering needed or max attempts reached

        Args:
            evolved_hypotheses: Initial list of evolved hypotheses
            goal: Research goal
            original_hypothesis: Original hypothesis being evolved
            critiques: Critiques to address
            evidence: Supporting evidence
            feedback: Scientist feedback
            iteration: Current iteration number
            system_prompt: System prompt for evolution
            output_schema: Output schema for structured generation
            context: Execution context with task_name

        Returns:
            Final list of evolved hypotheses after filtering and regeneration
        """
        logger.info(f"Starting evolved hypothesis filtering against failed records (threshold: {self.failed_similarity_threshold})")

        final_hypotheses = []
        current_hypotheses = evolved_hypotheses.copy()
        regeneration_attempt = 0

        while regeneration_attempt < self.max_regeneration_attempts:
            # Check each evolved hypothesis against failed records
            to_filter = []
            to_keep = []
            failed_records_for_filtered = []

            for hyp in current_hypotheses:
                should_filter, similar_failed = await self._check_evolved_hypothesis_against_failed_records(hyp)

                if should_filter:
                    to_filter.append(hyp)
                    failed_records_for_filtered.append(similar_failed)
                else:
                    to_keep.append(hyp)

            # Add kept hypotheses to final list
            final_hypotheses.extend(to_keep)

            # If nothing to filter, we're done
            if not to_filter:
                logger.info(f"No evolved hypotheses filtered in attempt {regeneration_attempt + 1}. Filtering complete.")
                break

            logger.info(f"Attempt {regeneration_attempt + 1}: Filtered {len(to_filter)} evolved hypotheses, "
                       f"kept {len(to_keep)} hypotheses")

            # Regenerate filtered hypotheses
            regenerated = await self._regenerate_filtered_evolved_hypotheses(
                count=len(to_filter),
                failed_records_list=failed_records_for_filtered,
                goal=goal,
                original_hypothesis=original_hypothesis,
                critiques=critiques,
                evidence=evidence,
                feedback=feedback,
                iteration=iteration,
                system_prompt=system_prompt,
                output_schema=output_schema,
                context=context
            )

            # Update current hypotheses for next iteration
            current_hypotheses = regenerated
            regeneration_attempt += 1

        logger.info(f"Evolved hypothesis filtering complete: {len(final_hypotheses)} hypotheses passed, "
                   f"{regeneration_attempt} regeneration attempts used")

        return final_hypotheses

    async def _regenerate_filtered_evolved_hypotheses(
        self,
        count: int,
        failed_records_list: List[List[Dict[str, Any]]],
        goal: Dict[str, Any],
        original_hypothesis: Dict[str, Any],
        critiques: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]],
        iteration: int,
        system_prompt: str,
        output_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Regenerate evolved hypotheses that were too similar to failed attempts.

        Args:
            count: Number of hypotheses to regenerate
            failed_records_list: List of failed records for each filtered hypothesis
            goal: Research goal
            original_hypothesis: Original hypothesis being evolved
            critiques: Critiques to address
            evidence: Supporting evidence
            feedback: Scientist feedback
            iteration: Current iteration number
            system_prompt: System prompt for evolution
            output_schema: Output schema
            context: Execution context

        Returns:
            List of regenerated evolved hypotheses
        """
        logger.info(f"Regenerating {count} evolved hypotheses to avoid failed directions")

        # Build avoidance guidance from failed records
        avoidance_guidance = "\n# IMPORTANT: Avoid These Failed Directions\n"
        avoidance_guidance += "The following approaches have been tried and FAILED. DO NOT evolve in these directions:\n\n"

        for i, failed_records in enumerate(failed_records_list, 1):
            if failed_records:
                avoidance_guidance += f"Failed Direction #{i}:\n"
                for record in failed_records[:2]:  # Show top 2 most similar failures
                    avoidance_guidance += f"- {record['name']} ({record['overall_improvement_rate']:.1%} performance change)\n"
                    avoidance_guidance += f"  Description: {record['description']}\n"
                avoidance_guidance += "\n"

        # Build regeneration prompt
        prompt = await self._build_evolution_prompt(
            goal=goal,
            hypothesis=original_hypothesis,
            critiques=critiques,
            evidence=evidence,
            feedback=feedback,
            iteration=iteration,
            count=count,
            context=None  # Don't add memory guidance again
        )

        # Append avoidance guidance
        prompt += avoidance_guidance
        prompt += "Generate NEW evolved hypotheses that take DIFFERENT approaches from the failed directions listed above.\n"

        # Call model to regenerate
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature
            )

            regenerated = response.get("evolved_hypotheses", [])
            logger.info(f"Successfully regenerated {len(regenerated)} evolved hypotheses")

            return regenerated

        except Exception as e:
            logger.error(f"Failed to regenerate evolved hypotheses: {e}")
            # Return empty list if regeneration fails
            return []

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve hypothesis by addressing critiques and incorporating feedback.

        Generates improved hypothesis versions that systematically address identified
        weaknesses while preserving core strengths. Incorporates evidence and feedback
        to produce scientifically stronger, more testable hypotheses. Documents specific
        improvements and changes made during evolution.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal and constraints
                - hypothesis (Dict): Hypothesis to evolve with text/rationale
                - critiques (List[Dict]): Identified weaknesses to address
                - evidence (List[Dict]): Supporting evidence (optional)
                - feedback (List[Dict]): Scientist feedback (optional)
                - iteration (int): Current iteration number
            params (Dict[str, Any]): Runtime parameters:
                - evolution_count (int): Override evolution count (optional)

        Returns:
            Dict[str, Any]: Evolution results containing:
                - evolved_hypotheses (List[Dict]): Improved versions with text/rationale/improvements
                - reasoning (str): Evolution strategy explanation
                - changes (List[Dict]): Documented changes by type
                - metadata (Dict): Evolution context

        Raises:
            AgentExecutionError: If goal/hypothesis missing or evolution fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        critiques = context.get("critiques", [])
        evidence = context.get("evidence", [])
        feedback = context.get("feedback", [])
        
        if not goal or not hypothesis:
            raise AgentExecutionError("Research goal and hypothesis are required for evolution")
        
        # Extract text from hypothesis
        hypothesis_text = hypothesis.get("text", "")
        if not hypothesis_text:
            raise AgentExecutionError("Hypothesis text is required for evolution")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        evolution_count = params.get("evolution_count", self.evolution_count)

        # Initialize memory retriever if enabled and not yet initialized
        if self.use_memory and context.get("task_name") and not self.memory_retriever:
            from ..tools.memory_retrieval import TaskMemoryRetriever
            self.memory_retriever = TaskMemoryRetriever(
                task_name=context.get("task_name"),
                memory_dir=self.memory_dir,
                top_k=self.memory_top_k,
                alpha=self.memory_alpha,
                include_details=self.memory_include_details,
                config=self.config  # Pass config for embedding model
            )
            logger.info(f"Memory retriever initialized for EvolutionAgent: {context.get('task_name')}")

        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "evolved_hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The evolved hypothesis statement"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Reasoning for the evolved hypothesis"
                            },
                            "improvements": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "Specific improvement made"
                                }
                            }
                        },
                        "required": ["text", "rationale", "improvements"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the evolution approach"
                },
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "Type of change (e.g., 'specification', 'generalization', 'mechanism')"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the change"
                            }
                        },
                        "required": ["type", "description"]
                    }
                }
            },
            "required": ["evolved_hypotheses", "reasoning", "changes"]
        }
        
        # Build the prompt
        prompt = await self._build_evolution_prompt(
            goal=goal,
            hypothesis=hypothesis,
            critiques=critiques,
            evidence=evidence,
            feedback=feedback,
            iteration=iteration,
            count=evolution_count,
            context=None  # Not needed for initial evolution
        )
        
        # Call the model
        system_prompt = self._build_system_prompt(self.creativity_level)
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature
            )
            
            # Process the response
            evolved_hypotheses = response.get("evolved_hypotheses", [])
            reasoning = response.get("reasoning", "")
            changes = response.get("changes", [])

            if not evolved_hypotheses:
                logger.warning("Evolution agent returned no evolved hypotheses")
            else:
                # Filter and regenerate evolved hypotheses if they're similar to failed attempts
                if self.use_memory and self.filter_failed_ideas and self.memory_retriever:
                    logger.info(f"Checking {len(evolved_hypotheses)} evolved hypotheses against failed attempts in memory")
                    evolved_hypotheses = await self._filter_and_regenerate_evolved_hypotheses(
                        evolved_hypotheses=evolved_hypotheses,
                        goal=goal,
                        original_hypothesis=hypothesis,
                        critiques=critiques,
                        evidence=evidence,
                        feedback=feedback,
                        iteration=iteration,
                        system_prompt=system_prompt,
                        output_schema=output_schema,
                        context=context
                    )
                    logger.info(f"After filtering: {len(evolved_hypotheses)} evolved hypotheses remain")

            # Build the result
            result = {
                "evolved_hypotheses": evolved_hypotheses,
                "reasoning": reasoning,
                "changes": changes,
                "metadata": {
                    "original_hypothesis_id": hypothesis.get("id", ""),
                    "iteration": iteration,
                    "count": len(evolved_hypotheses)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Evolution agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to evolve hypothesis: {str(e)}")
    
    async def _build_evolution_prompt(self,
                              goal: Dict[str, Any],
                              hypothesis: Dict[str, Any],
                              critiques: List[Dict[str, Any]],
                              evidence: List[Dict[str, Any]],
                              feedback: List[Dict[str, Any]],
                              iteration: int,
                              count: int,
                              context: Dict[str, Any] = None) -> str:
        """
        Construct comprehensive prompt for hypothesis evolution.

        Builds structured prompt incorporating original hypothesis, identified critiques,
        supporting evidence, and feedback. Adapts guidance based on iteration number for
        appropriate evolution focus (major fixes early, refinement later).

        Args:
            goal (Dict[str, Any]): Research goal with domain and constraints
            hypothesis (Dict[str, Any]): Original hypothesis with text/rationale
            critiques (List[Dict[str, Any]]): Weaknesses to address
            evidence (List[Dict[str, Any]]): Supporting evidence to incorporate
            feedback (List[Dict[str, Any]]): Scientist feedback entries
            iteration (int): Current iteration for adaptation
            count (int): Number of evolved versions to generate

        Returns:
            str: Structured evolution prompt with task guidelines
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        
        hypothesis_text = hypothesis.get("text", "")
        hypothesis_rationale = hypothesis.get("rationale", "")
        
        # Build the prompt
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add constraints if available
        if constraints:
            prompt += "# Constraints\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
            
        # Add the original hypothesis
        prompt += f"# Original Hypothesis\n{hypothesis_text}\n\n"
        
        # Add the rationale if available
        if hypothesis_rationale:
            prompt += f"# Original Rationale\n{hypothesis_rationale}\n\n"
            
        # Add critiques
        if critiques:
            prompt += "# Critiques\n"
            for i, critique in enumerate(critiques, 1):
                if isinstance(critique, dict):
                    category = critique.get("category", "")
                    point = critique.get("point", "")
                    severity = critique.get("severity", "")
                    
                    prompt += f"{i}. "
                    if category:
                        prompt += f"[{category}] "
                    prompt += point
                    if severity:
                        prompt += f" (Severity: {severity})"
                    prompt += "\n"
                else:
                    # Handle string critiques
                    prompt += f"{i}. {critique}\n"
            prompt += "\n"
            
        # Add evidence if available
        if evidence:
            prompt += "# Relevant Evidence\n"
            for i, item in enumerate(evidence, 1):
                if isinstance(item, dict):
                    source = item.get("source", "")
                    content = item.get("content", "")
                    relevance = item.get("relevance", "")
                    
                    prompt += f"{i}. "
                    if source:
                        prompt += f"[{source}] "
                    if content:
                        prompt += f"{content}"
                    if relevance:
                        prompt += f" (Relevance: {relevance})"
                    prompt += "\n"
                else:
                    # Handle string evidence
                    prompt += f"{i}. {item}\n"
            prompt += "\n"
            
        # Add recent feedback
        if feedback:
            prompt += "# Scientist Feedback\n"
            # Sort by iteration and take the most recent
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0),
                reverse=True
            )[:3]
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                
                if feedback_text:
                    prompt += f"From iteration {feedback_iter}: {feedback_text}\n\n"
        
        # Add task description
        prompt += "# Task\n"
        prompt += f"Create {count} improved versions of the original hypothesis that address the critiques, "
        prompt += "incorporate relevant evidence, and align with the scientist's feedback.\n\n"
        
        prompt += "For each evolved hypothesis:\n"
        prompt += "1. Refine the hypothesis statement to be more precise, testable, and aligned with the research goal\n"
        prompt += "2. Address specific weaknesses identified in the critiques\n"
        prompt += "3. Incorporate relevant evidence to strengthen the hypothesis\n"
        prompt += "4. Provide a clear rationale explaining the improvement\n"
        
        # Additional guidance based on iteration
        if iteration == 0:
            prompt += "\nThis is the first iteration, so focus on addressing major weaknesses while preserving the core insight.\n"
        elif iteration < 3:
            prompt += f"\nThis is iteration {iteration}, so focus on incremental improvements and refinements.\n"
        else:
            prompt += f"\nThis is a later iteration ({iteration}), so focus on nuanced improvements and polishing.\n"

        return prompt
    
    def _build_system_prompt(self, creativity_level: float) -> str:
        """
        Build system prompt adapted to creativity level.

        Creates instructions for hypothesis evolution with approach tailored to
        creativity setting: conservative (minimal changes), balanced (moderate
        innovation), or creative (bold restructuring).

        Args:
            creativity_level (float): Creativity 0-1 where:
                <0.3: conservative minimal changes
                <0.7: balanced moderate innovation
                >=0.7: creative bold restructuring

        Returns:
            str: System prompt with evolution guidelines
        """
        base_prompt = """You are a scientific hypothesis refinement specialist working with a researcher. 
Your task is to evolve and improve research hypotheses based on critiques and feedback.

Guidelines:
- Address specific weaknesses identified in critiques while preserving strengths
- Make each evolved hypothesis more precise, testable, and aligned with research goals
- Incorporate relevant evidence to strengthen the scientific foundation
- Provide clear explanations for how and why you've made each change
- Ensure evolved hypotheses remain coherent and logically sound
"""

        if creativity_level < 0.3:
            # Conservative evolution
            base_prompt += """
Take a conservative approach to refinement. Make minimal necessary changes to address 
critical weaknesses. Focus on clarification, specification, and logical consistency 
rather than introducing new elements."""
        elif creativity_level < 0.7:
            # Balanced evolution
            base_prompt += """
Balance addressing critiques with moderate innovation. Consider alternate mechanisms 
or explanations when helpful, but maintain the core thesis. Look for opportunities to 
strengthen the hypothesis while resolving inconsistencies."""
        else:
            # Creative evolution
            base_prompt += """
Be bold in your refinements while addressing critiques. Consider alternative mechanisms, 
unexpected connections, or paradigm shifts if they could strengthen the hypothesis. 
Don't hesitate to substantially reshape the hypothesis if doing so creates a stronger 
scientific proposition."""
            
        return base_prompt 
