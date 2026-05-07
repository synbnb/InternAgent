"""
Generation Agent for InternAgent

Implements the idea generation agent that creates novel research hypotheses
based on research goals, domain constraints, literature, and optional code baselines.
The agent supports iterative refinement through feedback incorporation and provides
configurable creativity levels for idea generation.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional

from ..tools import get_registry
from ..tools.utils import get_related_tools 
from ..models.base_model import BaseModel
from .base_agent import BaseAgent, AgentExecutionError
from .codeview_agent import get_repo_structure


logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Agent that generates novel, scientifically plausible research hypotheses.

    Creates multiple idea candidates based on research goals, domain knowledge,
    literature surveys, and optional code baselines. Supports iterative refinement
    through feedback and adjustable creativity levels. Can analyze file-level or
    project-level code to generate code-aware hypotheses.

    Attributes:
        do_survey (bool): Whether to include literature information
        generation_count (int): Number of hypotheses to generate per call
        creativity (float): Creativity level 0-1 (higher = more creative)
        diversity_threshold (float): Minimum diversity between hypotheses
        temperature (float): Model sampling temperature
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the generation agent with model and configuration.

        Args:
            model (BaseModel): Language model for idea generation
            config (Dict[str, Any]): Configuration with keys:
                - do_survey (bool): Include literature (default: False)
                - generation_count (int): Hypotheses per call (default: 5)
                - creativity (float): Creativity 0-1 (default: 0.9)
                - diversity_threshold (float): Min diversity (default: 0.3)
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

        self.do_survey = config.get("do_survey", False)
        self.generation_count = config.get("generation_count", 5)
        self.creativity = config.get("creativity", 0.9)  # Higher = more creative
        self.diversity_threshold = config.get("diversity_threshold", 0.3)
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

        self.tool_registry = get_registry()
        self.allowed_tools = config.get("allowed_tools", None)
        if self.allowed_tools:
            logger.info(f"Allowed tools: {self.allowed_tools} for GenerationAgent")
        else:
            logger.info(f"All tools available for GenerationAgent")

        if self.use_memory:
            logger.info(f"Task memory enabled: dir={self.memory_dir}, top_k={self.memory_top_k}")

        # Memory retriever instance (will be initialized in execute with task_name)
        self.memory_retriever = None

    async def get_allowed_tools(self) -> list:
        """Get all tool definitions in OpenAI format (with permission filtering applied)"""
        return await self.tool_registry.get_all_definitions(
            allowed_tools=self.allowed_tools
        )
        
    async def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """Execute tool function (routes to either function-based or MCP tool)"""
        logger.info(f"Executing tool: {function_name}")
        logger.debug(f"Arguments: {json.dumps(function_args, ensure_ascii=False)}")
        
        try:
            result = await self.tool_registry.execute(function_name, **function_args)
            logger.info(f"Tool execution successful: {function_name}")
            logger.info(f"Tool Call Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed for {function_name}: {e}")
            raise
        
    async def _check_hypothesis_against_failed_records(self, hypothesis: Dict[str, Any]) -> tuple:
        """
        Check if a hypothesis is similar to failed attempts in memory.

        Args:
            hypothesis: Hypothesis dict with 'text' and 'rationale'

        Returns:
            Tuple of (should_filter, similar_failed_records):
            - should_filter: True if hypothesis should be filtered/regenerated
            - similar_failed_records: List of similar failed records with similarity scores
        """
        if not self.use_memory or not self.memory_retriever or not self.filter_failed_ideas:
            return False, []

        try:
            # Query memory with hypothesis text
            query = hypothesis.get("text", "")
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
                logger.warning(f"Hypothesis similar to {len(similar_failed)} failed attempt(s): '{query[:100]}...'")
                for failed in similar_failed:
                    logger.warning(f"  - {failed['name']} (similarity: {failed['similarity_score']:.2f}, "
                                 f"improvement: {failed['overall_improvement_rate']:.1%})")

            return should_filter, similar_failed

        except Exception as e:
            logger.error(f"Error checking hypothesis against failed records: {e}")
            return False, []

    async def _filter_and_regenerate_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        goal: Dict[str, Any],
        system_prompt: str,
        output_schema: Dict[str, Any],
        base_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Filter hypotheses similar to failed attempts and regenerate them.

        This method iteratively:
        1. Checks each hypothesis against failed memory records
        2. Identifies hypotheses that are too similar to failed attempts
        3. Regenerates those hypotheses with explicit avoidance instructions
        4. Repeats until no more filtering needed or max attempts reached

        Args:
            hypotheses: Initial list of generated hypotheses
            goal: Research goal
            system_prompt: System prompt for generation
            output_schema: Output schema for structured generation
            base_prompt: Base prompt without memory guidance

        Returns:
            Final list of hypotheses after filtering and regeneration
        """
        logger.info(f"Starting hypothesis filtering against failed records (threshold: {self.failed_similarity_threshold})")

        final_hypotheses = []
        current_hypotheses = hypotheses.copy()
        regeneration_attempt = 0

        while regeneration_attempt < self.max_regeneration_attempts:
            # Check each hypothesis against failed records
            to_filter = []
            to_keep = []
            failed_records_for_filtered = []

            for i, hyp in enumerate(current_hypotheses):
                should_filter, similar_failed = await self._check_hypothesis_against_failed_records(hyp)

                if should_filter:
                    to_filter.append(hyp)
                    failed_records_for_filtered.append(similar_failed)
                else:
                    to_keep.append(hyp)

            # Add kept hypotheses to final list
            final_hypotheses.extend(to_keep)

            # If nothing to filter, we're done
            if not to_filter:
                logger.info(f"No hypotheses filtered in attempt {regeneration_attempt + 1}. Filtering complete.")
                break

            logger.info(f"Attempt {regeneration_attempt + 1}: Filtered {len(to_filter)} hypotheses, "
                       f"kept {len(to_keep)} hypotheses")

            # Regenerate filtered hypotheses
            regenerated = await self._regenerate_filtered_hypotheses(
                count=len(to_filter),
                failed_records_list=failed_records_for_filtered,
                goal=goal,
                system_prompt=system_prompt,
                output_schema=output_schema,
                base_prompt=base_prompt
            )

            if not regenerated:
                logger.warning(f"Failed to regenerate hypotheses in attempt {regeneration_attempt + 1}")
                # Keep original filtered hypotheses if regeneration fails
                final_hypotheses.extend(to_filter)
                break

            # Prepare for next iteration with regenerated hypotheses
            current_hypotheses = regenerated
            regeneration_attempt += 1

        # If we hit max attempts and still have hypotheses to check, add them
        if current_hypotheses and regeneration_attempt >= self.max_regeneration_attempts:
            logger.warning(f"Reached max regeneration attempts ({self.max_regeneration_attempts}). "
                         f"Adding remaining {len(current_hypotheses)} hypotheses without further filtering.")
            final_hypotheses.extend(current_hypotheses)

        logger.info(f"Hypothesis filtering complete: {len(final_hypotheses)} final hypotheses "
                   f"after {regeneration_attempt} regeneration attempt(s)")

        return final_hypotheses

    async def _regenerate_filtered_hypotheses(
        self,
        count: int,
        failed_records_list: List[List[Dict[str, Any]]],
        goal: Dict[str, Any],
        system_prompt: str,
        output_schema: Dict[str, Any],
        base_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Regenerate hypotheses that were filtered due to similarity with failed attempts.

        Args:
            count: Number of hypotheses to regenerate
            failed_records_list: List of lists of similar failed records for each filtered hypothesis
            goal: Research goal
            system_prompt: System prompt for generation
            output_schema: Output schema for structured generation
            base_prompt: Base prompt without memory guidance

        Returns:
            List of newly generated hypotheses
        """
        if count == 0:
            return []

        # Build a special prompt that explicitly tells model to avoid failed attempts
        avoid_prompt = "\n\n# CRITICAL: Avoid These Failed Approaches\n"
        avoid_prompt += "The following ideas were previously tried and resulted in performance decline. "
        avoid_prompt += "You MUST generate completely different ideas that avoid these failed patterns:\n\n"

        # Collect all unique failed records
        all_failed = {}
        for failed_records in failed_records_list:
            for record in failed_records:
                name = record.get('name', '')
                if name not in all_failed:
                    all_failed[name] = record

        # Format failed records
        for i, (name, record) in enumerate(all_failed.items(), 1):
            avoid_prompt += f"**Failed Attempt {i}: {name}**\n"
            avoid_prompt += f"- Description: {record.get('description', '')}\n"
            avoid_prompt += f"- Performance: {record.get('overall_improvement_rate', 0):.1%}\n"
            avoid_prompt += f"- Why to avoid: This approach has been tried and failed\n\n"

        avoid_prompt += "\n**Instructions:**\n"
        avoid_prompt += "1. DO NOT generate ideas similar to the failed attempts above\n"
        avoid_prompt += "2. Analyze what made these attempts fail and avoid those patterns\n"
        avoid_prompt += "3. Generate novel ideas in completely different directions\n"
        avoid_prompt += "4. Ensure your ideas are distinct from all failed approaches\n\n"

        # Combine base prompt with avoidance guidance
        regeneration_prompt = base_prompt + avoid_prompt

        # Add explicit task
        regeneration_prompt += f"# Regeneration Task\n"
        regeneration_prompt += f"Generate {count} NEW scientifically plausible hypotheses that are completely different from the failed attempts listed above.\n"

        logger.info(f"Regenerating {count} hypotheses to avoid {len(all_failed)} failed patterns")

        try:
            response = await self._call_model(
                prompt=regeneration_prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature,
            )

            new_hypotheses = response.get("hypotheses", [])
            logger.info(f"Successfully regenerated {len(new_hypotheses)} hypotheses")
            return new_hypotheses[:count]  # Return only requested count

        except Exception as e:
            logger.error(f"Failed to regenerate hypotheses: {e}")
            return []

    async def _get_memory_guidance(self, goal: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Retrieve and format historical experiment results as guidance for idea generation.

        Provides reference to successful approaches (improved performance) and
        failed approaches (declined performance) to guide new idea generation.
        Handles memory retriever initialization if needed.

        Args:
            goal: Research goal dictionary
            context: Execution context with task_name

        Returns:
            Formatted memory guidance string to append to generation prompt,
            or empty string if memory is disabled or unavailable
        """
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

        # Return early if memory is disabled or not available
        if not self.use_memory or not self.memory_retriever:
            return ""

        # Retrieve memory and build guidance prompt
        try:
            logger.info(f"Retrieving task memory for: {self.memory_retriever.task_name}")

            # Use pre-configured memory retrieval (only query parameter needed)
            memory_result = await self.memory_retriever.retrieve(query=goal)

            if memory_result.get("success") and memory_result.get("similar_records"):
                stats = memory_result.get("statistics", {})
                records = memory_result.get("similar_records", [])

                logger.info(f"Memory retrieved: {stats.get('total', 0)} records "
                            f"(+{stats.get('positive', 0)} / 0:{stats.get('neutral', 0)} / -{stats.get('negative', 0)})")

                # Concise memory guidance - show what worked and what didn't
                successful = [r for r in records if r['label'] == 1]
                failed = [r for r in records if r['label'] == -1]

                memory_prompt = "\n# Historical Results\n"

                if successful:
                    memory_prompt += "Reference (improved):\n"
                    for r in successful[:3]:
                        rate = r.get('overall_improvement_rate', 0)
                        desc = r.get('description', '')
                        memory_prompt += f"- {r['name']} ({rate:+.1%}): {desc}\n"

                if failed:
                    memory_prompt += "Avoid (declined):\n"
                    for r in failed[:3]:
                        rate = r.get('overall_improvement_rate', 0)
                        desc = r.get('description', '')
                        memory_prompt += f"- {r['name']} ({rate:.1%}): {desc}\n"

                memory_prompt += "\n"

                recommendation = memory_result.get("recommendation", "uncertain")
                logger.info(f"Memory recommendation: {recommendation}")

                return memory_prompt

            elif memory_result.get("success"):
                # No similar records found
                logger.info("No similar historical records found - this is a novel direction")
                return ""
            else:
                logger.warning(f"Memory retrieval failed: {memory_result.get('error', 'Unknown error')}")
                return ""

        except FileNotFoundError:
            logger.info(f"No memory found for task: {self.memory_retriever.task_name}. Proceeding without historical guidance.")
            return ""
        except Exception as e:
            logger.error(f"Error retrieving task memory: {e}")
            # Continue without memory if retrieval fails
            return ""

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate novel research ideas based on provided context.

        Generates multiple idea candidates using the configured language model,
        incorporating research goals, domain knowledge, literature, code baselines,
        and iterative feedback. Returns structured ideas with rationales.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal with description, domain, constraints
                - iteration (int): Current iteration number
                - feedback (List[Dict]): Previous feedback entries (optional)
                - paper_lst (List[Dict]): Literature papers if do_survey=True
            params (Dict[str, Any]): Runtime parameters:
                - count (int): Override generation_count (optional)
                - creativity (float): Override creativity level (optional)

        Returns:
            Dict[str, Any]: Results containing:
                - hypotheses (List[Dict]): Generated hypotheses with text/rationale
                - metadata (Dict): Generation info (count, creativity, reasoning)
                - baseline_summary (str): Code baseline summary if applicable

        Raises:
            AgentExecutionError: If goal missing or generation fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        if not goal or not goal.get("description"):
            raise AgentExecutionError("Research goal is required for idea generation")

        # Extract and override parameters if provided
        count = params.get("count", self.generation_count)
        creativity = params.get("creativity", self.creativity)
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        paper_lst = context.get("paper_lst", [])
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The idea statement"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Reasoning for why this idea is plausible"
                            }
                        },
                        "required": ["text", "rationale"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the generation approach"
                },
                "baseline_summary":{
                    "type": "string",
                    "description": "Summary of the baseline code understanding"
                }
            },
            "required": ["hypotheses", "reasoning"]
        }
        
        ref_code_path = goal.get("ref_code_path") or ""
        if ref_code_path and os.path.exists(ref_code_path):
            output_schema["required"] = ["hypotheses", "reasoning", "baseline_summary"]
                    
        # Build the prompt
        prompt = self._build_generation_prompt(
            goal=goal,
            count=count,
            iteration=iteration,
            feedback=feedback,
            paper_lst=paper_lst
        )
        
        system_prompt = self._build_system_prompt(creativity)
        
        # Call the model for tool context
        self.all_tools = await self.get_allowed_tools()
        self.related_tools = get_related_tools(
            query=prompt,
            tools=self.all_tools
        )
        
        if self.related_tools:
            logger.info(f"Related tools for GenerationAgent: {[tool['function']['name'] for tool in self.related_tools]}")
            tool_prompt = self._build_tool_prompt()
            try:
                response = await self._call_model_with_tools(
                    system_prompt=tool_prompt,
                    prompt=prompt,
                    tools=self.related_tools,
                    max_iterations=params.get("max_iterations", 10),
                    max_tool_calls=params.get("max_tool_calls", 20)
                )
                tool_call_response = {
                    "status": "success",
                    "answer": response["content"],
                    "tool_calls": response["tool_calls_made"],
                    "iterations": response["iterations"]
                }
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                tool_call_response = {"status": "error", "error": str(e)}
        else:
            logger.info("No related tools found for GenerationAgent.")
            tool_call_response = {"status": "warning", "message": "No related tools found."}
        
        if tool_call_response.get("status") == "success":
            prompt += "\n\n# Tool Usage Context\n"
            prompt += tool_call_response.get("answer", "")
            prompt += "\n\nIncorporate the information obtained from the tools above into your idea generation."

        # Save base prompt before adding memory guidance (needed for regeneration)
        base_prompt = prompt

        # Retrieve memory-based guidance for idea generation
        memory_guidance = await self._get_memory_guidance(goal, context)
        prompt += memory_guidance

        # Call the model for idea generation
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature,
            )

            # Validate the response
            hypotheses = response.get("hypotheses", [])
            if not hypotheses:
                logger.warning("Generation agent returned no hypotheses")

            # Filter and regenerate hypotheses similar to failed attempts
            if self.filter_failed_ideas and self.use_memory and self.memory_retriever:
                hypotheses = await self._filter_and_regenerate_hypotheses(
                    hypotheses=hypotheses,
                    goal=goal,
                    system_prompt=system_prompt,
                    output_schema=output_schema,
                    base_prompt=base_prompt
                )

            # Add metadata to the response
            result = {
                "hypotheses": hypotheses,
                "metadata": {
                    "count": len(hypotheses),
                    "creativity": creativity,
                    "iteration": iteration,
                    "reasoning": response.get("reasoning", "")
                },
                "baseline_summary": response.get("baseline_summary", "")
            }

            return result
            
        except Exception as e:
            logger.error(f"Generation agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate hypotheses: {str(e)}")
    
    def _build_tool_prompt(self) -> str:
        """
        Build tool usage context prompt.

        Constructs a prompt section that describes the tools available
        for the generation agent to use, including their names and descriptions.

        Returns:
            str: Formatted tool usage context prompt
        """
        tool_descriptions = "\n".join([f"- {tool['function']['name']}: {tool['function']['description']}" for tool in self.related_tools])
        tool_prompt = f"""
            You are a research intelligence specialist. Your task is to analyze the user's research request and systematically gather contextual information that will support scientific hypothesis generation.

            **IMPORTANT: Your role is NOT to generate hypotheses or ideas. You are preparing the contextual foundation by collecting relevant materials that will later be incorporated into a hypothesis generation prompt.**

            ## Understanding the Task

            The user will provide a research request that may include:
            - Research goal or question
            - Domain or field of study
            - Existing constraints or requirements
            - Background information or baseline methods
            - Reference code or literature
            - Specific areas of focus

            Your job is to:
            1. **Analyze** what additional context would strengthen hypothesis generation for this specific request
            2. **Identify** knowledge gaps that need to be filled
            3. **Use available tools** to gather relevant information
            4. **Organize** the collected materials for easy integration

            ## Available Tools

            {tool_descriptions}

            ## Context Categories to Consider

            Based on the user's request, determine which of these areas need investigation:

            **A. Domain Foundation**
            - Core principles, theories, and mechanisms in the field
            - Key terminology and standard definitions
            - Fundamental concepts the user may not have provided

            **B. Current Knowledge State**
            - Recent advances and state-of-the-art methods
            - Existing approaches to similar problems
            - Known limitations of current methods
            - Gaps or controversies in the field

            **C. Technical Methods**
            - Relevant experimental techniques or algorithms
            - Standard evaluation metrics and benchmarks
            - Available tools and frameworks
            - Implementation considerations

            **D. Constraints & Feasibility**
            - Technical limitations in the domain
            - Resource or computational requirements
            - Ethical considerations
            - Practical barriers

            **E. Cross-Domain Insights**
            - Related problems in adjacent fields
            - Analogous solutions from other domains
            - Interdisciplinary approaches that could apply
            - Novel perspectives from recent literature

            ## Your Workflow

            1. **Parse the User Request**
            - Identify the research goal and domain
            - Note what information the user has already provided
            - Determine what's missing or could be expanded

            2. **Identify Information Needs**
            - What context would make hypothesis generation more effective?
            - What background knowledge is assumed but not stated?
            - What recent developments might be relevant?
            - What related work should be considered?

            3. **Plan Tool Usage**
            - Decide which tools to use based on identified needs
            - Prioritize the most critical information gaps
            - Plan the sequence of queries

            4. **Gather Information**
            - Execute tool calls systematically
            - Collect relevant materials from multiple sources
            - Follow promising leads
            - Verify important findings

            5. **Synthesize and Organize**
            - Structure information into clear categories
            - Highlight most relevant findings
            - Note connections to the user's specific request
            - Identify remaining gaps

            ## Output Structure

            **1. Request Analysis**
            - Summary of the user's research goal and what they provided
            - Identified information gaps and needs
            - Rationale for your investigation approach

            **2. Gathered Context** (organized by relevant categories)
            For each category investigated:
            - Synthesized findings with clear subsections
            - Source references where applicable
            - Relevance to the user's specific request

            **3. Key Relevant Findings**
            - Most significant information for this specific research goal
            - Important connections or patterns
            - Recent advances or methods that apply

            **4. Identified Opportunities**
            - Areas where current knowledge is limited
            - Contradictions or debates in the literature
            - Gaps that the user's research could address
            - Unexplored angles or approaches

            **5. Investigation Summary**
            - Tools and sources consulted
            - Search strategy employed
            - Assessment of context completeness

            ## Quality Standards

            - **Targeted**: Focus on information directly relevant to the user's request
            - **Comprehensive**: Cover important aspects thoroughly
            - **Current**: Prioritize recent developments (last 3-5 years)
            - **Accurate**: Verify key information across sources
            - **Factual**: Present existing knowledge, not speculation
            - **Well-Organized**: Structure for easy integration into generation prompt
            - **Gap-Aware**: Clearly identify what information is missing

            ## Critical Boundaries

            **Do NOT:**
            - Generate or propose hypotheses yourself
            - Suggest specific research directions or experiments
            - Make claims beyond what sources support
            - Include personal opinions or speculation

            **DO:**
            - Analyze what context would be most valuable
            - Gather and synthesize relevant existing knowledge
            - Identify where knowledge gaps exist
            - Present information objectively and clearly
            - Focus on creating a strong foundation for hypothesis generation

            ## Example Reasoning Process

            When you receive a request, think:
            1. "What domain knowledge is essential but not provided?"
            2. "What recent work is relevant to this goal?"
            3. "What methods or techniques could apply here?"
            4. "What are the known challenges in this area?"
            5. "What related fields might offer insights?"
            6. "Which tools can help me find this information?"

            Your gathered context will be integrated into a prompt that guides hypothesis generation. Ensure the information is relevant, accurate, organized, and provides both grounding in established knowledge and awareness of current frontiers and opportunities.
        """
        return tool_prompt
        
    def _build_generation_prompt(self,
                               goal: Dict[str, Any],
                               count: int,
                               iteration: int,
                               feedback: List[Dict[str, Any]],
                               paper_lst: List[Dict]) -> str:
        """
        Construct comprehensive prompt for idea generation.

        Builds a structured prompt incorporating research goals, domain constraints,
        background information, literature, code baselines, and iterative feedback.
        Handles both file-level and project-level code analysis.

        Args:
            goal (Dict[str, Any]): Goal with description/domain/constraints/background
            count (int): Number of hypotheses to generate
            iteration (int): Current iteration number for feedback context
            feedback (List[Dict[str, Any]]): Previous feedback entries
            paper_lst (List[Dict]): Literature papers for survey mode

        Returns:
            str: Formatted multi-section prompt for the language model
        """
        
        # Extract goal information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        background = goal.get("background", "")
        
        # Start with the goal
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add background if available
        if background:
            prompt += "Background Information is a detailed description of the baseline method. Please analyze the provided Background Information and give novel hypotheses based on the research goal and the background information.\n\n"
            prompt += f"# Background Information\n{background}\n\n"
            
        # Add constraints if available
        if constraints:
            prompt += "# Constraints\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"

        # if do_survey, literature information is included in goal, and need prompt
        if self.do_survey and paper_lst:
            logger.info("Add literature information to prompt")
            literature_prompt = "# Literature Information\n"
            for paper in paper_lst:
                literature_prompt += f"- {paper['title']} ({paper['year']})\n {paper['abstract']} \n\n"
            prompt += literature_prompt
            prompt += "Please analyze the provided Literature Information and give novel hypotheses based on the research goal and the literature."
            
        # Add reference code if available
        # load code if exist, judge file/dir/not exist
        ref_code_path = goal.get("ref_code_path") or ""
        if ref_code_path and os.path.exists(ref_code_path):
            if os.path.isfile(ref_code_path):
                logger.info("Perform #file-level# code understanding and generation")
                with open(ref_code_path, 'r') as f:
                    ref_code = f.read()
                logger.info("Add reference code (RAW) to prompt")
                prompt += f"# Reference Code\n```python\n{ref_code}\n```\n\n"
                
            elif os.path.isdir(ref_code_path):
                logger.info("Perform #project-level# code understanding and generation")
                logger.info("Loading codeivew Agent ...")
                if os.path.exists(os.path.join(ref_code_path, "code_summary.json")):
                    logger.info("Code summary exists! Loading code summary from file.")
                    with open(os.path.join(ref_code_path, "code_summary.json"), 'r') as f:
                        ref_code = f.read()
                    ref_code = json.loads(ref_code)
                else:
                    logger.info("Code summary does not exist! Generating code summary.")
                    # Get exp_backend from global config
                    global_config = self.config.get("_global_config", {})
                    exp_backend = global_config.get("exp_backend")

                    if exp_backend == "claudecode":
                        # Import the function when needed
                        from .codeview_agent import get_repo_structure_claudecode

                        # Get proxy settings and model from config if available
                        proxy_settings = global_config.get("proxy_settings", None)
                        claude_model = global_config.get("experiment", {}).get("model", "claude-sonnet-4-5-20250929")

                        logger.info(f"Using Claude Code backend to generate code summary with model: {claude_model}")
                        ref_code = get_repo_structure_claudecode(
                            project_path=ref_code_path,
                            output_dir=ref_code_path,
                            output_name="code_summary.json",
                            proxy_settings=proxy_settings,
                            model=claude_model
                        )
                    else:
                        # Use codeview agent to generate code summary (for claudecode, iflow, or other backends)
                        logger.info(f"Using {exp_backend} backend to generate code summary")
                        ref_code = get_repo_structure(
                            project_path=ref_code_path,
                            output_dir=ref_code_path,
                            output_name="code_summary.json",
                            ignore_list=None,
                            model=self.model.model_name,
                            provider="user"
                        )
                # Format key_files list into a readable string
                key_files = ref_code.get('key_files', [])
                if isinstance(key_files, list):
                    key_files_str = "\n".join([
                        f"- {f.get('path', 'unknown')}: {f.get('description', '')}"
                        for f in key_files if isinstance(f, dict)
                    ])
                else:
                    key_files_str = str(key_files)
                ref_code = ref_code.get('summary', '') + "\n\nKey Files:\n" + key_files_str
                logger.info("Add reference code (CODEVIEW) to prompt")
                prompt += f"# Reference Code (Repo Summary) \n{ref_code}\n\n"
            
            prompt += "The Reference Code serves as the baseline code aligned with the Research Goal. The proposed idea should be innovative, building upon the Reference Code to enhance task performance."
            prompt += "Please analyze the provided Reference Code and give a brief summary from the following perspectives: \n 1. Methods and Concepts: Describe the main methods and concepts used in the code. How do they support the functionality? 2. Model Structure: If applicable, outline the model architecture and design choices. How does the structure serve its purpose? 3. Limitations: What are the limitations of the code? How can they be addressed?"
        else:
            ref_code = ""
            logger.error("Default: No reference Code, simple idea-gen")
        
        if background and ref_code:
            prompt += "The Background Information and Reference Code are closely related. The proposed idea should be innovative, building upon the Background Information and Reference Code to enhance task performance."
        
        # Add feedback from previous iterations
        if feedback and iteration > 0:
            prompt += "# Previous Feedback\n"
            # Take the most recent feedback entries, up to 3
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0), 
                reverse=True
            )[:3]
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                prompt += f"Iteration {feedback_iter}: {feedback_text}\n\n"
                
        # Add task description
        prompt += f"# Task\n"
        prompt += f"Generate {count} scientifically plausible hypotheses for the research goal above."
        
        if iteration > 0:
            prompt += f" This is iteration {iteration}, so incorporate the feedback provided."
        
        return prompt
    
    def _build_system_prompt(self, creativity: float) -> str:
        """
        Build system prompt tailored to creativity level.

        Creates system-level instructions that guide the model's idea generation
        style based on the creativity parameter, ranging from conservative to highly
        innovative approaches.

        Args:
            creativity (float): Creativity level 0-1 where:
                >0.8: highly innovative and out-of-the-box
                >0.5: creative but grounded in principles
                <=0.5: conservative and evidence-based

        Returns:
            str: System prompt with tone and quality guidelines
        """
        if creativity > 0.8:
            tone = "highly innovative and out-of-the-box"
        elif creativity > 0.5:
            tone = "creative but grounded in scientific principles"
        else:
            tone = "conservative and strictly evidence-based"
            
        return f"""You are a creative scientific idea generator. Your task is to generate {tone} scientific hypotheses based on all the information given by the user.
Ensure that each idea:
1.Are novel, not obvious from existing literature, and grounded in scientific principles.
2.Propose specific mechanisms or pathways and include a brief explanation of them.
3.Are specific, testable, and can be tested experimentally.
4.Address the research goal directly and have clear scientific significance.
5.Consider constraints and domain knowledge.

Be creative but scientifically rigorous. Propose hypotheses that make unexpected connections between concepts, challenge conventional wisdom, or apply principles from one field to another.

The different scientific hypotheses proposed need to remain separate and cannot be similar.

Your hypotheses should be detailed enough for a scientist in the field to understand and evaluate, but not so detailed that they require specialized knowledge outside the domain. 
"""

    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'GenerationAgent':
        """
        Factory method to create GenerationAgent from configuration.

        Args:
            config (Dict[str, Any]): Agent configuration dictionary
            model (BaseModel): Language model instance

        Returns:
            GenerationAgent: Configured instance
        """
        return cls(model, config) 
