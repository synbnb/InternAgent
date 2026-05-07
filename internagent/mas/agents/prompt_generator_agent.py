"""
Prompt Generator Agent for InternAgent

This agent generates new research directions (task) and backgrounds based on
accumulated experiences from the unified experience library.

It analyzes high-confidence experiences to:
1. Identify promising research directions from successful patterns
2. Generate new task descriptions that build on proven insights
3. Update background information with key learnings

The agent prioritizes high-confidence experiences and synthesizes them into
actionable research directions.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class PromptGeneratorAgent(BaseAgent):
    """
    Agent that generates new research prompts based on unified experience library.

    This agent synthesizes accumulated experiences to:
    - Identify promising research directions from high-confidence experiences
    - Generate new task descriptions that build on proven insights
    - Update background information with key learnings
    - Create focused, actionable research objectives

    Attributes:
        focus_on_best (bool): Whether to prioritize highest confidence experiences
        min_confidence (int): Minimum confidence threshold for considering experiences
    """

    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the prompt generator agent.

        Args:
            model (BaseModel): Language model for prompt generation
            config (Dict[str, Any]): Configuration with keys:
                - focus_on_best (bool): Prioritize high confidence experiences (default: True)
                - min_confidence (int): Minimum confidence threshold (default: 5)
        """
        super().__init__(model, config)

        self.focus_on_best = config.get("focus_on_best", True)
        self.min_confidence = config.get("min_confidence", 5)
        self.temperature = config.get("temperature", None)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate new task and background based on unified experience library.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - experience_library (Dict): Unified experience library with experiences
                - current_task (str): Current task description
                - current_background (str): Current background
                - domain (str): Research domain
            params (Dict[str, Any]): Runtime parameters:
                - generate_task (bool): Generate new task (default: True)
                - generate_background (bool): Generate new background (default: True)

        Returns:
            Dict[str, Any]: Generated prompts containing:
                - new_task (str): New task direction
                - new_background (str): Updated background
                - reasoning (str): Explanation of the generation
                - experiences_used (int): Number of experiences used

        Raises:
            AgentExecutionError: If required data is missing or generation fails
        """
        # Extract required data
        experience_library = context.get("experience_library", {})
        current_task = context.get("current_task", "")
        current_background = context.get("current_background", "")
        domain = context.get("domain", "machine learning")
        fix_direction = context.get("fix_direction", "")
        # Handle backward compatibility with positive_library format
        if not experience_library and context.get("positive_library"):
            experience_library = context.get("positive_library", {})

        if not experience_library:
            raise AgentExecutionError("Experience library is required")

        experiences = experience_library.get("experiences", [])
        if not experiences:
            raise AgentExecutionError("Experience library is empty")

        # Extract parameters
        generate_task = params.get("generate_task", True)
        generate_background = params.get("generate_background", True)

        try:
            # Filter and sort experiences by confidence
            high_confidence_exps = self._filter_high_confidence_experiences(experiences)

            if not high_confidence_exps:
                logger.warning(f"No high-confidence experiences found (threshold: {self.min_confidence})")
                high_confidence_exps = experiences[:10]  # Fallback to top 10

            logger.info(f"Using {len(high_confidence_exps)} high-confidence experiences for prompt generation")

            result = {}

            if generate_task:
                logger.info("Generating new task direction...")
                new_task = await self._generate_task(
                    high_confidence_exps,
                    current_task,
                    domain,
                    fix_direction
                )
                result["new_task"] = new_task

            if generate_background:
                logger.info("Generating new background...")
                new_background = await self._generate_background(
                    high_confidence_exps,
                    current_background,
                    domain
                )
                result["new_background"] = new_background

            result["experiences_used"] = len(high_confidence_exps)
           

            return result

        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate prompts: {str(e)}")

    def _filter_high_confidence_experiences(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and sort experiences by confidence.

        Args:
            experiences: List of experiences from library

        Returns:
            List of high-confidence experiences, sorted by confidence
        """
        # Filter by minimum confidence
        filtered = [
            exp for exp in experiences
            if exp.get("confidence", 0) >= self.min_confidence
        ]

        # Sort by confidence (descending)
        sorted_exps = sorted(filtered, key=lambda x: x.get("confidence", 0), reverse=True)

        # Take top experiences if focus_on_best
        if self.focus_on_best:
            return sorted_exps[:15]  # Top 15 experiences
        else:
            return sorted_exps[:25]  # Top 25 experiences

    async def _generate_task(
        self,
        experiences: List[Dict[str, Any]],
        current_task: str,
        domain: str,
        fix_direction
    ) -> str:
        """
        Generate new task direction based on high-confidence methodological experiences.

        Args:
            experiences: High-confidence experiences
            current_task: Current task description
            domain: Research domain

        Returns:
            New task description string
        """
        # Filter for methodological experiences only
        methodological_exps = [
            exp for exp in experiences
            if exp.get("type", "").lower() == "methodological"
        ]

        # If no methodological experiences found, log warning and use all experiences as fallback
        if not methodological_exps:
            logger.warning("No methodological experiences found, using all high-confidence experiences as fallback")
            methodological_exps = experiences
        else:
            logger.info(f"Using {len(methodological_exps)} methodological experiences out of {len(experiences)} total")

        # Format methodological experiences with their confidence scores
        experiences_formatted = []
        for i, exp in enumerate(methodological_exps, 1):
            name = exp.get("name", "")
            content = exp.get("content", "")
            confidence = exp.get("confidence", 0)
            experiences_formatted.append(f"{i}. [{confidence}/10] {name}: {content}")

        experiences_text = "\n".join(experiences_formatted)

        # Add fix_direction section if provided
        fix_direction_section = ""
        if fix_direction:
            fix_direction_section = f"""
## Fixed Research Direction (Constraint)
{fix_direction}

**Important**: All generated research directions MUST explore within this fixed direction. Do not deviate from this overarching theme.
"""

        prompt = f"""# Task: Generate New Research Direction
## Current Research Direction
{current_task}
{fix_direction_section}
## Accumulated Experiences from Experiments

The following experiences were extracted from comparative analysis of experimental methods. Each experience is rated on confidence (0-10 scale, where higher means more reliable evidence):

{experiences_text}

## Your Task

Based on the accumulated experiences above, propose promising research directions to explore in the next phase.
**Key Requirements**:

1. **Stay Within Fixed Direction**:  All explorations MUST be within that scope. Generate diverse sub-directions within this constraint." 
2. **Propose Directions, Not Methods**: Focus on WHAT to explore rather than HOW to implement.
3. **High-Level Exploration Goals**: Generate broad, strategic questions or areas to investigate, not specific technical implementations.
4. **Leverage High-Confidence Insights**: Use experiences with higher confidence to inform broader research directions.
5. **Leverage High-Confidence Insights**: Use experiences with higher confidence to identify gaps or promising areas worth deeper exploration.
6. **Identify Unexplored Territories**: Based on what has been learned, suggest what aspects or dimensions deserve further investigation.
7. **Explore Rich Variations**: Within the fixed direction, generate as diverse and rich explorations as possible, covering different aspects, dimensions, and perspectives.

Objective:
Generate a concise description of 2-3 potential research directions. Keep each direction high-level and strategic, not detailed or fine-grained.
"""

        result = await self._call_model(
            prompt=prompt,
            system_prompt="You are an expert at synthesizing research experiences into actionable research directions. You excel at identifying promising paths forward based on accumulated evidence, prioritizing high-confidence insights and creating focused, concrete research objectives.",
            temperature=self.temperature 
        )

        return result.strip()

    async def _generate_background(
        self,
        experiences: List[Dict[str, Any]],
        current_background: str,
        domain: str
    ) -> str:
        """
        Generate new background based on high-confidence experiences.

        Args:
            experiences: High-confidence experiences
            current_background: Current background text
            domain: Research domain

        Returns:
            New background description string
        """
        # Extract top insights from highest confidence experiences
        top_experiences = experiences[:10]

        insights_summary = []
        for exp in top_experiences:
            name = exp.get("name", "")
            content = exp.get("content", "")
            confidence = exp.get("confidence", 0)
            insights_summary.append(f"- [{confidence}/10] {name}: {content}")

        insights_text = "\n".join(insights_summary)

        prompt = f"""# Task: Update Research Background

## Domain
{domain}

## Current Background
{current_background}

## Key Insights from Recent Research (Top 10 by Confidence)
{insights_text}

## Your Task

Update the background section to incorporate the most important insights from recent research.

The updated background should:
1. **Retain Core Context**: Keep essential domain background from current version
2. **Integrate Key Learnings**: Weave in the high-confidence insights naturally
3. **Explain What We Learned**: Describe why certain approaches work or don't work
4. **Set the Stage**: Provide context for future research directions
5. **Be Cohesive**: Read as a unified, well-organized background section

**Important Guidelines**:
- Don't just append; synthesize new insights with existing knowledge
- Prioritize insights with higher confidence scores
- Emphasize causal understanding (why things work, not just what works)
- Reference specific findings when relevant
- Maintain professional academic tone
- Keep it concise but informative (3-5 paragraphs)

Generate an updated background section that reflects our current understanding.
"""

        result = await self._call_model(
            prompt=prompt,
            system_prompt="You are an expert at writing research backgrounds that synthesize prior work with new findings. You excel at creating cohesive narratives that inform future research, prioritizing high-confidence insights.",
            temperature=self.temperature 
        )

        return result.strip()


