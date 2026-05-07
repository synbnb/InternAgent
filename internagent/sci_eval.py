"""
Thin adapter over the official ResearchClawBench evaluation scorer.

All rubric, prompt construction, and per-item scoring logic lives in
rcb_evaluation (symlinked from ResearchClawBench/evaluation/).

This module only adapts the interface for InternAgent's experiment loop:
  score_run(workspace_dir, checklist_path, model) -> scores dict
  write_final_info(run_dir, scores) -> path to final_info.json
"""
import os
import os.path as osp
import json
import logging
from pathlib import Path
from typing import Dict, Any

from internagent.rcb_evaluation.score import (
    _read_report, _find_generated_images, _score_single_item,
)
from internagent.rcb_evaluation.utils import safe_resolve

logger = logging.getLogger(__name__)


def _get_env(name: str, *fallbacks: str) -> str:
    """Get env var, trying fallback names if primary is empty."""
    val = os.environ.get(name, "")
    if val:
        return val
    for fb in fallbacks:
        val = os.environ.get(fb, "")
        if val:
            return val
    return ""


def score_run(
    workspace_dir: str,
    checklist_path: str,
    model: str = "gpt-5.1",
) -> Dict[str, Any]:
    """Score all checklist items using the official ResearchClawBench scorer."""
    if not osp.exists(checklist_path):
        logger.error(f"Checklist not found: {checklist_path}")
        return {'total_score': 0}

    with open(checklist_path) as f:
        checklist = json.load(f)

    if not checklist:
        logger.warning("Empty checklist, returning score 0")
        return {'total_score': 0}

    workspace = Path(workspace_dir)
    report_text = _read_report(workspace)
    if not report_text:
        logger.warning(f"No report found in {workspace_dir}")
        return {'total_score': 0}

    instructions_path = workspace / "INSTRUCTIONS.md"
    instructions = ""
    if instructions_path.exists():
        instructions = instructions_path.read_text(encoding="utf-8", errors="replace")

    generated_images = _find_generated_images(workspace)

    from structai import LLMAgent, multi_thread

    agent = LLMAgent(
        api_key=_get_env("OPENAI_API_KEY"),
        api_base=_get_env("OPENAI_BASE_URL", "OPENAI_API_BASE_URL"),
        model_version=model,
        system_prompt=(
            "You are a strict scientific peer reviewer evaluating AI-generated "
            "research. Score the report against the criterion only — do not "
            "attempt to solve the research task yourself."
        ),
        temperature=0,
        max_tokens=500,
        time_limit=120,
        max_try=2,
    )

    target_base = workspace / "target_study"

    def score_item(index, item_data):
        target_path = None
        if item_data.get("type", "text") == "image":
            target_rel = item_data.get("path", "")
            if target_rel:
                target_path = safe_resolve(target_base, target_rel)
        return _score_single_item(
            agent, report_text, item_data, target_path, generated_images, instructions
        )

    inputs = [{"index": i, "item_data": item} for i, item in enumerate(checklist)]
    n = len(checklist)
    logger.info(f"Scoring {n} checklist items in parallel (model={model})")
    raw_results = multi_thread(
        inputs, score_item, max_workers=min(n, 16), use_tqdm=False
    )

    scores: Dict[str, Any] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for i, (item, result) in enumerate(zip(checklist, raw_results)):
        w = float(item.get('weight', 1.0))
        sr = result if result else {"score": 0, "reasoning": "Scoring failed."}
        scores[f'item_{i}_score'] = sr['score']
        scores[f'item_{i}_reasoning'] = sr['reasoning']
        weighted_sum += sr['score'] * w
        total_weight += w
        logger.info(f"  item_{i}: {sr['score']}/100 (weight={w:.2f})")

    scores['total_score'] = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0
    logger.info(f"Weighted total score: {scores['total_score']:.2f}/100")
    return scores


def write_final_info(run_dir: str, scores: Dict[str, Any]) -> str:
    """Write scores as final_info.json for InternAgent's loop-critic."""
    means = {k: v for k, v in scores.items() if not k.endswith('_reasoning')}
    final_info = {"sci_task": {"means": means}}

    os.makedirs(run_dir, exist_ok=True)
    path = osp.join(run_dir, "final_info.json")
    with open(path, 'w') as f:
        json.dump(final_info, f, indent=2)
    logger.info(f"Wrote final_info.json -> {path}")
    return path
