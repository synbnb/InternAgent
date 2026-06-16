"""
流水线人工介入钩子模块
提供文件系统状态轮询机制，实现非侵入式的人机交互

设计原则：
1. 非侵入式 — 不修改 internagent/ 核心代码
2. 文件系统通信 — 通过状态文件在进程间传递状态
3. 超时机制 — 所有等待点都有默认超时，避免流水线永久阻塞
4. 向前兼容 — 无人工审批时自动 fallback 到系统默认行为
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

REVIEW_DIR_NAME = ".human_review"

# 状态文件名称常量
STATE_FILE = "pipeline_state.json"
IDEAS_PENDING_FILE = "ideas_pending.json"
IDEAS_APPROVED_FILE = "ideas_approved.json"
CODE_REVIEW_PREFIX = "code_review_"
CODE_FEEDBACK_PREFIX = "code_feedback_"
RESULT_REVIEW_FILE = "result_review.json"
RESULT_FEEDBACK_FILE = "result_feedback.json"

# 状态值常量
STATUS_WAITING = "waiting"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"
STATUS_MODIFIED = "modified"
STATUS_TIMEOUT = "timeout"

# 默认超时配置（秒）
DEFAULT_TIMEOUTS = {
    "task_review": 1800,      # 30 分钟
    "idea_review": 3600,      # 60 分钟
    "code_review": 1800,      # 30 分钟
    "result_review": 1800,    # 30 分钟
}

# 默认轮询间隔（秒）
DEFAULT_POLL_INTERVAL = 5


# ============================================================================
# 核心函数
# ============================================================================

def get_review_dir(output_dir: str) -> str:
    """获取审批状态目录的路径"""
    return os.path.join(output_dir, REVIEW_DIR_NAME)


def ensure_review_dir(output_dir: str) -> str:
    """确保审批目录存在"""
    review_dir = get_review_dir(output_dir)
    os.makedirs(review_dir, exist_ok=True)
    return review_dir


def write_state(state_file: str, state: dict) -> None:
    """将流水线状态写入文件"""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    logger.debug(f"Wrote state to {state_file}: status={state.get('status')}")


def read_state(state_file: str) -> dict:
    """从文件读取流水线状态"""
    if not os.path.exists(state_file):
        return {}
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read state file {state_file}: {e}")
        return {}


def read_state_safe(state_file: str) -> dict:
    """安全读取状态，文件损坏时返回空字典"""
    return read_state(state_file)


# ============================================================================
# 等待函数
# ============================================================================

def wait_for_state(
    state_file: str,
    target_status: str,
    timeout: int = 3600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> dict:
    """
    轮询等待状态文件达到目标状态

    Args:
        state_file: 状态文件路径
        target_status: 期望状态值（如 'approved', 'rejected'）
        timeout: 超时秒数
        poll_interval: 轮询间隔秒数

    Returns:
        最终的状态字典，超时返回空 dict
    """
    start = time.time()
    while time.time() - start < timeout:
        state = read_state(state_file)
        if state.get("status") == target_status:
            logger.info(f"State reached target status '{target_status}' after "
                        f"{time.time() - start:.1f}s")
            return state

        # 检查是否被标记为超时跳过
        if state.get("status") == STATUS_TIMEOUT:
            logger.info(f"State marked as timeout, returning empty")
            return {}

        time.sleep(poll_interval)

    logger.warning(f"Timeout waiting for status '{target_status}' after {timeout}s")
    return {}


def wait_for_state_change(
    state_file: str,
    timeout: int = 3600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> dict:
    """
    轮询等待状态文件发生任意变化（从 waiting 变为其他状态）

    相比 wait_for_state，此函数接受 approved/rejected/skipped 任一状态

    Returns:
        最终的状态字典，超时返回空 dict
    """
    start = time.time()
    final_statuses = {STATUS_APPROVED, STATUS_REJECTED, STATUS_SKIPPED, STATUS_MODIFIED}

    while time.time() - start < timeout:
        state = read_state(state_file)
        status = state.get("status", "")
        if status in final_statuses:
            logger.info(f"State changed to '{status}' after "
                        f"{time.time() - start:.1f}s")
            return state

        if status == STATUS_TIMEOUT:
            return {}

        time.sleep(poll_interval)

    logger.warning(f"Timeout waiting for state change after {timeout}s")
    return {}


# ============================================================================
# 想法审批相关
# ============================================================================

def write_pending_ideas(
    output_dir: str,
    session_id: str,
    top_ideas: List[Dict[str, Any]],
) -> str:
    """
    将待审批的想法写入状态文件

    Args:
        output_dir: 流水线输出目录
        session_id: MAS 会话 ID
        top_ideas: 想法列表（包含 refined_method_details 等字段）

    Returns:
        状态文件路径
    """
    review_dir = ensure_review_dir(output_dir)
    ideas_file = os.path.join(review_dir, IDEAS_PENDING_FILE)

    # 标准化想法数据
    standardized_ideas = []
    for i, idea in enumerate(top_ideas):
        # 兼容两种数据格式：直接是 dict 或包含 id/text 的对象
        details = idea.get("refined_method_details", idea)

        standardized_ideas.append({
            "id": idea.get("id", details.get("id", f"idea_{i}")),
            "title": details.get("title", idea.get("title", "")),
            "description": details.get("description", idea.get("description", "")),
            "method": details.get("method", idea.get("method", "")),
            "rationale": idea.get("rationale", ""),
            "score": idea.get("score", 0.0),
            "scores": idea.get("scores", {}),
        })

    state = {
        "status": STATUS_WAITING,
        "session_id": session_id,
        "created_at": time.time(),
        "ideas": standardized_ideas,
        "total_count": len(standardized_ideas),
    }

    write_state(ideas_file, state)
    logger.info(f"Wrote {len(standardized_ideas)} pending ideas to {ideas_file}")
    return ideas_file


def read_pending_ideas(output_dir: str) -> dict:
    """读取待审批的想法"""
    return read_state(os.path.join(get_review_dir(output_dir), IDEAS_PENDING_FILE))


def approve_ideas(
    output_dir: str,
    selected_ids: List[str] = None,
    modifications: Dict[str, Dict] = None,
) -> dict:
    """
    用户审批想法，记录选择

    Args:
        output_dir: 流水线输出目录
        selected_ids: 用户选择的 idea ID 列表（None 表示全选）
        modifications: 对特定 idea 的修改 {idea_id: {field: new_value}}

    Returns:
        更新后的状态 dict
    """
    review_dir = ensure_review_dir(output_dir)
    pending = read_pending_ideas(output_dir)
    if not pending or pending.get("status") != STATUS_WAITING:
        return {"status": "error", "message": "No pending ideas found"}

    ideas = pending.get("ideas", [])

    # 确定选中的 ideas
    if selected_ids is None:
        selected = ideas
    else:
        selected = [idea for idea in ideas if idea.get("id") in selected_ids]

    # 应用修改
    if modifications:
        for idea in selected:
            idea_id = idea.get("id")
            if idea_id in modifications:
                idea.update(modifications[idea_id])

    approved_state = {
        "status": STATUS_APPROVED,
        "session_id": pending.get("session_id"),
        "approved_at": time.time(),
        "ideas": selected,
        "selected_count": len(selected),
        "total_count": pending.get("total_count", 0),
    }

    # 写两种文件：状态文件和审批结果文件
    ideas_file = os.path.join(review_dir, IDEAS_PENDING_FILE)
    approved_file = os.path.join(review_dir, IDEAS_APPROVED_FILE)

    write_state(ideas_file, {"status": STATUS_APPROVED})
    write_state(approved_file, approved_state)

    logger.info(f"Approved {len(selected)}/{pending.get('total_count', 0)} ideas")
    return approved_state


def reject_ideas(output_dir: str, reason: str = "") -> dict:
    """用户否决所有想法"""
    review_dir = ensure_review_dir(output_dir)
    ideas_file = os.path.join(review_dir, IDEAS_PENDING_FILE)
    write_state(ideas_file, {
        "status": STATUS_REJECTED,
        "rejected_at": time.time(),
        "reason": reason,
    })
    logger.info(f"Ideas rejected: {reason}")
    return {"status": STATUS_REJECTED, "reason": reason}


# ============================================================================
# 结果审查相关
# ============================================================================

def write_result_for_review(
    output_dir: str,
    idea_name: str,
    run_num: int,
    scores: dict,
    report_text: str,
    report_images: List[str],
) -> str:
    """将实验结果写入待审查状态"""
    review_dir = ensure_review_dir(output_dir)
    result_file = os.path.join(review_dir, RESULT_REVIEW_FILE)

    state = read_state(result_file) if os.path.exists(result_file) else {"results": []}
    state["status"] = STATUS_WAITING

    result_entry = {
        "idea_name": idea_name,
        "run_num": run_num,
        "timestamp": time.time(),
        "scores": scores,
        "report_preview": report_text[:2000] if report_text else "",
        "report_images": report_images,
    }
    state["results"].append(result_entry)
    state["current"] = result_entry

    write_state(result_file, state)
    return result_file


def submit_result_feedback(output_dir: str, feedback: dict) -> dict:
    """用户提交结果反馈"""
    review_dir = ensure_review_dir(output_dir)
    feedback_file = os.path.join(review_dir, RESULT_FEEDBACK_FILE)
    result_file = os.path.join(review_dir, RESULT_REVIEW_FILE)

    feedback_state = {
        "status": STATUS_APPROVED,
        "submitted_at": time.time(),
        "feedback": feedback,
    }
    write_state(feedback_file, feedback_state)
    write_state(result_file, {"status": STATUS_APPROVED})

    return feedback_state


# ============================================================================
# 流水线状态管理
# ============================================================================

def get_pipeline_status(output_dir: str) -> dict:
    """
    获取流水线的完整状态摘要

    检查所有状态文件，返回当前的总体状态
    """
    review_dir = get_review_dir(output_dir)
    if not os.path.exists(review_dir):
        return {"status": "unknown", "message": "No pipeline state found"}

    status = {"status": "running", "reviews": {}}

    # 检查想法审批状态
    ideas_pending = read_state(os.path.join(review_dir, IDEAS_PENDING_FILE))
    if ideas_pending:
        status["reviews"]["idea_review"] = ideas_pending.get("status", "unknown")

    # 检查结果审查状态
    result_review = read_state(os.path.join(review_dir, RESULT_REVIEW_FILE))
    if result_review:
        status["reviews"]["result_review"] = result_review.get("status", "pending")

    # 推断整体状态
    review_statuses = list(status["reviews"].values())
    if STATUS_WAITING in review_statuses:
        status["status"] = "waiting_approval"

    return status


def mark_timeout(output_dir: str, review_type: str) -> None:
    """标记某个审批点为超时，让流水线跳过等待"""
    review_dir = ensure_review_dir(output_dir)
    type_file_map = {
        "idea_review": IDEAS_PENDING_FILE,
        "result_review": RESULT_REVIEW_FILE,
    }
    filename = type_file_map.get(review_type)
    if not filename:
        return

    state_file = os.path.join(review_dir, filename)
    current = read_state(state_file)
    current["status"] = STATUS_TIMEOUT
    current["timeout_at"] = time.time()
    write_state(state_file, current)
    logger.info(f"Marked {review_type} as timeout in {output_dir}")
