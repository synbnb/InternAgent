"""
启动 InternAgent 发现流水线
包含想法生成（Idea Generation）和实验执行（Experiment Execution）两个阶段
支持多轮迭代发现（fresh / incremental 模式）以及断点续跑（resume）
"""
import os  # 操作系统接口：文件路径、目录操作等
import os.path as osp  # 路径操作别名，方便后续使用 osp.join / osp.exists 等
import sys  # 系统相关：退出进程、命令行参数等
import json  # JSON 文件的读写解析
import argparse  # 命令行参数解析框架
import asyncio  # 异步 IO 框架，用于运行异步的 idea 生成等协程
import logging  # 日志记录模块
import glob  # 文件通配符匹配（查找符合模式的文件路径）
import shutil  # 高级文件操作：复制目录树、删除目录等
import torch  # PyTorch 深度学习框架（用于 GPU 检测等）
import yaml  # YAML 配置文件解析
from datetime import datetime  # 日期时间处理，用于生成时间戳格式的目录名
from dotenv import load_dotenv  # 从 .env 文件加载环境变量

# 导入 MAS（多智能体系统）核心组件
from internagent.stage import IdeaGenerator, ExperimentRunner
from typing import List, Dict, Any, Optional  # 类型注解工具

# 导入流水线人工介入钩子
from paper_to_task.interaction import pipeline_hooks

# 尝试导入长期记忆模块（可选依赖，不存在时优雅降级）
try:
    from internagent.mas.memory.long_memory import MemoryModule, ExperienceGenerator
    LONG_MEMORY_AVAILABLE = True  # 标记长期记忆模块可用
except ImportError:
    LONG_MEMORY_AVAILABLE = False  # 标记不可用
    MemoryModule = None  # 置空以避免后续引用报错
    ExperienceGenerator = None

load_dotenv()  # 加载 .env 文件中的环境变量（如 API Key 等）

# ============================================================================
# 任务类型检测与标准化
# ============================================================================
def detect_task_type(task_dir: str) -> str:
    """
    检测任务目录的类型：
    - 'sci' 类型：包含 task_info.json（用于论文复现类任务）
    - 'auto' 类型：其他情况（包含 prompt.json 的自动化任务）

    参数:
        task_dir: 任务目录路径

    返回:
        'sci' 或 'auto'
    """
    # 如果目录中存在 task_info.json，则判定为科学复现类任务
    if osp.exists(osp.join(task_dir, "task_info.json")):
        return "sci"
    # 否则默认为自动化任务
    return "auto"

def normalize_sci_task(task_dir: str, output_path: str) -> dict:
    """
    读取 sci_task 目录中的 task_info.json 和 checklist.json，
    将其合成为 InternAgent MAS 流水线兼容的 prompt.json。

    参数:
        task_dir: sci_task 目录路径（例如 tasks/sci_tasks/Chemistry_000）
        output_path: 合成后的 prompt.json 输出路径

    返回:
        合成后的 prompt 字典（同时写入 output_path）
    """
    # 读取 task_info.json（包含任务描述、数据列表等）
    task_info_path = osp.join(task_dir, "task_info.json")
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)

    # 读取 checklist.json（评估指标清单，位于 target_study 子目录中）
    checklist_path = osp.join(task_dir, "target_study", "checklist.json")
    checklist = []  # 初始化为空列表
    if osp.exists(checklist_path):
        with open(checklist_path, 'r') as f:
            checklist = json.load(f)

    # 从目录名提取领域标签（例如 "Chemistry_000" → "Chemistry"）
    dir_name = osp.basename(task_dir.rstrip('/\\'))  # 去除尾部斜杠后取最后一级目录名
    domain = dir_name.split('_')[0] if '_' in dir_name else dir_name  # 取下划线前的部分作为领域

    # 构建数据清单字符串（列出所有可用数据文件及其描述）
    data_items = task_info.get('data', [])  # 获取数据文件列表
    if data_items:
        # 为每个数据项生成 "- name: description" 格式的行
        data_lines = [f"- {d['name']}: {d.get('description', '')}" for d in data_items]
        data_manifest = "\n".join(data_lines)  # 用换行符连接
    else:
        data_manifest = "No data files specified."  # 无数据文件时的占位文本

    # 构建 checklist 摘要（用于约束条件）
    constraints = []  # 初始化约束列表
    for i, item in enumerate(checklist):  # 遍历每个 checklist 项
        w = item.get('weight', 0)  # 获取该项权重
        t = item.get('type', 'text')  # 获取该项类型
        preview = item.get('content', '')[:200]  # 截取前 200 字符作为预览
        constraints.append(f"Item {i} (type={t}, weight={w:.2f}): {preview}")  # 拼接格式化字符串

    # 构建复合任务描述（包含研究领域、可用数据、评估标准、工作区布局等）
    task_description = (
        f"Reproduce the findings from a scientific paper in the {domain} domain.\n\n"
        f"## Research Task\n{task_info.get('task', '')}\n\n"  # 原始任务描述
        f"## Available Data\n{data_manifest}\n\n"  # 可用数据清单
        f"## Evaluation Criteria ({len(checklist)} checklist items)\n"  # 评估标准标题
        + "\n".join(constraints) +  # 拼接所有约束条件
        "\n\n## Workspace Layout\n"
        "- Write analysis code in `code/`\n"  # 代码目录
        "- Save intermediate outputs in `outputs/`\n"  # 中间输出目录
        "- Write final report as `report/report.md`\n"  # 最终报告路径
        "- Save generated figures in `report/images/`\n"  # 图片目录
        "- Reference papers are in `related_work/`, raw data in `data/`\n"  # 参考资料
    )

    # 组装最终的 prompt 数据字典
    prompt_data = {
        "system": f"You are a scientific researcher reproducing findings from a {domain} paper.",  # 系统提示词
        "task_description": task_description,  # 完整任务描述
        "domain": domain,  # 研究领域标签
        "background": f"Data files available:\n{data_manifest}",  # 背景信息（数据清单）
        "constraints": constraints,  # 约束条件列表
        "task_type": "sci",  # 任务类型标记
    }

    # 将合成后的 prompt 写入 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)

    return prompt_data  # 返回 prompt 字典供后续使用

# ============================================================================
# 辅助函数
# ============================================================================
def _find_best_experiment_result(results: List[Dict[str, Any]], logger) -> Optional[Dict[str, Any]]:
    """
    从实验结果列表中找到整体提升率最高的成功实验。

    参数:
        results: 实验结果字典列表
        logger: 日志记录器实例

    返回:
        最佳实验结果字典，若无成功实验则返回 None
    """
    # 过滤出所有成功的实验结果（success 字段为 True）
    successful_results = [r for r in results if r.get('success', False)]

    if not successful_results:  # 如果没有成功的实验
        return None

    # 遍历所有成功结果，寻找 overall_improvement_rate 最高的那个
    best_result = None  # 当前最佳结果
    best_performance = float('-inf')  # 当前最佳性能（初始化为负无穷）

    for result in successful_results:
        perf_data = result.get('performance', {})  # 获取性能数据
        improvement_rate = perf_data.get('overall_improvement_rate', 0)  # 获取整体提升率

        if improvement_rate > best_performance:  # 如果当前结果更好
            best_performance = improvement_rate  # 更新最佳性能值
            best_result = result  # 更新最佳结果

    return best_result  # 返回最佳结果

def _update_baseline_for_incremental(best_code_path: str, logger, task_type: str = 'auto') -> bool:
    """
    在增量模式下，用最佳运行的结果更新基线代码和指标。

    对于 'auto' 任务会更新：
      1. code/ - 主代码目录
      2. run_0/code/ - 基线代码备份
      3. run_0/final_info.json - 基线指标

    对于 'sci' 任务额外更新：
      4. outputs/ - 中间输出
      5. report/ - 报告和图表

    参数:
        best_code_path: 最佳实验结果所在的根目录
        logger: 日志记录器实例
        task_type: 任务类型，'auto' 或 'sci'

    返回:
        更新成功返回 True，否则返回 False
    """
    # 查找所有 run_[1-9] 开头的运行目录（排除 run_0 基线目录）
    run_dirs = sorted(glob.glob(osp.join(best_code_path, "run_[1-9]*")))

    if not run_dirs:  # 没有找到运行目录
        logger.warning(f"No run directories found in {best_code_path}")
        return False

    # 从所有 run 目录中找到最后一个有效的（包含 final_info.json 的）
    best_run_dir = None  # 最佳运行目录路径
    best_final_info = None  # 最佳运行的最终信息

    for run_dir in run_dirs:  # 遍历所有运行目录
        final_info_path = osp.join(run_dir, "final_info.json")  # 拼接 final_info.json 路径
        if osp.exists(final_info_path):  # 如果该文件存在
            try:
                with open(final_info_path, 'r') as f:
                    best_final_info = json.load(f)  # 读取最终信息
                best_run_dir = run_dir  # 记录当前有效目录（最后一个有效目录即为最佳）
            except Exception as e:
                logger.warning(f"Failed to load {final_info_path}: {e}")

    # 如果没有找到任何有效的 final_info.json，则无法更新
    if not best_run_dir or not best_final_info:
        logger.warning(f"No valid final_info.json found in run directories")
        return False

    # 更新 run_0/final_info.json（基线指标文件）
    run0_dir = osp.join(best_code_path, "run_0")  # run_0 目录路径
    os.makedirs(run0_dir, exist_ok=True)  # 确保 run_0 目录存在
    run0_final_info = osp.join(run0_dir, "final_info.json")  # 基线指标文件路径

    try:
        with open(run0_final_info, 'w') as f:
            json.dump(best_final_info, f, indent=2)  # 写入最佳运行指标
        logger.info(f"Updated baseline metrics: {osp.join(best_run_dir, 'final_info.json')} -> {run0_final_info}")
    except Exception as e:
        logger.error(f"Failed to update baseline metrics: {e}")
        return False

    # 用最佳运行的代码更新主代码目录 code/
    best_run_code_dir = osp.join(best_run_dir, "code")  # 最佳运行的代码目录
    main_code_dir = osp.join(best_code_path, "code")  # 主代码目录

    if osp.exists(best_run_code_dir) and osp.isdir(best_run_code_dir):  # 确保源代码目录存在
        try:
            if osp.exists(main_code_dir):  # 如果主代码目录已存在，先删除
                shutil.rmtree(main_code_dir)
            shutil.copytree(best_run_code_dir, main_code_dir)  # 复制最佳代码到主目录
            logger.info(f"Updated main code: {best_run_code_dir} -> {main_code_dir}")
        except Exception as e:
            logger.error(f"Failed to update main code: {e}")
            return False

        # 同步更新 run_0/code/（基线代码备份），保持与主代码一致
        run0_code_dir = osp.join(run0_dir, "code")  # 基线备份代码目录
        try:
            if osp.exists(run0_code_dir):  # 如果备份目录已存在，先删除
                shutil.rmtree(run0_code_dir)
            shutil.copytree(best_run_code_dir, run0_code_dir)  # 复制最佳代码到备份目录
            logger.info(f"Updated baseline code backup: {best_run_code_dir} -> {run0_code_dir}")
        except Exception as e:
            logger.warning(f"Failed to update baseline code backup: {e}")
            # 非致命错误，继续执行

    # 对于 sci 类型任务：额外传播 outputs/ 和 report/ 目录
    if task_type == 'sci':
        for dir_name in ['outputs', 'report']:  # 遍历需要传播的目录名
            best_run_dir_src = osp.join(best_run_dir, dir_name)  # 最佳运行中的源目录
            main_dir_dst = osp.join(best_code_path, dir_name)  # 主目录中的目标路径
            if osp.exists(best_run_dir_src) and osp.isdir(best_run_dir_src):  # 确保源目录存在
                try:
                    if osp.exists(main_dir_dst):  # 如果目标目录已存在，先删除
                        shutil.rmtree(main_dir_dst)
                    shutil.copytree(best_run_dir_src, main_dir_dst)  # 复制目录
                    logger.info(f"Updated {dir_name}/: {best_run_dir_src} -> {main_dir_dst}")
                except Exception as e:
                    logger.warning(f"Failed to update {dir_name}/: {e}")

    return True  # 所有更新成功完成

def _generate_experiences_for_round(args, memory, session_id, logger) -> bool:
    """
    从单轮实验中生成经验（experiences），供后续轮次的 prompt 演化使用。

    参数:
        args: 命令行参数对象
        memory: MemoryModule 实例（长期记忆模块）
        session_id: 当前会话 ID
        logger: 日志记录器实例

    返回:
        经验生成成功返回 True，否则返回 False
    """
    if memory is None:  # 如果记忆模块未初始化，直接返回
        return False

    try:
        from internagent.mas.memory.long_memory import ExperienceGenerator  # 尝试导入经验生成器
    except ImportError:
        logger.warning("Long memory not available, skipping experience generation")
        return False

    # 从 prompt.json 加载研究领域信息
    prompt_path = getattr(args, 'prompt_path', None) or osp.join(args.task_dir, "prompt.json")
    domain = "machine learning"  # 默认领域
    if osp.exists(prompt_path):  # 如果 prompt.json 存在
        try:
            with open(prompt_path, 'r') as f:
                prompt_data = json.load(f)
                domain = prompt_data.get("domain", domain)  # 从中读取领域标签
        except Exception as e:
            logger.warning(f"Failed to load domain from prompt.json: {e}, using default")

    # 从当前会话目录加载想法和实验笔记
    session_dir = osp.join(args.output_dir, session_id)  # 当前会话目录路径
    if osp.exists(session_dir):  # 如果会话目录存在
        # 加载该会话中生成的想法
        ideas_path = osp.join(session_dir, "ideas.json")  # 想法文件路径
        if osp.exists(ideas_path):
            memory.load_idea_generation_output(ideas_path)  # 将想法加载到记忆模块

        # 加载该会话中的实验笔记
        memory.load_all_notes_from_directory(session_dir, args.task_name)

    # 获取记忆模块摘要信息
    summary = memory.get_memory_summary()
    logger.info(f"Loaded {summary['total_ideas']} ideas and {summary['total_experiments']} experiments")

    # 如果有实验数据，则生成经验
    if summary['total_experiments'] > 0:
        experience_generator = ExperienceGenerator(logger=logger, config_path=args.config)  # 创建经验生成器

        # 异步运行经验生成流程
        result = asyncio.run(
            experience_generator.generate_experiences_from_memory(
                memory=memory,  # 传入记忆模块
                task_domain=domain,  # 传入任务领域
                output_dir=args.base_output_dir  # 传入输出目录
            )
        )

        new_experiences = result.get("new_experiences", [])  # 新生成的经验列表
        updated_library = result.get("updated_library", [])  # 更新后的经验库

        logger.info(f"Generated {len(new_experiences)} new experiences")  # 记录新生成数量
        logger.info(f"Experience library now has {len(updated_library)} total experiences")  # 记录总量
        return True
    else:
        logger.warning("No experiments found in this round, skipping experience generation")
        return False

# ============================================================================
# 日志配置
# ============================================================================
def setup_logging():
    """配置日志系统：同时输出到控制台和日志文件"""
    log_dir = 'logs'  # 日志文件存放目录
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    # 日志文件名包含时间戳，例如 20260112_101127_internagent.log
    log_file = osp.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_internagent.log')

    logging.basicConfig(
        level=logging.INFO,  # 日志级别：INFO 及以上
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',  # 日志格式
        handlers=[
            logging.StreamHandler(),  # 控制台输出处理器
            logging.FileHandler(log_file)  # 文件输出处理器
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 降低 httpx 库的日志级别，避免刷屏
    return logging.getLogger("InternAgent")  # 返回专属 logger

# ============================================================================
# 命令行参数解析
# ============================================================================
def parse_arguments():
    """定义并解析所有命令行参数"""
    parser = argparse.ArgumentParser(
        description="Integrated InternAgent Pipeline: Idea Generation + Experiment Execution"
    )

    # ========================================
    # 任务配置参数组
    # ========================================
    task_group = parser.add_argument_group('Task Configuration')
    task_group.add_argument(
        "--task",  # 任务名称或路径
        type=str,
        default="AutoSeg",  # 默认任务名
        help="Task name or path to task directory. If it's a name, will use tasks/{task}; if it's a path, will use it directly"
    )
    task_group.add_argument(
        "--output_dir",  # 结果输出目录
        type=str,
        default=None,  # 默认为 results/{task_name}
        help="Results output directory (defaults to results/{task_name})"
    )
    task_group.add_argument(
        "--config",  # 配置文件路径
        type=str,
        default='config/default_config.yaml',  # 默认配置文件
        help="Path to configuration file"
    )

    # ========================================
    # 想法生成阶段参数组
    # ========================================
    idea_group = parser.add_argument_group('Idea Generation Phase')
    idea_group.add_argument(
        "--skip_idea_generation",  # 跳过想法生成阶段
        action="store_true",  # 布尔标志，存在即为 True
        help="Skip idea generation and use existing ideas from idea_path"
    )
    idea_group.add_argument(
        "--idea_path",  # 已有想法文件路径
        type=str,
        default=None,
        help="Path to existing ideas JSON (used when skip_idea_generation=True)"
    )
    idea_group.add_argument(
        "--ref_code_path",  # 基线参考代码路径
        type=str,
        default=None,  # 默认为 {task_dir}/code
        help="Baseline reference code path (defaults to {task_dir}/experiment.py)"
    )
    idea_group.add_argument(
        "--offline_feedback",  # 离线反馈文件路径
        type=str,
        default='config/feedback_global.json',  # 默认全局反馈文件
        help="Offline feedback file for idea generation"
    )

    # ========================================
    # 实验执行阶段参数组
    # ========================================
    exp_group = parser.add_argument_group('Experiment Execution Phase')
    exp_group.add_argument(
        "--mode",  # 执行模式
        type=str,
        default="experiment",  # 默认为实验模式
        choices=["experiment", "report"],  # 可选值：实验模式或仅报告模式
        help="Execution mode: 'experiment' for running experiments, 'report' for generating reports only"
    )
    exp_group.add_argument(
        "--exp_backend",  # 实验后端
        type=str,
        required=True,  # 必填参数
        default="claudecode",  # 默认使用 Claude Code 后端
        choices=["openhands", "claudecode", "iflow"],  # 支持的后端列表
        help="Experiment backend to use (required for experiment mode)"
    )
    # 注意：模型配置通过配置文件中的 experiment.model 字段指定
    # 注意：GPU 配置通过 CUDA_VISIBLE_DEVICES 环境变量或自动检测
    # 注意：并行执行通过配置文件中的 experiment.max_parallel_experiments 和 experiment.gpu_per_experiment 指定
    # 注意：OpenHands 相关配置（mount_paths, uri_prefix）通过配置文件指定

    # ========================================
    # 断点续跑配置参数组
    # ========================================
    resume_group = parser.add_argument_group('Resume Configuration')
    resume_group.add_argument(
        "--resume",  # 恢复路径
        type=str,
        default=None,  # 默认不启用恢复模式
        help="Path to existing launch folder (e.g., results/TaskName/20260112_101127_launch) to resume from last completed loop"
    )

    return parser.parse_args()  # 解析并返回参数对象

# ============================================================================
# 断点续跑状态检测
# ============================================================================
def load_resume_state(resume_path: str, logger) -> dict:
    """
    从已有的启动文件夹中加载恢复状态。

    参数:
        resume_path: 启动文件夹路径（例如 results/TaskName/20260112_101127_launch）
        logger: 日志记录器实例

    返回:
        包含以下信息的字典：
        - completed_rounds: 已完成的轮次数
        - all_round_results: 已完成轮次的结果列表
        - all_session_ids: 已完成轮次的会话 ID 列表
        - best_code_path: 增量模式下的最佳代码路径（如有）
        - best_overall_performance: 增量模式下的最佳性能（如有）
        - launch_id: 启动 ID（从文件夹名提取）
        - config_overrides: 需要保留的配置覆盖值
    """
    # 初始化恢复状态字典（所有字段都设为默认值）
    resume_state = {
        'completed_rounds': 0,  # 已完成轮次
        'all_round_results': [],  # 所有轮次结果
        'all_session_ids': [],  # 所有会话 ID
        'best_code_path': None,  # 最佳代码路径
        'best_overall_performance': None,  # 最佳整体性能
        'launch_id': None,  # 启动 ID
        'loop_mode': 'fresh',  # 循环模式（默认为全新开始）
        'loop_rounds': 1,  # 总轮次数
        'original_task_dir': None,  # 原始任务目录
        'base_output_dir': None,  # 基础输出目录
        'prompt_path': None  # prompt.json 路径
    }

    if not osp.exists(resume_path):  # 如果恢复路径不存在
        logger.error(f"Resume path does not exist: {resume_path}")
        return resume_state

    # 从文件夹名提取 launch_id（例如 "20260112_101127_launch"）
    resume_state['launch_id'] = osp.basename(resume_path)
    # 基础输出目录为恢复路径的父目录
    resume_state['base_output_dir'] = osp.dirname(resume_path)

    # 尝试加载 discovery_summary.json（包含完整的运行摘要）
    summary_path = osp.join(resume_path, "discovery_summary.json")
    if osp.exists(summary_path):  # 如果摘要文件存在
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)  # 读取摘要数据

            # 从摘要中恢复各项状态
            resume_state['completed_rounds'] = summary.get('total_rounds', 0)  # 已完成轮次
            resume_state['all_round_results'] = summary.get('rounds', [])  # 轮次结果
            resume_state['all_session_ids'] = summary.get('sessions', [])  # 会话 ID
            resume_state['loop_mode'] = summary.get('loop_mode', 'fresh')  # 循环模式
            resume_state['loop_rounds'] = summary.get('loop_rounds', 1)  # 总轮次
            resume_state['original_task_dir'] = summary.get('original_task_dir', summary.get('task_dir'))  # 原始任务目录

            # 如果存在增量模式状态，恢复相关信息
            if 'incremental_mode' in summary:
                inc_state = summary['incremental_mode']
                resume_state['best_code_path'] = inc_state.get('final_best_code_path')  # 最佳代码路径
                resume_state['best_overall_performance'] = inc_state.get('final_best_performance')  # 最佳性能

            logger.info(f"Loaded resume state from discovery_summary.json")
            logger.info(f"  Completed rounds: {resume_state['completed_rounds']}/{resume_state['loop_rounds']}")
            logger.info(f"  Sessions: {resume_state['all_session_ids']}")

        except Exception as e:
            logger.warning(f"Failed to load discovery_summary.json: {e}")
            # 加载失败，回退到扫描目录的方式
            resume_state = _scan_completed_rounds(resume_path, resume_state, logger)
    else:
        # 没有摘要文件，通过扫描目录来检测已完成的轮次
        logger.info("No discovery_summary.json found, scanning directories...")
        resume_state = _scan_completed_rounds(resume_path, resume_state, logger)

    # 检查启动文件夹中是否存在 prompt.json（可能经过演化的版本）
    prompt_path = osp.join(resume_path, "prompt.json")
    if osp.exists(prompt_path):
        resume_state['prompt_path'] = prompt_path  # 记录演化后的 prompt 路径
        logger.info(f"Found evolved prompt at: {prompt_path}")

    return resume_state  # 返回恢复状态字典

def _scan_completed_rounds(resume_path: str, resume_state: dict, logger) -> dict:
    """
    当 discovery_summary.json 不可用时，通过扫描目录结构来检测已完成的轮次。

    参数:
        resume_path: 启动文件夹路径
        resume_state: 当前恢复状态字典
        logger: 日志记录器实例

    返回:
        更新后的恢复状态字典
    """
    # 查找所有 session_* 开头的会话目录
    session_dirs = glob.glob(osp.join(resume_path, "session_*"))
    session_dirs.sort()  # 按名称排序（名称中包含时间戳，排序即按时间顺序）

    completed_rounds = 0  # 已完成轮次计数器
    for session_dir in session_dirs:  # 遍历每个会话目录
        session_id = osp.basename(session_dir)  # 获取会话 ID（目录名）

        # 列出该会话目录下所有非 session_ 开头的子目录（即实验文件夹）
        experiment_folders = [d for d in os.listdir(session_dir)
                            if osp.isdir(osp.join(session_dir, d)) and not d.startswith('session_')]

        has_completed_experiments = False  # 标记是否有已完成的实验
        for exp_folder in experiment_folders:  # 遍历每个实验文件夹
            # 检查是否存在包含 final_info.json 的 run_* 子目录
            run_folders = glob.glob(osp.join(session_dir, exp_folder, "run_*", "final_info.json"))
            if run_folders:  # 如果找到了，说明该实验已完成
                has_completed_experiments = True
                break  # 找到一个就够了，跳出内层循环

        if has_completed_experiments:  # 如果该会话有已完成的实验
            completed_rounds += 1  # 增加已完成轮次计数
            resume_state['all_session_ids'].append(session_id)  # 记录会话 ID
            logger.info(f"  Found completed session: {session_id}")

    resume_state['completed_rounds'] = completed_rounds  # 更新已完成轮次
    logger.info(f"Detected {completed_rounds} completed rounds from directory scan")

    return resume_state  # 返回更新后的状态

# ============================================================================
# 主流水线
# ============================================================================
def main():
    """InternAgent 主流水线入口：配置初始化 → 多轮想法生成+实验执行 → 输出摘要"""
    logger = setup_logging()  # 初始化日志系统
    args = parse_arguments()  # 解析命令行参数

    # ========================================
    # 断点续跑模式处理
    # ========================================
    resume_state = None  # 恢复状态字典
    start_round = 1  # 默认从第 1 轮开始

    if args.resume:  # 如果启用了恢复模式
        logger.info("=" * 80)
        logger.info("RESUME MODE ENABLED")  # 记录恢复模式已启用
        logger.info(f"Resuming from: {args.resume}")  # 记录恢复路径
        logger.info("=" * 80)

        resume_state = load_resume_state(args.resume, logger)  # 加载恢复状态

        if resume_state['completed_rounds'] == 0:  # 如果没有已完成的轮次
            logger.warning("No completed rounds found, starting fresh from round 1")
        else:
            start_round = resume_state['completed_rounds'] + 1  # 从下一轮开始
            logger.info(f"Will resume from round {start_round}")

    # ========================================
    # 设置任务目录
    # ========================================
    # 如果 task 参数包含路径分隔符或本身就是一个目录，则直接使用；否则拼接 tasks/ 前缀
    if '/' in args.task or '\\' in args.task or osp.isdir(args.task):
        args.task_dir = args.task  # 直接使用给定路径
        args.task_name = osp.basename(args.task.rstrip('/\\'))  # 从路径提取任务名
    else:
        args.task_dir = osp.join("tasks", args.task)  # 拼接为 tasks/{task_name}
        args.task_name = args.task  # 直接使用任务名

    if not osp.exists(args.task_dir):  # 验证任务目录是否存在
        raise FileNotFoundError(f"Task directory not found: {args.task_dir}")

    # 检测任务类型：'sci'（含 task_info.json）或 'auto'（含 prompt.json）
    args.task_type = detect_task_type(args.task_dir)

    # 设置参考代码路径
    if args.ref_code_path is None:  # 如果用户未指定
        if args.task_type == 'sci':
            args.ref_code_path = None  # sci 任务不需要参考代码
        else:
            args.ref_code_path = osp.join(args.task_dir, "code")  # auto 任务使用 {task_dir}/code

    # ========================================
    # 设置输出目录
    # ========================================
    if args.resume and resume_state and resume_state['launch_id']:
        # 恢复模式：使用已有的启动文件夹
        launch_id = resume_state['launch_id']  # 复用原有的启动 ID
        base_output_dir = resume_state['base_output_dir']  # 复用基础输出目录
        args.output_dir = args.resume  # 输出目录指向恢复路径
        args.base_output_dir = base_output_dir  # 记录基础输出目录

        # 如果恢复状态中有 prompt.json，直接使用
        if resume_state['prompt_path'] and osp.exists(resume_state['prompt_path']):
            args.prompt_path = resume_state['prompt_path']
        else:
            # 回退方案：sci 任务重新生成 prompt.json，auto 任务使用原始 prompt.json
            if args.task_type == 'sci':
                fallback_prompt_path = osp.join(args.output_dir, "prompt.json")
                normalize_sci_task(args.task_dir, fallback_prompt_path)  # 重新合成 prompt.json
                args.prompt_path = fallback_prompt_path
                logger.info(f"Regenerated sci_task prompt.json for resume: {fallback_prompt_path}")
            else:
                args.prompt_path = osp.join(args.task_dir, "prompt.json")  # 使用原始 prompt

        logger.info(f"Resuming with existing launch folder: {launch_id}")
    else:
        # 全新启动：创建新的启动文件夹
        # 目录结构: results/{task_name}/{launch_id}/session_xxx/...
        launch_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳
        launch_id = f"{launch_time}_launch"  # 拼接启动 ID

        if args.output_dir is None:
            base_output_dir = osp.join("results", args.task_name)  # 默认输出到 results/{task_name}
        else:
            base_output_dir = osp.join("results", args.output_dir)  # 用户指定的输出目录

        os.makedirs(base_output_dir, exist_ok=True)  # 创建基础输出目录

        # 在基础输出目录下创建启动文件夹
        args.output_dir = osp.join(base_output_dir, launch_id)
        os.makedirs(args.output_dir, exist_ok=True)

        # 记录基础输出目录（用于共享资源，如经验库）
        args.base_output_dir = base_output_dir

        # 将 prompt.json 复制或生成到启动目录中
        launch_prompt_path = osp.join(args.output_dir, "prompt.json")
        if args.task_type == 'sci':
            # sci 任务：从 task_info.json 合成 prompt.json
            normalize_sci_task(args.task_dir, launch_prompt_path)
            args.prompt_path = launch_prompt_path
            logger.info(f"Generated synthetic prompt.json for sci_task: {launch_prompt_path}")
        else:
            # auto 任务：从任务目录复制 prompt.json
            original_prompt_path = osp.join(args.task_dir, "prompt.json")
            if osp.exists(original_prompt_path):
                shutil.copy2(original_prompt_path, launch_prompt_path)  # 复制文件（保留元数据）
                args.prompt_path = launch_prompt_path
            else:
                raise FileNotFoundError(f"prompt.json not found in task directory: {original_prompt_path}")

    # ========================================
    # 加载配置文件
    # ========================================
    config = {}  # 初始化配置字典
    if args.config and osp.exists(args.config):  # 如果配置文件存在
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith(('.yaml', '.yml')):  # YAML 格式
                    config = yaml.safe_load(f)
                else:  # JSON 格式
                    config = json.load(f)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # 从配置文件获取循环轮次和循环模式
    # 注意：始终以配置文件为准，恢复状态仅提供已完成轮次信息
    # 这允许用户在恢复时更改 loop_rounds（例如从 5 轮扩展到 10 轮）
    loop_rounds = config.get('workflow', {}).get('loop_rounds', 1)  # 总循环轮次
    loop_mode = config.get('workflow', {}).get('loop_mode', 'fresh')  # 循环模式

    # 如果恢复状态有有效的循环设置，可作参考，但配置文件可以覆盖
    if args.resume and resume_state:
        resume_loop_rounds = resume_state.get('loop_rounds', 0)  # 恢复状态中的轮次
        resume_loop_mode = resume_state.get('loop_mode')  # 恢复状态中的模式

        # 如果配置文件使用默认值（1 轮），则使用恢复状态的轮次
        if resume_loop_rounds > 0 and loop_rounds == 1:
            loop_rounds = resume_loop_rounds
            logger.info(f"Using loop_rounds={loop_rounds} from resume state")

        # 如果恢复状态的模式不是默认的 'fresh'，则使用恢复状态的模式
        if resume_loop_mode and resume_loop_mode != 'fresh':
            loop_mode = resume_loop_mode
            logger.info(f"Using loop_mode={loop_mode} from resume state")

    # 如果跳过想法生成，则只运行一轮（无需迭代发现）
    if args.skip_idea_generation:
        loop_rounds = 1
        logger.info("Skip idea generation is enabled, running single round only")

    # 验证循环模式的合法性
    if loop_mode not in ['fresh', 'incremental']:
        logger.warning(f"Invalid loop_mode '{loop_mode}', defaulting to 'fresh'")
        loop_mode = 'fresh'  # 无效值回退到默认

    # 检查是否所有轮次已经完成
    if start_round > loop_rounds:
        logger.info("=" * 80)
        logger.info("All rounds already completed! Nothing to resume.")  # 所有轮次已完成
        logger.info(f"Completed: {start_round - 1}/{loop_rounds} rounds")
        logger.info("=" * 80)
        return  # 直接退出

    # 打印流水线启动信息
    logger.info("=" * 80)
    logger.info("InternAgent Pipeline Started" + (" (RESUMED)" if args.resume else ""))
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Task Type: {args.task_type.upper()}")
    logger.info(f"Task Directory: {args.task_dir}")
    logger.info(f"Experiment Backend: {args.exp_backend}")
    logger.info(f"Launch ID: {launch_id}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Prompt Path: {args.prompt_path}")
    logger.info(f"Shared Resources: {args.base_output_dir}")
    logger.info(f"Loop Rounds: {loop_rounds}")
    logger.info(f"Loop Mode: {loop_mode.upper()}")
    if args.resume:
        logger.info(f"Resume: Starting from round {start_round}/{loop_rounds}")
    if loop_mode == 'incremental':
        logger.info("  → Each round starts from the best result of previous rounds")  # 增量模式说明
    else:
        logger.info("  → Each round starts fresh from the original baseline")  # 全新模式说明
    logger.info("=" * 80)

    # 初始化长期记忆模块（用于 IdeaGraph 和经验追踪）
    memory = None  # 记忆模块实例
    if LONG_MEMORY_AVAILABLE:  # 如果长期记忆模块可用
        try:
            long_memory_config = config.get("memory", {}).get("long_memory", {})  # 读取记忆配置
            if long_memory_config.get("enabled", True):  # 检查是否启用
                memory = MemoryModule(logger=logger)  # 创建记忆模块实例
                logger.info("Long memory module initialized")

                # 加载历史数据（实现跨启动的经验连续性）
                logger.info("Loading historical ideas and experiment results...")

                # 从基础输出目录中加载所有历史会话的 ideas.json
                ideas_files = glob.glob(osp.join(base_output_dir, "*_launch", "session_*", "ideas.json"))
                ideas_files.extend(glob.glob(osp.join(base_output_dir, "session_*", "ideas.json")))  # 兼容旧格式
                for ideas_file in ideas_files:
                    memory.load_idea_generation_output(ideas_file)  # 逐个加载

                # 从基础输出目录加载所有历史实验笔记
                memory.load_all_notes_from_directory(base_output_dir, args.task_name)

                # 打印加载摘要
                summary = memory.get_memory_summary()
                logger.info(f"Historical data loaded: {summary['total_ideas']} ideas, {summary['total_experiments']} experiments")
        except Exception as e:
            logger.warning(f"Failed to initialize long memory module: {e}")

    # 初始化轮次结果存储（如果恢复模式，则从恢复状态中加载）
    if args.resume and resume_state:
        all_round_results = resume_state.get('all_round_results', [])  # 恢复已有结果
        all_session_ids = resume_state.get('all_session_ids', [])  # 恢复已有会话 ID
        logger.info(f"Restored {len(all_round_results)} completed rounds from resume state")
    else:
        all_round_results = []  # 全新启动，结果列表为空
        all_session_ids = []  # 全新启动，会话 ID 列表为空

    # 追踪增量模式下的最佳代码路径
    original_task_dir = args.task_dir  # 保存原始任务目录
    if args.resume and resume_state and resume_state.get('best_code_path'):
        best_code_path = resume_state['best_code_path']  # 从恢复状态加载最佳代码路径
        best_overall_performance = resume_state.get('best_overall_performance')  # 加载最佳性能
        logger.info(f"Restored best code path from resume: {best_code_path}")
    else:
        best_code_path = original_task_dir  # 初始时最佳代码路径就是原始任务目录
        best_overall_performance = None  # 初始时没有最佳性能记录

    # 多轮发现循环（恢复模式从 start_round 开始）
    base_code_dir = None  # 追踪增量模式的代码目录
    for round_num in range(start_round, loop_rounds + 1):  # 从 start_round 循环到 loop_rounds
        # 增量模式：从第 2 轮开始使用最佳结果的代码作为基线
        if loop_mode == 'incremental' and round_num > 1 and best_code_path != original_task_dir:
            logger.info(f"Incremental Mode: Using best result from previous rounds as baseline")
            logger.info(f"  Previous best code: {best_code_path}")
            base_code_dir = best_code_path  # 使用最佳代码路径
        else:
            base_code_dir = args.task_dir  # 全新模式使用原始任务目录

        # 调试检查：确保 base_code_dir 有效
        if not base_code_dir:
            logger.error(f"ERROR: base_code_dir is empty! args.task_dir={args.task_dir}, best_code_path={best_code_path}")
            raise ValueError("base_code_dir cannot be empty")

        logger.info(f"Base code directory: {base_code_dir}")
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STARTING DISCOVERY ROUND {round_num}/{loop_rounds}")  # 开始新一轮
        logger.info("=" * 80)

        # 步骤 1：想法生成
        session_id = None  # 初始化会话 ID
        if args.skip_idea_generation and round_num == 1:  # 如果跳过想法生成且是第一轮
            logger.info("Skipping idea generation, loading existing ideas...")
            if not args.idea_path or not osp.exists(args.idea_path):  # 验证想法文件路径
                raise FileNotFoundError(f"Idea path not found: {args.idea_path}")

            with open(args.idea_path, 'r') as f:
                ideas_data = json.load(f)  # 读取已有的想法数据

            # 提取排名靠前的假设
            if 'hypotheses' in ideas_data and 'top_hypotheses' in ideas_data:
                # 从完整假设列表中筛选出 top_hypotheses 中指定的那些
                top_ideas = [
                    item for item in ideas_data['hypotheses']
                    if item['id'] in ideas_data['top_hypotheses']
                ]
            else:
                # 如果没有 top_hypotheses 字段，则假设整个文件就是想法列表
                top_ideas = ideas_data

            logger.info(f"Loaded {len(top_ideas)} ideas from {args.idea_path}")
            session_json = args.idea_path  # 记录想法文件路径

            # 尝试从文件路径中提取 session_id（如果路径中包含 session_ 前缀）
            import re
            match = re.search(r'session_(\d+)', args.idea_path)
            if match:
                session_id = match.group(1)  # 提取会话 ID 数字部分

        else:
            # 正常进行想法生成
            logger.info(f"Starting idea generation with MAS (Round {round_num})...")
            # 创建想法生成器实例
            idea_generator = IdeaGenerator(args, logger, round_num=round_num, config=config)

            try:
                # 异步运行想法生成流程，返回 top_ideas 和 session_json 路径
                top_ideas, session_json = asyncio.run(idea_generator.generate_ideas())
            except Exception as e:
                logger.error(f"Idea generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)  # 想法生成失败，退出程序

            # 保存会话 ID（已包含 'session_' 前缀）
            session_id = idea_generator.session_id

            # 将想法保存为标准格式（存放在会话目录中）
            session_dir = osp.join(args.output_dir, session_id)  # 会话目录路径
            os.makedirs(session_dir, exist_ok=True)  # 确保会话目录存在
            ideas_output = osp.join(session_dir, "ideas.json")  # 想法输出文件路径

            # 提取每个想法的 refined_method_details 字段
            aligned_ideas = [idea['refined_method_details'] for idea in top_ideas]
            with open(ideas_output, 'w') as f:
                json.dump(aligned_ideas, f, indent=4)  # 写入对齐后的想法列表

            logger.info(f"Ideas saved to {ideas_output}")

            # 清除记忆缓存以释放 GPU 显存
            try:
                from internagent.mas.tools.memory_retrieval import clear_memory_cache
                clear_memory_cache()  # 执行缓存清除
            except Exception as e:
                logger.warning(f"Failed to clear memory cache: {e}")

            # ★ 新增：人工想法审批环节
            human_review_enabled = config.get('human_review', {}).get('idea_review', True)
            if human_review_enabled:
                logger.info("─" * 60)
                logger.info("🧪 HUMAN REVIEW: Waiting for idea approval...")
                logger.info(f"   Generated {len(top_ideas)} ideas, writing to review state")
                logger.info("─" * 60)

                # 写入待审批状态
                pipeline_hooks.write_pending_ideas(
                    args.output_dir, session_id, top_ideas
                )

                # 获取超时配置
                idea_timeout = config.get('human_review', {}).get('idea_timeout', 3600)
                logger.info(f"   Review timeout: {idea_timeout}s (config: human_review.idea_timeout)")
                logger.info(f"   Open the web UI and navigate to Pipeline → Idea Review to approve")
                logger.info(f"   State directory: {pipeline_hooks.get_review_dir(args.output_dir)}")

                # 轮询等待审批结果
                approved_state = pipeline_hooks.wait_for_state_change(
                    os.path.join(pipeline_hooks.get_review_dir(args.output_dir),
                                 pipeline_hooks.IDEAS_PENDING_FILE),
                    timeout=idea_timeout,
                )

                if approved_state and approved_state.get('status') == pipeline_hooks.STATUS_APPROVED:
                    approved_ideas = approved_state.get('ideas', [])
                    approved_ids = [idea.get('id') for idea in approved_ideas]

                    # 从 top_ideas 中筛选出被批准的
                    if approved_ids:
                        top_ideas = [
                            idea for idea in top_ideas
                            if idea.get('id') in approved_ids
                        ]
                        logger.info(f"✓ Human approved {len(top_ideas)}/{approved_state.get('total_count', len(top_ideas))} ideas")

                    # 应用用户对 idea 的修改
                    for approved in approved_ideas:
                        modified_method = approved.get('method')
                        modified_desc = approved.get('description')
                        if modified_method or modified_desc:
                            for idea in top_ideas:
                                idea_details = idea.get('refined_method_details', {})
                                if modified_method and not idea_details.get('method', '').endswith(modified_method):
                                    idea_details['method'] = modified_method
                                if modified_desc and not idea_details.get('description', '').endswith(modified_desc):
                                    idea_details['description'] = modified_desc
                elif approved_state and approved_state.get('status') == pipeline_hooks.STATUS_REJECTED:
                    logger.warning("⚠ Human rejected all ideas, using system defaults")
                    # 保留原始 top_ideas 作为后备
                else:
                    logger.warning(f"⚠ Human review timeout or skipped, using system top {len(top_ideas)} ideas")
            else:
                logger.info("Human review disabled, continuing with all ideas")

        # 步骤 2：实验执行或报告生成
        logger.info("=" * 80)

        if args.mode == "report":  # 如果是报告模式
            logger.info("Starting report generation")
            logger.info(f"Number of ideas to process: {len(top_ideas)}")
            logger.info(f"Reports will be saved to: {args.output_dir}")
            logger.info("=" * 80)

            from internagent.stage import ReportWriter  # 导入报告生成器

            report_writer = ReportWriter(args, logger, config)  # 创建报告生成器实例

            try:
                # 批量生成报告
                results = report_writer.generate_reports(
                    results_dir=args.output_dir,
                    ideas=top_ideas
                )
            except Exception as e:
                logger.error(f"Report generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)  # 报告生成失败，退出程序

        else:  # 实验模式
            if not args.exp_backend:  # 实验模式必须指定后端
                logger.error("--exp_backend is required for experiment mode")
                sys.exit(1)

            logger.info(f"Starting experiment execution with {args.exp_backend} backend")
            logger.info(f"Number of ideas to test: {len(top_ideas)}")
            logger.info("=" * 80)

            # 验证后端特定要求
            if args.exp_backend == "openhands":
                # 检查 OpenHands 后端的配置
                openhands_config = config.get("experiment", {}).get("openhands", {})
                mount_paths = openhands_config.get("mount_paths", [])  # 挂载路径列表
                uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")  # WebSocket URI 前缀

                if not mount_paths:  # 未配置挂载路径时给出警告
                    logger.warning("No mount paths specified in config for OpenHands backend")
                else:
                    logger.info(f"OpenHands mount paths: {mount_paths}")
                logger.info(f"OpenHands URI prefix: {uri_prefix}")

            # 创建实验运行器实例
            experiment_runner = ExperimentRunner(args, logger, config, session_id=session_id, base_code_dir=base_code_dir)

            # 确定实验结果存放目录
            if session_id:
                # 如果有会话 ID，结果存放在输出目录下的会话子目录中
                experiment_results_dir = osp.join(args.output_dir, session_id)
            else:
                # 否则直接存放在输出目录中
                experiment_results_dir = args.output_dir

            try:
                # 运行实验：对每个想法执行实验
                results = experiment_runner.run_experiments(
                    base_dir=base_code_dir,  # 基础代码目录
                    results_dir=experiment_results_dir,  # 结果输出目录
                    ideas=top_ideas  # 待测试的想法列表
                )
            except Exception as e:
                logger.error(f"Experiment execution failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)  # 实验执行失败，退出程序

            # ★ 新增：实验结果人工审查环节
            result_review_enabled = config.get('human_review', {}).get('result_review', True)
            if result_review_enabled and results:
                logger.info("─" * 60)
                logger.info("🔬 HUMAN REVIEW: Experiment results ready for review...")
                logger.info(f"   {len(results)} experiments completed")
                logger.info("─" * 60)

                # 为每个成功的实验结果写入审查状态
                for result in results:
                    if result.get('success'):
                        code_path = result.get('code_path', '')
                        if code_path:
                            # 尝试读取 final_info.json 中的评分
                            final_info_path = osp.join(code_path, 'final_info.json')
                            scores = {}
                            if osp.exists(final_info_path):
                                try:
                                    with open(final_info_path) as f:
                                        final_info = json.load(f)
                                    scores = final_info.get('sci_task', {}).get('means', {})
                                except Exception:
                                    pass

                            # 读取 report.md 预览
                            report_text = ''
                            report_path = osp.join(code_path, 'report', 'report.md')
                            if osp.exists(report_path):
                                try:
                                    with open(report_path, 'r', encoding='utf-8') as f:
                                        report_text = f.read()[:3000]
                                except Exception:
                                    pass

                            # 查找图片
                            images_dir = osp.join(code_path, 'report', 'images')
                            images = []
                            if osp.exists(images_dir):
                                images = [osp.join(images_dir, img)
                                         for img in sorted(os.listdir(images_dir))[:5]
                                         if osp.isfile(osp.join(images_dir, img))]

                            pipeline_hooks.write_result_for_review(
                                args.output_dir,
                                idea_name=result.get('idea_name', 'unknown'),
                                run_num=0,
                                scores=scores,
                                report_text=report_text,
                                report_images=images,
                            )

                # 等待用户审查确认
                result_timeout = config.get('human_review', {}).get('result_timeout', 1800)
                logger.info(f"   Review timeout: {result_timeout}s")
                logger.info(f"   Open the web UI → Pipeline → Result Review to examine results")

                review_dir = pipeline_hooks.get_review_dir(args.output_dir)
                result_file = osp.join(review_dir, pipeline_hooks.RESULT_FEEDBACK_FILE)

                feedback_state = pipeline_hooks.wait_for_state(
                    result_file,
                    pipeline_hooks.STATUS_APPROVED,
                    timeout=result_timeout,
                )

                if feedback_state:
                    feedback = feedback_state.get('feedback', {})
                    if feedback.get('overrides'):
                        # 用户提供了评分覆盖
                        logger.info(f"✓ Human review submitted with score overrides")
                    elif feedback.get('selected_best'):
                        # 用户手动选择了最佳结果
                        selected_best = feedback.get('selected_best')
                        logger.info(f"✓ Human selected best result: {selected_best}")
                else:
                    logger.info("⚠ Result review timeout, using system auto-selection")

        # 存储本轮结果
        round_result = {
            'round': round_num,  # 当前轮次编号
            'session_id': session_id,  # 会话 ID
            'results': results,  # 实验结果列表
            'successful': sum(1 for r in results if r['success']),  # 成功实验数量
            'failed': len(results) - sum(1 for r in results if r['success'])  # 失败实验数量
        }
        all_round_results.append(round_result)  # 添加到总结果列表
        all_session_ids.append(session_id)  # 添加到会话 ID 列表

        # 打印本轮摘要
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num}/{loop_rounds} COMPLETED")  # 本轮完成
        logger.info(f"Session: {session_id}")
        logger.info(f"Successful: {round_result['successful']}/{len(results)}")
        logger.info(f"Failed: {round_result['failed']}/{len(results)}")
        logger.info("=" * 80)

        # 每轮结束后生成经验（用于后续轮次的 prompt 演化）
        if LONG_MEMORY_AVAILABLE and memory is not None:
            logger.info(f"Generating experiences from Round {round_num}...")
            _generate_experiences_for_round(args, memory, session_id, logger)

        # 如果不是最后一轮，为下一轮做准备
        if round_num < loop_rounds:
            logger.info(f"Preparing for Round {round_num + 1}...")

            # 增量模式：找到本轮最佳结果并更新基线
            if loop_mode == 'incremental':
                logger.info(f"Incremental Mode: Finding best result from Round {round_num}...")
                round_best = _find_best_experiment_result(results, logger)  # 查找本轮最佳

                if round_best:
                    round_best_perf = round_best.get('performance', {}).get('overall_improvement_rate', 0)  # 最佳性能
                    round_best_path = round_best.get('code_path', '')  # 最佳代码路径

                    logger.info(f"  Round {round_num} best: {round_best['idea_name']} "
                              f"(improvement: {round_best_perf:+.2f}%)")

                    # 如果本轮最佳超过历史最佳，更新基线
                    if best_overall_performance is None or round_best_perf > best_overall_performance:
                        best_overall_performance = round_best_perf  # 更新最佳性能
                        best_code_path = round_best_path  # 更新最佳代码路径
                        logger.info(f"  New best found! Updating baseline for next round...")
                        _update_baseline_for_incremental(best_code_path, logger, task_type=args.task_type)  # 更新基线文件
                    else:
                        logger.info(f"  Current best remains: {best_code_path} "
                                  f"(improvement: {best_overall_performance:+.2f}%)")
                else:
                    logger.warning(f"  No successful experiments in Round {round_num}")

            logger.info(f"Starting Round {round_num + 1} in next iteration...")

    # 注意：经验生成现在在每轮结束后执行（见上方循环内）
    # 这确保经验库在后续轮次的 prompt 演化前是最新的

    # 步骤 3：最终摘要（所有轮次完成后）
    logger.info("")
    logger.info("=" * 80)
    if args.mode == "report":
        logger.info("ALL REPORT GENERATION ROUNDS COMPLETED")  # 报告模式完成
    else:
        logger.info("ALL DISCOVERY ROUNDS COMPLETED")  # 实验模式完成
    logger.info("=" * 80)

    # 汇总所有轮次的统计数据
    total_successful = sum(round_result['successful'] for round_result in all_round_results)  # 总成功数
    total_ideas = sum(len(round_result['results']) for round_result in all_round_results)  # 总想法数
    total_failed = total_ideas - total_successful  # 总失败数

    logger.info(f"Total Rounds: {len(all_round_results)}")
    logger.info(f"Loop Mode: {loop_mode.upper()}")
    if loop_mode == 'incremental' and best_code_path != original_task_dir:
        logger.info(f"Final Best Code Path: {best_code_path}")  # 最终最佳代码路径
        logger.info(f"Final Best Performance: {best_overall_performance:+.2f}%")  # 最终最佳性能
    logger.info(f"Sessions: {', '.join(all_session_ids)}")  # 所有会话 ID

    if args.mode == "report":
        logger.info(f"Total Reports Generated: {total_ideas}")  # 报告模式统计
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Failed: {total_failed}")
    else:
        logger.info(f"Total Ideas Tested: {total_ideas}")  # 实验模式统计
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Failed: {total_failed}")

    # 打印每轮的详细结果
    logger.info("\nDetailed Results by Round:")
    for round_result in all_round_results:  # 遍历每轮结果
        logger.info(f"\n  Round {round_result['round']} (Session: {round_result['session_id']}):")
        for i, result in enumerate(round_result['results'], 1):  # 遍历该轮的每个实验结果
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"  # 成功或失败标记
            logger.info(f"    {i}. {result['idea_name']}: {status}")
            if 'error' in result:  # 如果有错误信息
                logger.info(f"       Error: {result['error']}")
            elif 'report_path' in result:  # 如果有报告路径
                logger.info(f"       Report: {result['report_path']}")

    # 保存综合摘要到 JSON 文件
    summary = {
        'timestamp': datetime.now().isoformat(),  # 当前时间戳
        'launch_id': launch_id,  # 启动 ID
        'task': args.task_name,  # 任务名称
        'task_dir': args.task_dir,  # 任务目录
        'task_type': args.task_type,  # 任务类型
        'original_task_dir': original_task_dir,  # 原始任务目录
        'mode': args.mode,  # 执行模式
        'output_dir': args.output_dir,  # 输出目录
        'base_output_dir': args.base_output_dir,  # 基础输出目录
        'skip_idea_generation': args.skip_idea_generation,  # 是否跳过想法生成
        'total_rounds': len(all_round_results),  # 实际完成的轮次数
        'loop_rounds': loop_rounds,  # 配置的总轮次数
        'loop_mode': loop_mode,  # 循环模式
        'sessions': all_session_ids,  # 所有会话 ID
        'total_ideas': total_ideas,  # 总测试想法数
        'total_successful': total_successful,  # 总成功数
        'total_failed': total_failed,  # 总失败数
        'rounds': all_round_results  # 每轮详细结果
    }

    # 如果是增量模式，额外记录最佳结果信息
    if loop_mode == 'incremental':
        summary['incremental_mode'] = {
            'final_best_code_path': best_code_path,  # 最终最佳代码路径
            'final_best_performance': best_overall_performance  # 最终最佳性能
        }

    # 如果是实验模式，记录后端和模型信息
    if args.mode == "experiment":
        summary['exp_backend'] = args.exp_backend  # 实验后端
        summary['model'] = (
            config.get("experiment", {}).get("model") or  # 从配置读取模型
            "anthropic/claude-3-7-sonnet-20250219"  # 默认模型
        )

    # 将摘要写入 discovery_summary.json
    summary_path = osp.join(args.output_dir, "discovery_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)  # 格式化写入

    logger.info(f"\nSummary saved to {summary_path}")  # 记录摘要保存位置
    logger.info("=" * 80)
    logger.info("All done!")  # 流水线完成

# ============================================================================
# 入口点
# ============================================================================
if __name__ == "__main__":
    try:
        main()  # 运行主流水线
    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断
        print("\n\nDiscovery pipeline interrupted by user")
        sys.exit(1)  # 以退出码 1 退出
    except Exception as e:
        # 捕获未预期的异常并打印详细信息
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整堆栈跟踪
        sys.exit(1)  # 以退出码 1 退出
