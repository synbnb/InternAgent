#!/usr/bin/env python3
"""
Prompt加载工具
支持从配置文件加载prompt，提供灵活的prompt管理
"""

import os
import importlib.util
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("PromptLoader")


def load_prompt(config: Dict[str, Any], 
                default_name: str, default_path: str = "prompts/default_prompts.py") -> str:
    """
    从配置加载prompt
    
    如果config中配置了prompt_path和prompt_name，从指定文件加载
    否则使用默认的路径和名称
    
    Args:
        config: 配置字典（应包含prompt_path和prompt_name键，可选）
        default_name: 默认prompt变量名
        default_path: 默认prompt文件路径（相对于项目根目录）
        
    Returns:
        加载的prompt字符串
        
    Example:
        # 从配置加载
        config = {
            "prompt_path": "agents/prompts.py",
            "prompt_name": "CUSTOM_PLANNER_PROMPT"
        }
        prompt = load_prompt(config, "GLOBAL_PLANNER_PROMPT", "agents/prompts.py")
        
        # 使用默认值
        config = {}
        prompt = load_prompt(config, "GLOBAL_PLANNER_PROMPT", "agents/prompts.py")
    """
    
    # 从config读取路径，如果没有或config为None则使用默认值
    if config is None:
        config = {}

    prompt_path = config.get("prompt_path") or default_path
    prompt_name = config.get("prompt_name") or default_name
    
    try:
        loaded_prompt = load_prompt_from_file(prompt_path, prompt_name)
        if config.get("prompt_path") or config.get("prompt_name"):
            logger.info(f"从配置加载prompt: {prompt_path} -> {prompt_name}")
        else:
            logger.debug(f"使用默认prompt: {prompt_path} -> {prompt_name}")
        return loaded_prompt
    except Exception as e:
        logger.error(f"加载prompt失败: {e}")
        raise


def load_prompt_from_file(prompt_path: str, prompt_name: str) -> str:
    """
    从Python文件中加载prompt变量
    
    Args:
        prompt_path: prompt文件路径（相对于项目根目录）
        prompt_name: prompt变量名
        
    Returns:
        prompt字符串
        
    Raises:
        FileNotFoundError: 文件不存在
        AttributeError: 变量不存在
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 构建完整路径
    full_path = os.path.join(project_root, prompt_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Prompt文件不存在: {full_path}")
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("prompt_module", full_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {full_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 获取prompt变量
    if not hasattr(module, prompt_name):
        raise AttributeError(f"模块 {prompt_path} 中不存在变量 {prompt_name}")
    
    prompt = getattr(module, prompt_name)
    
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt变量 {prompt_name} 不是字符串类型")
    
    return prompt




if __name__ == "__main__":
    # 测试代码
    print("测试prompt加载器...")
    
    # 测试从配置加载
    config1 = {
        "prompt_path": "prompts/default_prompts.py",
        "prompt_name": "GLOBAL_PLANNER_PROMPT"
    }
    try:
        prompt1 = load_prompt(config1, "GLOBAL_PLANNER_PROMPT")
        print(f"✓ 从配置加载成功，长度: {len(prompt1)}")
    except Exception as e:
        print(f"✗ 从配置加载失败: {e}")
    
    # 测试使用默认值
    config2 = {}
    try:
        prompt2 = load_prompt(config2, "GLOBAL_COORDINATOR_PROMPT")
        print(f"✓ 使用默认值加载成功，长度: {len(prompt2)}")
    except Exception as e:
        print(f"✗ 使用默认值加载失败: {e}")
    
    
    print("\n所有测试完成！")
