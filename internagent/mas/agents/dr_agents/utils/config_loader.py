#!/usr/bin/env python3
"""
简单的配置加载工具
只从 config.yaml 读取配置，返回字典
"""

import os
import yaml
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger("ConfigLoader")

# 全局配置缓存
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        配置字典
    """
    global _config_cache
    
    if config_path is None:
        # 默认配置文件路径
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "config_simple.yaml"
        )
    
    # 如果配置文件不存在，返回默认配置
    if not os.path.exists(config_path):
        print(f"Warning: 配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 填充默认值
    config = merge_with_defaults(config)

    logger.info(f"Config loaded: {config_path}")
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'model': {
            'default_model': 'deepseek-v4-pro',
            'global_planner_model': None,
            'global_execution_model': None,
            'coordinator_model': None,
            'synthesizer_model': None,
            'task_agent_model': None,
        },
        'main': {
            'max_iter': 10,
        },
        'global_planner': {
            'max_iter': 5,
            'max_nodes': 8,
            'max_retries': 3,
            'enable_multi_layer': True,
            'timeout': 300,
            'save_graph': True,
            'graph_output_dir': './tmp/graphs',
        },
        'global_execution': {
            'max_workers': 10,
            'planner': {
                'max_subtasks': 2,
            },
            'execution': {
                'max_tool_calls': 15,
            },
        },
        'coordinator': {
            'enable_graph_correction': True,
            'max_correction_attempts': 3,
            'enable_dynamic_nodes': True,
        },
        'synthesizer': {
            'output_format': 'markdown',
            'include_details': True,
            'include_failed_nodes': True,
            'max_output_length': 10000,
            'disable_multimodal': False,  # True 时禁用阶段2.5（插图生成）
        },
        'tools': {
            'enabled_tools': [
                'search', 'web_search', 'file_reader', 
                'code_executor', 'calculator', 'database_query',
                'image_processor', 'video_processor'
            ],
            'tool_timeout': 120,
            'search_config': {
                'max_results': 10,
                'search_engine': 'google',
            },
            'code_execution_config': {
                'sandbox': True,
                'max_execution_time': 60,
                'allowed_languages': ['python', 'javascript', 'bash'],
            },
        },
        'log': {
            'log_level': 'INFO',
            'log_dir': './logs',
            'console_output': True,
            'log_filename_format': '{timestamp}_{agent_type}.log',
            'save_execution_trace': True,
            'trace_output_dir': './logs/traces',
        },
        'path': {
            'project_root': '.',
            'dataset_dir': './gaia_dataset',
            'results_dir': './gaia_results',
            'tmp_dir': './tmp',
            'cache_dir': './tmp/cache',
        },
        'workflow': {
            'enable_answer_node_detection': True,
            'min_layers_before_answer': 1,
            'save_intermediate_results': True,
            'intermediate_results_dir': './tmp/intermediate',
            'enable_progress_tracking': True,
        },
    }


def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将用户配置与默认配置合并
    
    Args:
        config: 用户配置
    
    Returns:
        合并后的配置
    """
    default = get_default_config()
    return deep_merge(default, config)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典
    
    Args:
        base: 基础字典
        override: 覆盖字典
    
    Returns:
        合并后的字典
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config() -> Dict[str, Any]:
    """
    获取全局配置（带缓存）
    
    Returns:
        配置字典
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_config()
    
    return _config_cache


def reload_config(config_path: str = None) -> Dict[str, Any]:
    """
    重新加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    global _config_cache
    _config_cache = load_config(config_path)
    return _config_cache


def get_value(path: str, default: Any = None) -> Any:
    """
    通过路径获取配置值
    
    Args:
        path: 配置路径，用点分隔，如 "global_planner.max_iter"
        default: 默认值
    
    Returns:
        配置值
    """
    config = get_config()
    keys = path.split('.')
    
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


# 便捷访问函数
def get_model_config() -> Dict[str, Any]:
    """获取模型配置"""
    return get_config().get('model', {})


def get_main_config() -> Dict[str, Any]:
    """获取主配置"""
    return get_config().get('main', {})


def get_planner_config() -> Dict[str, Any]:
    """获取Planner配置"""
    return get_config().get('global_planner', {})


def get_execution_config() -> Dict[str, Any]:
    """获取Execution配置"""
    return get_config().get('global_execution', {})


def get_coordinator_config() -> Dict[str, Any]:
    """获取Coordinator配置"""
    return get_config().get('coordinator', {})


def get_synthesizer_config() -> Dict[str, Any]:
    """获取Synthesizer配置"""
    return get_config().get('synthesizer', {})


def get_tools_config() -> Dict[str, Any]:
    """获取工具配置"""
    return get_config().get('tools', {})


def get_log_config() -> Dict[str, Any]:
    """获取日志配置"""
    return get_config().get('log', {})


def get_path_config() -> Dict[str, Any]:
    """获取路径配置"""
    return get_config().get('path', {})


def get_workflow_config() -> Dict[str, Any]:
    """获取工作流配置"""
    return get_config().get('workflow', {})


if __name__ == "__main__":
    # 测试配置加载
    config = get_config()
    print("配置加载成功！")
    print(f"默认模型: {config['model']['default_model']}")
    print(f"Planner最大迭代次数: {config['global_planner']['max_iter']}")
    print(f"Execution最大工作线程: {config['global_execution']['max_workers']}")

