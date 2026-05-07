"""
统一的日志配置模块
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional


class LoggerConfig:
    """日志配置类"""
    
    def __init__(self, 
                 name: str = "InternResearch",
                 log_level: int = logging.INFO,
                 log_dir: str = "logs",
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 100,
                 debug_mode: bool = False,
                 additional_log_file: Optional[str] = None):
        """
        初始化日志配置
        
        Args:
            name: 日志器名称
            log_level: 日志级别
            log_dir: 日志文件目录
            max_bytes: 单个日志文件最大大小
            backup_count: 保留的日志文件数量
            debug_mode: 调试模式，为True时每次启动创建新子文件夹，但仍按大小分割日志文件
            additional_log_file: 可选的额外日志文件路径，日志会同时写入主日志和这个文件
        """
        self.name = name
        self.log_level = log_level
        self.debug_mode = debug_mode
        self.additional_log_file = additional_log_file
        
        # 调试模式下，每次启动创建带时间戳的子文件夹
        if debug_mode:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_dir = os.path.join(log_dir, f"debug_{timestamp}")
        else:
            self.log_dir = log_dir
        
        # 无论调试模式还是生产模式，都使用相同的文件大小限制
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 如果有额外的日志文件路径，确保其目录存在
        if self.additional_log_file:
            additional_log_dir = os.path.dirname(self.additional_log_file)
            if additional_log_dir:
                os.makedirs(additional_log_dir, exist_ok=True)
        
        # 创建日志器
        self.logger = self._create_logger()
    
    def _create_logger(self) -> logging.Logger:
        """创建并配置日志器"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器 - 统一使用RotatingFileHandler，按大小分割
        if self.debug_mode:
            # 调试模式：使用带时间戳的文件名
            log_file = os.path.join(self.log_dir, f"{self.name}.log")
        else:
            # 生产模式：使用带日期的文件名
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(self.log_dir, f"{self.name}_{today}.log")
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 如果指定了额外的日志文件路径，添加额外的文件处理器
        if self.additional_log_file:
            additional_handler = logging.handlers.RotatingFileHandler(
                self.additional_log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            additional_handler.setLevel(self.log_level)
            additional_handler.setFormatter(formatter)
            logger.addHandler(additional_handler)
        
        # 防止日志向上传播，避免重复输出
        logger.propagate = False
        
        return logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        获取日志器
        
        Args:
            name: 子日志器名称，如果为None则返回主日志器
            
        Returns:
            日志器实例
        """
        if name:
            logger = logging.getLogger(f"{self.name}.{name}")
            # 确保子logger也设置正确的级别
            logger.setLevel(self.log_level)
            return logger
        return self.logger


# 全局日志配置实例
_logger_config = None

# 存储节点特定的日志器
_node_loggers = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 子日志器名称
        
    Returns:
        日志器实例
    """
    global _logger_config
    if _logger_config is None:
        _logger_config = LoggerConfig(debug_mode=True)  # 默认使用调试模式
    return _logger_config.get_logger(name)


def setup_logger(name: str = "InternResearch",
                log_level: int = logging.INFO,
                log_dir: str = "logs",
                debug_mode: bool = True,
                additional_log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        name: 日志器名称
        log_level: 日志级别
        log_dir: 日志文件目录
        debug_mode: 调试模式，默认为True
        additional_log_file: 可选的额外日志文件路径，日志会同时写入主日志和这个文件
        
    Returns:
        日志器实例
    """
    global _logger_config
    _logger_config = LoggerConfig(name, log_level, log_dir, debug_mode=debug_mode, 
                                  additional_log_file=additional_log_file)
    return _logger_config.get_logger()


# 便捷的日志方法
def debug(msg: str, *args, **kwargs):
    """调试日志"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """信息日志"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """警告日志"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """错误日志"""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """严重错误日志"""
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """异常日志"""
    get_logger().exception(msg, *args, **kwargs)


def get_node_logger(task_id: Optional[str], node_id: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    获取节点特定的日志器，每个节点的日志会写入独立的文件
    
    Args:
        task_id: 任务ID（可选），如果为None则使用时间戳作为任务标识
        node_id: 节点ID
        log_level: 日志级别
        
    Returns:
        节点特定的日志器实例
    """
    global _logger_config, _node_loggers
    
    # 确保主logger配置已初始化
    if _logger_config is None:
        _logger_config = LoggerConfig(debug_mode=True)
    
    # 如果没有task_id，使用时间戳或默认值
    if not task_id:
        task_id = "default_task"
    
    # 创建唯一的logger标识
    logger_key = f"{task_id}_{node_id}"
    
    # 如果已经创建过，直接返回
    if logger_key in _node_loggers:
        return _node_loggers[logger_key]
    
    # 创建节点特定的日志目录：logs/debug_timestamp/task_id/
    task_log_dir = os.path.join(_logger_config.log_dir, task_id)
    os.makedirs(task_log_dir, exist_ok=True)
    
    # 创建节点特定的logger
    logger_name = f"InternResearch.{task_id}.{node_id}"
    node_logger = logging.getLogger(logger_name)
    node_logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if not node_logger.handlers:
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器（可选，如果不想在控制台显示节点日志，可以注释掉）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        node_logger.addHandler(console_handler)
        
        # 文件处理器 - 节点特定的日志文件
        log_file = os.path.join(task_log_dir, f"{node_id}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=_logger_config.max_bytes,
            backupCount=_logger_config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        node_logger.addHandler(file_handler)
        
        # 防止日志向上传播到父logger，避免重复记录到主日志文件
        node_logger.propagate = False
    
    # 缓存logger
    _node_loggers[logger_key] = node_logger
    
    return node_logger


def add_task_log_handler(logger: logging.Logger, task_id: str, outer_logs_dir: str) -> logging.Handler:
    """
    为现有的logger添加一个任务特定的文件处理器
    
    Args:
        logger: 要添加处理器的logger实例
        task_id: 任务ID
        outer_logs_dir: 外部日志目录路径
        
    Returns:
        添加的文件处理器实例，可用于后续移除
    """
    global _logger_config
    
    # 确保外部日志目录存在
    os.makedirs(outer_logs_dir, exist_ok=True)
    
    # 创建任务特定的日志文件路径
    log_file = os.path.join(outer_logs_dir, f"flowsearch_{task_id}.log")
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件处理器
    if _logger_config is None:
        _logger_config = LoggerConfig(debug_mode=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=_logger_config.max_bytes,
        backupCount=_logger_config.backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(formatter)
    
    # 添加到logger
    logger.addHandler(file_handler)
    
    return file_handler


def remove_log_handler(logger: logging.Logger, handler: logging.Handler):
    """
    从logger中移除指定的处理器
    
    Args:
        logger: logger实例
        handler: 要移除的处理器
    """
    try:
        logger.removeHandler(handler)
        handler.close()
    except Exception as e:
        # 忽略移除处理器时的错误
        pass 