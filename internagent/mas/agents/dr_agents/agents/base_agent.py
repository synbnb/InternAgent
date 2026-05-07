from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
import redis

from utils.logger import get_logger


class BaseAgent(ABC):
    """
    基类 Agent，提供基本的执行框架
    """
    
    def __init__(self, model: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, config: Optional[Any] = None):
        """
        初始化 Agent
        
        Args:
            model: 使用的模型名称或配置
            tools: 可用的工具列表
            config: Agent特定的配置对象（每个Agent接收自己的配置）
        """
        self.model = model or "deepseek-r1"
        self.tools = tools or []
        self.config = config
        
        # Redis相关
        self.task_id: Optional[str] = None
        self.redis_client: Optional[redis.Redis] = None
        self._logger = None
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        执行 Agent 的主要逻辑
        
        Args:
            input_data: 输入数据
            
        Returns:
            执行结果
        """
        pass
    
    def get_model(self) -> str:
        """
        获取当前使用的模型
        
        Returns:
            模型名称或配置
        """
        return self.model
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        获取可用的工具列表
        
        Returns:
            工具列表
        """
        return self.tools
    
    def add_tool(self, tool: Dict[str, Any]) -> None:
        """
        添加工具
        
        Args:
            tool: 工具配置字典
        """
        self.tools.append(tool)
    
    def set_model(self, model: str) -> None:
        """
        设置模型
        
        Args:
            model: 模型名称或配置
        """
        self.model = model
    
    def set_task_id(self, task_id: str) -> None:
        """
        设置任务ID并初始化Redis连接
        
        Args:
            task_id: 任务ID
        """
        self.task_id = task_id
        if task_id:
            try:
                redis_host = os.getenv('REDIS_HOST', 'localhost')
                redis_port = int(os.getenv('REDIS_PORT', 6379))
                redis_db = int(os.getenv('REDIS_DB', 0))
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                if self._logger:
                    self._logger.info(f"Redis connected for task {task_id} in {self.__class__.__name__}")
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Failed to connect to redis in {self.__class__.__name__}: {e}")
                self.redis_client = None
    
    def send_redis_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        发送事件到Redis Stream
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
            
        Returns:
            bool: 是否发送成功
        """
        if not self.redis_client or not self.task_id:
            return False
        
        try:
            event = {
                "event": event_type,
                "task_id": self.task_id,
                **event_data
            }
            self.redis_client.xadd(
                f"workflow:{self.task_id}",
                {"data": json.dumps(event, ensure_ascii=False)},
                maxlen=1000
            )
            if self._logger:
                self._logger.info(f"Sent {event_type} event to redis stream for task {self.task_id}")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to send {event_type} event to redis: {e}")
            return False
    
    def _set_logger(self, logger) -> None:
        """
        设置logger实例（供子类使用）
        
        Args:
            logger: logger实例
        """
        self._logger = logger
