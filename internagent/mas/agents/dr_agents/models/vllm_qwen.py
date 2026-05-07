import os
import json
import re
from openai import OpenAI
from typing import Dict, Any, Optional, List
from .base_model import BaseModel

# 导入日志模块
from utils.logger import get_logger
from utils.fix_json import repair_json_string
import httpx

logger = get_logger(__name__)


def _is_likely_json_response(text: str) -> bool:
    """
    判断响应文本是否可能是JSON格式
    
    Args:
        text: 响应文本
        
    Returns:
        bool: 如果可能是JSON格式返回True
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # 检查是否包含JSON代码块
    if "```json" in text.lower() or "```" in text:
        return True
    
    # 检查是否以JSON对象或数组开始和结束
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        return True
    
    # 检查是否包含典型的JSON模式
    json_patterns = [
        r'{\s*"[^"]+"\s*:',  # 对象开始模式
        r'\[\s*{',           # 对象数组开始模式
        r':\s*"[^"]*"',      # 键值对模式
        r':\s*\d+',          # 数字值模式
        r':\s*true|false|null'  # 布尔值和null模式
    ]
    
    for pattern in json_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


class QwenModel(BaseModel):
    """
    OpenAI模型调用类
    """
    
    def __init__(self, model_name: str = "Qwen3-8B", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 OpenAI 模型
        
        Args:
            model_name: 模型名称，默认为 gpt-4o
            api_key: API 密钥，如果为 None 则从环境变量获取
            base_url: API 基础 URL，如果为 None 则使用默认的 OpenAI API
        """
        self.api_key = api_key or os.getenv("VLLM_KEY_WORKFLOW")
        self.base_url = base_url or os.getenv("VLLM_URL_WORKFLOW")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("VLLM key is required. Set VLLM_KEY environment variable or pass api_key parameter.")
        
        # 初始化 OpenAI 客户端
        self.client = httpx.Client(
            # proxy=None
            trust_env=False
        )
        
        # # Create a custom OpenAI client that uses the httpx client for requests
        # self.openai_client = OpenAI(http_client=self.client)
        client_kwargs = {
            "base_url": self.base_url,
            # 注意：OpenAI SDK 默认用 Bearer，这里覆盖为 Basic
            "default_headers": {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            },
            "http_client": self.client
        }
            
        self.client = OpenAI(**client_kwargs)
    
    def generate(self, prompt: str, auto_fix_json: bool = True, **kwargs) -> str:
        """
        调用 OpenAI 模型生成响应
        
        Args:
            prompt: 输入提示词
            auto_fix_json: 是否自动检测并修复JSON响应，默认为True
            **kwargs: 其他参数，如 temperature, max_tokens 等
            
        Returns:
            生成的文本响应，如果检测到JSON格式且auto_fix_json为True，则返回修复后的JSON字符串
        """
        # 默认参数
        default_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
        # 更新参数
        default_params.update(kwargs)

        # logger.info(f"generate messages: {default_params}")
        
        try:
            response = self.client.chat.completions.create(**default_params)
            content = response.choices[0].message.content
            
            # 如果启用自动JSON修复且响应可能是JSON格式
            if auto_fix_json and content and _is_likely_json_response(content):
                try:
                    # 尝试使用修复函数修复JSON
                    fixed_json = repair_json_string(content)
                    logger.info("Successfully repaired JSON response")
                    return fixed_json
                except Exception as json_error:
                    logger.warning(f"Failed to repair JSON response: {json_error}")
                    # 修复失败时返回原始文本
                    return content
            
            return content
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def generate_with_system_prompt(self, system_prompt: str, user_prompt: str, auto_fix_json: bool = True, **kwargs) -> str:
        """
        使用系统提示词和用户提示词生成响应
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            auto_fix_json: 是否自动检测并修复JSON响应，默认为True
            **kwargs: 其他参数
            
        Returns:
            生成的文本响应，如果检测到JSON格式且auto_fix_json为True，则返回修复后的JSON字符串
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 默认参数
        default_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2000,
            "stream": False
        }
        
        # 更新参数
        default_params.update(kwargs)
        
        # logger.info(f"generate_with_system_prompt messages: {default_params}")
        
        try:
            response = self.client.chat.completions.create(**default_params)
            content = response.choices[0].message.content
            
            # 如果启用自动JSON修复且响应可能是JSON格式
            if auto_fix_json and content and _is_likely_json_response(content):
                try:
                    # 尝试使用修复函数修复JSON
                    fixed_json = repair_json_string(content)
                    logger.info("Successfully repaired JSON response")
                    return fixed_json
                except Exception as json_error:
                    logger.warning(f"Failed to repair JSON response: {json_error}")
                    # 修复失败时返回原始文本
                    return content
            
            return content
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def generate_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        使用工具调用生成响应
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            **kwargs: 其他参数
            
        Returns:
            包含工具调用的响应
        """
        # 默认参数
        default_params = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": False
        }
        
        # 更新参数
        default_params.update(kwargs)

        # logger.info(f"tool_call messages: {messages}")
        
        try:
            response = self.client.chat.completions.create(**default_params)
            # 返回完整的响应对象，包含工具调用信息
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content,
                        "tool_calls": response.choices[0].message.tool_calls
                    }
                }],
                "usage": response.usage,
                "model": response.model,
                "id": response.id
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}") 
        
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/home/PJLAB/huyusong/workspace/InternResearch/.env")

    from . import get_model
    model = get_model("Qwen3-8B")

    # messages = [{'role': 'user', 'content': "Retrieve today's temperature in Beijing from a reliable weather source using web search."}, 
    #             {'role': 'assistant', 'content': '', 'tool_calls': [{'index': 0, 'id': 'call_0_ea0ad23b-3941-4b46-891b-6053631255ac', 'type': 'function', 'function': {'name': 'search_web', 'arguments': '{"query":"today\'s temperature in Beijing"}'}}]}, 
    #             {'role': 'tool', 'tool_call_id': 'call_0_ea0ad23b-3941-4b46-891b-6053631255ac', 'content': "搜索结果 for 'today's temperature in Beijing': 找到相关网页和文档信息"}]

    # tools = [{'type': 'function', 'function': {'name': 'search_web', 'description': '搜索网络信息', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': '搜索查询'}}, 'required': ['query']}}}, {'type': 'function', 'function': {'name': 'calculate', 'description': '执行数学计算', 'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string', 'description': '数学表达式'}}, 'required': ['expression']}}}, {'type': 'function', 'function': {'name': 'get_weather', 'description': '获取指定城市的天气信息', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': '城市名称'}}, 'required': ['city']}}}]
    # result = model.generate_with_tools(messages, tools)
    result = model.generate("hello")
    print("result: ", result)