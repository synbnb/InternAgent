import os
import json
import re
from openai import OpenAI
from typing import Dict, Any, Optional, List, Union
from .base_model import BaseModel

# 导入日志模块
from utils.logger import get_logger
from utils.fix_json import repair_json_string

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


class GeminiModel(BaseModel):
    """
    Gemini模型调用类
    """
    
    def __init__(self, model_name: str = "gemini-3-pro-preview", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 Gemini 模型
        
        Args:
            model_name: 模型名称，默认为 gpt-4o
            api_key: API 密钥，如果为 None 则从环境变量获取
            base_url: API 基础 URL，如果为 None 则使用默认的 Gemini API
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # 初始化 Gemini 客户端
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = OpenAI(**client_kwargs)
    
    def generate(self, prompt: str, auto_fix_json: bool = True, **kwargs) -> str:
        """
        调用 Gemini 模型生成响应
        
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

            # logger.info(f"openai response: {content}")
            
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
            raise Exception(f"Gemini API call failed: {str(e)}")
    
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
            raise Exception(f"Gemini API call failed: {str(e)}")
    
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

        # print(f"tool_call messages: {default_params}")

        # logger.info(f"tool_call messages: {default_params}")
        
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
            raise Exception(f"Gemini API call failed: {str(e)}") 