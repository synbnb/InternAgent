import os
import json
import re
from openai import OpenAI
from typing import Dict, Any, Optional, List
from .base_model import BaseModel
import httpx

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


class InternS1Model(BaseModel):
    """
    InternLM S1模型调用类
    """
    
    def __init__(self, model_name: str = "intern-s1", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 InternLM S1 模型
        
        Args:
            model_name: 模型名称，默认为 intern-s1
            api_key: API 密钥，如果为 None 则从环境变量获取
            base_url: API 基础 URL，如果为 None 则使用默认的 InternLM API
        """
        self.api_key = api_key or os.getenv("INTERN_S1_API_KEY")
        self.base_url = base_url or "https://intern.openxlab.org.cn/api/v1/"
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("InternLM S1 API key is required. Set INTERN_S1_API_KEY environment variable or pass api_key parameter.")


        # 初始化 OpenAI 客户端，使用自定义 httpx 客户端来禁用代理
        http_client = httpx.Client(trust_env=False)
        
        client_kwargs = {
            "api_key": self.api_key,
            "http_client": http_client
        }

        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = OpenAI(**client_kwargs)
    
    def generate(self, prompt: str, auto_fix_json: bool = True, **kwargs) -> str:
        """
        调用 InternLM S1 模型生成响应
        
        Args:
            prompt: 输入提示词
            auto_fix_json: 是否自动检测并修复JSON响应，默认为True
            **kwargs: 其他参数，如 temperature, max_tokens, thinking_mode 等
            
        Returns:
            生成的文本响应，如果检测到JSON格式且auto_fix_json为True，则返回修复后的JSON字符串
        """
        # 默认参数
        default_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # 处理 thinking_mode 参数，默认开启
        thinking_mode = kwargs.pop("thinking_mode", True)
        if thinking_mode:
            default_params["extra_body"] = {"thinking_mode": True}
        
        # 更新其他参数
        default_params.update(kwargs)

        logger.info(f"InternS1 generate request: {default_params}")
        
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
            logger.error(f"InternS1 API call failed: {str(e)}")
            raise Exception(f"InternS1 API call failed: {str(e)}")
    
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
        }
        
        # 处理 thinking_mode 参数，默认开启
        thinking_mode = kwargs.pop("thinking_mode", True)
        if thinking_mode:
            default_params["extra_body"] = {"thinking_mode": True}
        
        # 更新其他参数
        default_params.update(kwargs)
        
        logger.info(f"InternS1 generate_with_system_prompt request: {default_params}")
        
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
            logger.error(f"InternS1 API call failed: {str(e)}")
            raise Exception(f"InternS1 API call failed: {str(e)}")
    
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
        }
        
        # 处理 thinking_mode 参数，默认开启
        thinking_mode = kwargs.pop("thinking_mode", True)
        if thinking_mode:
            default_params["extra_body"] = {"thinking_mode": True}
        
        # 更新其他参数
        default_params.update(kwargs)

        logger.info(f"InternS1 tool_call request: {default_params}")
        
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
            logger.error(f"InternS1 API call failed: {str(e)}")
            raise Exception(f"InternS1 API call failed: {str(e)}")
 