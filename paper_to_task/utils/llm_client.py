"""
LLM客户端 - 用于调用语言模型进行论文分析
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path


class LLMClient:
    """统一的LLM调用接口"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM客户端

        Args:
            config: 配置字典，包含model, api_key, base_url等信息
        """
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 4000)
        self.cache_enabled = config.get("cache_enabled", True)

        # 简单的内存缓存
        self._cache: Dict[str, str] = {}

        # 根据配置选择后端
        self.backend = config.get("backend", "openai")

    def call(self, prompt: str, use_cache: bool = True,
             system_prompt: Optional[str] = None) -> str:
        """
        调用LLM生成响应

        Args:
            prompt: 用户提示词
            use_cache: 是否使用缓存
            system_prompt: 系统提示词（可选）

        Returns:
            LLM的响应文本
        """
        # 检查缓存
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(prompt, system_prompt)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # 优化提示词
        optimized_prompt = self._optimize_prompt(prompt)

        # 调用API
        response = self._make_api_call(optimized_prompt, system_prompt)

        # 缓存结果
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(prompt, system_prompt)
            self._cache[cache_key] = response

        return response

    def call_with_json(self, prompt: str, use_cache: bool = True,
                      system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        调用LLM并期望返回JSON格式

        Args:
            prompt: 用户提示词
            use_cache: 是否使用缓存
            system_prompt: 系统提示词

        Returns:
            解析后的JSON字典
        """
        # 添加JSON格式要求
        json_prompt = prompt + "\n\n请严格按照JSON格式返回结果，不要包含其他文本。"

        response = self.call(json_prompt, use_cache, system_prompt)

        # 尝试解析JSON
        try:
            # 清理可能的markdown代码块标记
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            # 如果解析失败，返回空字典
            print(f"警告: JSON解析失败 - {e}")
            return {}

    def _make_api_call(self, prompt: str,
                      system_prompt: Optional[str] = None) -> str:
        """执行实际的API调用"""

        if self.backend == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.backend == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        elif self.backend == "deepseek":
            return self._call_deepseek(prompt, system_prompt)
        elif self.backend == "mock":
            # 用于测试的模拟后端
            return self._mock_response(prompt)
        else:
            raise ValueError(f"不支持的backend: {self.backend}")

    def _call_openai(self, prompt: str,
                    system_prompt: Optional[str] = None) -> str:
        """调用OpenAI API"""
        try:
            import openai

            # 配置客户端
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 调用API
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except ImportError:
            # 如果没有安装openai包，使用模拟响应
            print("警告: 未安装openai包，使用模拟响应")
            return self._mock_response(prompt)
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return self._mock_response(prompt)

    def _call_anthropic(self, prompt: str,
                       system_prompt: Optional[str] = None) -> str:
        """调用Anthropic API"""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

            # 调用API
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "你是一个科研助手",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except ImportError:
            print("警告: 未安装anthropic包，使用模拟响应")
            return self._mock_response(prompt)
        except Exception as e:
            print(f"Anthropic API调用失败: {e}")
            return self._mock_response(prompt)

    def _call_deepseek(self, prompt: str,
                       system_prompt: Optional[str] = None) -> str:
        """调用DeepSeek API"""
        try:
            import openai

            # DeepSeek使用OpenAI兼容的API
            # 默认base_url为 https://api.deepseek.com
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.deepseek.com"
            )

            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 调用API
            response = client.chat.completions.create(
                model=self.model or "deepseek-chat",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except ImportError:
            print("警告: 未安装openai包，DeepSeek需要openai包，使用模拟响应")
            return self._mock_response(prompt)
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """
        生成模拟响应（用于测试）

        这是一个简化版本的实现，实际使用时需要真实的LLM调用
        """
        # 根据提示词关键词生成简单响应
        if "研究目标" in prompt or "goal" in prompt.lower():
            return json.dumps({
                "goal": "验证论文中的核心发现",
                "hypothesis": "论文描述的方法有效",
                "background": "该领域的研究背景",
                "research_field": "生物学"
            }, ensure_ascii=False)

        elif "方法" in prompt or "method" in prompt.lower():
            return json.dumps({
                "main_methods": "实验方法概述",
                "algorithms": "相关算法",
                "tools": "分析工具",
                "experimental_setup": "实验设置"
            }, ensure_ascii=False)

        elif "实验设计" in prompt or "experimental" in prompt.lower():
            return json.dumps({
                "phases": {
                    "phase_1": {
                        "name": "第一阶段",
                        "description": "初步实验",
                        "methods": ["方法1", "方法2"]
                    }
                },
                "comparison_groups": "对照组设置",
                "dependent_variables": ["因变量1"],
                "independent_variables": ["自变量1"]
            }, ensure_ascii=False)

        else:
            return '{"result": "模拟响应"}'

    def _optimize_prompt(self, prompt: str) -> str:
        """优化提示词，提高效率"""
        # 压缩过长的提示词
        if len(prompt) > 10000:
            # 保留开头和结尾，中间压缩
            prompt = prompt[:3000] + "\n...[内容省略]...\n" + prompt[-3000:]

        return prompt

    def _get_cache_key(self, prompt: str,
                      system_prompt: Optional[str] = None) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{system_prompt or ''}::{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": 1 if self.cache_enabled else 0
        }
