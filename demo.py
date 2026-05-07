
import os
import asyncio
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

from internagent.mas.agents.base_agent import BaseAgent
from internagent.mas.models.base_model import BaseModel
from internagent.mas.models.openai_model import OpenAIModel
from internagent.mas.tools import get_registry


load_dotenv()
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MyAgent(BaseAgent):
    """简化的助手Agent，使用函数式工具"""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # 获取工具注册表
        self.tool_registry = get_registry()
        
        logger.info(f"MyAgent initialized with {len(self.tool_registry)} tools: {self.tool_registry.get_all_names()}")
    
    async def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """
        执行工具函数
        
        Args:
            function_name: 工具名称
            function_args: 工具参数
        
        Returns:
            工具执行结果
        """
        logger.info(f"🔧 Executing: {function_name}")
        logger.info(f"📝 Args: {json.dumps(function_args, ensure_ascii=False)}")
        
        # 通过注册表执行工具
        result = await self.tool_registry.execute(function_name, **function_args)
        
        logger.info(f"✅ Result: {result}")
        return result
    
    def get_registered_tools(self) -> list:
        """获取所有工具的OpenAI格式定义"""
        return self.tool_registry.get_all_definitions()
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Agent任务
        
        Args:
            context: {"query": "用户问题"}
            params: {"temperature": 0.7}
        """
        query = context.get("query", "")
        
        if not query:
            return {"status": "error", "error": "No query provided"}
        
        logger.info(f"📥 Query: {query}")
        
        try:
            # 获取所有工具定义
            tools = self.get_registered_tools()
            
            # 调用模型（假设base_agent中有这个方法）
            response = await self._call_model_with_tools(
                prompt=query,
                tools=tools,
                temperature=params.get("temperature", 0.7),
                max_iterations=params.get("max_iterations", 10),
                max_tool_calls=params.get("max_tool_calls", 20)
            )
            
            return {
                "status": "success",
                "answer": response["content"],
                "tool_calls": response["tool_calls_made"],
                "iterations": response["iterations"]
            }
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {"status": "error", "error": str(e)}

"""
Simple Function-based Tools Demo
"""

async def main():
    print("\n" + "="*60)
    print("MyAgent Demo - Function-based Tools")
    print("="*60)
    
    # 1. 初始化
    model = OpenAIModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        temperature=0.7
    )
    
    agent = MyAgent(
        model=model,
        config={"name": "my_assistant"}
    )
    
    print(f"\n✅ Loaded {len(agent.get_registered_tools())} tools")
    
    # 2. 测试
    tests = [
        "Design a novel organocatalyst for enhancing CO2 conversion in carbon capture processes. The following reference functions are available: 1) Research existing organocatalysts using `query2smiles` to obtain their SMILES structures. 2) Generate structural variants using `modify_mol`. 3) Verify novelty with `patent_check`. Finally, propose a novel organocatalyst with detailed rationale for its design and expected performance. Ensure all steps are documented and justified. ",
        # "What's the weather in Beijing?",
        # "Calculate (10 + 20) * 3",
        # "Get weather for Tokyo and Paris, then calculate the average temperature"
    ]
    
    for i, query in enumerate(tests, 1):
        print(f"\n{'─'*60}")
        print(f"Test {i}: {query}")
        print(f"{'─'*60}")
        
        result = await agent.execute(
            context={"query": query},
            params={"temperature": 0.7}
        )
        
        if result["status"] == "success":
            print(f"\n💬 {result['answer']}")
            print(f"\n🔧 Used {len(result['tool_calls'])} tool(s)")
        else:
            print(f"\n❌ Error: {result['error']}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
