import sys
import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import BaseAgent
from models import get_model
from utils.graph import DirectedGraph, NodeExecutionStatus

from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from tools import get_tool_manager

from dotenv import load_dotenv
load_dotenv()

logger = get_logger("coordinator_agent")


class CoordinatorAgent(BaseAgent):
    """
    协调器代理，负责接收global_execution_agent的graph，
    调用大模型进行分析和修改，输出修改后的graph
    """
    
    def __init__(self, model: str = "deepseek-r1", config=None, **model_kwargs):
        """
        初始化协调器代理
        
        Args:
            model: 使用的模型名称
            config: CoordinatorConfig配置对象
            **model_kwargs: 传递给模型的参数
        """
        super().__init__(model=model, config=config)
        self.model_instance = get_model(model, **model_kwargs)
        self.logger = logger
        self._set_logger(logger)  # 设置logger到基类
        self.model_kwargs = model_kwargs
        self.tool_manager = get_tool_manager(config=config)
        
        # 从配置加载prompt（如果配置中没有指定，使用默认值）
        self.config = config or {}
        self.prompt_template = load_prompt(
            self.config,
            default_name="GLOBAL_COORDINATOR_PROMPT"
        )
    
    def execute(self, input_data: Any) -> Any:
        """
        执行协调器的主要逻辑
        
        Args:
            input_data: 
            {
                "graph": graph, (DirectedGraph)
                "query": query
            }
            
        Returns:
            修改后的graph
        """

        if isinstance(input_data, dict):
            graph = input_data.get('graph')
            query = input_data.get('query')
        else:
            graph = input_data

        tools = self.tool_manager.list_tools()
        tool_info_list = []
        for tool in tools:
            tool_info = self.tool_manager.get_simple_tool_info(tool)
            tool_info_list.append(tool_info)
        
        if not graph:
            raise ValueError("输入数据中必须包含graph")
        
        return self._call_llm_for_modification(graph, query, tool_info_list)
    
    def _call_llm_for_modification(self, graph: DirectedGraph, query: str, tool_info_list: List[Dict[str, Any]], additional_info: str = None):
        """
        调用大模型来修改graph
        
        Args:
            graph: 原始graph
            context: 上下文信息
            additional_info: 额外的文档信息(Optional)
            
        Returns:
            修改后的graph
        """
        # 构建prompt参数
        graph_json = json.dumps(graph.to_dict(), ensure_ascii=False, indent=2)
        
        # 确保所有参数都是字符串类型，避免 tuple 等类型导致的错误
        query_str = str(query) if query is not None else ""
        additional_info_str = str(additional_info) if additional_info else ""
        tools_str = json.dumps(tool_info_list, ensure_ascii=False)
        
        # 使用format方法统一格式化prompt
        prompt = self.prompt_template.format(
            graph=graph_json,
            query=query_str,
            tools=tools_str,
            additional_info=additional_info_str if additional_info_str else ""
        )

        # print("prompt: ", prompt)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 调用大模型
                response = self.model_instance.generate(prompt, **self.model_kwargs)


            
                # 解析响应并应用修改
                modified_graph = self._parse_llm_response(graph, response)
                
                # 验证并修复图的结构问题
                modified_graph.validate_and_fix_graph()

                # 发送coordinator响应到Redis
                self.send_redis_event("get_coordinator_modification", {
                    "task_id": self.task_id,
                    "modification": json.loads(response)
                })

                # 发送修改后的图到Redis
                modified_graph_dict = modified_graph.to_dict()
                self.send_redis_event("get_graph_after_coordinator", {
                    "task_id": self.task_id,
                    "nodes": modified_graph_dict['nodes'],
                    "edges": modified_graph_dict['edges']
                })

                # 如果解析成功，返回修改后的graph
                return modified_graph
                    
            except Exception as e:
                self.logger.warning(f"Coordinator第{attempt + 1}次尝试失败: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"重试{max_retries}次后仍然失败，返回原始graph")
                    return copy.deepcopy(graph)
                else:
                    self.logger.info(f"Coordinator准备进行第{attempt + 2}次重试...")
                    continue
        
        # 理论上不会到达这里，但为了安全起见
        return copy.deepcopy(graph)
    
    def _parse_llm_response(self, original_graph: DirectedGraph, response: str) -> DirectedGraph:
        """
        解析大模型的响应并应用修改
        
        Args:
            original_graph: 原始graph
            response: 大模型的响应
            
        Returns:
            修改后的graph
        """
        try:
            modifications = json.loads(response)       
            # 创建graph的副本
            modified_graph = copy.deepcopy(original_graph)
            
            self._apply_modification(modified_graph, modifications)
            
            return modified_graph
            
        except Exception as e:
            self.logger.error(f"解析或应用修改失败: {e}")
            self.logger.error(f"响应内容: {response}")
            self.logger.error(f"原始graph: {original_graph.to_dict()}")
            raise e
    
    def _apply_modification(self, graph: DirectedGraph, modifications: List[Dict[str, Any]]) -> None:
        """
        应用单个修改到graph
        
        Args:
            graph: 要修改的graph
            modification: 修改指令
        """
        for modification in modifications:
            mod_type = modification.get('action')
            reason = modification.get('reason', '未提供原因')
            
            self.logger.info(f"应用修改: {mod_type} - {modification}")
            
            if mod_type == 'add_node':
                node_id = modification.get('node_id')
                attributes = modification.get('attributes', {})
                if node_id:
                    graph.add_node(node_id, **attributes)
                    self.logger.info(f"添加节点: {node_id}")
            
            elif mod_type == 'remove_node':
                node_id = modification.get('node_id')
                if node_id and node_id in graph.nodes:
                    # 移除相关的边
                    edges_to_remove = []
                    for edge_id, edge_data in graph.edges.items():
                        if edge_data['from_node'] == node_id or edge_data['to_node'] == node_id:
                            edges_to_remove.append(edge_id)
                    
                    for edge_id in edges_to_remove:
                        graph.remove_edge(edge_id)
                    
                    # 移除节点
                    del graph.nodes[node_id]
                    self.logger.info(f"移除节点: {node_id}")
            
            elif mod_type == 'add_edge':
                from_node = modification.get('from_node')
                to_node = modification.get('to_node')
                attributes = modification.get('attributes', {})
                if from_node and to_node:
                    edge_id = graph.add_edge(from_node, to_node, **attributes)
                    if edge_id:
                        self.logger.info(f"添加边: {from_node} -> {to_node}")
            
            elif mod_type == 'remove_edge':
                from_node = modification.get('from_node')
                to_node = modification.get('to_node')
                if from_node and to_node:
                    # 找到对应的边
                    for edge_id, edge_data in graph.edges.items():
                        if edge_data['from_node'] == from_node and edge_data['to_node'] == to_node:
                            graph.remove_edge(edge_id)
                            self.logger.info(f"移除边: {from_node} -> {to_node}")
                            break
            
            elif mod_type == 'modify_node':
                node_id = modification.get('node_id')
                attributes = modification.get('attributes', {})
                if node_id and node_id in graph.nodes:
                    # 更新节点属性
                    graph.nodes[node_id]['attributes'].update(attributes)
                    self.logger.info(f"修改节点: {node_id}")

            elif mod_type == 'modify_edge':
                from_node = modification.get('from_node')
                to_node = modification.get('to_node')
                attributes = modification.get('attributes', {})
                if from_node and to_node:
                    # 找到对应的边并更新属性
                    for edge_id, edge_data in graph.edges.items():
                        if edge_data['from_node'] == from_node and edge_data['to_node'] == to_node:
                            edge_data['attributes'].update(attributes)
                            self.logger.info(f"修改边: {from_node} -> {to_node}")
                            break
        
    def get_modified_graph_info(self, graph: DirectedGraph) -> Dict[str, Any]:
        """
        获取修改后graph的信息
        
        Args:
            graph: 修改后的graph
            
        Returns:
            graph信息字典
        """
        return graph.get_graph_info()



def main():
    """
    主函数
    """
    test_agent = CoordinatorAgent(model="o4-mini")

    graph_dict = {'nodes': [{'node_id': 'n1', 'status': 'executed', 'task': 'Find the first place mentioned by name in the Book of Esther (NIV)', 'type': 'search', 'final_response': 'In Esther 1:1 (NIV), India is the first place mentioned by name in the Book of Esther; the original question was specialized to: "In April 1977, who was the Prime Minister of India?"', 'success': True, 'reasoning': ''}, {'node_id': 'n2a', 'status': 'pending', 'task': 'Search for the historical region corresponding to the place identified in n1', 'type': 'search', 'final_response': '', 'success': False, 'reasoning': ''}, {'node_id': 'n2b', 'status': 'pending', 'task': 'Determine the modern country of the historical region found in n2a', 'type': 'solve', 'final_response': '', 'success': False, 'reasoning': ''}, {'node_id': 'n3', 'status': 'pending', 'task': 'Find who was the Prime Minister of the determined country in April 1977', 'type': 'search', 'final_response': '', 'success': False, 'reasoning': ''}, {'node_id': 'task', 'status': 'pending', 'task': 'In April of 1977, who was the Prime Minister of the first place mentioned by name in the Book of Esther (in the New International Version)?', 'type': 'answer', 'final_response': '', 'success': False, 'reasoning': ''}], 'edges': [{'from': 'n1', 'to': 'n2a', 'relationship': 'provides place name'}, {'from': 'n2a', 'to': 'n2b', 'relationship': 'provides historical region'}, {'from': 'n2b', 'to': 'n3', 'relationship': 'provides country for PM search'}, {'from': 'n3', 'to': 'task', 'relationship': 'supplies answer'}]}
    graph = DirectedGraph.from_dict(graph_dict) 
    input_data = {
        "graph" : graph,
        "query":"In April of 1977, who was the Prime Minister of the first place mentioned by name in the Book of Esther (in the New International Version)?"
    }
    
    response = test_agent.execute(input_data)
    print(response.to_dict())
    


if __name__ == "__main__":
    main() 