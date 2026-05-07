import sys
import os
import re
import json
from typing import Dict, Any, List, Optional
from xml.etree import ElementTree

# Add paths to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import BaseAgent
from agents.task.planner_agent import PlannerAgent
from agents.task.execution_agent import ExecutionAgent
from models import get_model
from utils.logger import get_logger, get_node_logger
from utils.prompt_loader import load_prompt

# 使用utils logger
logger = get_logger("task_workflow")


class TaskWorkflow(BaseAgent):
    """
    任务工作流，协调规划器和执行器
    """
    
    def __init__(self, model = "deepseek-r1", tools: List[Dict[str, Any]] = None, tool_mode: str = "local", config=None, summarizer_model=None, reference_manager=None, **model_kwargs):
        """
        初始化任务工作流
        
        Args:
            model: 使用的模型名称
            tools: 可用的工具列表
            tool_mode: 工具模式，'test'使用测试工具，其他值使用本地工具
            config: TaskWorkflowConfig配置对象
            summarizer_model: 用于总结的模型名称（如果为None则使用model）
            reference_manager: 参考文献管理器（如果为None则创建新的）
            **model_kwargs: 传递给模型的参数
        """
        logger.info(f"初始化任务工作流，模型: {model}, 总结模型: {summarizer_model}, 工具模式: {tool_mode}")
        
        # 调用基类初始化
        super().__init__(model=model, tools=tools, config=config)
        self._set_logger(logger)  # 设置logger到基类
        
        self.tool_mode = tool_mode
        self.model_kwargs = model_kwargs
        self.node_id = None  # 节点ID
        self.config = config or {}
        
        # 从配置加载task summary prompt（如果配置中没有指定，使用默认值）
        self.task_summary_prompt = load_prompt(
            self.config,
            default_name="TASK_SUMMARY_PROMPT"
        )
        
        # 初始化代理
        logger.info("初始化planner agent")
        self.planner_agent = PlannerAgent(model=model, tools=tools, tool_mode=tool_mode, config=config.get('planner', {}), **model_kwargs)
        logger.info("初始化execution agent")
        self.execution_agent = ExecutionAgent(model=model, tools=tools, tool_mode=tool_mode, config=config.get('execution', {}), summarizer_model=summarizer_model, reference_manager=reference_manager, **model_kwargs)

        self.summarizer_model = summarizer_model
        self.reference_manager = reference_manager
        
        logger.info("任务工作流初始化完成")
    
    def set_node_id(self, node_id: str):
        """
        设置节点ID，并切换到节点特定的logger
        
        Args:
            node_id: 节点ID
        """
        self.node_id = node_id
        
        # 切换到节点特定的logger（task_id可选）
        if node_id:
            self.node_logger = get_node_logger(self.task_id, node_id)
            self._set_logger(self.node_logger)  # 更新基类的logger
            
            # 同时更新子agent的logger和task_id
            self.planner_agent._set_logger(self.node_logger)
            self.execution_agent._set_logger(self.node_logger)
            
            # 确保子agents也有task_id（用于发送Redis事件）
            if self.task_id:
                self.planner_agent.set_task_id(self.task_id)
                self.execution_agent.set_task_id(self.task_id)
            
            task_info = f"task_id={self.task_id}" if self.task_id else "no task_id"
            self.node_logger.info(f"TaskWorkflow set node_id: {node_id} ({task_info}), switched to node-specific logger")
        else:
            logger.info(f"TaskWorkflow set node_id: {node_id}")
    
    def execute(self, task_input: Dict[str, Any], use_planner: bool = True) -> Dict[str, Any]:
        """
        执行完整的工作流
        
        Args:
            task_input: 包含以下键的字典：
                - task: 任务内容
                - knowledge_info: 知识信息（可选）
                - file_path: 文件路径（可选）
                - query: 查询内容（可选）
                - summary_type: 总结类型，'search'或'answer'（可选）
            use_planner: 是否使用规划器
                
        Returns:
            工作流执行结果
        """
        # 使用节点特定的logger（如果已设置），否则使用全局logger
        active_logger = getattr(self, 'node_logger', logger)
        
        try:
            active_logger.info("开始执行任务工作流")
            active_logger.info(f"任务内容: {task_input.get('task', '')}")

            if use_planner:
                # 步骤1: 使用规划器分解任务
                subtasks = self._plan_subtasks(task_input)

                # 发送子任务规划结果到Redis
                self.send_redis_event("get_subtasks", {
                    "task_id": self.task_id,
                    "subtasks": subtasks,
                    "node_id": self.node_id
                })
                
                # 步骤2: 依次执行每个子任务
                results = []
                task = task_input.get("task", "")

                
                for i, subtask in enumerate(subtasks):
                    active_logger.info(f"执行子任务 {i+1}/{len(subtasks)}: {subtask}")
                    
                    # 构建历史子任务列表和结果
                    history_subtasks = []
                    for j, subtask_desc in enumerate(subtasks):
                        if j < i:
                            # 已完成的子任务
                            prev_result = results[j]
                            history_subtasks.append({
                                "subtask": subtask_desc,
                                "completed": True,
                                "summary": prev_result.get("summary", "执行成功"),
                                "success": prev_result.get("success", False)
                            })
                        elif j == i:
                            # 当前子任务
                            history_subtasks.append({
                                "subtask": subtask_desc,
                                "completed": False,
                                "summary": "正在执行"
                            })
                        else:
                            # 未完成的子任务
                            history_subtasks.append({
                                "subtask": subtask_desc,
                                "completed": False,
                                "summary": "未开始"
                            })
                    
                    # 构建上下文信息
                    context = task_input.get("context", {})
                    context.update({
                        "history_subtasks": history_subtasks,
                        "task": task,  # 传入总体任务
                        "knowledge_info": task_input.get("knowledge_info", ""),
                        "file_path": task_input.get("file_path", ""),
                        "node_id": self.node_id,  # 传入节点ID
                        "subtask_id": i + 1  # 传入子任务ID
                    })
                    
                    result = self.execution_agent.execute(subtask = subtask, context = context, query = task_input['query'])
                    # 发送子任务执行结果到Redis
                    self.send_redis_event("get_subtask_result", {
                        "task_id": self.task_id,
                        "node_id": self.node_id,
                        "subtask_id": i + 1,
                        "subtask": subtask,
                        "result": {"success": result.get("success", False), "summary": result.get("summary", "")}
                    })
                    results.append(result)
                    active_logger.info(f"子任务 {i+1} 执行完成: {result.get('success', False)}")
            # else:
            #     result = self.execution_agent.execute(task_input)
            #     results.append(result)
            #     active_logger.info(f"任务执行完成: {result.get('success', False)}")
            
            # 步骤3: 汇总结果
            active_logger.info("任务结果总结")
            final_result = self._summarize_results(task_input, subtasks, results, task_input.get("summary_type", "search"))
            active_logger.info(f"任务总结结果: {final_result}")

            # 发送任务总结结果到Redis
            self.send_redis_event("get_node_result", {
                "task_id": self.task_id,
                "node_id": self.node_id,
                "result": {"success": final_result.get("success", False), "summary": final_result.get("final_answer", ""), "reasoning": final_result.get("reasoning", "")}
            })
            
            active_logger.info("任务工作流执行完成")
            return final_result
            
        except Exception as e:
            active_logger.error(f"任务工作流执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "subtasks": [],
                "results": []
            }
    
    def _plan_subtasks(self, task_input: Dict[str, Any]) -> List[str]:
        """
        使用规划器分解任务为子任务
        
        Args:
            task_input: 任务输入
            
        Returns:
            子任务列表
        """
        active_logger = getattr(self, 'node_logger', logger)
        
        try:
            active_logger.info("开始规划子任务")
            # 调用规划器
            planner_result = self.planner_agent.execute(task_input)
            active_logger.info(f"规划器返回结果: {planner_result}")
            
            # 解析XML格式的子任务
            subtasks = self._parse_subtasks_xml(planner_result)
            
            if not subtasks:
                # 如果解析失败，使用默认子任务
                subtasks = ["分析任务需求", "执行任务", "生成结果"]
                active_logger.warning("XML解析失败，使用默认子任务")
            
            return subtasks
            
        except Exception as e:
            active_logger.error(f"规划器执行失败: {e}")
            # 返回默认子任务
            return ["分析任务需求", "执行任务", "生成结果"]
    
    def _parse_subtasks_xml(self, xml_content: str) -> List[str]:
        """
        解析XML格式的子任务
        
        Args:
            xml_content: XML格式的子任务内容
            
        Returns:
            子任务列表
        """
        active_logger = getattr(self, 'node_logger', logger)
        
        try:
            # 清理XML内容
            xml_content = xml_content.strip()
            
            # 尝试解析XML
            root = ElementTree.fromstring(xml_content)
            
            # 查找所有task标签
            tasks = []
            for task_elem in root.findall('.//task'):
                task_text = task_elem.text.strip() if task_elem.text else ""
                if task_text:
                    tasks.append(task_text)
            
            active_logger.info(f"XML解析成功，找到 {len(tasks)} 个子任务")
            return tasks
            
        except ElementTree.ParseError as e:
            active_logger.warning(f"XML解析错误: {e}")
            active_logger.info("尝试使用正则表达式提取子任务")
            # 尝试使用正则表达式提取
            return self._extract_subtasks_regex(xml_content)
        except Exception as e:
            active_logger.warning(f"子任务解析失败: {e}")
            return []
    
    def _extract_subtasks_regex(self, content: str) -> List[str]:
        """
        使用正则表达式提取子任务
        
        Args:
            content: 包含子任务的内容
            
        Returns:
            子任务列表
        """
        active_logger = getattr(self, 'node_logger', logger)
        
        try:
            # 匹配 <task>...</task> 格式
            pattern = r'<task>(.*?)</task>'
            matches = re.findall(pattern, content, re.DOTALL)
            
            # 清理匹配结果
            subtasks = []
            for match in matches:
                task = match.strip()
                if task:
                    subtasks.append(task)
            
            active_logger.info(f"正则表达式提取成功，找到 {len(subtasks)} 个子任务")
            return subtasks
            
        except Exception as e:
            active_logger.warning(f"正则表达式提取失败: {e}")
            return []
    

    
    def _summarize_results(self, task_input: Dict[str, Any], subtasks: List[str], results: List[Dict[str, Any]], summary_type: str = "search") -> Dict[str, Any]:
        """
        汇总所有子任务的结果
        
        Args:
            task_input: 原始任务输入
            subtasks: 子任务列表
            results: 子任务结果列表
            
        Returns:
            汇总结果
        """
        active_logger = getattr(self, 'node_logger', logger)
        
        try:
            active_logger.info("开始汇总子任务结果")
            
            # 统计信息
            total_subtasks = len(subtasks)
            successful_subtasks = sum(1 for r in results if r.get("success", False))
            total_tool_calls = sum(r.get("tool_calls", 0) for r in results)
            
            active_logger.info(f"总子任务数: {total_subtasks}, 成功: {successful_subtasks}, 工具调用总数: {total_tool_calls}")
            
            # 构建汇总提示

            subtask_trace = ""

            for i, (subtask, result) in enumerate(zip(subtasks, results)):
                subtask_trace += f"""
            Subtask {i+1}: {subtask}
            Status: {"Success" if result.get('success') else "Failure"}
            Summary: {result.get("summary","No summary")}
            """
                

            summary_prompt = self.task_summary_prompt.format(
            task=task_input.get("task", ""),
            subtask_trace=subtask_trace
            )
            # 调用summarizer模型生成最终答案
            active_logger.info(f"使用模型 {self.summarizer_model} 生成任务总结")
            model_instance = get_model(self.summarizer_model, **self.model_kwargs)
            final_answer_text = model_instance.generate(summary_prompt)
            
            # print(summary_prompt)
            # print(final_answer_text)
            # import sys;sys.exit()
            # 尝试解析JSON响应
            try:
                # 尝试直接解析JSON
                final_answer = json.loads(final_answer_text)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试提取JSON部分
                active_logger.warning("直接JSON解析失败，尝试提取JSON部分")
                import re
                json_match = re.search(r'\{.*\}', final_answer_text, re.DOTALL)
                if json_match:
                    try:
                        final_answer = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        active_logger.error("JSON提取和解析都失败")
                        final_answer = {"success": False, "result": "JSON解析失败"}
                else:
                    active_logger.error("未找到JSON格式内容")
                    final_answer = {"success": False, "result": "未找到有效的JSON响应"}
            
            active_logger.info(f"任务结果总结完成: {task_input.get('task', '')} {final_answer}")
            
            return {
                "completed": True,
                "success": final_answer.get("success", True),
                "final_answer": final_answer.get("summary", ""),
                "reasoning": final_answer.get("reasoning", ""),
                "task": task_input.get("task", ""),
                # "total_tool_calls": total_tool_calls,
                # "subtasks": subtasks,
                # "subtask_results": results,
                # "subtask_trace": subtask_trace,
                # "summary": final_answer.get("summary", "")
                
            }
            
        except Exception as e:
            active_logger.error(f"汇总结果失败: {e}")
            return {
                "completed": False,
                "success": False,
                "error": f"汇总结果时出现错误: {str(e)}",
                "task": task_input.get("task", ""),
                "subtasks": subtasks,
                "subtask_results": results,
                "subtask_trace": subtask_trace
            }


def create_task_workflow(model: str = "deepseek-r1", tools: List[Dict[str, Any]] = None, tool_mode: str = "local", **model_kwargs) -> TaskWorkflow:
    """
    创建任务工作流实例
    
    Args:
        model: 使用的模型名称
        tools: 可用的工具列表
        tool_mode: 工具模式，'test'使用测试工具，其他值使用本地工具
        **model_kwargs: 传递给模型的参数
        
    Returns:
        任务工作流实例
    """
    logger.info(f"创建任务工作流，模型: {model}, 总结模型: {summarizer_model}, 工具模式: {tool_mode}")
    return TaskWorkflow(model=model, tools=tools, tool_mode=tool_mode, summarizer_model=summarizer_model, **model_kwargs)


# 使用示例
if __name__ == "__main__":
    # 创建示例工具配置
    example_tools = [
        {
            "name": "web_search",
            "description": "搜索网络信息",
            "required_parameters": ["query"],
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                }
            }
        },
        {
            "name": "calculator",
            "description": "执行数学计算",
            "required_parameters": ["expression"],
            "parameters": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            }
        }
    ]
    
    # 创建工作流
    workflow = create_task_workflow(
        model="deepseek-v3",
        tools=example_tools
    )
 