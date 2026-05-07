import sys
import os
import json
import re
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from models import get_model
from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from utils.graph import DirectedGraph
from datetime import datetime



class SynthesizerAgent(BaseAgent):
    """
    合成器代理，负责基于工作流执行轨迹合成最终答案
    使用两阶段方法：
    1. 生成报告大纲（包含每段内容相关的节点）
    2. 并行地独立写作每个段落
    """
    
    def __init__(self, model: str = "deepseek-r1", config=None, max_workers: int = 10, max_retries: int = 3, **model_kwargs):
        """
        初始化合成器代理
        
        Args:
            model: 使用的模型名称
            config: SynthesizerConfig配置对象
            max_workers: 并行写作的最大线程数
            max_retries: 段落写作失败时的最大重试次数
            **model_kwargs: 传递给模型的参数
        """
        super().__init__(model=model, config=config)
        self.outline_prompt = load_prompt(
            config,
            default_name="OUTLINE_GENERATION_PROMPT"
        )
        self.section_prompt = load_prompt(
            config,
            default_name="SECTION_WRITING_PROMPT"
        )
        self.introduction_prompt = load_prompt(
            config,
            default_name="INTRODUCTION_SECTION_PROMPT"
        )
        self.polishing_prompt = load_prompt(
            config,
            default_name="SECTION_POLISHING_PROMPT"
        )
        self.qa_prompt = load_prompt(
            config,
            default_name="QA_SYNTHESIZER_PROMPT"
        )
        self.model_instance = get_model(model, **model_kwargs)
        self.logger = get_logger("SynthesizerAgent")
        self._set_logger(self.logger)  # 设置logger到基类
        self.model_kwargs = model_kwargs
        self.max_workers = max_workers
        self.max_retries = max_retries

        # 从配置中读取模式和润色设置
        if isinstance(config, dict):
            self.mode = config.get('mode', 'report')
            self.enable_polish = config.get('polish', True)
        elif config:
            self.mode = getattr(config, 'mode', 'report')
            self.enable_polish = getattr(config, 'polish', True)
        else:
            self.mode = 'report'
            self.enable_polish = True
        self.logger.info(f"合成模式: {self.mode}")
        self.logger.info(f"润色功能: {'启用' if self.enable_polish else '禁用'}")

        # 用于存储生成时间戳，保证report使用相同的时间戳
        self.timestamp = None

    
    def execute(self, input_data: Any) -> Any:
        """
        执行合成器的主要逻辑
        
        Args:
            input_data: 包含question和graph_dict的字典
                {
                    "question": str,  # 原始问题
                    "graph": DirectedGraph  # 工作流执行轨迹
                    "node_id": str  # 当前节点id
                    "reference_manager": ReferenceManager  # 参考文献管理器（可选）
                    "timestamp": str  # 时间戳（可选）
                }
            
        Returns:
            合成的最终答案
        """
        if isinstance(input_data, dict):
            question = input_data.get('question')
            graph = input_data.get('graph')
            node_id = input_data.get('node_id')
            reference_manager = input_data.get('reference_manager')
            timestamp = input_data.get('timestamp')  # 获取可选的timestamp
        else:
            raise ValueError("输入数据必须是包含question和graph, node_id的字典")
        
        if not question or not graph:
            raise ValueError("输入数据中必须包含question和graph, node_id")
        
        return self._synthesize_answer(question, graph, node_id, reference_manager, timestamp)
    
    def _synthesize_answer(self, question: str, graph: DirectedGraph, node_id: str, reference_manager=None, timestamp: str = None) -> str:
        """
        基于问题和工作流轨迹合成答案（五阶段方法）
        
        阶段1: 生成报告大纲（包含标题）
        阶段2: 并行写作各个段落
        阶段2.5: 为部分段落生成插图，并把图片 markdown 插入到 section_content
        阶段3: 顺序润色各个段落（引言不润色，其他段落基于前文润色）
        阶段4: 合并段落并添加标题
        阶段5: 参考文献去重并追加参考文献列表
        
        Args:
            question: 原始问题
            graph: 工作流执行轨迹
            node_id: 当前节点ID
            reference_manager: 参考文献管理器（可选）
            timestamp: 时间戳（可选），如果提供则使用外部时间戳，否则自动生成
            
        Returns:
            合成的最终答案
        """
        graph_dict = graph.to_dict()
        dependent_node_ids = graph.get_dependent_nodes(node_id)

        if self.mode == 'qa':
            return self._synthesize_qa(question, graph, node_id, reference_manager)

        # 使用外部时间戳或生成新的时间戳
        if timestamp:
            self.timestamp = timestamp
            self.logger.info(f"使用外部时间戳: {self.timestamp}")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logger.info(f"生成新时间戳: {self.timestamp}")

        try:
            # 阶段1: 生成报告大纲（包含标题）
            self.logger.info("阶段1: 开始生成报告大纲和标题")
            outline_data = self._generate_outline(question, graph_dict, dependent_node_ids)
            report_title = outline_data.get('title', '')
            outline = outline_data.get('outline', [])
            self.logger.info(f"成功生成标题: {report_title}")
            self.logger.info(f"成功生成大纲，共 {len(outline)} 个段落")
            
            # 阶段2: 并行写作各个段落
            self.logger.info("阶段2: 开始并行写作各个段落")
            sections = self._write_sections_parallel(question, graph_dict, outline)
            self.logger.info(f"成功完成 {len(sections)} 个段落的写作")

            # 阶段3: 顺序润色各个段落（引言不润色）- 根据配置决定是否启用
            if self.enable_polish:
                self.logger.info("阶段3: 开始顺序润色各个段落")
                polished_sections = self._polish_sections_sequential(sections)
                self.logger.info(f"成功完成 {len(polished_sections)} 个段落的润色")
            else:
                self.logger.info("阶段3: 润色功能已禁用，跳过润色阶段")
                polished_sections = sections
            
            # 阶段4: 合并所有段落（添加标题）
            merged_report = self._merge_sections(polished_sections, outline, report_title)
            self.logger.info("成功合并所有段落")
            
            # 阶段5: 参考文献去重并追加参考文献列表
            self.logger.info("阶段5: 开始参考文献重新编号和追加过程")
            
            # # 步骤5.1: 去除重复的参考文献引用（保留前2次出现，删除之后的）
            # self.logger.info("步骤5.1: 去除重复的参考文献引用")
            # deduplicated_report = self._deduplicate_references(merged_report)
            # self.logger.info("成功完成参考文献去重")
            
            # 步骤5.2: 提取报告中的参考文献标签并追加到报告尾部
            # self.logger.info("步骤5.2: 追加参考文献列表到报告尾部")
            final_answer_with_refs, cited_references = self._append_references_to_report(merged_report, graph_dict, reference_manager)
            
            # 发送Redis事件（直接使用返回的参考文献列表）
            if cited_references:
                self._send_references_event(cited_references)
            
            return final_answer_with_refs
                
        except Exception as e:
            self.logger.error(f"调用模型合成答案时发生错误: {str(e)}")
            raise RuntimeError(f"合成答案失败: {str(e)}")

    def _synthesize_qa(self, question: str, graph, node_id: str, reference_manager=None) -> str:
        """Single-stage QA synthesis: one prompt call, returns a concise plain-text answer."""
        graph_dict = graph.to_dict()
        dependent_node_ids = graph.get_dependent_nodes(node_id)
        prompt = self.qa_prompt.format(
            question=question,
            graph_dict=json.dumps(graph_dict, ensure_ascii=False, indent=2),
            dependent_node_ids=json.dumps(dependent_node_ids, ensure_ascii=False),
            additional_info=""
        )
        for attempt in range(self.max_retries):
            try:
                response = self.model_instance.generate(prompt)
                result = json.loads(response)
                return result.get("result", response)
            except (json.JSONDecodeError, KeyError):
                match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1)).get("result", response)
                    except json.JSONDecodeError:
                        pass
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"QA synthesis attempt {attempt + 1} failed, retrying...")
        self.logger.warning("QA synthesis: all retries exhausted, returning raw response")
        return response

    def _generate_outline(self, question: str, graph_dict: Dict, dependent_node_ids: List[str]) -> Dict:
        """
        生成报告大纲和标题
        
        Args:
            question: 原始问题
            graph_dict: 工作流图字典
            dependent_node_ids: 依赖节点ID列表
            
        Returns:
            包含标题和大纲的字典：
            {
                "title": str,  # 报告标题
                "outline": [    # 大纲列表
                    {
                        "section_title": str,
                        "description": str,
                        "relevant_node_ids": List[str]
                    },
                    ...
                ]
            }
        """
        prompt = self.outline_prompt.format(
            question=question,
            graph_dict=json.dumps(graph_dict, ensure_ascii=False, indent=2),
            dependent_node_ids=json.dumps(dependent_node_ids, ensure_ascii=False)
        )
        
        try:
            response = self.model_instance.generate(prompt, **self.model_kwargs)
            
            # 解析JSON响应
            result = json.loads(response)
            if "outline" in result and isinstance(result["outline"], list):
                # 兼容旧格式：如果没有title字段，使用问题作为标题
                if "title" not in result:
                    result["title"] = question
                self.logger.info(f"成功生成大纲: {[item for item in result['outline']]}")
                return result
            else:
                self.logger.error("大纲生成响应格式不正确")
                raise ValueError("大纲生成响应格式不正确")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"解析大纲JSON失败: {str(e)}")
            raise ValueError(f"解析大纲JSON失败: {str(e)}")
    
    def _write_sections_parallel(self, question: str, graph_dict: Dict, outline: List[Dict]) -> List[Dict]:
        """
        并行写作各个段落（第一段使用特殊的引言prompt）
        
        Args:
            question: 原始问题
            graph_dict: 工作流图字典（完整图）
            outline: 大纲列表
            
        Returns:
            段落列表，每个元素包含：
            - section_index: 段落索引
            - section_content: 段落内容
        """
        sections = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有写作任务
            future_to_index = {}
            for idx, section_outline in enumerate(outline):
                # 提取该段落相关的节点ID
                relevant_node_ids = section_outline.get('relevant_node_ids', [])
                
                # 从完整图中筛选出相关节点的子图
                subgraph_dict = self._extract_subgraph(graph_dict, relevant_node_ids)

                # self.logger.info(f"子图: {idx}: {subgraph_dict}")
                
                # 第一段使用特殊的引言写作方法，其他段落使用常规方法
                if idx == 0:
                    future = executor.submit(
                        self._write_introduction_section,
                        question,
                        subgraph_dict,
                        section_outline,
                        outline,  # 传入完整大纲
                        idx
                    )
                else:
                    future = executor.submit(
                        self._write_single_section,
                        question,
                        subgraph_dict,
                        section_outline,
                        idx
                    )
                future_to_index[future] = idx
            
            # 收集结果
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    section_data = future.result()
                    sections.append(section_data)
                    self.logger.info(f"完成段落 {idx + 1}/{len(outline)}: {section_data.get('section_title', '')}")
                except Exception as e:
                    self.logger.error(f"写作段落 {idx} 时发生错误: {str(e)}")
                    # 添加错误占位符
                    sections.append({
                        "section_index": idx,
                        "section_title": outline[idx].get("section_title", f"Section {idx}"),
                        "section_content": f"<!-- Error generating this section: {str(e)} -->\n\n",
                        "error": True
                    })
        
        # 按索引排序
        sections.sort(key=lambda x: x["section_index"])
        return sections
    
    def _extract_subgraph(self, graph_dict: Dict, node_ids: List[str]) -> Dict:
        """
        从完整图中提取包含指定节点的子图
        
        Args:
            graph_dict: 完整的工作流图字典
            node_ids: 需要包含的节点ID列表
            
        Returns:
            子图字典，只包含指定的节点
        """
        if not node_ids:
            return {"nodes": [], "edges": []}
        
        # 将node_ids转换为集合以便快速查找
        node_id_set = set(node_ids)
        
        # 筛选节点
        all_nodes = graph_dict.get('nodes', [])
        filtered_nodes = [node for node in all_nodes if node.get('node_id') in node_id_set]
        
        # 筛选边：只保留两端节点都在node_id_set中的边
        all_edges = graph_dict.get('edges', [])
        filtered_edges = [
            edge for edge in all_edges 
            if edge.get('from') in node_id_set and edge.get('to') in node_id_set
        ]
        
        subgraph = {
            "nodes": filtered_nodes,
            "edges": filtered_edges
        }
        
        self.logger.debug(f"提取子图: {len(filtered_nodes)} 个节点, {len(filtered_edges)} 条边")
        
        return subgraph
    
    def _write_single_section(self, question: str, graph_dict: Dict, section_outline: Dict, section_index: int) -> Dict:
        """
        写作单个段落（带重试机制）
        
        Args:
            question: 原始问题
            graph_dict: 工作流子图字典（只包含该段落相关的节点）
            section_outline: 段落大纲
            section_index: 段落索引
            
        Returns:
            段落数据字典
        """
        prompt = self.section_prompt.format(
            question=question,
            section_outline=json.dumps(section_outline, ensure_ascii=False, indent=2),
            graph_dict=json.dumps(graph_dict, ensure_ascii=False, indent=2)
        )
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.model_instance.generate(prompt, **self.model_kwargs)
                
                # 解析JSON响应
                result = json.loads(response)
                if "section_content" in result:
                    if attempt > 0:
                        self.logger.info(f"段落 {section_index} 在第 {attempt + 1} 次尝试后成功")
                    return {
                        "section_index": section_index,
                        "section_title": section_outline.get("section_title", f"Section {section_index}"),
                        "section_content": result["section_content"],
                        "error": False
                    }
                else:
                    raise ValueError("段落响应格式不正确，缺少section_content字段")
                    
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                self.logger.warning(f"段落 {section_index} 第 {attempt + 1}/{self.max_retries} 次尝试失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"正在重试段落 {section_index}...")
                    continue
                else:
                    self.logger.error(f"段落 {section_index} 在 {self.max_retries} 次尝试后仍然失败")
                    raise ValueError(f"解析段落JSON失败（已重试{self.max_retries}次）: {str(last_error)}")
    
    def _polish_sections_sequential(self, sections: List[Dict]) -> List[Dict]:
        """
        顺序润色各个段落（引言不润色，其他段落基于前文润色）
        
        Args:
            sections: 原始段落列表（已按索引排序）
            
        Returns:
            润色后的段落列表
        """
        polished_sections = []
        
        if not sections:
            return polished_sections
        
        # 第一段（引言）不润色，直接加入
        introduction = sections[0]
        polished_sections.append(introduction)
        self.logger.info(f"段落 0 (引言) 不需要润色，直接使用")
        
        # 从第二段开始，依次润色
        for idx in range(1, len(sections)):
            current_section = sections[idx]
            
            try:
                # 润色当前段落（基于引言和之前已润色的段落）
                polished_section = self._polish_single_section(
                    introduction_content=introduction.get('section_content', ''),
                    previous_sections=polished_sections[1:],  # 不包括引言
                    current_section=current_section
                )
                polished_sections.append(polished_section)
                
                self.logger.info(f"完成段落 {idx} 的润色")
                
            except Exception as e:
                self.logger.error(f"润色段落 {idx} 时发生错误: {str(e)}")
                # 出错时使用原始段落
                polished_sections.append(current_section)
                self.logger.warning(f"段落 {idx} 润色失败，使用原始内容")
        
        return polished_sections
    
    def _polish_single_section(self, introduction_content: str, previous_sections: List[Dict], 
                               current_section: Dict) -> Dict:
        """
        润色单个段落
        
        Args:
            introduction_content: 引言内容
            previous_sections: 之前已润色的段落列表（不包括引言）
            current_section: 当前需要润色的段落
            
        Returns:
            润色后的段落数据字典
        """
        # 构建之前段落的文本
        previous_sections_text = ""
        for prev_section in previous_sections:
            prev_content = prev_section.get('section_content', '')
            if prev_content:
                previous_sections_text += prev_content + "\n\n"
        
        # 如果没有之前的段落，只使用空字符串
        if not previous_sections_text:
            previous_sections_text = "(No previous sections yet)"
        
        current_section_content = current_section.get('section_content', '')
        
        prompt = self.polishing_prompt.format(
            introduction_content=introduction_content,
            previous_sections=previous_sections_text,
            current_section=current_section_content
        )
        
        last_error = None
        section_idx = current_section.get("section_index", "unknown")
        
        for attempt in range(self.max_retries):
            try:
                response = self.model_instance.generate(prompt, **self.model_kwargs)
                
                # 解析JSON响应
                result = json.loads(response)
                if "polished_section_content" in result:
                    if attempt > 0:
                        self.logger.info(f"段落 {section_idx} 润色在第 {attempt + 1} 次尝试后成功")
                    # self.logger.info(f"润色前段落: {current_section_content}")
                    # self.logger.info(f"润色后段落: {result['polished_section_content']}")
                    return {
                        "section_index": current_section.get("section_index"),
                        "section_title": current_section.get("section_title"),
                        "section_content": result["polished_section_content"],
                        "error": False
                    }
                else:
                    raise ValueError("润色响应格式不正确，缺少polished_section_content字段")
                    
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                self.logger.warning(f"段落 {section_idx} 润色第 {attempt + 1}/{self.max_retries} 次尝试失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"正在重试段落 {section_idx} 的润色...")
                    continue
                else:
                    self.logger.error(f"段落 {section_idx} 润色在 {self.max_retries} 次尝试后仍然失败")
                    raise ValueError(f"解析润色段落JSON失败（已重试{self.max_retries}次）: {str(last_error)}")
    
    def _write_introduction_section(self, question: str, graph_dict: Dict, section_outline: Dict, 
                                        full_outline: List[Dict], section_index: int) -> Dict:
            """
            写作引言/背景段落（使用特殊的引言prompt，带重试机制）
            
            Args:
                question: 原始问题
                graph_dict: 工作流子图字典（只包含该段落相关的节点）
                section_outline: 段落大纲
                full_outline: 完整报告大纲（用于提供上下文）
                section_index: 段落索引（应该是0）
                
            Returns:
                段落数据字典
            """
            prompt = self.introduction_prompt.format(
                question=question,
                section_outline=json.dumps(section_outline, ensure_ascii=False, indent=2),
                full_outline=json.dumps(full_outline, ensure_ascii=False, indent=2),
                graph_dict=json.dumps(graph_dict, ensure_ascii=False, indent=2)
            )
            
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = self.model_instance.generate(prompt, **self.model_kwargs)
                    
                    # 解析JSON响应
                    result = json.loads(response)
                    if "section_content" in result:
                        if attempt > 0:
                            self.logger.info(f"引言段落在第 {attempt + 1} 次尝试后成功")
                        return {
                            "section_index": section_index,
                            "section_title": section_outline.get("section_title", f"Section {section_index}"),
                            "section_content": result["section_content"],
                            "error": False
                        }
                    else:
                        raise ValueError("引言段落响应格式不正确，缺少section_content字段")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    self.logger.warning(f"引言段落第 {attempt + 1}/{self.max_retries} 次尝试失败: {str(e)}")
                    if attempt < self.max_retries - 1:
                        self.logger.info(f"正在重试引言段落...")
                        continue
                    else:
                        self.logger.error(f"引言段落在 {self.max_retries} 次尝试后仍然失败")
                        raise ValueError(f"解析引言段落JSON失败（已重试{self.max_retries}次）: {str(last_error)}")

    def _merge_sections(self, sections: List[Dict], outline: List[Dict], title: str = "") -> str:
        """
        合并所有段落成完整报告（在开头添加标题）
        
        Args:
            sections: 段落列表
            outline: 原始大纲（用于验证）
            title: 报告标题
            
        Returns:
            完整的报告文本
        """
        report_parts = []
        
        # 如果有标题，添加到最开头
        if title and title.strip():
            report_parts.append(f"# {title.strip()}")
        
        for section in sections:
            content = section.get("section_content", "")
            if content:
                # 确保段落之间有适当的间隔
                report_parts.append(content.strip())
        
        # 用双换行符连接所有段落
        final_report = "\n\n".join(report_parts)
        
        return final_report
    
    # def _deduplicate_references(self, report_text: str, max_occurrences: int = 2) -> str:
    #     """
    #     去除报告中重复的参考文献引用
    #     当一个参考文献出现超过max_occurrences次后，删除之后的重复引用
        
    #     Args:
    #         report_text: 报告文本
    #         max_occurrences: 每个参考文献允许出现的最大次数（默认2次）
            
    #     Returns:
    #         去重后的报告文本
    #     """
    #     try:
    #         # 1. 找到所有引用标签及其位置
    #         citation_pattern = r'\[\[(\d+)\]\]'
    #         matches = list(re.finditer(citation_pattern, report_text))
            
    #         if not matches:
    #             self.logger.info("报告中未发现任何参考文献标签，无需去重")
    #             return report_text
            
    #         self.logger.info(f"报告中共发现 {len(matches)} 个引用标签")
            
    #         # 2. 统计每个标签的出现次数和位置
    #         tag_occurrences = {}  # {tag: [match1, match2, ...]}
    #         for match in matches:
    #             tag = match.group(1)
    #             if tag not in tag_occurrences:
    #                 tag_occurrences[tag] = []
    #             tag_occurrences[tag].append(match)
            
    #         # 3. 找出需要删除的引用（超过max_occurrences次的）
    #         matches_to_remove = []
    #         for tag, occurrence_list in tag_occurrences.items():
    #             if len(occurrence_list) > max_occurrences:
    #                 # 保留前max_occurrences次出现，删除之后的
    #                 matches_to_remove.extend(occurrence_list[max_occurrences:])
    #                 self.logger.info(f"标签 [[{tag}]] 出现 {len(occurrence_list)} 次，将删除后 {len(occurrence_list) - max_occurrences} 次")
            
    #         if not matches_to_remove:
    #             self.logger.info("所有引用标签出现次数均未超过限制，无需去重")
    #             return report_text
            
    #         # 4. 按位置从后往前删除（避免位置偏移）
    #         matches_to_remove.sort(key=lambda m: m.start(), reverse=True)
            
    #         result_text = report_text
    #         for match in matches_to_remove:
    #             # 删除该引用标签
    #             start = match.start()
    #             end = match.end()
    #             result_text = result_text[:start] + result_text[end:]
            
    #         self.logger.info(f"成功删除 {len(matches_to_remove)} 个重复的引用标签")
    #         return result_text
            
    #     except Exception as e:
    #         self.logger.error(f"去重参考文献时发生错误: {str(e)}")
    #         return report_text  # 出错时返回原始报告
    
    def _append_references_to_report(self, report_text: str, graph_dict: Dict, reference_manager=None) -> tuple:
        """
        从报告中提取参考文献标签，并将完整的参考文献列表追加到报告尾部
        
        Args:
            report_text: 生成的报告文本
            graph_dict: 工作流图字典，包含所有节点信息
            reference_manager: ReferenceManager实例（可选）
            
        Returns:
            tuple: (追加了参考文献列表的报告文本, 引用的参考文献列表)
        """
        try:

            # self.logger.info(f"原始报告: {report_text}")
            # 1. 从报告中提取所有引用标签 [[1]], [[2]], [[3]] 等，保持首次出现的顺序
            citation_pattern = r'\[\[(\d+)\]\]'
            all_matches = re.findall(citation_pattern, report_text)
            
            # 使用字典保持顺序并去重（Python 3.7+字典保持插入顺序）
            cited_tags_ordered = list(dict.fromkeys(all_matches))
            
            if not cited_tags_ordered:
                self.logger.info("报告中未发现任何参考文献标签，不添加参考文献部分")
                return report_text, []
            
            self.logger.info(f"从报告中提取到 {len(cited_tags_ordered)} 个引用标签（按首次出现顺序）: {cited_tags_ordered}")
            
            # 2. 优先从ReferenceManager中获取文献信息，否则从graph_dict中收集
            if reference_manager is not None:
                self.logger.info("从ReferenceManager中获取文献信息")
                all_references_list = reference_manager.get_reference_list()
                # 转换为字典格式，key为tag
                all_references = {ref['tag']: ref for ref in all_references_list}
                self.logger.info(f"从ReferenceManager中获取到 {len(all_references)} 条文献信息")
            else:
                self.logger.info("从graph_dict中收集文献信息")
                all_references = self._collect_references_from_graph(graph_dict)
                self.logger.info(f"从graph中收集到 {len(all_references)} 条文献信息")
            
            if not all_references:
                self.logger.warning("未能找到任何文献信息，不添加参考文献部分")
                return report_text, []
            
            # 3. 筛选出报告中引用的文献，按首次出现顺序排列
            cited_references = []
            for tag in cited_tags_ordered:
                if tag in all_references:
                    ref_info = all_references[tag]
                    cited_references.append({
                        "old_tag": tag,  # 保存原始标签
                        "url": ref_info.get('url', ''),
                        "title": ref_info.get('title', ''),
                        "type": ref_info.get('type', 'unknown')
                    })
                else:
                    self.logger.warning(f"标签 {tag} 在文献列表中未找到对应信息")
            
            if not cited_references:
                self.logger.warning("未能匹配到任何引用的文献，不添加参考文献部分")
                return report_text, []
            
            self.logger.info(f"成功匹配 {len(cited_references)} 条引用文献")
            
            # 4. 重新编号:创建旧标签到新标签的映射
            tag_mapping = {}
            for new_idx, ref in enumerate(cited_references, start=1):
                old_tag = ref['old_tag']
                new_tag = str(new_idx)
                tag_mapping[old_tag] = new_tag
                ref['tag'] = new_tag  # 更新为新标签
            
            self.logger.info(f"标签映射: {tag_mapping}")
            
            # 5. 替换报告中的所有旧标签为新标签
            # 使用两步替换法避免标签冲突：
            # 步骤1: 先将所有旧标签替换为临时占位符 [[TEMP_1]], [[TEMP_2]], ...
            temp_report = report_text
            for old_tag, new_tag in tag_mapping.items():
                old_pattern = r'\[\[' + re.escape(old_tag) + r'\]\]'
                temp_placeholder = f'[[TEMP_{new_tag}]]'
                temp_report = re.sub(old_pattern, temp_placeholder, temp_report)
            
            # 步骤2: 将所有临时占位符替换为最终的新标签
            final_report = temp_report
            for old_tag, new_tag in tag_mapping.items():
                temp_pattern = r'\[\[TEMP_' + re.escape(new_tag) + r'\]\]'
                new_citation = f'[[{new_tag}]]'
                final_report = re.sub(temp_pattern, new_citation, final_report)
            
            report_text = final_report
            self.logger.info("成功更新报告中的所有引用标签")
            
            # 6. 构建参考文献部分
            references_section = self._build_references_section(cited_references)
            
            # 7. 检查报告是否已经包含参考文献部分
            if '## References' in report_text or '## 参考文献' in report_text:
                self.logger.info("报告中已存在参考文献部分，将替换为新的参考文献列表")
                # 移除现有的参考文献部分
                report_text = re.sub(r'\n## References.*$', '', report_text, flags=re.DOTALL)
                report_text = re.sub(r'\n## 参考文献.*$', '', report_text, flags=re.DOTALL)
            
            # 8. 追加参考文献到报告尾部
            final_report = report_text.rstrip() + '\n\n' + references_section
            self.logger.info("成功将参考文献追加到报告尾部")
            
            return final_report, cited_references
            
        except Exception as e:
            self.logger.error(f"追加参考文献到报告时发生错误: {str(e)}")
            return report_text, []  # 出错时返回原始报告和空列表
    
    def _build_references_section(self, references: List[Dict]) -> str:
        """
        构建参考文献部分的文本
        
        Args:
            references: 参考文献列表
            
        Returns:
            格式化的参考文献文本
        """
        lines = ["## References", ""]
        
        for ref in references:
            tag = ref['tag']
            title = ref['title']
            url = ref['url']
            ref_type = ref.get('type', 'unknown')
            
            # 根据类型选择图标
            icon = '📄' if ref_type == 'paper' else '🌐'
            
            # 格式: [[1]] 📄 Title - URL
            if title and url:
                lines.append(f"[[{tag}]] {icon} {title} - {url}")
            elif url:
                lines.append(f"[[{tag}]] {icon} {url}")
            else:
                lines.append(f"[[{tag}]] {icon} (无标题)")
            
            # 在每个参考文献后添加空行，使其在Markdown中换行显示
            lines.append("")
        
        return '\n'.join(lines)
    
    def _collect_references_from_graph(self, graph_dict: Dict) -> Dict[str, Dict]:
        """
        从graph字典中收集所有文献信息
        
        Args:
            graph_dict: 工作流图字典
            
        Returns:
            字典，key为标签(如'1', '2')，value为文献信息字典
        """
        references = {}
        
        try:
            # 遍历所有节点
            nodes = graph_dict.get('nodes', [])
            for node in nodes:
                # 检查节点的各个字段中是否包含文献信息
                # 文献信息可能存在于 final_response, summary, subtask_traces 等字段
                
                # 尝试从节点的 references 字段获取（如果存在）
                if 'references' in node:
                    node_refs = node['references']
                    if isinstance(node_refs, list):
                        for ref in node_refs:
                            if isinstance(ref, dict) and 'tag' in ref:
                                tag = ref['tag']
                                references[tag] = {
                                    'url': ref.get('url', ''),
                                    'title': ref.get('title', ''),
                                    'type': ref.get('type', 'unknown')
                                }
                
                # 也可以从节点的文本内容中提取文献信息
                # 查找类似 "[R1] url - title" 的格式
                text_fields = ['final_response', 'summary', 'final_answer']
                for field in text_fields:
                    if field in node and node[field]:
                        text = str(node[field])
                        # 查找 ## References 部分
                        if '## References' in text:
                            ref_section = text.split('## References', 1)[1]
                            # 匹配格式: [[1]] 📄 title - url 或 [[1]] title - url
                            ref_pattern = r'\[\[(\d+)\]\]\s*(?:📄|🌐)?\s*(.+?)\s*-\s*(https?://\S+)'
                            matches = re.findall(ref_pattern, ref_section)
                            for tag, title, url in matches:
                                if tag not in references:
                                    references[tag] = {
                                        'url': url.strip(),
                                        'title': title.strip(),
                                        'type': 'paper' if '📄' in ref_section else 'unknown'
                                    }
            
            return references
            
        except Exception as e:
            self.logger.error(f"从graph收集文献信息时发生错误: {str(e)}")
            return {}
    
    def _send_references_event(self, references: List[Dict]) -> None:
        """
        通过Redis事件发送参考文献列表
        
        Args:
            references: 参考文献列表
        """
        try:
            event_data = {
                "references": references,
            }
            
            success = self.send_redis_event("get_cited_references", event_data)
            
            if success:
                self.logger.info(f"成功发送 {len(references)} 条引用文献到Redis")
            else:
                self.logger.warning("发送引用文献到Redis失败（可能未配置Redis连接）")
                
        except Exception as e:
            self.logger.error(f"发送Redis事件时发生错误: {str(e)}")
    