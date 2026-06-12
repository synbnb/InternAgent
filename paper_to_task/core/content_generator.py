"""
内容生成模块 - 使用LLM生成所有内容
"""

from typing import Dict, List, Any


class SciTaskGenerator:
    """sci任务生成器 - 完全基于LLM"""

    def __init__(self):
        pass

    def generate_task_info(self, research_info: Dict[str, Any],
                          paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成task_info.json（包含论文来源引用）

        Args:
            research_info: 提取的研究信息（含 _paper_sources）
            paper_metadata: 论文元数据

        Returns:
            task_info字典（每个数据项含 _source 字段）
        """
        # 直接从research_info中获取信息
        title = research_info.get('title', '相关论文')
        datasets = research_info.get('datasets', [])
        sources = research_info.get('_paper_sources', {})

        # 从 sources 中构建数据集引用
        dataset_sources_map = {}
        if 'datasets' in sources and isinstance(sources['datasets'], list):
            for ds_src in sources['datasets']:
                dataset_sources_map[ds_src['name']] = ds_src['source']

        # 构建任务描述及其来源
        task_desc = f"复现论文《{title}》的核心发现"
        task_source = sources.get('title', '')

        return {
            "task": task_desc,
            "task_source": task_source,
            "data": [
                {
                    "name": dataset.get('name', f"dataset_{i+1}"),
                    "path": f"data/{dataset.get('name', f'dataset_{i+1}')}",
                    "description": dataset.get('description', ''),
                    "_source": dataset_sources_map.get(dataset.get('name', ''), dataset.get('source', '该数据集由系统从论文中识别'))
                }
                for i, dataset in enumerate(datasets)
            ]
        }

    def generate_checklist(self, research_info: Dict[str, Any],
                           llm_client=None) -> List[Dict]:
        """
        使用LLM生成针对性的checklist.json，包含论文原文依据

        Args:
            research_info: 研究信息
            llm_client: LLM客户端（需要传入）

        Returns:
            checklist列表，每项含 _source 引用原文
        """
        if not llm_client:
            # 如果没有LLM客户端，使用基础模板
            return self._generate_basic_checklist(research_info)

        # 使用LLM生成针对性checklist
        prompt = f"""基于以下研究信息，生成具体的评分checklist。**每项必须提供 `_source` 字段**，从下面的研究信息中摘录支持该评分项的原始依据。

!!! 重要规则 !!!
1. 原文摘录必须**保留原文语言**（英文论文→英文摘录，中文论文→中文摘录）。**禁止翻译！**
2. 每个摘录必须是**完整段落或至少3-5句连贯内容**，不能只截取一句话。
3. 每个摘录前面必须标注**出自论文的哪个部分/章节**，格式为 `【章节名】`。

研究信息：
标题：{research_info.get('title', '')}
研究目标：{research_info.get('research_goal', '')}
方法：{research_info.get('methods', {}).get('main_methods', '')}
数据集：{', '.join([ds.get('name', '') for ds in research_info.get('datasets', [])])}
预期结果：{'; '.join(research_info.get('key_findings', [])[:3])}

请生成5个具体的评分项，要求：
1. 每个评分项都要针对这个具体研究
2. 权重要合理分配（总和为1.0）
3. 内容要具体，不要用通用套话
4. **每项必须提供 `_source` 字段**，格式为 `【章节名】原文完整段落（保留原文语言，至少3-5句话）`

返回JSON格式：
{{
  "items": [
    {{
      "type": "text",
      "weight": 0.25,
      "content": "针对具体研究的评分内容（中文）",
      "_source": "【Section X】原文完整段落（保留原始语言，至少3-5句话）"
    }}
  ]
}}"""

        response = llm_client.call_with_json(prompt)

        if response and 'items' in response:
            checklist_items = response['items']
            # 确保ID连续，确保每个item有_source
            for i, item in enumerate(checklist_items):
                item['id'] = f"item_{i}"
                if '_source' not in item:
                    item['_source'] = '系统基于论文研究信息生成'
            return checklist_items
        else:
            # LLM失败时使用基础模板
            return self._generate_basic_checklist(research_info)

    def _generate_basic_checklist(self, research_info: Dict[str, Any]) -> List[Dict]:
        """生成基础checklist（后备方案），含来源引用"""
        # 根据研究信息生成稍微具体的checklist
        methods = research_info.get('methods', {})
        main_methods = methods.get('main_methods', '')
        datasets = [ds.get('name', '') for ds in research_info.get('datasets', [])]
        sources = research_info.get('_paper_sources', {})

        checklist_items = [
            {
                "type": "text",
                "weight": 0.25,
                "content": f"报告应详细描述{main_methods[:50] if main_methods else '实验方法'}的具体实现过程",
                "_source": sources.get('methods', '基于论文方法部分概述')
            },
            {
                "type": "text",
                "weight": 0.15,
                "content": f"报告应包含数据处理步骤，涉及{', '.join(datasets[:3]) if datasets else '相关'}数据集的使用说明",
                "_source": f"基于论文使用数据集: {', '.join(datasets[:3]) if datasets else '未明确列出'}"
            },
            {
                "type": "text",
                "weight": 0.20,
                "content": "报告应展示量化的性能结果，包括性能指标、准确率、效率等指标",
                "_source": sources.get('key_findings', '基于论文预期结果部分') if isinstance(sources.get('key_findings'), str) else (sources.get('key_findings', ['基于论文预期结果部分'])[0] if isinstance(sources.get('key_findings'), list) and sources.get('key_findings') else '基于论文预期结果部分')
            },
            {
                "type": "text",
                "weight": 0.15,
                "content": "报告应讨论实验结果与论文关键发现的一致性",
                "_source": "基于论文的研究目标和假说"
            },
            {
                "type": "text",
                "weight": 0.10,
                "content": "代码应具有良好的可读性和可重复性",
                "_source": "基于通用科研实践标准"
            }
        ]

        # 确保ID连续
        for i, item in enumerate(checklist_items):
            item['id'] = f"item_{i}"
        return checklist_items

    def generate_research_doc(self, research_info: Dict[str, Any],
                             paper_metadata: Dict[str, Any]) -> str:
        """
        生成研究详情文档（Markdown格式）

        Args:
            research_info: 提取的研究信息
            paper_metadata: 论文元数据

        Returns:
            Markdown格式的文档内容
        """
        title = research_info.get('title', '相关论文')
        authors = research_info.get('authors', [])
        year = research_info.get('year', '')
        doi = research_info.get('doi', '')

        doc_lines = []
        doc_lines.append(f"# {title}\n")

        # 论文信息
        if authors or year or doi:
            doc_lines.append("## 论文信息\n")
            if authors:
                doc_lines.append(f"**作者**: {', '.join(authors[:5])}")
                if len(authors) > 5:
                    doc_lines.append(f" 等 ({len(authors)}位作者)")
            if year:
                doc_lines.append(f"\n**年份**: {year}")
            if doi:
                doc_lines.append(f"\n**DOI**: {doi}")
            doc_lines.append("\n")

        # 研究背景
        background = research_info.get("background", "")
        if background:
            doc_lines.append("## 研究背景\n")
            doc_lines.append(f"{background}\n")

        # 研究目标
        research_goal = research_info.get("research_goal", "")
        if research_goal:
            doc_lines.append("## 研究目标\n")
            doc_lines.append(f"{research_goal}\n")

        # 研究假设
        hypothesis = research_info.get("hypothesis", "")
        if hypothesis:
            doc_lines.append("## 研究假设\n")
            doc_lines.append(f"{hypothesis}\n")

        # 方法概述
        methods = research_info.get("methods", {})
        main_methods = methods.get("main_methods", "")
        if main_methods:
            doc_lines.append("## 方法概述\n")
            doc_lines.append(f"{main_methods}\n")

        # 实验设计
        experimental_design = research_info.get("experimental_design", {})
        if experimental_design:
            doc_lines.append("## 实验设计\n")
            phases = experimental_design.get('phases', {})
            if phases:
                for phase_name, phase_info in phases.items():
                    if isinstance(phase_info, dict) and phase_info.get('name'):
                        doc_lines.append(f"### {phase_info['name']}\n")
                        doc_lines.append(f"{phase_info.get('description', '')}\n")

            comparison = experimental_design.get('comparison_groups', '')
            if comparison:
                doc_lines.append(f"**对比组**: {comparison}\n")

            variables = experimental_design.get('variables', [])
            if variables:
                doc_lines.append("\n**变量**:\n")
                for var in variables:
                    doc_lines.append(f"- {var}\n")

        # 预期结果
        key_findings = research_info.get("key_findings", [])
        if key_findings:
            doc_lines.append("## 预期结果\n")
            for finding in key_findings:
                doc_lines.append(f"- {finding}\n")

        # 约束条件
        constraints = research_info.get("constraints", [])
        if constraints:
            doc_lines.append("## 约束条件\n")
            for constraint in constraints:
                doc_lines.append(f"- {constraint}\n")

        # 成功标准
        success_criteria = research_info.get("success_criteria", [])
        if success_criteria:
            doc_lines.append("## 成功标准\n")
            for criterion in success_criteria:
                doc_lines.append(f"- {criterion}\n")

        return '\n'.join(doc_lines)

    def balance_checklist_weights(self, checklist: List[Dict]) -> List[Dict]:
        """
        平衡checklist权重

        Args:
            checklist: 原始checklist

        Returns:
            权重平衡后的checklist
        """
        if not checklist:
            return checklist

        # 计算当前总权重
        total_weight = sum(item.get('weight', 0) for item in checklist)

        # 如果总权重为1.0，无需调整
        if abs(total_weight - 1.0) < 0.01:
            return checklist

        # 归一化权重
        if total_weight > 0:
            for item in checklist:
                current_weight = item.get('weight', 0)
                item['weight'] = round(current_weight / total_weight, 2)
        else:
            # 如果没有权重，平均分配
            weight = 1.0 / len(checklist)
            for item in checklist:
                item['weight'] = round(weight, 2)

        return checklist
