"""
信息提取模块 - 从论文内容中提取研究信息
"""

import json
from typing import Dict, List, Any, Optional


class ResearchInfoExtractor:
    """研究信息提取器"""

    def __init__(self, llm_client):
        """
        初始化信息提取器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client

        # 系统提示词
        self.system_prompt = """你是一个专业的科研论文分析助手，擅长从论文中提取关键研究信息。
请严格按照要求返回JSON格式的结果，不要添加任何额外的解释。"""

    def extract_all_info(self, paper_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取所有关键研究信息

        Args:
            paper_content: 解析后的论文内容

        Returns:
            提取的研究信息字典
        """
        sections = paper_content.get('sections', {})
        metadata = paper_content.get('metadata', {})

        # 提取基础信息
        basic_info = self._extract_basic_info(sections, metadata)

        # 提取方法信息
        methods_info = self._extract_methods_info(sections)

        # 提取实验设计
        experimental_design = self._extract_experimental_design(sections)

        # 提取数据集信息
        datasets_info = self._extract_datasets_info(sections, paper_content.get('tables', []))

        # 提取评估指标
        metrics_info = self._extract_metrics_info(sections)

        # 提取关键发现
        key_findings = self._extract_key_findings(sections)

        # 组合所有信息
        research_info = {
            'title': metadata.get('title', ''),
            'authors': metadata.get('authors', []),
            'year': metadata.get('year'),
            'doi': metadata.get('doi', ''),
            'research_goal': basic_info.get('goal', ''),
            'hypothesis': basic_info.get('hypothesis', ''),
            'background': basic_info.get('background', ''),
            'research_field': basic_info.get('research_field', ''),
            'methods': methods_info,
            'experimental_design': experimental_design,
            'datasets': datasets_info,
            'metrics': metrics_info,
            'key_findings': key_findings,
            'constraints': self._infer_constraints(experimental_design, basic_info),
            'success_criteria': self._derive_success_criteria(key_findings, metrics_info)
        }

        return research_info

    def _extract_basic_info(self, sections: Dict, metadata: Dict) -> Dict[str, str]:
        """提取基础研究信息"""
        abstract = sections.get('abstract', '')
        introduction = sections.get('introduction', '')
        title = metadata.get('title', '')

        prompt = f"""分析以下论文的标题、摘要和引言，提取关键信息：

论文标题：{title}

摘要：
{abstract[:1000] if abstract else '无摘要'}

引言：
{introduction[:1500] if introduction else '无引言'}

请提取并返回以下JSON格式的信息：
{{
    "goal": "研究的主要目标（一句话描述）",
    "hypothesis": "研究假设或核心思想（一句话）",
    "background": "研究背景和动机（2-3句话）",
    "research_field": "研究领域（如：生物学、化学、物理学、计算机科学等）"
}}"""

        response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)

        # 提供默认值
        return {
            'goal': response.get('goal', f"研究{title[:30]}...的核心发现"),
            'hypothesis': response.get('hypothesis', '论文提出的方法是有效的'),
            'background': response.get('background', abstract[:200] if abstract else '研究背景'),
            'research_field': response.get('research_field', 'Science')
        }

    def _extract_methods_info(self, sections: Dict) -> Dict[str, str]:
        """提取方法信息"""
        methods_section = sections.get('methods', '')
        experiments_section = sections.get('experiments', '')

        # 合并方法相关章节
        methods_text = methods_section + "\n" + experiments_section

        if not methods_text.strip():
            return {
                'main_methods': '实验方法',
                'algorithms': '',
                'tools': '',
                'experimental_setup': ''
            }

        prompt = f"""分析以下论文的方法部分，提取关键信息：

{methods_text[:2000]}

请提取并返回以下JSON格式的信息：
{{
    "main_methods": "主要方法的概述（1-2句话）",
    "algorithms": "使用的主要算法或模型（列举2-3个）",
    "tools": "使用的工具、软件或平台（列举2-3个）",
    "experimental_setup": "实验设置和环境的描述（1-2句话）"
}}"""

        response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)

        return {
            'main_methods': response.get('main_methods', '实验方法'),
            'algorithms': response.get('algorithms', ''),
            'tools': response.get('tools', ''),
            'experimental_setup': response.get('experimental_setup', '')
        }

    def _extract_experimental_design(self, sections: Dict) -> Dict[str, Any]:
        """提取实验设计信息"""
        methods = sections.get('methods', '')
        experiments = sections.get('experiments', '')

        combined_text = methods + "\n" + experiments

        if not combined_text.strip():
            return {'phases': {}, 'comparison_groups': '', 'variables': []}

        prompt = f"""分析以下论文的实验设计，识别实验阶段和变量：

{combined_text[:2500]}

请提取并返回以下JSON格式的信息：
{{
    "phases": {{
        "phase_1": {{
            "name": "第一阶段名称（如：数据准备、基线实验等）",
            "description": "第一阶段的详细描述（1-2句话）"
        }},
        "phase_2": {{
            "name": "第二阶段名称",
            "description": "第二阶段的详细描述（1-2句话）"
        }}
    }},
    "comparison_groups": "对照组和实验组的设置说明",
    "variables": ["自变量1", "因变量1", "控制变量1"]
}}"""

        response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)

        # 清理和验证响应
        phases = response.get('phases', {})
        if not isinstance(phases, dict):
            phases = {}

        return {
            'phases': phases,
            'comparison_groups': response.get('comparison_groups', ''),
            'variables': response.get('variables', [])
        }

    def _extract_datasets_info(self, sections: Dict, tables: List) -> List[Dict]:
        """提取数据集信息"""
        methods = sections.get('methods', '')

        # 从文本中识别数据集
        if methods.strip():
            prompt = f"""分析以下内容，识别使用的数据集：

{methods[:2000]}

请提取并返回以下JSON格式的信息：
{{
    "datasets": [
        {{
            "name": "数据集名称",
            "description": "数据集的简要描述（1句话）",
            "size": "数据规模（如：1000个样本）",
            "source": "数据来源（如：公开数据库、论文提供等）"
        }}
    ]
}}"""

            response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)
            datasets = response.get('datasets', [])
        else:
            datasets = []

        # 如果从文本中没有提取到数据集，尝试从表格推断
        if not datasets and tables:
            for i, table in enumerate(tables[:3]):  # 最多取3个表格
                datasets.append({
                    'name': f"Table_from_paper_page_{table['page']}",
                    'description': f"论文表格数据，包含{table.get('rows', 0)}行",
                    'size': f"{table.get('rows', 0)} rows",
                    'source': '论文表格'
                })

        # 如果仍然没有数据集，提供默认建议
        if not datasets:
            datasets.append({
                'name': 'research_data',
                'description': '根据论文描述准备的实验数据',
                'size': '根据实验需要确定',
                'source': '需要根据论文补充材料获取'
            })

        return datasets

    def _extract_metrics_info(self, sections: Dict) -> List[str]:
        """提取评估指标"""
        results = sections.get('results', '')
        discussion = sections.get('discussion', '')

        combined = results + "\n" + discussion

        if not combined.strip():
            return []

        prompt = f"""分析以下论文的结果部分，识别使用的评估指标：

{combined[:1500]}

请提取并返回以下JSON格式的信息：
{{
    "metrics": ["指标1", "指标2", "指标3"]
}}

注意：只提取具体的性能指标，如准确率、精确率、召回率、F1分数等。"""

        response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)
        metrics = response.get('metrics', [])

        # 如果没有提取到指标，提供常见的默认指标
        if not metrics:
            metrics = ['性能指标', '准确率', '效率']

        return metrics

    def _extract_key_findings(self, sections: Dict) -> List[str]:
        """提取关键发现"""
        results = sections.get('results', '')
        conclusion = sections.get('conclusion', '')

        combined = results + "\n" + conclusion

        if not combined.strip():
            return []

        prompt = f"""分析以下论文的结果和结论，提取3-5个关键发现：

{combined[:2000]}

请提取并返回以下JSON格式的信息：
{{
    "findings": [
        "关键发现1（一句话）",
        "关键发现2（一句话）",
        "关键发现3（一句话）"
    ]
}}"""

        response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)
        findings = response.get('findings', [])

        # 如果没有提取到发现，提供默认值
        if not findings:
            findings = ['论文的核心发现', '方法的验证结果', '与对比方法的性能比较']

        return findings

    def _infer_constraints(self, experimental_design: Dict, basic_info: Dict) -> List[str]:
        """推断实验约束条件"""
        constraints = []

        # 基于实验设计推断约束
        phases = experimental_design.get('phases', {})
        if phases:
            constraints.append(f"实验包含{len(phases)}个阶段，需按顺序执行")

        datasets = experimental_design.get('datasets', [])
        if datasets:
            constraints.append(f"依赖{len(datasets)}个数据集，需确保数据可用性")

        # 基于研究内容推断约束
        research_field = basic_info.get('research_field', '').lower()
        if 'biology' in research_field or 'protein' in research_field or 'gene' in research_field:
            constraints.append("生物数据可能需要专业处理和验证")
        elif 'physics' in research_field or 'quantum' in research_field:
            constraints.append("物理实验可能需要特殊设备或环境")

        # 通用约束
        constraints.extend([
            "需确保代码可重复运行",
            "结果应具有统计学意义"
        ])

        return constraints

    def _derive_success_criteria(self, key_findings: List, metrics: List) -> List[str]:
        """从关键发现推导成功标准"""
        criteria = []

        # 从关键发现推导
        for i, finding in enumerate(key_findings[:3]):
            criteria.append(f"能够验证：{finding}")

        # 从指标推导
        if metrics:
            criteria.append(f"能够计算并报告{', '.join(metrics[:3])}等指标")

        # 通用标准
        criteria.extend([
            "代码能够正常运行并生成结果",
            "生成的研究报告包含完整的实验描述",
            "结果与论文描述一致"
        ])

        return list(set(criteria))  # 去重

    def extract_domain_info(self, research_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取领域特定信息

        Args:
            research_info: 研究信息

        Returns:
            领域信息
        """
        research_field = research_info.get('research_field', '').lower()
        background = research_info.get('background', '').lower()

        # 领域关键词映射
        domain_keywords = {
            'ProteinBio': ['protein', 'biology', 'gene', 'dna', 'rna', 'sequence', 'structure'],
            'Chemistry': ['chemical', 'molecule', 'reaction', 'synthesis', 'compound'],
            'Physics': ['physics', 'quantum', 'particle', 'force', 'energy'],
            'Medicine': ['medical', 'clinical', 'patient', 'treatment', 'disease'],
            'Material': ['material', 'crystal', 'structure', 'property'],
            'ComputerScience': ['algorithm', 'computation', 'network', 'data']
        }

        # 查找匹配的领域
        combined_text = research_field + " " + background
        matched_domain = 'Science'  # 默认

        for domain, keywords in domain_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                matched_domain = domain
                break

        return {
            'domain': matched_domain,
            'domain_keywords': domain_keywords.get(matched_domain, []),
            'requires_special_data': matched_domain in ['ProteinBio', 'Chemistry', 'Medicine'],
            'requires_computation': matched_domain in ['ComputerScience', 'Physics']
        }
