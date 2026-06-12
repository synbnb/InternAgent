"""
信息提取模块 - 使用LLM从Markdown中提取所有研究信息
"""

from typing import Dict, List, Any


class ResearchInfoExtractor:
    """研究信息提取器 - 完全基于LLM"""

    def __init__(self, llm_client):
        """
        初始化信息提取器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client

    def extract_all_info(self, paper_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM提取所有关键研究信息，包含论文原文出处

        Args:
            paper_content: 解析后的论文内容（包含Markdown）

        Returns:
            提取的研究信息字典（每个字段附带 _source 引用原文）
        """
        markdown_text = paper_content.get('markdown_content', '')

        # 使用LLM一次性提取所有信息，并要求给出原文出处
        prompt = f"""分析以下论文内容，提取关键研究信息。**对于每个提取的字段，必须提供原文出处（excerpt）**。

!!! 重要规则 !!!
1. 原文摘录必须**保留原文语言**（如果论文是英文，摘录就用英文；中文论文就用中文）。**禁止**把英文翻译成中文或做任何语言转换。
2. 每个摘录必须是**完整段落或至少3-5句连贯内容**，不能只截取一句话。
3. 每个摘录前面必须标注**出自论文的哪个部分/章节**，格式为: `【章节名】完整段落原文...`
   例如: `【Abstract】We present a novel approach...` 或 `【Section 3: Methodology】Our model consists of...`

论文内容：
{markdown_text[:10000]}

请以JSON格式返回以下信息：
{{
    "title": "论文标题",
    "title_source": "【出处章节】论文原文标题的完整原文",
    "authors": ["作者1", "作者2", "作者3"],
    "research_field": "研究领域",
    "research_field_source": "【出处章节】论文原文中提及研究领域的完整段落原文",
    "background": "研究背景（2-3句话，中文描述）",
    "background_source": "【出处章节】论文原文中研究背景相关的完整段落原文（保留原文语言）",
    "research_goal": "研究目标（一句话，中文描述）",
    "research_goal_source": "【出处章节】论文原文中明确研究目标的完整段落原文（保留原文语言）",
    "hypothesis": "研究假设（一句话，中文描述）",
    "hypothesis_source": "【出处章节】论文原文中假设陈述的完整段落原文（保留原文语言）",
    "methods": {{
        "main_methods": "主要方法概述（1-2句话，中文描述）",
        "main_methods_source": "【出处章节】论文原文中方法描述的完整段落原文（保留原文语言）",
        "algorithms": ["算法1", "算法2"],
        "tools": ["工具1", "工具2"]
    }},
    "experimental_design": {{
        "phases": {{
            "phase_1": {{
                "name": "第一阶段名称",
                "description": "第一阶段描述"
            }}
        }},
        "comparison_groups": "对照组和实验组设置",
        "variables": ["变量1", "变量2", "变量3"]
    }},
    "datasets": [
        {{
            "name": "数据集名称",
            "description": "数据集描述（中文）",
            "size": "数据规模",
            "source": "【出处章节】论文原文中提及该数据集的完整段落原文（保留原文语言）"
        }}
    ],
    "metrics": ["指标1", "指标2"],
    "key_findings": ["发现1（中文）", "发现2（中文）"],
    "key_findings_sources": ["【出处章节】原文支持发现1的完整段落原文（保留原文语言）", "【出处章节】原文支持发现2的完整段落原文（保留原文语言）"],
    "constraints": ["约束1", "约束2"],
    "success_criteria": ["标准1", "标准2"]
}}

注意：
1. 提取真实的数据集名称，不要编造
2. 每个 _source 字段的格式必须为: `【章节名称】完整原文段落...`，章节名如 Abstract, Introduction, Section 2, Section 3.1 Methodology, Section 4 Experiments, Section 5 Results 等
3. 摘录必须保留论文原始语言（英文论文→英文摘录，中文论文→中文摘录），**绝不要翻译**
4. 摘录必须足够长，至少3-5句话或一个完整段落，提供充分的上下文
5. 如果论文中没有明确提到某些信息，使用合理的默认值，并在 _source 注明"推断，原文无明确依据"
6. 确保JSON格式正确，可以被解析"""

        response = self.llm_client.call_with_json(prompt)

        # 构建带_paper_sources的结果
        result = {
            'title': response.get('title', '相关论文'),
            'authors': response.get('authors', []),
            'year': response.get('year'),
            'doi': response.get('doi', ''),
            'research_field': response.get('research_field', 'Science'),
            'background': response.get('background', ''),
            'research_goal': response.get('research_goal', ''),
            'hypothesis': response.get('hypothesis', ''),
            'methods': response.get('methods', {}),
            'experimental_design': response.get('experimental_design', {}),
            'datasets': response.get('datasets', []),
            'metrics': response.get('metrics', []),
            'key_findings': response.get('key_findings', []),
            'constraints': response.get('constraints', []),
            'success_criteria': response.get('success_criteria', []),
        }

        # 收集所有 source 信息到 _paper_sources 字典
        sources = {}
        for key in ['title', 'background', 'research_goal', 'hypothesis', 'research_field']:
            source_key = f'{key}_source'
            if source_key in response:
                sources[key] = response[source_key]

        if response.get('methods', {}).get('main_methods_source'):
            sources['methods'] = response['methods']['main_methods_source']

        # 数据集来源
        if 'datasets' in response and isinstance(response['datasets'], list):
            dataset_sources = []
            for ds in response['datasets']:
                if ds.get('source'):
                    dataset_sources.append({
                        'name': ds.get('name', ''),
                        'source': ds['source']
                    })
            if dataset_sources:
                sources['datasets'] = dataset_sources

        # key findings 来源
        if 'key_findings_sources' in response:
            sources['key_findings'] = response['key_findings_sources']

        result['_paper_sources'] = sources

        return result
