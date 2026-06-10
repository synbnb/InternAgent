"""
内容生成模块 - 生成task_info.json和checklist.json
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class SciTaskGenerator:
    """sci任务生成器"""

    def __init__(self):
        # task_info模板
        self.task_info_template = {
            "task": "",
            "background": "",
            "research_goal": "",
            "hypothesis": "",
            "data": [],
            "experimental_design": {},
            "expected_outcomes": [],
            "constraints": [],
            "success_criteria": []
        }

        # checklist模板
        self.checklist_template = {
            "items": []
        }

    def generate_task_info(self, research_info: Dict[str, Any],
                          paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成task_info.json

        Args:
            research_info: 提取的研究信息
            paper_metadata: 论文元数据

        Returns:
            task_info字典
        """
        task_info = self.task_info_template.copy()

        # 基础信息
        title = paper_metadata.get('title', '相关论文')
        task_info["task"] = f"复现论文《{title[:50]}...》的核心发现"
        task_info["background"] = research_info.get("background", "")
        task_info["research_goal"] = research_info.get("research_goal", "")
        task_info["hypothesis"] = research_info.get("hypothesis", "")

        # 数据信息
        datasets = research_info.get("datasets", [])
        task_info["data"] = self._generate_data_section(datasets)

        # 实验设计
        experimental_design = research_info.get("experimental_design", {})
        task_info["experimental_design"] = self._format_experimental_design(experimental_design)

        # 预期结果
        key_findings = research_info.get("key_findings", [])
        task_info["expected_outcomes"] = [
            f"成功验证：{finding}" for finding in key_findings
        ]

        # 约束条件
        constraints = research_info.get("constraints", [])
        task_info["constraints"] = constraints

        # 成功标准
        success_criteria = research_info.get("success_criteria", [])
        task_info["success_criteria"] = success_criteria

        return task_info

    def _generate_data_section(self, datasets: List[Dict]) -> List[Dict]:
        """生成数据部分"""
        data_section = []

        for i, dataset in enumerate(datasets):
            data_item = {
                "name": dataset.get("name", f"dataset_{i+1}"),
                "path": f"data/{dataset.get('name', f'dataset_{i+1}')}",
                "description": dataset.get("description", "")
            }
            data_section.append(data_item)

        # 如果没有数据集，添加默认数据项
        if not data_section:
            data_section.append({
                "name": "research_data",
                "path": "data/research_data",
                "description": "根据论文描述准备的实验数据"
            })

        return data_section

    def _format_experimental_design(self, experimental_design: Dict) -> Dict:
        """格式化实验设计"""
        formatted = {}

        phases = experimental_design.get('phases', {})
        if phases:
            formatted['phases'] = phases

        comparison = experimental_design.get('comparison_groups', '')
        if comparison:
            formatted['comparison_groups'] = comparison

        variables = experimental_design.get('variables', [])
        if variables:
            formatted['variables'] = variables

        return formatted

    def generate_checklist(self, research_info: Dict[str, Any],
                           paper_content: Dict[str, Any]) -> List[Dict]:
        """
        生成checklist.json

        Args:
            research_info: 研究信息
            paper_content: 论文内容

        Returns:
            checklist列表
        """
        checklist_items = []

        # 1. 方法实现评分项
        methods = research_info.get("methods", {})
        if methods.get("main_methods"):
            checklist_items.append({
                "id": f"item_{len(checklist_items)}",
                "type": "text",
                "weight": 0.25,
                "content": f"报告应详细描述{methods['main_methods']}的具体实现过程",
                "keywords": ["方法实现", "实验设计", "具体步骤"],
                "evaluation_criteria": "是否清楚说明了实验方法的具体实现过程和参数设置"
            })

        # 2. 数据处理评分项
        datasets = research_info.get("datasets", [])
        if datasets:
            checklist_items.append({
                "id": f"item_{len(checklist_items)}",
                "type": "text",
                "weight": 0.15,
                "content": f"报告应包含数据处理步骤，涉及{len(datasets)}个数据集的使用说明",
                "keywords": ["数据处理", "数据集使用", "预处理"],
                "evaluation_criteria": "是否描述了如何处理和使用实验数据"
            })

        # 3. 结果展示评分项
        metrics = research_info.get("metrics", [])
        if metrics:
            checklist_items.append({
                "id": f"item_{len(checklist_items)}",
                "type": "text",
                "weight": 0.20,
                "content": f"报告应展示量化的性能结果，包括{', '.join(metrics[:3])}等指标",
                "keywords": ["性能指标", "量化结果", "统计分析"] + metrics,
                "evaluation_criteria": "是否提供具体的数值结果和统计分析"
            })

        # 4. 图表生成评分项
        figures = paper_content.get("figures", [])
        if len(figures) >= 2:
            checklist_items.append({
                "id": f"item_{len(checklist_items)}",
                "type": "image",
                "weight": 0.15,
                "content": "生成类似论文中的可视化结果图表",
                "path": "images/result_visualization.png",
                "keywords": ["可视化", "结果图表", "性能对比"],
                "evaluation_criteria": "图表是否清晰展示了实验结果"
            })

        # 5. 讨论分析评分项
        key_findings = research_info.get("key_findings", [])
        if key_findings:
            checklist_items.append({
                "id": f"item_{len(checklist_items)}",
                "type": "text",
                "weight": 0.15,
                "content": "报告应讨论实验结果与论文关键发现的一致性",
                "keywords": ["结果分析", "发现验证", "一致性讨论"],
                "evaluation_criteria": "是否分析了实验结果与预期发现的符合程度"
            })

        # 6. 代码质量评分项
        checklist_items.append({
            "id": f"item_{len(checklist_items)}",
            "type": "text",
            "weight": 0.10,
            "content": "代码应具有良好的可读性和可重复性",
            "keywords": ["代码质量", "可重复性", "文档"],
            "evaluation_criteria": "代码结构是否清晰，能否重复运行"
        })

        return checklist_items

    def generate_project_structure(self, task_name: str,
                                   task_info: Dict,
                                   checklist: List[Dict],
                                   domain: str = "Science") -> Dict[str, str]:
        """
        生成项目结构

        Args:
            task_name: 任务名称
            task_info: 任务信息
            checklist: 检查清单
            domain: 领域名称

        Returns:
            项目结构字典
        """
        # 生成任务ID
        task_id = self._generate_task_id(domain, task_name)

        # 创建目录结构
        directory_structure = {
            "root": f"sci_tasks/tasks/{task_id}",
            "files": {},
            "directories": {}
        }

        # 添加文件
        directory_structure["files"] = {
            "task_info.json": json.dumps(task_info, indent=2, ensure_ascii=False),
            "target_study/checklist.json": json.dumps(checklist, indent=2, ensure_ascii=False),
            "DATA_README.md": self._generate_data_readme(task_info),
            "TASK_STATUS.md": "# Task Status\n\n初始化完成，等待执行。"
        }

        # 添加目录
        directory_structure["directories"] = {
            "data/": "数据文件目录",
            "target_study/images/": "参考图表目录",
            "target_study/paper/": "参考论文目录"
        }

        return directory_structure

    def _generate_task_id(self, domain: str, task_name: str) -> str:
        """生成任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        # 简化版任务ID生成
        return f"{domain}_{timestamp}_001"

    def _generate_data_readme(self, task_info: Dict) -> str:
        """生成数据说明文件"""
        readme = "# 数据文件说明\n\n"

        data_items = task_info.get("data", [])
        if data_items:
            readme += "## 可用数据文件\n\n"
            for item in data_items:
                readme += f"### {item['name']}\n"
                readme += f"- **路径**: `{item['path']}`\n"
                readme += f"- **描述**: {item['description']}\n\n"
        else:
            readme += "## 数据文件\n\n"
            readme += "请根据论文描述准备相应的数据文件。\n\n"

        readme += "## 数据准备指南\n\n"
        readme += "1. 从论文补充材料或相关网站下载数据\n"
        readme += "2. 将数据文件放置在 `data/` 目录下\n"
        readme += "3. 更新 `task_info.json` 中的文件路径\n\n"

        readme += "## 数据格式要求\n\n"
        readme += "- 确保数据格式与论文描述一致\n"
        readme += "- 包含必要的元数据和标签\n"
        readme += "- 检查数据完整性和质量\n\n"

        return readme

    def optimize_task_description(self, task_info: Dict,
                                  research_info: Dict) -> Dict:
        """
        优化任务描述

        Args:
            task_info: 原始任务信息
            research_info: 研究信息

        Returns:
            优化后的任务信息
        """
        optimized = task_info.copy()

        # 优化任务描述
        task_desc = optimized.get("task", "")
        if len(task_desc) < 50:
            # 扩展简短的任务描述
            research_goal = research_info.get("research_goal", "")
            if research_goal:
                optimized["task"] = f"{task_desc}：{research_goal}"

        # 确保必需字段存在
        if not optimized.get("background"):
            optimized["background"] = research_info.get("background", "研究背景待补充")

        if not optimized.get("research_goal"):
            optimized["research_goal"] = research_info.get("research_goal", "验证论文核心发现")

        return optimized

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

    def generate_metadata(self, research_info: Dict,
                          paper_metadata: Dict) -> Dict[str, Any]:
        """
        生成元数据

        Args:
            research_info: 研究信息
            paper_metadata: 论文元数据

        Returns:
            元数据字典
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "paper_title": paper_metadata.get("title", ""),
            "paper_authors": paper_metadata.get("authors", []),
            "paper_year": paper_metadata.get("year"),
            "paper_doi": paper_metadata.get("doi", ""),
            "research_field": research_info.get("research_field", ""),
            "generator_version": "1.0.0"
        }
