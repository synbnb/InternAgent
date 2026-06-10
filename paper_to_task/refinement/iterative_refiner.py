"""
迭代优化模块 - 基于用户反馈改进生成内容
"""

import json
from typing import Dict, List, Any, Optional


class IterativeRefiner:
    """迭代优化器"""

    def __init__(self, llm_client):
        """
        初始化迭代优化器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client

        self.system_prompt = """你是一个内容优化助手，擅长根据用户反馈改进研究任务描述。
请严格按照要求返回JSON格式的结果，不要添加任何额外的解释。"""

    def refine_content(self, current_content: Dict[str, Any],
                      feedback: str) -> Dict[str, Any]:
        """
        基于用户反馈改进内容

        Args:
            current_content: 当前生成的内容
            feedback: 用户反馈

        Returns:
            改进后的内容和改进信息
        """
        # 分析反馈
        feedback_analysis = self._analyze_feedback(feedback, current_content)

        # 根据反馈类型选择改进策略
        if feedback_analysis['type'] == 'missing_info':
            refined_content = self._add_missing_info(current_content, feedback_analysis)
        elif feedback_analysis['type'] == 'inaccurate':
            refined_content = self._correct_inaccuracy(current_content, feedback_analysis)
        elif feedback_analysis['type'] == 'structure_issue':
            refined_content = self._improve_structure(current_content, feedback_analysis)
        elif feedback_analysis['type'] == 'content_enhancement':
            refined_content = self._enhance_content(current_content, feedback_analysis)
        else:
            refined_content = self._general_improvement(current_content, feedback_analysis)

        # 验证改进后的内容
        validation_result = self._validate_refined_content(refined_content)

        return {
            'task_info': refined_content['task_info'],
            'checklist': refined_content['checklist'],
            'improvements_made': feedback_analysis.get('improvements', []),
            'feedback_analysis': feedback_analysis,
            'validation': validation_result
        }

    def _analyze_feedback(self, feedback: str,
                        current_content: Dict) -> Dict[str, Any]:
        """分析用户反馈"""
        prompt = f"""分析以下用户反馈，确定反馈类型和改进方向：

用户反馈：{feedback}

当前内容摘要：
- 任务描述：{current_content.get('task_info', {}).get('task', '无')[:100]}
- 数据项数量：{len(current_content.get('task_info', {}).get('data', []))}
- 评分项数量：{len(current_content.get('checklist', []))}

请返回JSON格式的分析结果：
{{
    "type": "反馈类型",
    "target": "需要改进的部分（task_info/checklist/both）",
    "specific_issues": ["具体问题1", "具体问题2"],
    "improvements": ["改进建议1", "改进建议2"],
    "priority": "优先级（high/medium/low）"
}}

反馈类型可选值：
- missing_info: 缺少信息
- inaccurate: 描述不准确
- structure_issue: 结构问题
- content_enhancement: 内容增强
- general: 一般改进"""

        response = self.llm_client.call_with_json(prompt, system_prompt=self.system_prompt)

        # 提供默认分析
        return {
            'type': response.get('type', 'general'),
            'target': response.get('target', 'both'),
            'specific_issues': response.get('specific_issues', ['需要进一步优化']),
            'improvements': response.get('improvements', ['根据反馈进行改进']),
            'priority': response.get('priority', 'medium')
        }

    def _add_missing_info(self, content: Dict, analysis: Dict) -> Dict:
        """添加缺失信息"""
        task_info = content['task_info'].copy()
        checklist = content['checklist'].copy()

        target = analysis.get('target', 'both')

        if target in ['task_info', 'both']:
            # 补充task_info
            if not task_info.get('background'):
                task_info['background'] = "请提供研究背景信息"

            if not task_info.get('data'):
                task_info['data'] = [{
                    "name": "data_file",
                    "path": "data/data_file",
                    "description": "请描述数据文件内容"
                }]

        if target in ['checklist', 'both']:
            # 补充checklist
            if len(checklist) < 3:
                additional_items = [
                    {
                        "id": f"item_{len(checklist)}",
                        "type": "text",
                        "weight": 0.1,
                        "content": "补充评分项：详细说明实验方法",
                        "keywords": ["方法", "实现"],
                        "evaluation_criteria": "是否详细描述了实验方法"
                    }
                ]
                checklist.extend(additional_items)

        return {'task_info': task_info, 'checklist': checklist}

    def _correct_inaccuracy(self, content: Dict, analysis: Dict) -> Dict:
        """修正不准确描述"""
        task_info = content['task_info'].copy()
        checklist = content['checklist'].copy()

        # 根据具体问题进行修正
        specific_issues = analysis.get('specific_issues', [])

        for issue in specific_issues:
            if '任务' in issue or 'task' in issue.lower():
                # 改进任务描述
                task_info['task'] = self._improve_task_description(task_info['task'], issue)
            elif '数据' in issue or 'data' in issue.lower():
                # 改进数据描述
                task_info['data'] = self._improve_data_description(task_info.get('data', []), issue)
            elif '评分' in issue or 'checklist' in issue.lower():
                # 改进checklist
                checklist = self._improve_checklist(checklist, issue)

        return {'task_info': task_info, 'checklist': checklist}

    def _improve_structure(self, content: Dict, analysis: Dict) -> Dict:
        """改进结构"""
        task_info = content['task_info'].copy()
        checklist = content['checklist'].copy()

        # 重新组织checklist结构
        if 'checklist' in analysis.get('target', ''):
            checklist = self._reorganize_checklist(checklist)

        # 确保所有必需字段存在
        for field in ['task', 'background', 'research_goal']:
            if not task_info.get(field):
                task_info[field] = f"请补充{field}信息"

        return {'task_info': task_info, 'checklist': checklist}

    def _enhance_content(self, content: Dict, analysis: Dict) -> Dict:
        """增强内容"""
        task_info = content['task_info'].copy()
        checklist = content['checklist'].copy()

        # 增强task_info描述
        if 'task_info' in analysis.get('target', ''):
            task_info = self._enhance_task_info_details(task_info)

        # 增强checklist
        if 'checklist' in analysis.get('target', ''):
            checklist = self._enhance_checklist_details(checklist)

        return {'task_info': task_info, 'checklist': checklist}

    def _general_improvement(self, content: Dict, analysis: Dict) -> Dict:
        """通用改进"""
        task_info = content['task_info'].copy()
        checklist = content['checklist'].copy()

        # 根据反馈建议进行改进
        improvements = analysis.get('improvements', [])

        for improvement in improvements:
            if '详细' in improvement or 'detail' in improvement.lower():
                task_info = self._add_more_details(task_info)
            elif '具体' in improvement or 'specific' in improvement.lower():
                checklist = self._make_checklist_more_specific(checklist)
            elif '权重' in improvement or 'weight' in improvement.lower():
                checklist = self._adjust_checklist_weights(checklist)

        return {'task_info': task_info, 'checklist': checklist}

    def _improve_task_description(self, current_task: str, issue: str) -> str:
        """改进任务描述"""
        if len(current_task) < 30:
            return f"{current_task}。需要更详细地描述研究内容和目标。"
        return current_task + "。根据反馈进行了优化。"

    def _improve_data_description(self, data_list: List, issue: str) -> List:
        """改进数据描述"""
        for data_item in data_list:
            if not data_item.get('description') or len(data_item.get('description', '')) < 10:
                data_item['description'] = f"数据文件：{data_item.get('name', '')}。{issue}"
        return data_list

    def _improve_checklist(self, checklist: List, issue: str) -> List:
        """改进checklist"""
        for item in checklist:
            if len(item.get('content', '')) < 20:
                item['content'] = f"{item.get('content', '')}。{issue}"
        return checklist

    def _reorganize_checklist(self, checklist: List) -> List:
        """重新组织checklist"""
        # 按权重排序
        sorted_checklist = sorted(checklist,
                                  key=lambda x: x.get('weight', 0),
                                  reverse=True)
        return sorted_checklist

    def _enhance_task_info_details(self, task_info: Dict) -> Dict:
        """增强task_info细节"""
        # 增强background
        if task_info.get('background'):
            if len(task_info['background']) < 100:
                task_info['background'] += "。需要提供更详细的研究背景，包括相关工作、研究动机等。"

        # 增强success_criteria
        criteria = task_info.get('success_criteria', [])
        if len(criteria) < 3:
            criteria.extend([
                "代码具有良好的可读性和可重复性",
                "实验结果具有统计显著性",
                "报告包含完整的方法描述"
            ])
            task_info['success_criteria'] = criteria

        return task_info

    def _enhance_checklist_details(self, checklist: List) -> List:
        """增强checklist细节"""
        for item in checklist:
            # 增强content
            if len(item.get('content', '')) < 30:
                item['content'] += "。需要提供更详细的描述和具体的评估标准。"

            # 添加keywords
            if not item.get('keywords'):
                item['keywords'] = ["质量", "完整性", "准确性"]

        return checklist

    def _add_more_details(self, task_info: Dict) -> Dict:
        """添加更多细节"""
        task_info['task'] += "。包括详细的方法描述、数据分析过程和结果解释。"

        if not task_info.get('constraints'):
            task_info['constraints'] = [
                "确保实验的可重复性",
                "使用适当的数据分析方法",
                "报告具有统计学意义的结果"
            ]

        return task_info

    def _make_checklist_more_specific(self, checklist: List) -> List:
        """使checklist更具体"""
        for item in checklist:
            content = item.get('content', '')
            if '应该' in content and '具体' not in content:
                item['content'] = content.replace('应该', '应该具体详细地')
        return checklist

    def _adjust_checklist_weights(self, checklist: List) -> List:
        """调整checklist权重"""
        # 确保权重总和为1.0
        total_weight = sum(item.get('weight', 0) for item in checklist)

        if total_weight > 0:
            for item in checklist:
                current_weight = item.get('weight', 0)
                item['weight'] = round(current_weight / total_weight, 2)

        return checklist

    def _validate_refined_content(self, content: Dict) -> Dict[str, Any]:
        """验证改进后的内容"""
        task_info = content.get('task_info', {})
        checklist = content.get('checklist', [])

        # 基本验证
        validation = {
            'task_info_valid': bool(task_info.get('task')),
            'has_data': bool(task_info.get('data')),
            'checklist_valid': len(checklist) >= 2,
            'weights_balanced': self._check_weight_balance(checklist)
        }

        validation['overall_valid'] = all(validation.values())

        return validation

    def _check_weight_balance(self, checklist: List) -> bool:
        """检查权重平衡"""
        total_weight = sum(item.get('weight', 0) for item in checklist)
        return abs(total_weight - 1.0) < 0.2

    def get_improvement_summary(self, refinement_result: Dict) -> str:
        """
        获取改进摘要

        Args:
            refinement_result: 改进结果

        Returns:
            改进摘要文本
        """
        summary = []
        summary.append("=== 内容改进摘要 ===\n")

        # 改进内容
        improvements = refinement_result.get('improvements_made', [])
        if improvements:
            summary.append("已执行的改进:")
            for i, improvement in enumerate(improvements, 1):
                summary.append(f"  {i}. {improvement}")
        else:
            summary.append("未执行具体改进")

        # 验证结果
        validation = refinement_result.get('validation', {})
        summary.append("\n验证结果:")
        summary.append(f"  - 任务描述有效: {'✅' if validation.get('task_info_valid') else '❌'}")
        summary.append(f"  - 包含数据: {'✅' if validation.get('has_data') else '❌'}")
        summary.append(f"  - 评分项有效: {'✅' if validation.get('checklist_valid') else '❌'}")
        summary.append(f"  - 权重平衡: {'✅' if validation.get('weights_balanced') else '❌'}")

        # 整体状态
        overall = validation.get('overall_valid', False)
        summary.append(f"\n整体状态: {'✅ 通过' if overall else '❌ 需要进一步改进'}")

        return "\n".join(summary)

    def suggest_improvements(self, content: Dict) -> List[str]:
        """
        建议改进方向

        Args:
            content: 当前内容

        Returns:
            改进建议列表
        """
        suggestions = []

        task_info = content.get('task_info', {})
        checklist = content.get('checklist', [])

        # task_info改进建议
        if len(task_info.get('task', '')) < 20:
            suggestions.append("任务描述过于简短，建议扩展到至少20个字符")

        if not task_info.get('background'):
            suggestions.append("缺少研究背景，建议添加背景信息")

        if not task_info.get('data'):
            suggestions.append("缺少数据文件信息，建议添加数据描述")

        # checklist改进建议
        if len(checklist) < 3:
            suggestions.append("评分项数量过少，建议至少3个评分项")

        if not any(item.get('type') == 'image' for item in checklist):
            suggestions.append("建议添加至少一个图像类型的评分项")

        total_weight = sum(item.get('weight', 0) for item in checklist)
        if abs(total_weight - 1.0) > 0.2:
            suggestions.append(f"评分权重总和({total_weight:.2f})建议接近1.0")

        if not suggestions:
            suggestions.append("内容质量良好，无需特别改进")

        return suggestions
