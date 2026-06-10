"""
质量评分模块 - 对生成内容进行多维度的质量评分
"""

from typing import Dict, List, Any, Tuple
import re


class QualityScorer:
    """质量评分器"""

    def __init__(self):
        # 各维度的权重
        self.dimension_weights = {
            'completeness': 0.3,
            'accuracy': 0.25,
            'clarity': 0.25,
            'feasibility': 0.2
        }

    def score_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        对内容进行全面质量评分

        Args:
            content: 包含task_info和checklist的字典

        Returns:
            评分结果
        """
        task_info = content.get('task_info', {})
        checklist = content.get('checklist', [])

        # 计算各维度分数
        dimension_scores = {
            'completeness': self._score_completeness(task_info, checklist),
            'accuracy': self._score_accuracy(task_info, checklist),
            'clarity': self._score_clarity(task_info, checklist),
            'feasibility': self._score_feasibility(task_info)
        }

        # 计算加权总分
        overall_score = sum(
            score * self.dimension_weights[dim]
            for dim, score in dimension_scores.items()
        )

        # 生成评级
        grade = self._get_grade(overall_score)

        # 生成建议
        suggestions = self._generate_suggestions(dimension_scores, task_info, checklist)

        return {
            'overall_score': round(overall_score, 3),
            'grade': grade,
            'dimension_scores': {k: round(v, 3) for k, v in dimension_scores.items()},
            'suggestions': suggestions,
            'passed': overall_score >= 0.7
        }

    def _score_completeness(self, task_info: Dict, checklist: List) -> float:
        """评分完整性"""
        score = 0.0

        # task_info必需字段 (50分)
        required_fields = {
            'task': (bool(task_info.get('task')), 20),
            'data': (isinstance(task_info.get('data'), list) and task_info['data'], 15),
            'background': (bool(task_info.get('background')), 8),
            'research_goal': (bool(task_info.get('research_goal')), 7)
        }

        for field, (present, points) in required_fields.items():
            if present:
                score += points

        # checklist完整性 (50分)
        if checklist:
            # 数量评分 (20分)
            num_items = len(checklist)
            if num_items >= 5:
                score += 20
            elif num_items >= 3:
                score += 15
            elif num_items >= 2:
                score += 10

            # 必需字段评分 (15分)
            valid_items = sum(
                1 for item in checklist
                if all(key in item for key in ['content', 'weight', 'type'])
            )
            if valid_items == len(checklist):
                score += 15
            elif valid_items >= len(checklist) * 0.8:
                score += 10

            # 多样性评分 (15分)
            has_text = any(item.get('type') == 'text' for item in checklist)
            has_image = any(item.get('type') == 'image' for item in checklist)

            if has_text and has_image:
                score += 15
            elif has_text:
                score += 8

        return score / 100

    def _score_accuracy(self, task_info: Dict, checklist: List) -> float:
        """评分准确性"""
        score = 0.0

        # task_info数据准确性 (40分)
        if isinstance(task_info.get('data'), list):
            data_items = task_info['data']
            if data_items:
                # 检查数据项完整性
                valid_data = sum(
                    1 for item in data_items
                    if all(key in item for key in ['name', 'path', 'description'])
                )

                if valid_data == len(data_items):
                    score += 20
                elif valid_data >= len(data_items) * 0.5:
                    score += 10

                # 检查描述质量
                good_descriptions = sum(
                    1 for item in data_items
                    if len(item.get('description', '')) > 10
                )

                if good_descriptions / len(data_items) >= 0.7:
                    score += 20
                elif good_descriptions / len(data_items) >= 0.3:
                    score += 10

        # checklist准确性 (60分)
        if checklist:
            # 检查权重分布
            total_weight = sum(item.get('weight', 0) for item in checklist)

            if 0.9 <= total_weight <= 1.1:
                score += 25
            elif 0.7 <= total_weight <= 1.3:
                score += 15

            # 检查类型有效性
            valid_types = sum(
                1 for item in checklist
                if item.get('type') in ['text', 'image']
            )

            if valid_types == len(checklist):
                score += 20

            # 检查image类型的path字段
            image_items = [item for item in checklist if item.get('type') == 'image']
            if image_items:
                with_path = sum(1 for item in image_items if item.get('path'))
                if with_path == len(image_items):
                    score += 15
                elif with_path >= len(image_items) * 0.5:
                    score += 8

        return score / 100

    def _score_clarity(self, task_info: Dict, checklist: List) -> float:
        """评分清晰度"""
        score = 0.0

        # task_info描述清晰度 (40分)
        # 任务描述长度和质量
        task_desc = task_info.get('task', '')
        desc_length_score = min(len(task_desc) / 100, 1) * 15
        score += desc_length_score

        # 检查描述是否包含关键词
        keywords = ['复现', '验证', '研究', '实验', '方法']
        keyword_count = sum(1 for kw in keywords if kw in task_desc)
        score += (keyword_count / len(keywords)) * 10

        # background长度
        background = task_info.get('background', '')
        if len(background) > 50:
            score += 10
        elif len(background) > 20:
            score += 5

        # checklist描述清晰度 (60分)
        if checklist:
            # 内容长度评分
            content_lengths = [len(item.get('content', '')) for item in checklist]
            avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0

            if avg_length >= 40:
                score += 25
            elif avg_length >= 25:
                score += 15
            elif avg_length >= 15:
                score += 10

            # 关键词覆盖率
            items_with_keywords = sum(
                1 for item in checklist
                if item.get('keywords') and isinstance(item.get('keywords'), list)
            )

            if items_with_keywords == len(checklist):
                score += 20
            elif items_with_keywords >= len(checklist) * 0.5:
                score += 10

            # 评估标准覆盖率
            items_with_criteria = sum(
                1 for item in checklist
                if item.get('evaluation_criteria')
            )

            if items_with_criteria >= len(checklist) * 0.7:
                score += 15
            elif items_with_criteria >= len(checklist) * 0.3:
                score += 8

        return score / 100

    def _score_feasibility(self, task_info: Dict) -> float:
        """评分可行性"""
        score = 0.5  # 基础分

        # 数据可用性 (20分)
        data_items = task_info.get('data', [])
        if data_items:
            score += 0.1

            # 检查数据描述的可行性
            feasible_data = sum(
                1 for item in data_items
                if self._is_feasible_data_description(item.get('description', ''))
            )

            if feasible_data == len(data_items):
                score += 0.1
            elif feasible_data >= len(data_items) * 0.5:
                score += 0.05

        # 约束条件合理性 (15分)
        constraints = task_info.get('constraints', [])
        if constraints:
            # 检查约束条件的合理性
            reasonable_constraints = sum(
                1 for constraint in constraints
                if self._is_reasonable_constraint(constraint)
            )

            if reasonable_constraints >= len(constraints) * 0.7:
                score += 0.15

        # 成功标准明确性 (15分)
        success_criteria = task_info.get('success_criteria', [])
        if success_criteria and len(success_criteria) >= 2:
            score += 0.15

        return min(score, 1.0)

    def _is_feasible_data_description(self, description: str) -> bool:
        """检查数据描述是否可行"""
        # 检查是否包含具体的来源或获取方式
        feasible_indicators = [
            '下载', '公开', '提供', '网站', '数据库',
            'download', 'available', 'public', 'dataset'
        ]

        return any(indicator in description.lower() for indicator in feasible_indicators)

    def _is_reasonable_constraint(self, constraint: str) -> bool:
        """检查约束条件是否合理"""
        # 合理的约束条件不应该过于严格或不切实际
        unreasonable_indicators = [
            '完美', '绝对', '100%', 'impossible', 'perfect'
        ]

        constraint_lower = constraint.lower()
        return not any(indicator in constraint_lower for indicator in unreasonable_indicators)

    def _get_grade(self, score: float) -> str:
        """根据分数获取评级"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _generate_suggestions(self, dimension_scores: Dict,
                             task_info: Dict,
                             checklist: List) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 完整性建议
        if dimension_scores['completeness'] < 0.7:
            if not task_info.get('background'):
                suggestions.append("添加研究背景信息以提高完整性")
            if len(checklist) < 3:
                suggestions.append("增加评分项数量至至少3个")

        # 准确性建议
        if dimension_scores['accuracy'] < 0.7:
            total_weight = sum(item.get('weight', 0) for item in checklist)
            if abs(total_weight - 1.0) > 0.2:
                suggestions.append(f"调整评分权重，当前总和为{total_weight:.2f}，建议接近1.0")

        # 清晰度建议
        if dimension_scores['clarity'] < 0.7:
            if len(task_info.get('task', '')) < 30:
                suggestions.append("扩展任务描述，增加更多细节")

        # 可行性建议
        if dimension_scores['feasibility'] < 0.7:
            if not task_info.get('data'):
                suggestions.append("添加数据文件信息以提高可行性")

        return suggestions

    def compare_scores(self, score1: Dict, score2: Dict) -> Dict[str, Any]:
        """
        比较两次评分

        Args:
            score1: 第一次评分
            score2: 第二次评分

        Returns:
            比较结果
        """
        diff = score2['overall_score'] - score1['overall_score']

        dimension_diffs = {}
        for dim in score1['dimension_scores']:
            dimension_diffs[dim] = (
                score2['dimension_scores'][dim] - score1['dimension_scores'][dim]
            )

        return {
            'improved': diff > 0,
            'score_difference': round(diff, 3),
            'dimension_differences': {k: round(v, 3) for k, v in dimension_diffs.items()},
            'improvement_percentage': round((diff / score1['overall_score']) * 100, 1) if score1['overall_score'] > 0 else 0
        }

    def get_quality_summary(self, score_result: Dict) -> str:
        """
        获取质量摘要

        Args:
            score_result: 评分结果

        Returns:
            质量摘要文本
        """
        summary = []
        summary.append("=== 质量评分报告 ===\n")

        # 总体评分
        overall_score = score_result['overall_score']
        grade = score_result['grade']
        summary.append(f"总体评分: {overall_score:.3f}/1.000 (评级: {grade})")

        passed = score_result.get('passed', False)
        status = "✅ 通过" if passed else "❌ 不通过"
        summary.append(f"状态: {status}\n")

        # 维度评分
        summary.append("维度评分:")
        dimension_scores = score_result['dimension_scores']
        for dim, score in dimension_scores.items():
            dim_name = {
                'completeness': '完整性',
                'accuracy': '准确性',
                'clarity': '清晰度',
                'feasibility': '可行性'
            }.get(dim, dim)

            # 进度条表示
            bar_length = int(score * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            summary.append(f"  {dim_name}: {bar} {score:.3f}")

        # 改进建议
        suggestions = score_result.get('suggestions', [])
        if suggestions:
            summary.append("\n改进建议:")
            for i, suggestion in enumerate(suggestions, 1):
                summary.append(f"  {i}. {suggestion}")

        return "\n".join(summary)
