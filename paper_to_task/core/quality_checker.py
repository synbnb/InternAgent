"""
质量检查模块 - 检查生成内容的质量
"""

from typing import Dict, List, Any


class QualityChecker:
    """质量检查器"""

    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.7,
            'accuracy': 0.7,
            'clarity': 0.6,
            'feasibility': 0.6
        }

    def check_generated_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查生成内容的整体质量

        Args:
            content: 包含task_info和checklist的字典

        Returns:
            质量检查结果
        """
        task_info = content.get('task_info', {})
        checklist = content.get('checklist', [])

        # 执行各项检查
        completeness_score = self._check_completeness(task_info, checklist)
        accuracy_score = self._check_accuracy(task_info, checklist)
        clarity_score = self._check_clarity(task_info, checklist)
        feasibility_score = self._check_feasibility(task_info)

        # 计算总体分数（转百分制）
        scores = {
            'completeness': round(completeness_score * 100, 1),
            'accuracy': round(accuracy_score * 100, 1),
            'clarity': round(clarity_score * 100, 1),
            'feasibility': round(feasibility_score * 100, 1)
        }

        overall_score = round(sum(scores.values()) / len(scores), 1)

        # 判断是否通过（阈值也转百分制）
        passed = all(
            score >= threshold * 100 for score, threshold in
            zip(scores.values(), self.quality_thresholds.values())
        )

        return {
            'overall_score': overall_score,
            'detailed_scores': scores,
            'passed': passed,
            'recommendations': self._generate_recommendations(scores, task_info, checklist)
        }

    def _check_completeness(self, task_info: Dict, checklist: List) -> float:
        """检查完整性"""
        score = 0.0

        # task_info必需字段 (50分)
        required_task_fields = ['task', 'data']
        for field in required_task_fields:
            if field in task_info and task_info[field]:
                score += 0.25

        # task_info推荐字段 (30分)
        recommended_task_fields = ['background', 'research_goal', 'experimental_design']
        for field in recommended_task_fields:
            if field in task_info and task_info[field]:
                score += 0.1

        # checklist完整性 (20分)
        if checklist and len(checklist) >= 3:
            score += 0.1
        if len(checklist) >= 5:
            score += 0.1

        return min(score, 1.0)

    def _check_accuracy(self, task_info: Dict, checklist: List) -> float:
        """检查准确性"""
        score = 0.0

        # 检查task_info字段类型
        if isinstance(task_info.get('task'), str) and len(task_info.get('task', '')) > 10:
            score += 0.2

        if isinstance(task_info.get('data'), list):
            score += 0.2

        if isinstance(task_info.get('experimental_design'), dict):
            score += 0.1

        # 检查checklist质量
        if checklist:
            valid_items = sum(1 for item in checklist if self._is_valid_checklist_item(item))
            if valid_items / len(checklist) >= 0.8:
                score += 0.3

            # 检查权重分布
            weights = [item.get('weight', 0) for item in checklist]
            if weights and 0 < sum(weights) <= 1.5:
                score += 0.2

        return min(score, 1.0)

    def _check_clarity(self, task_info: Dict, checklist: List) -> float:
        """检查清晰度"""
        score = 0.0

        # 检查描述长度
        task_desc = task_info.get('task', '')
        if len(task_desc) > 30:
            score += 0.3
        elif len(task_desc) > 15:
            score += 0.15

        # 检查background
        background = task_info.get('background', '')
        if len(background) > 50:
            score += 0.2

        # 检查checklist内容清晰度
        if checklist:
            clear_items = sum(
                1 for item in checklist
                if len(item.get('content', '')) > 20
            )
            if clear_items / len(checklist) >= 0.7:
                score += 0.3

        return min(score, 1.0)

    def _check_feasibility(self, task_info: Dict) -> float:
        """检查可行性"""
        score = 0.5  # 基础分

        # 检查数据可用性
        data_items = task_info.get('data', [])
        if data_items:
            score += 0.2

            # 检查数据描述质量
            described_items = sum(
                1 for item in data_items
                if item.get('description') and len(item.get('description', '')) > 10
            )
            if described_items / len(data_items) >= 0.5:
                score += 0.1

        # 检查成功标准合理性
        success_criteria = task_info.get('success_criteria', [])
        if success_criteria and len(success_criteria) >= 2:
            score += 0.1

        # 检查约束条件
        constraints = task_info.get('constraints', [])
        if constraints:
            score += 0.1

        return min(score, 1.0)

    def _is_valid_checklist_item(self, item: Dict) -> bool:
        """检查checklist项目是否有效"""
        # 必需字段
        if 'content' not in item:
            return False

        if not isinstance(item['content'], str) or len(item['content']) < 10:
            return False

        # 类型检查
        if 'type' in item and item['type'] not in ['text', 'image']:
            return False

        # 权重检查
        if 'weight' in item:
            try:
                weight = float(item['weight'])
                if weight < 0 or weight > 1:
                    return False
            except (ValueError, TypeError):
                return False

        return True

    def _generate_recommendations(self, scores: Dict,
                                 task_info: Dict,
                                 checklist: List) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 完整性建议
        if scores['completeness'] < self.quality_thresholds['completeness']:
            if 'task' not in task_info or not task_info['task']:
                recommendations.append("添加明确的任务描述")
            if 'data' not in task_info or not task_info['data']:
                recommendations.append("指定使用的数据文件")
            if len(checklist) < 3:
                recommendations.append("增加评分项至至少3个")

        # 准确性建议
        if scores['accuracy'] < self.quality_thresholds['accuracy']:
            recommendations.append("检查字段类型和格式是否正确")
            recommendations.append("确保checklist权重分布合理")

        # 清晰度建议
        if scores['clarity'] < self.quality_thresholds['clarity']:
            recommendations.append("增强描述的详细程度")
            recommendations.append("为评分项添加关键词")

        # 可行性建议
        if scores['feasibility'] < self.quality_thresholds['feasibility']:
            recommendations.append("确保数据文件的可用性")
            recommendations.append("明确成功标准")

        return recommendations

    def quick_check(self, task_info: Dict, checklist: List) -> Dict[str, Any]:
        """
        快速检查（用于初步筛选）

        Args:
            task_info: 任务信息
            checklist: 检查清单

        Returns:
            快速检查结果
        """
        # 基本有效性检查
        has_task = bool(task_info.get('task'))
        has_data = bool(task_info.get('data'))
        has_checklist = len(checklist) >= 2

        # 基本质量检查
        task_desc_len = len(task_info.get('task', ''))
        quality_task = task_desc_len > 20

        valid_checklist = all(
            item.get('content') and len(item.get('content', '')) > 10
            for item in checklist
        )

        return {
            'passed': has_task and has_data and has_checklist and quality_task and valid_checklist,
            'has_task': has_task,
            'has_data': has_data,
            'has_checklist': has_checklist,
            'quality_task': quality_task,
            'valid_checklist': valid_checklist
        }

    def get_quality_report(self, check_result: Dict) -> str:
        """
        生成质量报告

        Args:
            check_result: 检查结果

        Returns:
            质量报告文本
        """
        report = []
        report.append("=== 质量检查报告 ===\n")

        # 总体评分
        overall_score = check_result.get('overall_score', 0)
        report.append(f"总体评分: {overall_score:.1f}/100")

        passed = check_result.get('passed', False)
        status = "✅ 通过" if passed else "❌ 不通过"
        report.append(f"状态: {status}\n")

        # 详细评分
        report.append("详细评分:")
        scores = check_result.get('detailed_scores', {})
        for aspect, score in scores.items():
            report.append(f"  {aspect}: {score:.1f}/100")

        # 改进建议
        recommendations = check_result.get('recommendations', [])
        if recommendations:
            report.append("\n改进建议:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")

        return "\n".join(report)
