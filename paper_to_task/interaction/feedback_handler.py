"""
反馈处理模块 - 处理用户反馈
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict


class FeedbackHandler:
    """反馈处理器"""

    def __init__(self):
        # 反馈类型
        self.feedback_types = {
            'missing_info': '缺少信息',
            'inaccurate': '描述不准确',
            'structure_issue': '结构问题',
            'quality_low': '质量不足',
            'general': '一般建议'
        }

        # 反馈历史
        self.feedback_history = []

    def parse_feedback(self, feedback_text: str) -> Dict[str, Any]:
        """
        解析用户反馈

        Args:
            feedback_text: 用户反馈文本

        Returns:
            解析后的反馈信息
        """
        # 分析反馈类型
        feedback_type = self._classify_feedback(feedback_text)

        # 提取目标
        target = self._extract_target(feedback_text)

        # 提取具体问题
        issues = self._extract_issues(feedback_text)

        # 提取改进建议
        suggestions = self._extract_suggestions(feedback_text)

        # 评估优先级
        priority = self._assess_priority(feedback_text)

        return {
            'type': feedback_type,
            'text': feedback_text,
            'target': target,
            'issues': issues,
            'suggestions': suggestions,
            'priority': priority,
            'timestamp': self._get_timestamp()
        }

    def _classify_feedback(self, feedback: str) -> str:
        """分类反馈类型"""
        feedback_lower = feedback.lower()

        # 关键词匹配
        type_keywords = {
            'missing_info': ['缺少', '缺失', '没有', 'less', 'missing'],
            'inaccurate': ['不准确', '错误', '错误', 'wrong', 'inaccurate'],
            'structure_issue': ['结构', '组织', '排列', 'structure', 'organize'],
            'quality_low': ['质量', '详细', '完善', 'quality', 'improve']
        }

        for fb_type, keywords in type_keywords.items():
            if any(keyword in feedback_lower for keyword in keywords):
                return fb_type

        return 'general'

    def _extract_target(self, feedback: str) -> str:
        """提取反馈目标"""
        feedback_lower = feedback.lower()

        if any(word in feedback_lower for word in ['task', '任务', '描述']):
            return 'task_info'

        if any(word in feedback_lower for word in ['checklist', '评分', '标准']):
            return 'checklist'

        if any(word in feedback_lower for word in ['data', '数据', '数据文件']):
            return 'data'

        if any(word in feedback_lower for word in ['both', '全部', '所有']):
            return 'both'

        return 'general'

    def _extract_issues(self, feedback: str) -> List[str]:
        """提取具体问题"""
        issues = []

        # 简单的句子分割
        sentences = feedback.split('。')

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:
                # 避免重复
                if sentence not in issues:
                    issues.append(sentence)

        return issues

    def _extract_suggestions(self, feedback: str) -> List[str]:
        """提取改进建议"""
        suggestions = []

        # 查找建议关键词
        suggestion_keywords = ['建议', '希望', '需要', 'should', 'suggest', 'want']

        sentences = feedback.split('。')

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in suggestion_keywords):
                suggestion = sentence.strip()
                if len(suggestion) > 5 and suggestion not in suggestions:
                    suggestions.append(suggestion)

        return suggestions

    def _assess_priority(self, feedback: str) -> str:
        """评估优先级"""
        feedback_lower = feedback.lower()

        # 高优先级关键词
        high_priority_keywords = ['重要', '关键', '必须', 'urgent', 'critical', 'must']

        # 中优先级关键词
        medium_priority_keywords = ['建议', '可以', 'should', 'could']

        if any(keyword in feedback_lower for keyword in high_priority_keywords):
            return 'high'
        elif any(keyword in feedback_lower for keyword in medium_priority_keywords):
            return 'medium'
        else:
            return 'low'

    def add_feedback(self, parsed_feedback: Dict):
        """
        添加反馈到历史记录

        Args:
            parsed_feedback: 解析后的反馈
        """
        self.feedback_history.append(parsed_feedback)

    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        获取反馈摘要

        Returns:
            反馈摘要统计
        """
        if not self.feedback_history:
            return {
                'total_feedbacks': 0,
                'type_distribution': {},
                'target_distribution': {},
                'priority_distribution': {}
            }

        # 类型分布
        type_counts = defaultdict(int)
        for feedback in self.feedback_history:
            type_counts[feedback['type']] += 1

        # 目标分布
        target_counts = defaultdict(int)
        for feedback in self.feedback_history:
            target_counts[feedback['target']] += 1

        # 优先级分布
        priority_counts = defaultdict(int)
        for feedback in self.feedback_history:
            priority_counts[feedback['priority']] += 1

        return {
            'total_feedbacks': len(self.feedback_history),
            'type_distribution': dict(type_counts),
            'target_distribution': dict(target_counts),
            'priority_distribution': dict(priority_counts),
            'recent_feedbacks': self.feedback_history[-5:]  # 最近5条
        }

    def get_common_issues(self) -> List[Dict[str, Any]]:
        """
        获取常见问题

        Returns:
            常见问题列表
        """
        issue_counts = defaultdict(int)

        for feedback in self.feedback_history:
            for issue in feedback.get('issues', []):
                issue_counts[issue] += 1

        # 排序并返回前10个
        common_issues = [
            {'issue': issue, 'count': count}
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        return common_issues[:10]

    def clear_history(self):
        """清空反馈历史"""
        self.feedback_history = []

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def format_feedback_report(self) -> str:
        """
        格式化反馈报告

        Returns:
            反馈报告文本
        """
        summary = self.get_feedback_summary()

        report = []
        report.append("=== 用户反馈报告 ===\n")

        # 总体统计
        report.append(f"总反馈数量: {summary['total_feedbacks']}")

        # 类型分布
        type_dist = summary.get('type_distribution', {})
        if type_dist:
            report.append("\n反馈类型分布:")
            for fb_type, count in type_dist.items():
                type_name = self.feedback_types.get(fb_type, fb_type)
                report.append(f"  {type_name}: {count}")

        # 目标分布
        target_dist = summary.get('target_distribution', {})
        if target_dist:
            report.append("\n反馈目标分布:")
            for target, count in target_dist.items():
                report.append(f"  {target}: {count}")

        # 常见问题
        common_issues = self.get_common_issues()
        if common_issues:
            report.append("\n常见问题:")
            for i, issue in enumerate(common_issues[:5], 1):
                report.append(f"  {i}. {issue['issue']} (出现{issue['count']}次)")

        return "\n".join(report)

    def suggest_improvements_from_feedback(self) -> List[str]:
        """
        从反馈历史中建议改进

        Returns:
            改进建议列表
        """
        suggestions = []

        # 从常见问题中提取建议
        common_issues = self.get_common_issues()

        if common_issues:
            suggestions.append("重点关注以下常见问题:")
            for i, issue in enumerate(common_issues[:3], 1):
                suggestions.append(f"  {i}. {issue['issue']}")

        # 从类型分布中获取建议
        summary = self.get_feedback_summary()
        type_dist = summary.get('type_distribution', {})

        if type_dist.get('missing_info', 0) > 2:
            suggestions.append("考虑添加更多详细信息到task_info和checklist")

        if type_dist.get('inaccurate', 0) > 2:
            suggestions.append("注意检查描述的准确性")

        if type_dist.get('structure_issue', 0) > 2:
            suggestions.append("考虑优化内容结构")

        return suggestions
