"""
验证器 - 验证生成的task_info和checklist的正确性
"""

from typing import Dict, List, Any


def validate_task_info(task_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证task_info.json的必需字段

    Args:
        task_info: 任务信息字典

    Returns:
        验证结果字典，包含valid, errors, warnings字段
    """
    errors = []
    warnings = []

    # 必需字段检查
    if 'task' not in task_info:
        errors.append("缺少必需字段: 'task'")
    elif not task_info['task'] or not isinstance(task_info['task'], str):
        errors.append("字段'task'必须是非空字符串")

    # 推荐字段检查
    if 'data' not in task_info:
        warnings.append("缺少推荐字段: 'data'")
    else:
        # 验证data字段格式
        if not isinstance(task_info['data'], list):
            errors.append("字段'data'必须是列表")
        else:
            for i, data_item in enumerate(task_info['data']):
                if not isinstance(data_item, dict):
                    errors.append(f"data[{i}]必须是字典")
                    continue

                # 检查data子字段
                if 'name' not in data_item:
                    errors.append(f"data[{i}]缺少必需字段: 'name'")
                if 'description' not in data_item:
                    warnings.append(f"data[{i}]缺少推荐字段: 'description'")

    # 可选字段类型检查
    optional_fields = {
        'background': str,
        'research_goal': str,
        'hypothesis': str,
        'experimental_design': dict,
        'expected_outcomes': list,
        'constraints': list,
        'success_criteria': list
    }

    for field, expected_type in optional_fields.items():
        if field in task_info:
            if not isinstance(task_info[field], expected_type):
                errors.append(f"字段'{field}'应该是{expected_type.__name__}类型")

    # 检查权重和完整性
    completeness_score = _calculate_completeness(task_info)

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'completeness_score': completeness_score
    }


def validate_checklist(checklist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证checklist.json的必需字段

    Args:
        checklist: 检查清单列表

    Returns:
        验证结果字典，包含valid, errors, warnings字段
    """
    errors = []
    warnings = []

    # 基本类型检查
    if not isinstance(checklist, list):
        errors.append("checklist必须是列表")
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings,
            'total_weight': 0
        }

    if len(checklist) == 0:
        errors.append("checklist不能为空")

    # 检查每个item
    total_weight = 0.0
    has_image = False
    has_text = False

    for i, item in enumerate(checklist):
        if not isinstance(item, dict):
            errors.append(f"item[{i}]必须是字典")
            continue

        # 必需字段检查
        if 'content' not in item:
            errors.append(f"item[{i}]缺少必需字段: 'content'")
        elif not item['content'] or not isinstance(item['content'], str):
            errors.append(f"item[{i}]的'content'必须是非空字符串")

        # 字段类型检查
        if 'type' in item:
            if item['type'] not in ['text', 'image']:
                errors.append(f"item[{i}]的'type'必须是'text'或'image'")

            if item['type'] == 'image':
                has_image = True
                # image类型必须有path字段
                if 'path' not in item:
                    errors.append(f"item[{i}]的image类型必须有'path'字段")
            elif item['type'] == 'text':
                has_text = True

        # 权重检查
        if 'weight' in item:
            weight = item['weight']
            if not isinstance(weight, (int, float)):
                errors.append(f"item[{i}]的'weight'必须是数字")
            elif weight < 0 or weight > 1:
                errors.append(f"item[{i}]的'weight'必须在0-1之间")
            else:
                total_weight += weight
        else:
            warnings.append(f"item[{i}]缺少推荐字段: 'weight'")

        # 可选字段检查
        optional_fields = ['id', 'keywords', 'evaluation_criteria']
        for field in optional_fields:
            if field in item:
                if field == 'keywords' and not isinstance(item[field], list):
                    errors.append(f"item[{i}]的'keywords'必须是列表")
                if field == 'id' and not isinstance(item[field], str):
                    errors.append(f"item[{i}]的'id'必须是字符串")

    # 权重和建议检查
    if total_weight > 0 and abs(total_weight - 1.0) > 0.1:
        warnings.append(f"权重总和({total_weight:.2f})建议接近1.0")

    if not has_image and len(checklist) > 5:
        warnings.append("建议包含至少一个image类型的评分项")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'total_weight': total_weight,
        'has_image': has_image,
        'has_text': has_text
    }


def _calculate_completeness(task_info: Dict[str, Any]) -> float:
    """计算task_info的完整性分数"""
    required_fields = ['task']
    recommended_fields = ['data', 'background', 'research_goal']
    optional_fields = ['hypothesis', 'experimental_design',
                      'expected_outcomes', 'constraints', 'success_criteria']

    score = 0.0

    # 必需字段 (50%)
    for field in required_fields:
        if field in task_info and task_info[field]:
            score += 0.5

    # 推荐字段 (30%)
    for field in recommended_fields:
        if field in task_info and task_info[field]:
            score += 0.1

    # 可选字段 (20%)
    for field in optional_fields:
        if field in task_info and task_info[field]:
            score += 0.04

    return min(score, 1.0)


def validate_generated_content(task_info: Dict[str, Any],
                               checklist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证整体生成内容的质量

    Args:
        task_info: 任务信息
        checklist: 检查清单

    Returns:
        整体验证结果
    """
    task_validation = validate_task_info(task_info)
    checklist_validation = validate_checklist(checklist)

    # 计算整体质量分数
    quality_scores = {
        'task_completeness': task_validation.get('completeness_score', 0),
        'task_valid': 1 if task_validation['valid'] else 0,
        'checklist_valid': 1 if checklist_validation['valid'] else 0,
        'checklist_balance': _check_checklist_balance(checklist),
        'content_richness': _check_content_richness(task_info, checklist)
    }

    overall_score = sum(quality_scores.values()) / len(quality_scores)

    return {
        'overall_score': overall_score,
        'quality_scores': quality_scores,
        'task_validation': task_validation,
        'checklist_validation': checklist_validation,
        'passed': overall_score >= 0.7 and task_validation['valid'] and checklist_validation['valid']
    }


def _check_checklist_balance(checklist: List[Dict[str, Any]]) -> float:
    """检查checklist的平衡性"""
    if not checklist:
        return 0.0

    total_weight = sum(item.get('weight', 0) for item in checklist)
    if total_weight == 0:
        return 0.5

    # 检查权重分布是否合理
    target_weight = 1.0
    diff = abs(total_weight - target_weight)

    # 越接近1.0越好
    return max(0, 1.0 - diff)


def _check_content_richness(task_info: Dict[str, Any],
                           checklist: List[Dict[str, Any]]) -> float:
    """检查内容丰富度"""
    score = 0.0

    # 检查task_info的字段数量
    task_fields = len([f for f in task_info if task_info[f]])
    score += min(task_fields * 0.1, 0.5)

    # 检查checklist的项目数量
    checklist_items = len(checklist)
    score += min(checklist_items * 0.1, 0.3)

    # 检查描述长度
    task_desc_len = len(task_info.get('task', ''))
    if task_desc_len > 20:
        score += 0.2

    return min(score, 1.0)
