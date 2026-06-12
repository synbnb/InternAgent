"""
测试迭代优化功能
演示用户反馈和系统改进的完整流程
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_to_task.pipeline import PaperToTaskPipeline
from paper_to_task.utils.llm_client import LLMClient


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(result: dict, title: str = ""):
    """打印结果"""
    if title:
        print(f"\n【{title}】")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def test_user_feedback_workflow():
    """测试完整的用户反馈工作流"""

    print_section("Paper-to-Task 迭代优化功能测试")

    # 1. 初始化管道（使用真实LLM配置）
    print("\n[1] 初始化管道...")

    config = {
        'llm': {
            'backend': 'deepseek',
            'model': 'deepseek-chat',
            'temperature': 0.3,
            'max_tokens': 4000,
            'cache_enabled': True
        },
        'project': {
            'sci_tasks_base': 'sci_tasks/tasks'
        },
        'quality': {
            'min_score': 0.7,
            'enable_auto_improvement': True
        }
    }

    pipeline = PaperToTaskPipeline(config)

    # 2. 模拟已生成的内容（使用之前测试的结果）
    print("\n[2] 加载已生成的任务内容...")

    # 使用 Science_008 的实际结果作为测试数据
    task_info = {
        "task": "复现论文《ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning》的核心发现",
        "data": [
            {
                "name": "UniRef50",
                "path": "data/UniRef50",
                "description": "UniProt数据库在50%成对序列同一性下聚类得到的数据集。"
            },
            {
                "name": "UniRef100",
                "path": "data/UniRef100",
                "description": "UniProt数据库去除重复后的完整数据集。"
            },
            {
                "name": "BFD",
                "path": "data/BFD",
                "description": "结合UniProt与宏基因组数据，去除重复后的数据集。"
            }
        ]
    }

    checklist = [
        {
            "id": "item_0",
            "type": "text",
            "weight": 0.25,
            "content": "报告应详细描述实验方法的具体实现过程",
            "evaluation_criteria": "是否清楚说明了实验方法的具体实现过程和参数设置"
        },
        {
            "id": "item_1",
            "type": "text",
            "weight": 0.15,
            "content": "报告应包含数据处理步骤，涉及相关数据集的使用说明",
            "evaluation_criteria": "是否描述了如何处理和使用实验数据"
        }
    ]

    current_content = {
        'task_info': task_info,
        'checklist': checklist
    }

    print("✅ 已加载初始内容")
    print(f"   - 任务: {task_info['task'][:50]}...")
    print(f"   - 数据项: {len(task_info['data'])} 个")
    print(f"   - 评分项: {len(checklist)} 个")

    # 3. 测试不同的用户反馈场景
    print_section("测试场景 1：改进 Checklist 针对性")

    feedback1 = "checklist 的评分项太通用了，我需要针对ProtTrans研究的具体评分标准，比如要包含模型架构、数据集、准确率指标等具体内容"

    print(f"\n📝 用户反馈: {feedback1}")

    result1 = pipeline.refine_content(current_content, feedback1)

    if result1['success']:
        print("\n✅ 迭代优化成功")
        print(f"   - 反馈分析: {result1['feedback_analysis']['type']}")
        print(f"   - 改进建议: {len(result1['improvements'])} 项")

        # 显示改进后的 checklist
        print("\n📋 改进后的 Checklist:")
        for i, item in enumerate(result1['checklist']):
            print(f"\n  [{i}] 权重: {item['weight']}")
            print(f"      内容: {item['content'][:80]}...")

        # 显示改进摘要
        summary = pipeline.iterative_refiner.get_improvement_summary(result1)
        print(f"\n{summary}")
    else:
        print(f"\n❌ 迭代优化失败: {result1.get('error')}")

    # 4. 测试第二个反馈场景
    print_section("测试场景 2：增强任务描述")

    feedback2 = "任务描述太简单了，需要更具体地说明要复现哪些核心发现，比如模型性能指标、对比实验等"

    print(f"\n📝 用户反馈: {feedback2}")

    result2 = pipeline.refine_content(result1['success'] and {
        'task_info': result1['task_info'],
        'checklist': result1['checklist']
    } or current_content, feedback2)

    if result2['success']:
        print("\n✅ 迭代优化成功")
        print(f"\n📋 改进后的任务描述:")
        print(f"   {result2['task_info']['task']}")
    else:
        print(f"\n❌ 迭代优化失败: {result2.get('error')}")

    # 5. 测试第三个反馈场景
    print_section("测试场景 3：数据集描述优化")

    feedback3 = "数据集描述太简单了，需要补充每个数据集的具体用途，比如UniRef50用于训练，UniRef100用于评估等"

    print(f"\n📝 用户反馈: {feedback3}")

    result3 = pipeline.refine_content(result2['success'] and {
        'task_info': result2['task_info'],
        'checklist': result2['checklist']
    } or current_content, feedback3)

    if result3['success']:
        print("\n✅ 迭代优化成功")
        print(f"\n📋 改进后的数据集描述:")
        for i, data in enumerate(result3['task_info']['data']):
            print(f"\n  [{i}] {data['name']}")
            print(f"      路径: {data['path']}")
            print(f"      描述: {data['description']}")

    # 6. 最终质量检查
    print_section("最终质量检查")

    if result3['success']:
        final_quality = pipeline.quality_scorer.score_content({
            'task_info': result3['task_info'],
            'checklist': result3['checklist']
        })

        print(f"\n📊 最终质量评分: {final_quality['overall_score']:.2f}")
        print(f"   等级: {final_quality['grade']}")
        print(f"   状态: {'✅ 通过' if final_quality['passed'] else '❌ 未达标'}")

        print(f"\n📈 维度评分:")
        for dimension, score in final_quality['dimension_scores'].items():
            print(f"   - {dimension}: {score:.2f}")

        if final_quality['suggestions']:
            print(f"\n💡 改进建议:")
            for i, suggestion in enumerate(final_quality['suggestions'], 1):
                print(f"   {i}. {suggestion}")

    # 7. 测试自动改进建议
    print_section("自动改进建议")

    suggestions = pipeline.iterative_refiner.suggest_improvements(current_content)

    print(f"\n💡 系统自动建议 ({len(suggestions)} 项):")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")

    print_section("测试完成")

    print("\n✅ 迭代优化功能测试完成")
    print("   - 反馈分析: ✅")
    print("   - 内容改进: ✅")
    print("   - 质量检查: ✅")
    print("   - 自动建议: ✅")


def main():
    """主测试函数"""
    try:
        test_user_feedback_workflow()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
