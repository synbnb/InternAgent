"""
迭代优化功能演示
展示用户反馈和系统改进的完整流程（使用模拟LLM响应）
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_to_task.pipeline import PaperToTaskPipeline


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_user_feedback_workflow():
    """演示完整的用户反馈工作流"""

    print_section("Paper-to-Task 迭代优化功能演示")

    # 1. 初始化管道（使用默认配置，包含mock后端）
    print("\n[1] 初始化管道...")
    pipeline = PaperToTaskPipeline()
    print("✅ 管道初始化完成（使用 mock LLM 后端用于演示）")

    # 2. 加载初始内容（使用真实数据但简化版）
    print("\n[2] 加载初始任务内容...")

    initial_task_info = {
        "task": "复现论文《ProtTrans》的核心发现",
        "data": [
            {
                "name": "protein_data",
                "path": "data/protein_data",
                "description": "蛋白质序列数据"
            }
        ]
    }

    initial_checklist = [
        {
            "id": "item_0",
            "type": "text",
            "weight": 0.5,
            "content": "报告应该描述实验方法",
            "keywords": ["方法", "实验"],
            "evaluation_criteria": "是否描述了方法"
        },
        {
            "id": "item_1",
            "type": "text",
            "weight": 0.5,
            "content": "报告应该包含结果",
            "keywords": ["结果", "数据"],
            "evaluation_criteria": "是否包含结果"
        }
    ]

    current_content = {
        'task_info': initial_task_info,
        'checklist': initial_checklist
    }

    print("✅ 已加载初始内容")
    print(f"   任务: {initial_task_info['task']}")
    print(f"   数据项: {len(initial_task_info['data'])} 个")
    print(f"   评分项: {len(initial_checklist)} 个")

    # 3. 演示迭代优化流程
    print_section("演示：用户反馈驱动的迭代改进")

    feedbacks = [
        "checklist 的评分项太通用，需要针对蛋白质语言模型的具体评估标准",
        "数据集描述太简单，需要说明UniRef50、UniRef100等具体数据集的用途",
        "任务描述需要更具体，要说明复现哪些核心发现，比如Q3准确率、对比实验等"
    ]

    for i, feedback in enumerate(feedbacks, 1):
        print(f"\n{'='*70}")
        print(f"  第 {i} 轮改进")
        print(f"{'='*70}")

        print(f"\n📝 用户反馈:")
        print(f"   \"{feedback}\"")

        # 展示当前内容的问题
        print(f"\n📊 当前内容分析:")
        if i == 1:
            print(f"   ❌ 评分项过于通用: '{current_content['checklist'][0]['content']}'")
            print(f"   ❌ 缺少具体的研究细节")
        elif i == 2:
            print(f"   ❌ 数据集描述不具体: '{current_content['task_info']['data'][0]['description']}'")
            print(f"   ❌ 没有说明具体数据集名称和用途")

        # 模拟改进效果
        print(f"\n🔧 系统改进中...")

        # 为演示目的，手动模拟改进效果
        if i == 1:
            # 模拟LLM生成针对性checklist
            improved_checklist = [
                {
                    "id": "item_0",
                    "type": "text",
                    "weight": 0.30,
                    "content": "模型架构与训练策略：评估是否使用了Transformer-XL、BERT等模型，并在UniRef数据集上进行了充分预训练",
                    "keywords": ["Transformer", "UniRef", "预训练"],
                    "evaluation_criteria": "明确列出使用的模型类型和训练数据集"
                },
                {
                    "id": "item_1",
                    "type": "text",
                    "weight": 0.30,
                    "content": "二级结构预测性能：检查ProtT5模型的Q3准确率是否达到81%-87%，并与使用进化信息的方法进行对比",
                    "keywords": ["Q3准确率", "二级结构", "ProtT5"],
                    "evaluation_criteria": "提供具体的Q3数值和对比结果"
                },
                {
                    "id": "item_2",
                    "type": "text",
                    "weight": 0.20,
                    "content": "生物物理特征捕获：评估模型在亚细胞定位和膜蛋白识别任务上的表现",
                    "keywords": ["亚细胞定位", "膜蛋白", "生物物理"],
                    "evaluation_criteria": "提供Q10和Q2准确率数值"
                },
                {
                    "id": "item_3",
                    "type": "text",
                    "weight": 0.10,
                    "content": "可复现性：检查是否提供了模型权重下载和实验代码",
                    "keywords": ["模型权重", "代码", "可复现性"],
                    "evaluation_criteria": "提供权重和代码的访问链接"
                },
                {
                    "id": "item_4",
                    "type": "text",
                    "weight": 0.10,
                    "content": "实验设计完整性：确保包含了与最先进方法的对比实验",
                    "keywords": ["对比实验", "baseline", "评估"],
                    "evaluation_criteria": "明确列出对比的方法和评估指标"
                }
            ]
            current_content['checklist'] = improved_checklist

        elif i == 2:
            # 模拟改进数据集描述
            improved_data = [
                {
                    "name": "UniRef50",
                    "path": "data/UniRef50",
                    "description": "UniProt数据库在50%序列同一性下聚类，用于模型预训练"
                },
                {
                    "name": "UniRef100",
                    "path": "data/UniRef100",
                    "description": "UniProt完整去重数据集，用于模型评估和下游任务"
                },
                {
                    "name": "BFD",
                    "path": "data/BFD",
                    "description": "结合UniProt与宏基因组数据的大规模训练集"
                }
            ]
            current_content['task_info']['data'] = improved_data

        elif i == 3:
            # 模拟改进任务描述
            current_content['task_info']['task'] = "复现ProtTrans论文的核心发现：验证蛋白质语言模型在二级结构预测中达到Q3=81%-87%的性能，且无需多序列比对或进化信息"

        # 展示改进结果
        print(f"\n✅ 改进完成")

        if i == 1:
            print(f"\n📋 改进后的评分项:")
            for j, item in enumerate(current_content['checklist'][:2], 1):
                print(f"\n   [{j}] 权重: {item['weight']}")
                print(f"       内容: {item['content'][:60]}...")

        elif i == 2:
            print(f"\n📋 改进后的数据集:")
            for j, data in enumerate(current_content['task_info']['data'], 1):
                print(f"\n   [{j}] {data['name']}")
                print(f"       描述: {data['description']}")
        elif i == 3:
            print(f"\n📋 改进后的任务:")
            print(f"   {current_content['task_info']['task']}")

        # 质量评分
        quality_score = pipeline.quality_scorer.score_content(current_content)
        print(f"\n📊 质量评分: {quality_score['overall_score']:.2f} ({quality_score['grade']})")

    # 4. 最终结果展示
    print_section("最终改进结果")

    print(f"\n✅ 经过 {len(feedbacks)} 轮迭代改进后的内容:")

    print(f"\n📄 任务描述:")
    print(f"   {current_content['task_info']['task']}")

    print(f"\n📁 数据集 ({len(current_content['task_info']['data'])}个):")
    for data in current_content['task_info']['data']:
        print(f"   - {data['name']}: {data['description']}")

    print(f"\n📋 评分项 ({len(current_content['checklist'])}个):")
    for i, item in enumerate(current_content['checklist'], 1):
        print(f"\n   [{i}] 权重: {item['weight']} | {item['content'][:50]}...")

    # 最终质量评分
    final_quality = pipeline.quality_scorer.score_content(current_content)

    print(f"\n📊 最终质量评分:")
    print(f"   总分: {final_quality['overall_score']:.2f}")
    print(f"   等级: {final_quality['grade']}")
    print(f"   状态: {'✅ 通过' if final_quality['passed'] else '❌ 未达标'}")

    print(f"\n📈 维度评分:")
    dimensions = final_quality['dimension_scores']
    print(f"   - 完整性: {dimensions['completeness']:.2f}")
    print(f"   - 准确性: {dimensions['accuracy']:.2f}")
    print(f"   - 清晰度: {dimensions['clarity']:.2f}")
    print(f"   - 可行性: {dimensions['feasibility']:.2f}")

    # 5. 功能说明
    print_section("迭代优化功能说明")

    print(f"""
🔄 迭代优化工作流程:

1. 用户提供反馈 → 2. 系统分析反馈 → 3. LLM生成改进 → 4. 质量检查

📝 反馈类型支持:
   - 缺少信息 (missing_info)
   - 描述不准确 (inaccurate)
   - 结构问题 (structure_issue)
   - 内容增强 (content_enhancement)
   - 一般改进 (general)

🎯 改进目标:
   - task_info: 更具体的研究目标、详细的数据描述
   - checklist: 针对性评分项、可量化的评估标准

⚠️ 实际使用要求:
   - 配置真实LLM API密钥 (DeepSeek/OpenAI等)
   - 在配置文件中设置: llm.backend = 'deepseek'
   - 提供有效的 api_key

💡 演示说明:
   - 本演示使用模拟数据展示改进效果
   - 实际使用时LLM会根据具体论文内容生成针对性改进
   - 系统支持多轮迭代直到质量达标
""")

    print_section("演示完成")

    print("\n✅ 迭代优化功能演示完成！")
    print("   - 反馈分析: ✅")
    print("   - 内容改进: ✅")
    print("   - 质量检查: ✅")
    print("   - 多轮迭代: ✅")


if __name__ == "__main__":
    demo_user_feedback_workflow()
