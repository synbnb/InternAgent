"""
命令行界面 - 提供命令行交互功能
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class CLIInterface:
    """命令行交互界面"""

    def __init__(self, pipeline):
        """
        初始化CLI界面

        Args:
            pipeline: PaperToTaskPipeline实例
        """
        self.pipeline = pipeline

    def run_interactive(self, pdf_path: str,
                       auto_create: bool = False) -> Dict[str, Any]:
        """
        运行交互式处理流程

        Args:
            pdf_path: PDF文件路径
            auto_create: 是否自动创建项目

        Returns:
            最终处理结果
        """
        self._print_header()

        # 第一步：处理PDF
        print(f"📄 正在处理PDF: {pdf_path}")
        print("-" * 60)

        result = self.pipeline.process_pdf(pdf_path, auto_improve=True)

        if not result['success']:
            print(f"❌ 处理失败: {result.get('error')}")
            return result

        # 显示生成结果
        self._display_generation_result(result)

        # 第二步：用户确认循环
        if not auto_create:
            confirmation_result = self._confirmation_loop(result)
            if not confirmation_result['confirmed']:
                return confirmation_result

            result = confirmation_result['result']

        # 第三步：创建项目
        if auto_create or self._ask_yes_no("\n是否创建sci_tasks项目？"):
            creation_result = self._create_project(result, pdf_path)
            result.update(creation_result)

        return result

    def run_batch(self, pdf_dir: str,
                 output_dir: str,
                 pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        批量处理PDF文件

        Args:
            pdf_dir: PDF文件目录
            output_dir: 输出目录
            pattern: 文件匹配模式

        Returns:
            批量处理结果
        """
        import glob

        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            return {
                'success': False,
                'error': f"目录不存在: {pdf_dir}"
            }

        # 查找PDF文件
        pdf_files = list(pdf_path.glob(pattern))

        if not pdf_files:
            return {
                'success': False,
                'error': f"未找到PDF文件: {pdf_dir}/{pattern}"
            }

        print(f"找到 {len(pdf_files)} 个PDF文件")
        print("-" * 60)

        # 批量处理
        results = self.pipeline.batch_process(
            [str(f) for f in pdf_files],
            output_dir
        )

        return {
            'success': True,
            'total_files': len(pdf_files),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }

    def _print_header(self):
        """打印标题头"""
        print("=" * 60)
        print("📚 Paper-to-Task 自动化系统")
        print("   将论文PDF转换为sci_tasks任务")
        print("=" * 60)
        print()

    def _display_generation_result(self, result: Dict):
        """显示生成结果"""
        print("\n📊 生成结果:")
        print("-" * 60)

        # 质量评分
        quality = result.get('quality', {})
        print(f"质量评分: {quality.get('score', 0):.2f}/1.00 (评级: {quality.get('grade', 'N/A')})")
        print(f"状态: {'✅ 通过' if quality.get('passed') else '❌ 未达标'}")

        # 任务信息摘要
        task_info = result.get('task_info', {})
        print(f"\n任务描述: {task_info.get('task', '')[:80]}...")

        # 数据文件
        data_items = task_info.get('data', [])
        print(f"数据文件: {len(data_items)} 个")

        # 评分项
        checklist = result.get('checklist', [])
        print(f"评分项: {len(checklist)} 个")

        # 维度评分
        dimension_scores = quality.get('dimension_scores', {})
        if dimension_scores:
            print("\n维度评分:")
            for dim, score in dimension_scores.items():
                dim_name = {
                    'completeness': '完整性',
                    'accuracy': '准确性',
                    'clarity': '清晰度',
                    'feasibility': '可行性'
                }.get(dim, dim)
                print(f"  {dim_name}: {score:.2f}")

    def _confirmation_loop(self, result: Dict) -> Dict:
        """用户确认循环"""
        current_content = {
            'task_info': result['task_info'],
            'checklist': result['checklist']
        }

        while True:
            print("\n" + "=" * 60)
            print("请选择操作:")
            print("  1. 查看详细内容")
            print("  2. 确认并继续")
            print("  3. 提供反馈并改进")
            print("  4. 取消")
            print("=" * 60)

            choice = input("\n请输入选项 (1-4): ").strip()

            if choice == '1':
                self._display_detailed_content(current_content)

            elif choice == '2':
                # 确认并继续
                result['task_info'] = current_content['task_info']
                result['checklist'] = current_content['checklist']
                return {
                    'confirmed': True,
                    'result': result
                }

            elif choice == '3':
                # 提供反馈并改进
                feedback = input("\n请提供改进建议: ").strip()
                if feedback:
                    print("\n正在根据反馈改进内容...")
                    refinement_result = self.pipeline.refine_content(
                        current_content,
                        feedback
                    )

                    if refinement_result['success']:
                        current_content = {
                            'task_info': refinement_result['task_info'],
                            'checklist': refinement_result['checklist']
                        }

                        # 显示改进摘要
                        print("\n" + refinement_result.get('summary', ''))

                        # 显示新评分
                        new_quality = refinement_result.get('quality', {})
                        print(f"\n新评分: {new_quality.get('score', 0):.2f}/1.00")
                    else:
                        print(f"❌ 改进失败: {refinement_result.get('error')}")
                else:
                    print("未提供反馈，返回主菜单")

            elif choice == '4':
                # 取消
                return {
                    'confirmed': False,
                    'message': '用户取消操作'
                }

            else:
                print("❌ 无效选项，请重新输入")

    def _display_detailed_content(self, content: Dict):
        """显示详细内容"""
        print("\n" + "=" * 60)
        print("📋 task_info.json 内容:")
        print("=" * 60)
        print(json.dumps(content['task_info'], indent=2, ensure_ascii=False))

        print("\n" + "=" * 60)
        print("📊 checklist.json 内容:")
        print("=" * 60)
        print(json.dumps(content['checklist'], indent=2, ensure_ascii=False))

    def _create_project(self, result: Dict, pdf_path: str) -> Dict:
        """创建项目"""
        print("\n" + "-" * 60)

        # 获取任务名称
        task_name = input("请输入任务名称 (直接回车使用默认名称): ").strip()
        if not task_name:
            task_name = f"AutoTask_{result.get('paper_metadata', {}).get('year', '2024')}"

        # 获取领域
        research_info = result.get('research_info', {})
        domain = research_info.get('domain', 'Science')

        print(f"\n正在创建项目...")
        print(f"  任务名称: {task_name}")
        print(f"  领域: {domain}")

        creation_result = self.pipeline.create_project(
            task_name=task_name,
            task_info=result['task_info'],
            checklist=result['checklist'],
            pdf_path=pdf_path,
            domain=domain
        )

        if creation_result['success']:
            print("\n✅ 项目创建成功!")
            print(f"📁 项目路径: {creation_result['task_path']}")

            # 显示下一步操作
            next_steps = creation_result.get('next_steps', [])
            if next_steps:
                print("\n下一步操作:")
                for step in next_steps:
                    print(f"  {step}")

        return creation_result

    def _ask_yes_no(self, question: str) -> bool:
        """询问是/否问题"""
        while True:
            response = input(f"{question} (y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                return True
            elif response in ['n', 'no', '否']:
                return False
            else:
                print("请输入 y/n 或 是/否")

    def run_quick_process(self, pdf_path: str,
                         task_name: Optional[str] = None,
                         domain: str = "Science") -> Dict[str, Any]:
        """
        快速处理模式（非交互）

        Args:
            pdf_path: PDF文件路径
            task_name: 任务名称
            domain: 领域名称

        Returns:
            处理结果
        """
        print(f"📄 快速处理模式: {pdf_path}")
        print("-" * 60)

        # 处理PDF
        result = self.pipeline.process_pdf(pdf_path, auto_improve=True)

        if not result['success']:
            return result

        # 显示简要结果
        quality = result.get('quality', {})
        print(f"✅ 处理完成 (评分: {quality.get('score', 0):.2f})")

        # 自动创建项目
        if not task_name:
            task_name = f"AutoTask_{result.get('paper_metadata', {}).get('year', '2024')}"

        print(f"\n正在创建项目: {task_name}")

        creation_result = self.pipeline.create_project(
            task_name=task_name,
            task_info=result['task_info'],
            checklist=result['checklist'],
            pdf_path=pdf_path,
            research_doc=result.get('research_doc', ''),
            domain=domain
        )

        if creation_result['success']:
            print(f"✅ 项目创建成功: {creation_result['task_path']}")

        result.update(creation_result)
        return result

    def display_status(self):
        """显示系统状态"""
        status = self.pipeline.get_status()

        print("\n" + "=" * 60)
        print("🔧 系统状态")
        print("=" * 60)

        print(f"\n就绪状态: {'✅' if status['ready'] else '❌'}")
        print(f"当前文件: {status['current_file'] or '无'}")
        print(f"上次处理时间: {status['last_processing_time']}s")

        print("\n组件状态:")
        for component, state in status['components'].items():
            print(f"  {component}: {state}")

        print("\n配置信息:")
        config = status['config']
        print(f"  LLM后端: {config['llm_backend']}")
        print(f"  质量阈值: {config['quality_threshold']}")
        print(f"  自动改进: {'✅' if config['auto_improvement'] else '❌'}")

        print()
