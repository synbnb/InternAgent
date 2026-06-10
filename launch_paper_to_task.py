#!/usr/bin/env python3
"""
Paper-to-Task 主入口文件
将论文PDF自动转换为sci_tasks任务
"""

import argparse
import sys
import json
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from paper_to_task.pipeline import PaperToTaskPipeline
from paper_to_task.interaction.cli_interface import CLIInterface


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Paper-to-Task: 将论文PDF自动转换为sci_tasks任务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 交互式处理单个PDF
  python launch_paper_to_task.py --pdf paper.pdf

  # 快速处理模式（自动创建项目）
  python launch_paper_to_task.py --pdf paper.pdf --quick

  # 批量处理目录中的所有PDF
  python launch_paper_to_task.py --pdf-dir papers/ --output output/

  # 指定任务名称和领域
  python launch_paper_to_task.py --pdf paper.pdf --task-name "MyTask" --domain Biology

  # 查看系统状态
  python launch_paper_to_task.py --status

  # 显示帮助信息
  python launch_paper_to_task.py --help
        """
    )

    # 基本参数
    parser.add_argument('--pdf', type=str,
                       help='要处理的PDF文件路径')

    parser.add_argument('--pdf-dir', type=str,
                       help='批量处理：PDF文件目录')

    parser.add_argument('--output', type=str, default='output/',
                       help='输出目录（默认: output/）')

    # 任务参数
    parser.add_argument('--task-name', type=str,
                       help='任务名称')

    parser.add_argument('--domain', type=str, default='Science',
                       help='研究领域（默认: Science）')

    # 处理模式
    parser.add_argument('--quick', action='store_true',
                       help='快速处理模式（自动创建项目）')

    parser.add_argument('--interactive', action='store_true', default=True,
                       help='交互式模式（默认: True）')

    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')

    # 系统参数
    parser.add_argument('--status', action='store_true',
                       help='显示系统状态')

    parser.add_argument('--config', type=str,
                       help='配置文件路径')

    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')

    # LLM配置
    parser.add_argument('--llm-backend', type=str, default='mock',
                       choices=['mock', 'openai', 'anthropic', 'deepseek'],
                       help='LLM后端（默认: mock）')

    parser.add_argument('--model', type=str,
                       help='LLM模型名称')

    return parser


def load_config(args):
    """加载配置"""
    import os
    config = {}

    # 从配置文件加载
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                config.update(yaml.safe_load(f))

    # 从命令行参数更新
    if args.llm_backend:
        config.setdefault('llm', {})['backend'] = args.llm_backend

    if args.model:
        config.setdefault('llm', {})['model'] = args.model

    # 根据后端设置API密钥和base_url
    backend = config.get('llm', {}).get('backend', args.llm_backend or 'mock')

    if backend == 'deepseek':
        # 从环境变量读取DeepSeek配置
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key:
            config.setdefault('llm', {})['api_key'] = api_key
        base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        config.setdefault('llm', {})['base_url'] = base_url

    elif backend == 'openai':
        # 从环境变量读取OpenAI配置
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            config.setdefault('llm', {})['api_key'] = api_key

    elif backend == 'anthropic':
        # 从环境变量读取Anthropic配置
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            config.setdefault('llm', {})['api_key'] = api_key

    return config


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 加载配置
    config = load_config(args)

    # 创建管道
    pipeline = PaperToTaskPipeline(config)

    # 创建CLI界面
    cli = CLIInterface(pipeline)

    # 显示状态
    if args.status:
        cli.display_status()
        return 0

    # 批量处理模式
    if args.batch and args.pdf_dir:
        print("📚 批量处理模式")
        print("=" * 60)

        result = cli.run_batch(
            pdf_dir=args.pdf_dir,
            output_dir=args.output
        )

        if result['success']:
            print(f"\n✅ 批量处理完成")
            print(f"成功: {result['successful']}")
            print(f"失败: {result['failed']}")
            return 0
        else:
            print(f"\n❌ 批量处理失败: {result.get('error')}")
            return 1

    # 单文件处理模式
    if args.pdf:
        pdf_path = Path(args.pdf)

        # 检查文件是否存在
        if not pdf_path.exists():
            print(f"❌ 错误: PDF文件不存在: {args.pdf}")
            return 1

        # 快速模式
        if args.quick:
            print("🚀 快速处理模式")
            print("=" * 60)

            result = cli.run_quick_process(
                pdf_path=str(pdf_path),
                task_name=args.task_name,
                domain=args.domain
            )

            if result.get('success') and result.get('task_path'):
                print(f"\n✅ 处理完成: {result['task_path']}")
                return 0
            else:
                print(f"\n❌ 处理失败")
                return 1

        # 交互模式
        else:
            result = cli.run_interactive(
                pdf_path=str(pdf_path),
                auto_create=not args.interactive
            )

            if result.get('success'):
                print(f"\n✅ 处理完成")
                return 0
            else:
                print(f"\n❌ 处理取消或失败")
                return 1

    # 没有指定PDF文件
    if not args.pdf and not args.pdf_dir and not args.status:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        if args.verbose if 'args' in dir() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)
