"""
主流程管道 - 协调整个Paper-to-Task处理流程
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from .core.pdf_parser import PaperParser
from .core.info_extractor import ResearchInfoExtractor
from .core.content_generator import SciTaskGenerator
from .core.quality_checker import QualityChecker
from .core.project_creator import ProjectCreator
from .refinement.iterative_refiner import IterativeRefiner
from .refinement.quality_scorer import QualityScorer
from .utils.llm_client import LLMClient
from .utils.validators import validate_generated_content
from .utils.file_utils import format_size, get_file_size


class PaperToTaskPipeline:
    """Paper-to-Task主流程管道"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化流程管道

        Args:
            config: 配置字典
        """
        self.config = config or self._default_config()

        # 初始化组件
        self.llm_client = LLMClient(self.config.get('llm', {}))
        self.pdf_parser = PaperParser()
        self.info_extractor = ResearchInfoExtractor(self.llm_client)
        self.content_generator = SciTaskGenerator()
        self.quality_checker = QualityChecker()
        self.iterative_refiner = IterativeRefiner(self.llm_client)
        self.quality_scorer = QualityScorer()
        self.project_creator = ProjectCreator(self.config.get('project', {}))

        # 处理状态
        self.current_file = None
        self.processing_time = 0

    def _default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'llm': {
                'backend': 'mock',  # 默认使用模拟后端用于测试
                'model': 'gpt-4',
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
            },
            'paper_to_task': {
                'auto_improve': True,
                'batch_output': 'output/'
            }
        }

    def process_pdf(self, pdf_path: str,
                   auto_improve: bool = True) -> Dict[str, Any]:
        """
        处理PDF文件的主要流程

        Args:
            pdf_path: PDF文件路径
            auto_improve: 是否自动改进质量

        Returns:
            处理结果字典
        """
        import time
        start_time = time.time()

        self.current_file = pdf_path

        try:
            # 第一步：解析PDF
            print(f"[1/6] 正在解析PDF: {pdf_path}")
            paper_content = self._parse_pdf(pdf_path)
            print(f"✅ PDF解析完成 ({paper_content['page_count']}页)")

            # 第二步：提取研究信息（所有工作交给LLM）
            print("[2/6] 正在提取研究信息...")
            research_info = self._extract_research_info(paper_content)
            print(f"✅ 研究信息提取完成 (领域: {research_info.get('research_field', 'Unknown')})")

            # 第三步：生成内容
            print("[3/6] 正在生成task_info和checklist...")
            generated_content = self._generate_content(research_info, paper_content)
            print(f"✅ 内容生成完成")

            # 第四步：质量检查
            print("[4/6] 正在进行质量检查...")
            quality_result = self._check_quality(generated_content)
            print(f"✅ 质量检查完成 (评分: {quality_result['overall_score']:.1f}/100)")

            # 第五步：自动改进（可选）
            min_score = self.config.get('quality', {}).get('min_score', 0.7)
            if auto_improve and quality_result['overall_score'] < min_score:
                print("[5/6] 质量未达标，正在自动改进...")
                generated_content = self._auto_improve(generated_content, quality_result)
                print(f"✅ 自动改进完成")

            else:
                print("[5/6] 质量检查通过，跳过改进")

            # 第六步：准备结果
            print("[6/6] 正在准备结果...")
            result = self._prepare_result(
                generated_content,
                research_info,
                paper_content,
                quality_result
            )
            print(f"✅ 处理完成")

            # 记录处理时间
            self.processing_time = time.time() - start_time
            result['processing_time'] = round(self.processing_time, 2)
            result['pdf_file'] = pdf_path
            result['pdf_size'] = format_size(get_file_size(Path(pdf_path)))

            return result

        except Exception as e:
            # 处理错误
            error_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'pdf_file': pdf_path
            }

            self.processing_time = time.time() - start_time
            error_result['processing_time'] = round(self.processing_time, 2)

            print(f"❌ 处理失败: {e}")
            return error_result

    def _parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """解析PDF文件"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        paper_content = self.pdf_parser.parse_pdf(pdf_path)

        # 检查解析质量
        if paper_content['page_count'] == 0:
            raise ValueError("PDF解析失败：无法提取页面")

        if not paper_content.get('raw_text'):
            raise ValueError("PDF解析失败：无法提取文本内容")

        return paper_content

    def _extract_research_info(self, paper_content: Dict) -> Dict[str, Any]:
        """提取研究信息（完全交给LLM）"""
        return self.info_extractor.extract_all_info(paper_content)

    def _generate_content(self, research_info: Dict,
                        paper_content: Dict) -> Dict[str, Any]:
        """生成task_info、checklist和research_doc"""
        # 生成task_info（简化版）
        task_info = self.content_generator.generate_task_info(
            research_info,
            research_info  # 直接使用research_info作为metadata
        )

        # 生成checklist（使用LLM生成针对性内容）
        checklist = self.content_generator.generate_checklist(
            research_info,
            self.llm_client
        )

        # 平衡权重
        checklist = self.content_generator.balance_checklist_weights(checklist)

        # 生成研究详情文档
        research_doc = self.content_generator.generate_research_doc(
            research_info,
            research_info  # 直接使用research_info作为metadata
        )

        return {
            'task_info': task_info,
            'checklist': checklist,
            'research_doc': research_doc
        }

    def _check_quality(self, content: Dict) -> Dict[str, Any]:
        """检查内容质量"""
        # 使用QualityChecker
        quality_check = self.quality_checker.check_generated_content(content)

        # 使用QualityScorer获取详细评分
        quality_score = self.quality_scorer.score_content(content)

        # 使用验证器
        validation = validate_generated_content(
            content['task_info'],
            content['checklist']
        )

        return {
            'overall_score': quality_score['overall_score'],
            'grade': quality_score['grade'],
            'passed': quality_score['passed'],
            'quality_check': quality_check,
            'quality_score': quality_score,
            'validation': validation
        }

    def _auto_improve(self, content: Dict, quality_result: Dict) -> Dict:
        """自动改进内容质量"""
        # 获取改进建议
        suggestions = self.quality_checker.get_quality_report(quality_result)

        # 构造自动反馈
        feedback = f"系统自动改进：{suggestions}"

        # 使用迭代优化器
        refinement_result = self.iterative_refiner.refine_content(
            content,
            feedback
        )

        return {
            'task_info': refinement_result['task_info'],
            'checklist': refinement_result['checklist']
        }

    def _prepare_result(self, content: Dict,
                       research_info: Dict,
                       paper_content: Dict,
                       quality_result: Dict) -> Dict[str, Any]:
        """准备最终结果"""
        return {
            'success': True,
            'task_info': content['task_info'],
            'checklist': content['checklist'],
            'research_doc': content.get('research_doc', ''),
            'research_info': research_info,
            'paper_metadata': research_info,  # 使用research_info作为元数据
            'paper_sources': research_info.get('_paper_sources', {}),  # 论文原文引用
            'raw_markdown': paper_content.get('markdown_content', ''),  # 原始论文Markdown
            'quality': {
                'score': quality_result['overall_score'],
                'grade': quality_result['grade'],
                'passed': quality_result['passed'],
                'dimension_scores': quality_result['quality_score']['dimension_scores'],
                'suggestions': quality_result['quality_score']['suggestions']
            },
            'validation': quality_result['validation'],
            'next_steps': [
                '检查生成的task_info.json和checklist.json',
                '查看RESEARCH_DETAILS.md了解详细研究信息',
                '如需修改，使用refine功能进行改进',
                '确认后使用create_project创建项目'
            ]
        }

    def refine_content(self, current_content: Dict,
                     feedback: str) -> Dict[str, Any]:
        """
        根据用户反馈改进内容

        Args:
            current_content: 当前生成的内容
            feedback: 用户反馈

        Returns:
            改进后的内容
        """
        print("正在根据反馈改进内容...")

        try:
            # 使用迭代优化器
            refinement_result = self.iterative_refiner.refine_content(
                current_content,
                feedback
            )

            # 重新检查质量
            new_quality = self._check_quality(refinement_result)

            # 生成改进摘要
            summary = self.iterative_refiner.get_improvement_summary(refinement_result)

            return {
                'success': True,
                'task_info': refinement_result['task_info'],
                'checklist': refinement_result['checklist'],
                'improvements': refinement_result['improvements_made'],
                'feedback_analysis': refinement_result['feedback_analysis'],
                'quality': new_quality,
                'summary': summary
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'改进失败: {e}'
            }

    def create_project(self, task_name: str,
                      task_info: Dict,
                      checklist: List[Dict],
                      pdf_path: Optional[str] = None,
                      research_doc: str = "",
                      domain: str = "Science") -> Dict[str, Any]:
        """
        创建sci_tasks项目

        Args:
            task_name: 任务名称
            task_info: 任务信息
            checklist: 检查清单
            pdf_path: PDF文件路径
            research_doc: 研究详情文档
            domain: 领域名称

        Returns:
            创建结果
        """
        print(f"正在创建项目: {task_name}")

        try:
            # 使用项目创建器
            creation_result = self.project_creator.create_project(
                task_name=task_name,
                task_info=task_info,
                checklist=checklist,
                pdf_path=pdf_path,
                research_doc=research_doc,
                domain=domain
            )

            if creation_result['success']:
                print(f"✅ 项目创建成功: {creation_result['task_id']}")
                print(f"📁 路径: {creation_result['task_path']}")
            else:
                print(f"❌ 项目创建失败: {creation_result.get('error')}")

            return creation_result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'创建项目失败: {e}'
            }

    def get_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        status = {
            'ready': True,
            'current_file': self.current_file,
            'last_processing_time': self.processing_time,
            'components': {
                'pdf_parser': '✅',
                'info_extractor': '✅',
                'content_generator': '✅',
                'quality_checker': '✅',
                'iterative_refiner': '✅',
                'quality_scorer': '✅',
                'project_creator': '✅'
            },
            'config': {
                'llm_backend': self.config.get('llm', {}).get('backend', 'mock'),
                'quality_threshold': self.config.get('quality', {}).get('min_score', 0.7),
                'auto_improvement': self.config.get('quality', {}).get('enable_auto_improvement', True)
            }
        }

        return status

    def batch_process(self, pdf_paths: List[str],
                     output_dir: str) -> List[Dict]:
        """
        批量处理PDF文件

        Args:
            pdf_paths: PDF文件路径列表
            output_dir: 输出目录

        Returns:
            处理结果列表
        """
        results = []

        print(f"开始批量处理 {len(pdf_paths)} 个PDF文件")
        print("=" * 60)

        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"\n[{i}/{len(pdf_paths)}] 处理: {pdf_path}")
            print("-" * 60)

            try:
                result = self.process_pdf(pdf_path, auto_improve=False)
                results.append(result)

                # 保存结果
                if result['success']:
                    self._save_batch_result(result, output_dir, i)

            except Exception as e:
                error_result = {
                    'success': False,
                    'pdf_file': pdf_path,
                    'error': str(e)
                }
                results.append(error_result)
                print(f"❌ 处理失败: {e}")

        # 生成摘要
        print("\n" + "=" * 60)
        print("批量处理完成")
        print(f"总计: {len(results)} 个文件")
        print(f"成功: {sum(1 for r in results if r['success'])}")
        print(f"失败: {sum(1 for r in results if not r['success'])}")

        return results

    def _save_batch_result(self, result: Dict, output_dir: str, index: int):
        """保存批量处理结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存task_info
        task_info_file = output_path / f"task_info_{index}.json"
        with open(task_info_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(result['task_info'], f, indent=2, ensure_ascii=False)

        # 保存checklist
        checklist_file = output_path / f"checklist_{index}.json"
        with open(checklist_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(result['checklist'], f, indent=2, ensure_ascii=False)
