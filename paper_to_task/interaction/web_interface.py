"""
Web界面模块 - 提供Web界面支持（简化版）
"""

from typing import Dict, Any, Optional
from pathlib import Path


class WebInterface:
    """Web界面处理器"""

    def __init__(self, pipeline):
        """
        初始化Web界面

        Args:
            pipeline: PaperToTaskPipeline实例
        """
        self.pipeline = pipeline
        self.uploaded_files = {}

    def upload_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        处理文件上传

        Args:
            file_data: 文件数据
            filename: 文件名

        Returns:
            上传结果
        """
        try:
            # 保存临时文件
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "paper_to_task_uploads"
            temp_dir.mkdir(exist_ok=True)

            temp_path = temp_dir / filename
            with open(temp_path, 'wb') as f:
                f.write(file_data)

            # 记录上传文件
            file_id = str(len(self.uploaded_files) + 1)
            self.uploaded_files[file_id] = str(temp_path)

            return {
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'message': '文件上传成功'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'文件上传失败: {e}'
            }

    def process_file(self, file_id: str) -> Dict[str, Any]:
        """
        处理上传的文件

        Args:
            file_id: 文件ID

        Returns:
            处理结果
        """
        if file_id not in self.uploaded_files:
            return {
                'success': False,
                'error': 'File not found',
                'message': '文件不存在'
            }

        pdf_path = self.uploaded_files[file_id]

        try:
            # 使用管道处理
            result = self.pipeline.process_pdf(pdf_path, auto_improve=True)

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'处理失败: {e}'
            }

    def refine_content(self, file_id: str,
                      current_content: Dict,
                      feedback: str) -> Dict[str, Any]:
        """
        改进内容

        Args:
            file_id: 文件ID
            current_content: 当前内容
            feedback: 用户反馈

        Returns:
            改进结果
        """
        if file_id not in self.uploaded_files:
            return {
                'success': False,
                'error': 'File not found'
            }

        try:
            result = self.pipeline.refine_content(current_content, feedback)
            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def create_project(self, file_id: str,
                      task_name: str,
                      task_info: Dict,
                      checklist: Dict,
                      domain: str = "Science") -> Dict[str, Any]:
        """
        创建项目

        Args:
            file_id: 文件ID
            task_name: 任务名称
            task_info: 任务信息
            checklist: 检查清单
            domain: 领域名称

        Returns:
            创建结果
        """
        if file_id not in self.uploaded_files:
            return {
                'success': False,
                'error': 'File not found'
            }

        try:
            pdf_path = self.uploaded_files[file_id]
            result = self.pipeline.create_project(
                task_name=task_name,
                task_info=task_info,
                checklist=checklist,
                pdf_path=pdf_path,
                domain=domain
            )
            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return self.pipeline.get_status()

    def cleanup_temp_files(self):
        """清理临时文件"""
        import tempfile
        import shutil

        temp_dir = Path(tempfile.gettempdir()) / "paper_to_task_uploads"

        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self.uploaded_files = {}
            except Exception as e:
                print(f"清理临时文件失败: {e}")
