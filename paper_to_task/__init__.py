"""
Paper-to-Task 自动化系统
将论文PDF自动转换为sci_tasks所需的task_info.json和checklist.json
"""

__version__ = "1.0.0"

from .pipeline import PaperToTaskPipeline

__all__ = ['PaperToTaskPipeline']
