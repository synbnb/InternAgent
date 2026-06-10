"""
核心处理模块
"""

from .pdf_parser import PaperParser
from .info_extractor import ResearchInfoExtractor
from .content_generator import SciTaskGenerator
from .quality_checker import QualityChecker
from .project_creator import ProjectCreator

__all__ = [
    'PaperParser',
    'ResearchInfoExtractor',
    'SciTaskGenerator',
    'QualityChecker',
    'ProjectCreator'
]
