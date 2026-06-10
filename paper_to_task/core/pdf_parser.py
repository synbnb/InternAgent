"""
PDF解析模块 - 解析论文PDF并提取结构化内容
"""

import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class PaperParser:
    """论文PDF解析器"""

    def __init__(self):
        # 章节识别模式
        self.section_patterns = {
            'abstract': r'^(abstract|摘要)$',
            'introduction': r'^(introduction|引言|前言)$',
            'related_work': r'^(related\s+work|相关工作|literature\s+review)$',
            'methods': r'^(methods?|methodology|experimental\s+methods?|实验方法|材料与方法)$',
            'experiments': r'^(experiments?|实验|实验设计)$',
            'results': r'^(results?|结果|结果与讨论)$',
            'discussion': r'^(discussion|讨论)$',
            'conclusion': r'^(conclusions?|结论|总结)$',
            'references': r'^(references?|参考文献)$'
        }

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        解析PDF文件，返回结构化内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            包含论文内容的字典
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        # 尝试使用不同的PDF解析库
        content = self._try_parse_with_pdfplumber(pdf_path)
        if not content:
            content = self._try_parse_with_pypdf(pdf_path)

        if not content:
            raise ValueError(f"无法解析PDF文件: {pdf_path}")

        # 识别论文结构
        sections = self._identify_sections(content.get('full_text', ''))

        # 提取元数据
        metadata = self._extract_metadata(content, sections)

        return {
            'raw_text': content.get('full_text', ''),
            'sections': sections,
            'figures': content.get('figures', []),
            'tables': content.get('tables', []),
            'metadata': metadata,
            'page_count': content.get('page_count', 0)
        }

    def _try_parse_with_pdfplumber(self, pdf_path: Path) -> Optional[Dict]:
        """尝试使用pdfplumber解析"""
        try:
            import pdfplumber

            full_text = ""
            figures = []
            tables = []
            page_count = 0

            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

                    # 提取图片
                    if page.images:
                        for img in page.images:
                            figures.append({
                                'page': i + 1,
                                'bbox': img.get('bbox', []),
                                'description': f"Figure on page {i + 1}"
                            })

                    # 提取表格
                    page_tables = page.extract_tables()
                    for j, table in enumerate(page_tables):
                        tables.append({
                            'page': i + 1,
                            'table_num': j + 1,
                            'data': table,
                            'rows': len(table) if table else 0
                        })

            return {
                'full_text': full_text,
                'figures': figures,
                'tables': tables,
                'page_count': page_count
            }

        except ImportError:
            print("警告: 未安装pdfplumber包")
            return None
        except Exception as e:
            print(f"pdfplumber解析失败: {e}")
            return None

    def _try_parse_with_pypdf(self, pdf_path: Path) -> Optional[Dict]:
        """尝试使用PyPDF解析"""
        try:
            import PyPDF2

            full_text = ""
            page_count = 0

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

            return {
                'full_text': full_text,
                'figures': [],
                'tables': [],
                'page_count': page_count
            }

        except ImportError:
            print("警告: 未安装PyPDF2包")
            return None
        except Exception as e:
            print(f"PyPDF解析失败: {e}")
            return None

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """
        识别论文章节结构

        Args:
            text: 论文全文

        Returns:
            章节字典，键为章节名，值为章节内容
        """
        sections = {}

        # 使用更灵活的章节分割
        # 首先尝试识别Abstract，它通常在开头
        abstract_match = re.search(r'abstract\s*[-—]\s*(.+?)(?=\n\s*(?:introduction|1\.|keywords|\n\s*\n|\d+\.))', text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()

        # 识别Introduction
        intro_match = re.search(r'introduction\s+(.+?)(?=\n\s*(?:2\.|related work|methods?|conclusion))', text, re.IGNORECASE | re.DOTALL)
        if intro_match:
            sections['introduction'] = intro_match.group(1).strip()

        # 识别Methods
        methods_match = re.search(r'(?:methods?|methodology|experimental\s+methods?)\s+(.+?)(?=\n\s*(?:results?|discussion|conclusion|\d+\.\s))', text, re.IGNORECASE | re.DOTALL)
        if methods_match:
            sections['methods'] = methods_match.group(1).strip()

        # 识别Results
        results_match = re.search(r'results?\s+(.+?)(?=\n\s*(?:discussion|conclusion|references?))', text, re.IGNORECASE | re.DOTALL)
        if results_match:
            sections['results'] = results_match.group(1).strip()

        # 识别Conclusion
        conclusion_match = re.search(r'(?:conclusions?|discussion\s+and\s+conclusion)\s+(.+?)(?=\n\s*(?:references?|acknowledg))', text, re.IGNORECASE | re.DOTALL)
        if conclusion_match:
            sections['conclusion'] = conclusion_match.group(1).strip()

        # 如果没有找到任何章节，将全文作为unknown
        if not sections:
            sections['unknown'] = text[:5000]  # 限制长度

        return sections

    def _extract_metadata(self, content: Dict, sections: Dict) -> Dict[str, Any]:
        """提取论文元数据"""
        metadata = {
            'title': '',
            'authors': [],
            'year': None,
            'doi': '',
            'keywords': [],
            'abstract': ''
        }

        full_text = content.get('full_text', '')

        # 提取标题（通常是前几行中较长的行）
        lines = full_text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # 简单的启发式：避免包含"Abstract"、"Introduction"等词
                if not any(word in line.lower() for word in ['abstract', 'introduction', 'keywords']):
                    metadata['title'] = line
                    break

        # 提取DOI
        doi_pattern = r'(?:DOI|doi)\s*[:=]\s*(10\.\d+/[^\s\)]+)'
        doi_match = re.search(doi_pattern, full_text)
        if doi_match:
            metadata['doi'] = doi_match.group(1)

        # 提取年份
        year_pattern = r'(?:©|\(c\)|Published)\s*(\d{4})'
        year_match = re.search(year_pattern, full_text)
        if year_match:
            metadata['year'] = int(year_match.group(1))

        # 提取关键词
        keywords_pattern = r'(?:keywords?|关键词)\s*[:=]\s*([^\n]+)'
        keywords_match = re.search(keywords_pattern, full_text, re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            # 分割关键词（常见的分隔符：,、；;）
            keywords = re.split(r'[,，、；;]\s*', keywords_text)
            metadata['keywords'] = [k.strip() for k in keywords if k.strip()]

        # 提取摘要
        if 'abstract' in sections:
            metadata['abstract'] = sections['abstract'][:500]  # 限制长度

        # 提取作者（简化版）
        authors_pattern = r'(?:authors?|作者)\s*[:=]\s*([^\n]+)'
        authors_match = re.search(authors_pattern, full_text, re.IGNORECASE)
        if authors_match:
            authors_text = authors_match.group(1)
            # 简单分割
            authors = re.split(r'[,，、和;]\s*', authors_text)
            metadata['authors'] = [a.strip() for a in authors if a.strip()][:10]  # 限制数量

        return metadata

    def extract_figures_context(self, content: Dict) -> List[Dict[str, str]]:
        """
        提取图表的上下文信息

        Args:
            content: 解析后的论文内容

        Returns:
            图表上下文列表
        """
        figures_context = []
        sections = content.get('sections', {})
        figures = content.get('figures', [])

        # 从各个章节中查找图表引用
        for section_name, section_content in sections.items():
            # 查找图表引用模式
            # Fig. 1, Figure 2, 图1, 图2等
            figure_refs = re.findall(r'(?:fig|figure|图)\s*(\d+)', section_content, re.IGNORECASE)

            for fig_num in figure_refs:
                figures_context.append({
                    'figure_number': fig_num,
                    'section': section_name,
                    'context': self._extract_sentence_around(section_content, f"Fig {fig_num}")
                })

        return figures_context

    def _extract_sentence_around(self, text: str, keyword: str, window: int = 100) -> str:
        """提取关键词周围的句子"""
        index = text.lower().find(keyword.lower())
        if index == -1:
            return ""

        start = max(0, index - window)
        end = min(len(text), index + len(keyword) + window)
        return text[start:end].strip()

    def get_section_summary(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        获取各章节的摘要

        Args:
            sections: 章节内容字典

        Returns:
            章节摘要字典
        """
        summary = {}

        for section_name, section_content in sections.items():
            # 获取前500字作为摘要
            summary[section_name] = section_content[:500] + "..." if len(section_content) > 500 else section_content

        return summary
