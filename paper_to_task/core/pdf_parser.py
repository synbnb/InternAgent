"""
PDF解析模块 - 使用markitdown将PDF转换为Markdown
"""

import subprocess
from typing import Dict, Any
from pathlib import Path


class PaperParser:
    """论文PDF解析器 - 简化版，只负责转换Markdown"""

    def __init__(self):
        self.markitdown_cmd = "markitdown"

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        解析PDF文件，返回Markdown内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            包含Markdown内容的字典
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        # 使用markitdown转换PDF为Markdown
        markdown_content = self._convert_with_markitdown(pdf_path)

        if not markdown_content:
            raise ValueError(f"无法解析PDF文件: {pdf_path}")

        return {
            'raw_text': markdown_content,
            'markdown_content': markdown_content,
            'page_count': self._estimate_page_count(markdown_content)
        }

    def _convert_with_markitdown(self, pdf_path: Path) -> str:
        """使用markitdown转换PDF为Markdown"""
        try:
            # 使用临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
                temp_output_path = temp_file.name

            # 调用markitdown命令
            result = subprocess.run(
                [self.markitdown_cmd, str(pdf_path), '-o', temp_output_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # 读取转换后的Markdown内容
                with open(temp_output_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()

                # 清理临时文件
                Path(temp_output_path).unlink()

                return markdown_content
            else:
                print(f"markitdown转换失败: {result.stderr}")
                return ""

        except subprocess.TimeoutExpired:
            print("markitdown转换超时")
            return ""
        except Exception as e:
            print(f"markitdown转换出错: {e}")
            return ""

    def _estimate_page_count(self, markdown_content: str) -> int:
        """估算页数"""
        return max(1, len(markdown_content) // 3000)
