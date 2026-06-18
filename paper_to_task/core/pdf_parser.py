"""
PDF解析模块 - 使用PaddleOCR API将PDF转换为Markdown
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional
from pathlib import Path


PADDLE_OCR_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
PADDLE_OCR_MODEL = "PaddleOCR-VL-1.6"
PADDLE_OCR_MAX_POLL_TIME = 600  # 最长等待10分钟
PADDLE_OCR_POLL_INTERVAL = 5    # 每5秒轮询一次


class PaperParser:
    """论文PDF解析器 - 使用PaddleOCR API"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get(
            "PADDLE_OCR_TOKEN",
            "846ce46725ba29d1df4fc26f1c744f2cc183f905"
        )

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

        markdown_content = self._convert_with_paddleocr(pdf_path)

        if not markdown_content:
            raise ValueError(f"无法解析PDF文件: {pdf_path}")

        return {
            'raw_text': markdown_content,
            'markdown_content': markdown_content,
            'page_count': self._estimate_page_count(markdown_content)
        }

    def _convert_with_paddleocr(self, pdf_path: Path) -> str:
        """使用PaddleOCR API转换PDF为Markdown"""
        headers = {
            "Authorization": f"bearer {self.token}",
        }

        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }

        try:
            # 上传PDF文件
            print(f"正在上传PDF到PaddleOCR: {pdf_path.name}")
            data = {
                "model": PADDLE_OCR_MODEL,
                "optionalPayload": json.dumps(optional_payload),
            }

            with open(str(pdf_path), "rb") as f:
                files = {"file": f}
                job_response = requests.post(
                    PADDLE_OCR_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=120,
                )

            if job_response.status_code != 200:
                print(f"PaddleOCR上传失败: {job_response.status_code} {job_response.text}")
                return ""

            job_id = job_response.json()["data"]["jobId"]
            print(f"PaddleOCR任务提交成功, jobId: {job_id}")

            # 轮询等待处理结果
            jsonl_url = ""
            start_time = time.time()

            while time.time() - start_time < PADDLE_OCR_MAX_POLL_TIME:
                job_result_response = requests.get(
                    f"{PADDLE_OCR_URL}/{job_id}",
                    headers=headers,
                    timeout=30,
                )

                if job_result_response.status_code != 200:
                    print(f"查询任务状态失败: {job_result_response.status_code}")
                    time.sleep(PADDLE_OCR_POLL_INTERVAL)
                    continue

                state = job_result_response.json()["data"]["state"]

                if state == "pending":
                    print("PaddleOCR: 任务排队中...")
                elif state == "running":
                    try:
                        progress = job_result_response.json()["data"]["extractProgress"]
                        print(
                            f"PaddleOCR: 正在处理 "
                            f"({progress.get('extractedPages', 0)}/{progress.get('totalPages', '?')}页)"
                        )
                    except KeyError:
                        print("PaddleOCR: 正在处理...")
                elif state == "done":
                    extracted_pages = job_result_response.json()["data"]["extractProgress"]["extractedPages"]
                    print(f"PaddleOCR处理完成, 共{extracted_pages}页")
                    jsonl_url = job_result_response.json()["data"]["resultUrl"]["jsonUrl"]
                    break
                elif state == "failed":
                    error_msg = job_result_response.json()["data"]["errorMsg"]
                    print(f"PaddleOCR处理失败: {error_msg}")
                    return ""
                else:
                    print(f"PaddleOCR未知状态: {state}")

                time.sleep(PADDLE_OCR_POLL_INTERVAL)

            if not jsonl_url:
                print("PaddleOCR轮询超时")
                return ""

            # 下载处理结果
            print("正在下载PaddleOCR处理结果...")
            jsonl_response = requests.get(jsonl_url, timeout=60)
            jsonl_response.raise_for_status()

            # 合并所有页的Markdown
            lines = jsonl_response.text.strip().split("\n")
            all_markdown = []
            page_num = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)["result"]
                    for res in result.get("layoutParsingResults", []):
                        md_text = res.get("markdown", {}).get("text", "")
                        if md_text:
                            all_markdown.append(md_text)
                        page_num += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"解析PaddleOCR结果行失败: {e}")
                    continue

            if not all_markdown:
                print("PaddleOCR未能提取到任何文字内容")
                return ""

            full_markdown = "\n\n---\n\n".join(all_markdown)
            print(f"PaddleOCR解析完成, 共{page_num}页, {len(full_markdown)}字符")
            return full_markdown

        except requests.Timeout:
            print("PaddleOCR请求超时")
            return ""
        except Exception as e:
            print(f"PaddleOCR转换出错: {e}")
            return ""

    def _estimate_page_count(self, markdown_content: str) -> int:
        """估算页数"""
        return max(1, len(markdown_content) // 3000)
