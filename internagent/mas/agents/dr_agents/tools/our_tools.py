import sys
import os

# 添加camel模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camel.toolkits import FunctionTool
from camel.models import ModelFactory
from agents.task.execution_agent import ExecutionAgent
from utils.reference_manager import ReferenceManager
from camel.types import(
    ModelPlatformType,
    ModelType
)
from dotenv import load_dotenv

load_dotenv(override=True)

import json
from typing import List, Dict, Any
from utils.logger import get_logger
import requests
import re
from typing import Optional
import time
from datetime import datetime, timezone
from calendar import monthrange
from urllib.parse import quote, urlparse, parse_qs
import threading
import asyncio

logger = get_logger(__name__)

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "aiohttp"])
    import aiohttp

try:
    from bs4 import BeautifulSoup
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "beautifulsoup4", "lxml"])
    from bs4 import BeautifulSoup

from dataclasses import dataclass, asdict
from datetime import datetime
import urllib.parse

# PaperMetadata, CitationManager, and LiteratureSearch classes have been moved to info_processing_tools.py
# They are imported above

# OLD CODE - REMOVED (moved to info_processing_tools.py):
import mimetypes
import subprocess
from urllib.parse import urlparse, unquote
from typing import Optional, Tuple

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 延迟导入 PaddleOCR，避免在模块加载时触发依赖错误
# from paddleocr import PaddleOCR

# ---------- OCR 单例管理 ----------

class OCRSingleton:
    """PaddleOCR 单例管理器，确保只创建一个实例并提供线程安全的访问"""
    
    _instance = None
    _lock = threading.Lock()
    _ocr_lock = threading.Lock()  # OCR推理锁，确保并行安全
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OCRSingleton, cls).__new__(cls)
                    cls._instance._ocr = None
        return cls._instance
    
    def get_ocr(self):
        """获取OCR实例，线程安全"""
        if self._ocr is None:
            with self._lock:
                if self._ocr is None:
                    logger.info("初始化 PaddleOCR 实例...")
                    # 延迟导入，只在实际需要时导入
                    from paddleocr import PaddleOCR
                    self._ocr = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                    logger.info("PaddleOCR 实例初始化完成")
        return self._ocr
    
    def predict_safe(self, image_path: str) -> str:
        """线程安全的OCR预测"""
        ocr = self.get_ocr()
        with self._ocr_lock:  # 确保OCR推理的线程安全
            result = ocr.predict(image_path)
            for res in result:
                return res['rec_texts']
        return ""

# 全局OCR单例实例
_ocr_singleton = OCRSingleton()

# ---------- 工具函数 ----------

def _extract_wikipedia_image_url(url: str) -> Optional[str]:
    """
    从Wikipedia媒体URL中提取实际的图片URL
    
    Args:
        url: Wikipedia媒体URL，格式如 https://de.wikipedia.org/wiki/Page#/media/Datei:filename.jpg
        
    Returns:
        直接的图片URL，如果解析失败则返回None
    """
    try:
        # 解析URL以确定是否为Wikipedia媒体URL
        parsed = urlparse(url)
        if 'wikipedia.org' not in parsed.netloc:
            return None
            
        # 检查是否包含媒体片段
        if not parsed.fragment or not parsed.fragment.startswith('/media/'):
            return None
            
        # 提取文件名
        # 格式: #/media/Datei:filename 或 #/media/File:filename
        fragment = parsed.fragment
        if '/media/Datei:' in fragment:
            filename = fragment.split('/media/Datei:', 1)[1]
        elif '/media/File:' in fragment:
            filename = fragment.split('/media/File:', 1)[1]
        else:
            return None
            
        # URL解码文件名
        filename = unquote(filename)
        
        # 确定Wikipedia API的语言版本
        domain_parts = parsed.netloc.split('.')
        if len(domain_parts) >= 3 and domain_parts[0] != 'www':
            lang = domain_parts[0]  # de, en, fr, etc.
        else:
            lang = 'en'  # 默认英文版
            
        # 尝试多种方法获取图片URL
        session = _requests_session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 方法1：尝试本地Wikipedia API
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'titles': f'File:{filename}',
            'prop': 'imageinfo',
            'iiprop': 'url',
            'format': 'json'
        }
        
        try:
            response = session.get(api_url, params=params, timeout=session.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':  # 页面不存在
                    continue
                imageinfo = page_data.get('imageinfo', [])
                if imageinfo and 'url' in imageinfo[0]:
                    return imageinfo[0]['url']
        except Exception as e:
            logger.warning(f"Local Wikipedia API failed: {e}")
        
        # 方法2：尝试Wikimedia Commons API（很多文件都存储在Commons）
        commons_api = "https://commons.wikimedia.org/w/api.php"
        params = {
            'action': 'query',
            'titles': f'File:{filename}',
            'prop': 'imageinfo',
            'iiprop': 'url',
            'format': 'json'
        }
        
        try:
            response = session.get(commons_api, params=params, timeout=session.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':  # 页面不存在
                    continue
                imageinfo = page_data.get('imageinfo', [])
                if imageinfo and 'url' in imageinfo[0]:
                    return imageinfo[0]['url']
        except Exception as e:
            logger.warning(f"Commons API failed: {e}")
                
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract Wikipedia image URL: {e}")
        return None

def _convert_wikimedia_thumb_to_original(url: str) -> Optional[str]:
    """
    将Wikimedia缩略图URL转换为原始高分辨率图片URL
    
    Args:
        url: Wikimedia缩略图URL，格式如 https://upload.wikimedia.org/wikipedia/commons/thumb/...
        
    Returns:
        原始图片URL，如果转换失败则返回None
    """
    try:
        parsed = urlparse(url)
        if 'wikimedia.org' not in parsed.netloc:
            return None
            
        # 检查是否为缩略图URL
        if '/thumb/' not in parsed.path:
            return None
            
        # 缩略图URL格式: /wikipedia/commons/thumb/a/ab/filename.jpg/NNNpx-filename.jpg
        # 原图URL格式: /wikipedia/commons/a/ab/filename.jpg
        
        path_parts = parsed.path.split('/thumb/')
        if len(path_parts) != 2:
            return None
            
        # 提取基础路径和文件信息
        base_path = path_parts[0]  # /wikipedia/commons
        thumb_info = path_parts[1]  # a/ab/filename.jpg/NNNpx-filename.jpg
        
        # 分割出目录结构和原始文件名
        thumb_parts = thumb_info.split('/')
        if len(thumb_parts) < 4:  # 至少需要 a/ab/filename.jpg/thumb-name
            return None
            
        # 重构原始URL：base_path + / + 目录结构 + / + 原始文件名
        dir_structure = '/'.join(thumb_parts[:-1])  # a/ab/filename.jpg
        original_url = f"https://{parsed.netloc}{base_path}/{dir_structure}"
        
        return original_url
        
    except Exception as e:
        logger.warning(f"Failed to convert thumbnail to original URL: {e}")
        return None

def _requests_session(
    total_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    timeout: Tuple[int, int] = (10, 30),
):
    """带重试的 Requests 会话。"""
    sess = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["HEAD", "GET"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.request_timeout = timeout
    return sess

def _guess_filename_from_headers(headers) -> Optional[str]:
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not cd:
        return None
    # 解析 filename / filename*（RFC 5987）
    fname = None
    m = re.search(r'filename\*=\s*UTF-8\'\'([^;]+)', cd, flags=re.I)
    if m:
        fname = unquote(m.group(1).strip('"\' '))
    else:
        m = re.search(r'filename=\s*"?([^";]+)"?', cd, flags=re.I)
        if m:
            fname = m.group(1).strip('"\' ')
    return fname

def _safe_filename(name: str) -> str:
    # 去除不合法字符
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()

def _filename_from_url(url: str) -> Optional[str]:
    path = urlparse(url).path
    if not path:
        return None
    base = os.path.basename(path)
    return _safe_filename(unquote(base)) or None

def _ensure_extension(fname: str, content_type: Optional[str]) -> str:
    if not content_type:
        return fname
    # 已有扩展名就不再补
    if os.path.splitext(fname)[1]:
        return fname
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    # 修正常见视频/音频的 mime -> 扩展名不准确的问题
    mapping = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "image/jpeg": ".jpg",
    }
    ext = mapping.get(content_type.split(";")[0].strip(), ext)
    return fname + ext

def _human_size(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def _supports_range(headers) -> bool:
    return headers.get("Accept-Ranges", "").lower() == "bytes" or "content-range" in {k.lower() for k in headers}

def _print_progress(downloaded, total):
    if total:
        pct = downloaded / total * 100
        sys.stdout.write(f"\rDownloading: {_human_size(downloaded)} / {_human_size(total)} ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\rDownloading: {_human_size(downloaded)}")
    sys.stdout.flush()

# ---------- 直接文件下载（图片/视频/音频/二进制） ----------

def _download_binary(url: str, dest_path: Optional[str] = None, chunk_size: int = 1 << 20) -> str:
    sess = _requests_session()
    
    # 为Wikimedia URLs设置特殊头部
    if 'wikimedia.org' in url or 'wikipedia.org' in url:
        sess.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://commons.wikimedia.org/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    # 先 HEAD 探测
    head = sess.head(url, allow_redirects=True, timeout=sess.request_timeout)
    content_type = head.headers.get("Content-Type", "").lower()
    content_len = head.headers.get("Content-Length")
    total_size = int(content_len) if content_len and content_len.isdigit() else None

    # 生成文件名
    fname = _guess_filename_from_headers(head.headers) or _filename_from_url(url) or "download"
    fname = _ensure_extension(fname, content_type) if content_type else fname
    if dest_path:
        out = dest_path if os.path.isdir(dest_path) else os.path.abspath(dest_path)
        if os.path.isdir(dest_path):
            out = os.path.join(dest_path, fname)
    else:
        out = os.path.abspath(fname)

    # 断点续传
    resume_pos = 0
    mode = "wb"
    headers = dict(sess.headers)  # 复制session的headers
    if os.path.exists(out) and total_size:
        current_size = os.path.getsize(out)
        if 0 < current_size < total_size and _supports_range(head.headers):
            resume_pos = current_size
            headers["Range"] = f"bytes={current_size}-"
            mode = "ab"

    with sess.get(url, stream=True, headers=headers, allow_redirects=True, timeout=sess.request_timeout) as r:
        r.raise_for_status()
        # 若 GET 返回不同的 Content-Type，用 GET 的
        content_type = r.headers.get("Content-Type", content_type)
        if not os.path.splitext(out)[1]:
            out = _ensure_extension(out, content_type)

        downloaded = resume_pos
        total = total_size
        if total is None:
            # 尝试从 GET 头里再取一次
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                total = int(cl) + resume_pos

        # 确保目录存在
        os.makedirs(os.path.dirname(out), exist_ok=True)
        
        with open(out, mode) as f:
            last_print = time.time()
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # 限制刷新频率
                    if time.time() - last_print > 0.25:
                        _print_progress(downloaded, total)
                        last_print = time.time()
            _print_progress(downloaded, total)
            print()  # 换行

    return out

# ---------- 使用 yt-dlp 解析网页视频（可选） ----------

def _yt_dlp_available() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def _download_with_ytdlp(url: str, out_dir: str = ".", format_code: str = "bv*+ba/b") -> str:
    """
    使用 yt-dlp 下载网页视频：
    - format_code 默认优先选择最佳画质带音轨（可按需调整）
    - 返回最终文件（或目录）路径
    """
    cmd = [
        "yt-dlp",
        "-f", format_code,
        "-o", os.path.join(out_dir, "%(title).200B [%(id)s].%(ext)s"),
        "--no-playlist",
        "--restrict-filenames",
        "--retries", "3",
    ]   
    
    result = subprocess.run(cmd + [url], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = result.stdout

    if result.returncode != 0:
        # 返回错误信息而不是抛出异常
        return f"yt-dlp 下载失败：\n{out}"

    # 从日志里尽力找出生成的文件名（不保证 100%）
    m = re.search(r'\[Merger\] Merging formats into "(.*)"', out)
    if not m:
        m = re.search(r'\[download\] Destination: (.*)', out)
    if m:
        path = m.group(1).strip()
        return os.path.abspath(path)

    # 找不到就返回输出目录
    return os.path.abspath(out_dir)

# ---------- 对外主函数 ----------

def download_media_from_url(url: str,
                 dest: Optional[str] = None,
                 ) -> Dict[str, Any]:
    """
    Download any given URL (image, video, audio, document, or webpage).
    - If it's a direct link (file), download with requests.
    - If it's a webpage and yt-dlp is available, try yt-dlp.
    - Otherwise, save the HTML page.

    Args:
        url: Target URL
        dest: Output directory (default: tmp/)

    Returns:
        Dict with 'success' (bool) and 'path' (str) keys.
        On success: {"success": True, "path": "/path/to/file"}
        On failure: {"success": False, "path": error_message}
    """
    # 如果没有指定dest，默认使用tmp目录
    if dest is None:
        dest = "tmp"
        # 确保tmp目录存在
        os.makedirs(dest, exist_ok=True)
    
    sess = _requests_session()
    try:
        # 特殊处理：优先获取高分辨率版本
        original_url = url
        
        # 1. 如果是Wikimedia缩略图，转换为原图
        if '/thumb/' in url and 'wikimedia.org' in url:
            converted = _convert_wikimedia_thumb_to_original(url)
            if converted:
                logger.info(f"Converting thumbnail to original: {converted}")
                original_url = converted
            else:
                logger.info("Thumbnail conversion failed, using original URL")
        
        # 2. 如果是Wikipedia媒体页面URL，提取直接图片URL
        elif '#/media/' in url and 'wikipedia.org' in url:
            extracted = _extract_wikipedia_image_url(url)
            if extracted:
                logger.info(f"Extracted Wikipedia image URL: {extracted}")
                original_url = extracted
                
                # 如果提取的URL还是缩略图，再次转换
                if '/thumb/' in original_url:
                    converted = _convert_wikimedia_thumb_to_original(original_url)
                    if converted:
                        logger.info(f"Further converting to original: {converted}")
                        original_url = converted
        
        # 3. 特殊处理：Wikimedia直接图片URL (upload.wikimedia.org)
        parsed_url = urlparse(original_url)
        if 'wikimedia.org' in parsed_url.netloc or 'wikipedia.org' in parsed_url.netloc:
            # 对于Wikimedia域名，直接尝试下载，因为很可能是图片
            try:
                path = _download_binary(original_url, dest)
                return {"success": True, "path": path}
            except Exception as e:
                logger.warning(f"Direct Wikimedia download failed: {e}")
                # 继续到下面的逻辑
        
        # 先尝试 HEAD 看看是不是直接的媒体文件
        try:
            head = sess.head(original_url, allow_redirects=True, timeout=sess.request_timeout)
            ctype = (head.headers.get("Content-Type") or "").lower()
            is_media = any(
                ctype.startswith(prefix) for prefix in ("image/", "video/", "audio/")
            ) or ctype.startswith("application/") or "octet-stream" in ctype

            if is_media:
                path = _download_binary(original_url, dest)
                return {"success": True, "path": path}
        except Exception as e:
            logger.warning(f"HEAD request failed: {e}")
            # 如果HEAD失败，尝试直接下载（有些服务器不支持HEAD）
            try:
                path = _download_binary(original_url, dest)
                return {"success": True, "path": path}
            except Exception as e2:
                logger.warning(f"Direct download also failed: {e2}")
                # 继续到下面的逻辑

        # 非直接媒体：可能是网页
        if _yt_dlp_available():
            try:
                out_dir = dest if (dest and os.path.isdir(dest)) else (os.path.dirname(dest) if (dest and os.path.dirname(dest)) else ".")
                path = _download_with_ytdlp(url, out_dir=out_dir)
                # Check if yt-dlp returned an error message
                if path.startswith("yt-dlp 下载失败："):
                    return {"success": False, "path": path}
                return {"success": True, "path": path}
            except RuntimeError as e:
                # yt-dlp 不支持该网站，返回错误信息
                return {"success": False, "path": f"错误：{str(e)}"}
        else:
            # 回退为下载网页 HTML（有时也有用）
            html = sess.get(url, allow_redirects=True, timeout=sess.request_timeout)
            html.raise_for_status()
            fname = _guess_filename_from_headers(html.headers) or _filename_from_url(url) or "page.html"
            if not os.path.splitext(fname)[1]:
                fname += ".html"
            out = dest if (dest and not os.path.isdir(dest)) else os.path.join(dest or ".", _safe_filename(fname))
            with open(out, "wb") as f:
                f.write(html.content)
            return {"success": True, "path": os.path.abspath(out)}

    except requests.RequestException as e:
        return {"success": False, "path": f"网络错误：{e}"}
    



def search_wiki_revision(entity: str, year: int, month: int):
    """
    Search Wikipedia to get the latest Wikipedia revision *at or before* the end of the given (year, month).

    Args:
        entity: Wikipedia page title, e.g. "Penguin"
        year:  e.g. 2022
        month: 1-12

    Returns:
        dict | None
        {
            "timestamp": ISO 8601 UTC string,
            "oldid":     int,
            "url":       str  # direct URL to that revision
        }
        or None if not found.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if not (1 <= month <= 12):
                raise ValueError("month must be in 1..12")

            # End of month (UTC)
            last_day = monthrange(year, month)[1]
            end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            end_iso = end.isoformat().replace("+00:00", "Z")

            api = f"https://en.wikipedia.org/w/api.php"
            session = requests.Session()
            session.headers.update({"User-Agent": "RevisionFetcher/1.1 (contact: example@example.com)"})

            params = {
                "action": "query",
                "prop": "revisions",
                "titles": entity,
                "rvprop": "ids|timestamp",
                "rvstart": end_iso,      # anchor at end-of-month
                "rvdir": "older",        # walk backwards in time
                "rvlimit": 1,            # only need the closest one
                "redirects": "1",        # follow redirects to the canonical page
                "format": "json",
                "formatversion": "2",
            }

            r = session.get(api, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            pages = data.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                return None

            page = pages[0]
            revs = page.get("revisions") or []
            if not revs:
                # Page exists but has no revisions prior to end-of-month (extremely rare) -> None
                return None

            rev = revs[0]
            oldid = rev["revid"]
            ts = rev["timestamp"]

            # Build revision URL with canonicalized title
            title = page.get("title", entity).replace(" ", "_")
            encoded_title = quote(title, safe=":_()'!*")
            url = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&oldid={oldid}"

            return {"timestamp": ts, "oldid": oldid, "url": url}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避：1, 2, 4 秒
            else:
                logger.error(f"Wiki revision search failed after {max_retries} attempts")
                raise

    return None



def ocr2text(image_path: str) -> str:
    """
    OCR the image and return the text
    
    Args:
        image_path: the path of the image

    Returns:
        the text of the image
    """

    return _ocr_singleton.predict_safe(image_path)


# ---------- 文献搜索工具 ----------
def construct_our_tools() -> List[FunctionTool]:
    """
    构建我们的工具列表
    
    Returns:
        工具列表
    """
    
    # 定义工具列表
    tools = [
        FunctionTool(download_media_from_url),  # 现在包含高分辨率下载功能
        FunctionTool(search_wiki_revision),
        FunctionTool(ocr2text),
        FunctionTool(search_academic_papers),  # 多源学术论文搜索
        FunctionTool(summarize_paper),  # 对指定论文进行内容提取和摘要（本地文件或URL）
        FunctionTool(summarize_webpage),  # 对指定网页进行内容提取和摘要
        FunctionTool(search_and_summarize_papers),  # 集成搜索、下载、提取功能
        FunctionTool(search_and_summarize_webpages),  # 集成搜索、提取、分析功能
    ]
    
    return tools


if __name__ == "__main__":
    # 测试工具
    tools = construct_our_tools()
    print(f"成功创建 {len(tools)} 个工具:")
    for i, tool in enumerate(tools):
        print(f"{i+1}. {tool.func.__name__}")
    
    print("\n" + "="*80)
    print("测试新增的 summarize_paper 函数")
    print("="*80)
    
    # 测试1: 论文名称（会自动搜索）
    # print("\n[测试1] 使用论文名称:")
    # paper_result = summarize_paper("Attention Is All You Need")
    # print("paper_result: ", paper_result)

    # 测试2: 论文URL（如果需要的话）
    print("\n[测试2] 使用论文URL:")
    paper_result = summarize_paper("https://arxiv.org/pdf/2510.08521")
    print("paper_result: ", paper_result)
    
    # 测试3: 本地文件（如果有的话）
    # print("\n[测试3] 使用本地文件:")
    # paper_result = summarize_paper("path/to/paper")
    # print("paper_result: ", paper_result)
    
    # print("\n" + "="*80)
    # print("测试新增的 summarize_webpage 函数")
    # print("="*80)
    
    # # 测试网页URL
    # webpage_result = summarize_webpage("https://www.ajc.com/education/2026/01/atlanta-demonstrators-join-ice-protests-breaking-out-across-the-nation/")
    # print("webpage_result: ", webpage_result)
    
    # # 之前的测试代码
    # print("\n" + "="*80)
    # print("测试 search_and_summarize_papers 集成函数")
    # print("="*80)
    
    # # 测试集成函数
    # results = search_and_summarize_papers("Yangtze River Flood", max_number=1)
    
    # print(f"\n搜索结果: {len(results)} 篇论文")
    # print("="*80)
    
    # for i, paper in enumerate(results):
    #     print(f"\n论文 {i+1}:")
    #     print(f"  标题: {paper.get('title', 'N/A')[:80]}")
    #     print(f"  作者: {', '.join(paper.get('authors', [])[:3])}")
    #     print(f"  年份: {paper.get('year', 'N/A')}")
    #     print(f"  来源: {paper.get('source', 'N/A')}")
    #     print(f"  已下载: {'✓' if paper.get('downloaded') else '✗'}")
        
    #     if paper.get('downloaded'):
    #         print(f"  本地路径: {paper.get('local_path', 'N/A')}")
            
    #         if paper.get('summary'):
    #             summary = paper['summary']
    #             print(f"  摘要提取:")
    #             for key in ['problem_and_background', 'method_and_approach', 'experiments_and_results', 
    #                        'conclusions_and_insights', 'limitations_and_future_work']:
    #                 value = summary.get(key)
    #                 status = '✓' if value else '✗'
    #                 print(f"    {status} {key}: {(value if value is not None else 'None')}")
    #     print("-"*80)

    # print("results: ", results)
