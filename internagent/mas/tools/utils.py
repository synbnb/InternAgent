"""
Utility Tools for Scientific Literature Management

This module provides a comprehensive suite of utility functions and classes for managing
scientific literature, including:
- Paper metadata structures (PaperMetadata dataclass)
- Multi-source paper search (Semantic Scholar, arXiv, PubMed)
- PDF downloading and text extraction
- Paper filtering and deduplication
- Citation formatting (APA, BibTeX)
- Query parsing and execution
- DOI resolution and publisher page scraping

These utilities support the literature search and survey capabilities of the InternAgent system.
"""

import logging
import re
import os
import time
import json
import requests
import httpx
import subprocess
from pathlib import Path
import pdfplumber
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import random
import yaml

from bs4 import BeautifulSoup
from .literature_search import PaperMetadata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)

# Define the paper search endpoint URL
search_url = 'https://api.semanticscholar.org/graph/v1/paper/search/'
graph_url = 'https://api.semanticscholar.org/graph/v1/paper/'
rec_url = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"

# Similarity threshold for tool relevance
def get_related_tools(query: Union[str, List[str]], 
                        tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据查询（query）从工具列表（tools）中筛选相关的工具。

    使用 TF-IDF 和余弦相似度计算相关性。
    返回的工具按相关性得分降序排列，并受 MAX_RETURNED_TOOLS 限制。

    参数:
    query (Union[str, List[str]]): 
        查询词或查询词列表。
    tools (List[Dict[str, Any]]): 
        工具定义的列表。

    返回:
    List[Dict[str, Any]]: 
        与查询相关的工具列表（按相关性排序）。
    """
    MAX_RETURNED_TOOLS = 128
    SIMILARITY_THRESHOLD = 0.1
    if not tools:
        return []

    # 1. 标准化查询输入
    query_string: str
    if isinstance(query, str):
        query_string = query
    elif isinstance(query, list):
        query_string = " ".join(q for q in query if isinstance(q, str))
    else:
        sys.stderr.write("错误：查询类型必须是 str 或 List[str]。\n")
        return []

    if not query_string.strip():
        return []

    # 2. 构建“文档”语料库 (Corpus)
    corpus = []
    valid_tools = []
    for tool in tools:
        try:
            func_data = tool['function']
            name = func_data.get('name', '')
            description = func_data.get('description', '')
            searchable_text = f"{name} {description}"
            corpus.append(searchable_text)
            valid_tools.append(tool)
        except (AttributeError, TypeError, KeyError):
            continue
            
    if not valid_tools:
        return []

    # 3. TF-IDF 向量化
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query_string])
    except ValueError as e:
        sys.stderr.write(f"TF-IDF 向量化时发生错误: {e}\n")
        return []

    # 4. 计算余弦相似度
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    scores = cosine_similarities[0]

    # 5. 筛选、排序和限制
    scored_tools = []
    for i, score in enumerate(scores):
        if score > SIMILARITY_THRESHOLD:
            scored_tools.append((score, valid_tools[i]))

    # 按得分（score）降序排列
    scored_tools.sort(key=lambda x: x[0], reverse=True)

    # --- 关键修改 ---
    # 截取列表，只返回最多 MAX_RETURNED_TOOLS 个工具
    limited_scored_tools = scored_tools[:MAX_RETURNED_TOOLS]
    # -----------------

    # 仅返回工具本身
    related_tools = [tool for score, tool in limited_scored_tools]

    return related_tools

def select_papers(paper_bank, max_papers, rag_read_depth):
    selected_for_deep_read = []
    count = 0
    for paper in sorted(paper_bank, key=lambda x: x['score'], reverse=True):
        if count >= rag_read_depth:
            break
        url = None

        if 'url' in paper:
            url = paper['url']
        elif 'doi' in paper:
            url = paper['doi']
        
        if url:
            selected_for_deep_read.append(paper)
            count += 1

    selected_for_deep_read = selected_for_deep_read[:max_papers]
    return selected_for_deep_read

def parse_io_description(output):
    match_input = re.match(r'Input\("([^"]+)"\)', output)
    input_description = match_input.group(1) if match_input else None
    match_output = re.match(r'.*Output\("([^"]+)"\)', output)
    output_description = match_output.group(1) if match_output else None
    return input_description, output_description

def format_papers_for_printing_next_query(paper_lst, include_abstract=True, include_score=True, include_id=True):
    """
    Convert a list of papers to a string for printing or as part of a prompt.
    """
    output_str = ""
    for idx, paper in enumerate(paper_lst):
        if include_id:
            output_str += "paperId: " + str(idx) + "\n" 
        elif include_id and "title" in paper:
            output_str += "paperId: " + paper["title"].strip() + "\n"
        
        output_str += "title: " + paper.get("title", "").strip() + "\n"
        
        output_str += "\n"
    
    return output_str

def _is_valid_pdf_link(url):
    """
    Check if a URL is likely to be a valid PDF link.
    Filters out known invalid patterns like stamp.jsp, citation pages, etc.
    
    Args:
        url: URL to check
    
    Returns:
        True if the URL is likely a valid PDF link, False otherwise
    """
    if not url:
        return False
    
    url_lower = url.lower()
    
    # 排除已知的无效模式
    invalid_patterns = [
        'stamp.jsp',           # IEEE stamp页面，不是真正的PDF
        'citation.cfm',        # 引用页面
        'abstract',            # 摘要页面
        'login',               # 登录页面
        'register',            # 注册页面
        'javascript:',         # JavaScript链接
        '#',                   # 锚点链接
    ]
    
    for pattern in invalid_patterns:
        if pattern in url_lower:
            return False
    
    # 检查是否包含PDF相关的有效模式
    valid_patterns = [
        '.pdf',                # 直接的PDF文件
        '/pdf/',               # PDF路径
        'arxiv.org/pdf',       # arXiv PDF
        'openreview.net/pdf',  # OpenReview PDF
        '/ielx',               # IEEE直接PDF链接
        'pdfserve',            # PDF服务
        'download',            # 下载链接
    ]
    
    for pattern in valid_patterns:
        if pattern in url_lower:
            return True
    
    return False

def download_pdf(pdf_url, save_folder="pdfs", max_retries=2):
    """
    Download a PDF from a given URL.
    Supports direct PDF links and can extract PDF links from academic paper pages
    (e.g., Semantic Scholar, arXiv, publisher pages).
    
    Args:
        pdf_url: URL to download the PDF from
        save_folder: Directory to save the PDF
        max_retries: Maximum number of retries for recursive calls (to prevent infinite loops)
    
    Returns:
        Path to the downloaded PDF file, or None if download failed
    """
    logger.info(f"downloading pdf from {pdf_url}")
    
    if not pdf_url:
        return None
    
    os.makedirs(save_folder, exist_ok=True)
    
    file_name = pdf_url.split("/")[-1].split("?")[0]  # 移除URL参数
    if not file_name.endswith('.pdf'):
        file_name = file_name + '.pdf'
    save_path = os.path.join(save_folder, file_name)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    
    try:
        response = httpx.get(url=pdf_url, headers=headers, timeout=30, verify=False, follow_redirects=True)
        
        if response.status_code != 200:
            logger.error(f"Failed to download PDF from {pdf_url}: {response.status_code}")
            return None
        
        # 检查Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        
        # 检查响应内容的前几个字节是否为PDF标识符
        content = response.content
        
        # 如果内容太小，可能不是有效的PDF
        if len(content) < 1024:
            logger.warning(f"Downloaded content from {pdf_url} is too small ({len(content)} bytes)")
            return None
        
        is_pdf = content.startswith(b'%PDF')
        
        # 如果内容不是PDF格式，尝试从HTML页面中提取PDF链接
        if not is_pdf:
            logger.warning(f"Downloaded content from {pdf_url} is not a valid PDF (Content-Type: {content_type})")
            
            # 如果还有重试次数，尝试从HTML页面中提取PDF链接
            if max_retries > 0:
                try:
                    # 尝试解析为HTML
                    if b'<html' in content[:1000].lower() or b'<!doctype' in content[:1000].lower():
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # 查找可能的PDF下载链接
                        pdf_link = None
                        
                        # 特殊处理Semantic Scholar
                        if 'semanticscholar.org' in pdf_url:
                            logger.info("Extracting PDF link from Semantic Scholar page")
                            
                            # 方法1: 尝试使用Semantic Scholar API
                            # 从URL中提取paper ID
                            paper_id_match = re.search(r'/paper/([a-f0-9]+)', pdf_url)
                            paper_doi = None
                            if paper_id_match:
                                paper_id = paper_id_match.group(1)
                                try:
                                    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=openAccessPdf,externalIds,isOpenAccess"
                                    api_response = httpx.get(api_url, headers=headers, timeout=10, follow_redirects=True)
                                    if api_response.status_code == 200:
                                        paper_data = api_response.json()
                                        
                                        # 尝试获取开放获取的PDF
                                        if paper_data.get('openAccessPdf') and paper_data['openAccessPdf'].get('url'):
                                            pdf_link = paper_data['openAccessPdf']['url']
                                            logger.info(f"Found PDF link via Semantic Scholar API: {pdf_link}")
                                        
                                        # 保存DOI以备后用
                                        if paper_data.get('externalIds') and paper_data['externalIds'].get('DOI'):
                                            paper_doi = paper_data['externalIds']['DOI']
                                            logger.info(f"Found DOI via Semantic Scholar API: {paper_doi}")
                                        
                                        # 尝试从arXiv ID构造PDF链接
                                        if not pdf_link and paper_data.get('externalIds') and paper_data['externalIds'].get('ArXiv'):
                                            arxiv_id = paper_data['externalIds']['ArXiv']
                                            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                                            logger.info(f"Constructed arXiv PDF link: {pdf_link}")
                                except Exception as e:
                                    logger.warning(f"Failed to use Semantic Scholar API: {e}")
                            
                            # 方法2: 从页面的JSON数据中提取
                            if not pdf_link:
                                try:
                                    # 查找页面中的JSON-LD或其他结构化数据
                                    scripts = soup.find_all('script', type='application/ld+json')
                                    for script in scripts:
                                        try:
                                            data = json.loads(script.string)
                                            if isinstance(data, dict) and 'url' in data:
                                                url_val = data.get('url', '')
                                                if url_val.endswith('.pdf'):
                                                    pdf_link = url_val
                                                    logger.info(f"Found PDF link in JSON-LD: {pdf_link}")
                                                    break
                                        except:
                                            pass
                                except Exception as e:
                                    logger.debug(f"Failed to parse JSON-LD: {e}")
                            
                            # 方法3: 查找meta标签
                            if not pdf_link:
                                meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
                                if meta_pdf and meta_pdf.get('content'):
                                    pdf_link = meta_pdf['content']
                                    logger.info(f"Found PDF link in meta tag: {pdf_link}")
                            
                            # 方法4: 查找所有链接并智能筛选
                            if not pdf_link:
                                candidate_links = []
                                for link in soup.find_all('a', href=True):
                                    href = link['href']
                                    full_href = href if href.startswith('http') else urljoin(pdf_url, href)
                                    
                                    # 只添加真正有效的PDF链接
                                    if _is_valid_pdf_link(href):
                                        candidate_links.append(full_href)
                                
                                if candidate_links:
                                    logger.info(f"Found {len(candidate_links)} candidate PDF links")
                                    # 输出前5个候选链接用于调试
                                    for i, link in enumerate(candidate_links[:5]):
                                        logger.debug(f"Candidate {i+1}: {link}")
                                
                                # 优先级1: arXiv PDF链接
                                for link in candidate_links:
                                    if 'arxiv.org/pdf' in link:
                                        pdf_link = link
                                        logger.info(f"Selected arXiv PDF link: {pdf_link}")
                                        break
                                
                                # 优先级2: OpenReview PDF链接
                                if not pdf_link:
                                    for link in candidate_links:
                                        if 'openreview.net/pdf' in link:
                                            pdf_link = link
                                            logger.info(f"Selected OpenReview PDF link: {pdf_link}")
                                            break
                                
                                # 优先级3: 直接的.pdf文件（但排除stamp.jsp等）
                                if not pdf_link:
                                    for link in candidate_links:
                                        if link.endswith('.pdf') and 'stamp.jsp' not in link:
                                            pdf_link = link
                                            logger.info(f"Selected .pdf link: {pdf_link}")
                                            break
                                
                                # 优先级4: /ielx路径（IEEE直接PDF）
                                if not pdf_link:
                                    for link in candidate_links:
                                        if '/ielx' in link and link.endswith('.pdf'):
                                            pdf_link = link
                                            logger.info(f"Selected IEEE ielx PDF link: {pdf_link}")
                                            break
                        
                        # 特殊处理IEEE页面
                        elif 'ieee.org' in pdf_url or 'ieeexplore' in pdf_url:
                            logger.info("Extracting PDF link from IEEE page")
                            
                            # IEEE的PDF链接通常在meta标签或特定的链接中
                            meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
                            if meta_pdf and meta_pdf.get('content'):
                                pdf_link = meta_pdf['content']
                            
                            # 查找iframe中的PDF链接
                            if not pdf_link:
                                for iframe in soup.find_all('iframe'):
                                    src = iframe.get('src', '')
                                    if src.endswith('.pdf') or '/ielx' in src:
                                        pdf_link = src if src.startswith('http') else urljoin(pdf_url, src)
                                        break
                            
                            # 查找直接的PDF下载链接
                            if not pdf_link:
                                for link in soup.find_all('a', href=True):
                                    href = link['href']
                                    if '/ielx' in href and href.endswith('.pdf'):
                                        pdf_link = href if href.startswith('http') else urljoin(pdf_url, href)
                                        break
                        
                        # 特殊处理DOI链接
                        elif 'doi.org' in pdf_url:
                            logger.info("Extracting PDF link from DOI page")
                            
                            # 查找meta标签中的PDF链接
                            meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
                            if meta_pdf and meta_pdf.get('content'):
                                pdf_link = meta_pdf['content']
                            
                            # 查找PDF下载按钮或链接
                            if not pdf_link:
                                for link in soup.find_all('a', href=True):
                                    href = link['href']
                                    text = link.get_text().lower().strip()
                                    if _is_valid_pdf_link(href) and any(kw in text for kw in ['pdf', 'download', 'full text']):
                                        pdf_link = href if href.startswith('http') else urljoin(pdf_url, href)
                                        break
                        
                        # 通用方法: 查找meta标签中的PDF链接
                        if not pdf_link:
                            meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
                            if meta_pdf and meta_pdf.get('content'):
                                pdf_link = meta_pdf['content']
                        
                        # 通用方法: 查找明确标记为PDF的链接
                        if not pdf_link:
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                text = link.get_text().lower().strip()
                                
                                # 检查链接文本或href中是否包含pdf相关关键词
                                if any(keyword in text for keyword in ['pdf', 'download pdf', 'view pdf', 'full text pdf']):
                                    if _is_valid_pdf_link(href):
                                        pdf_link = href if href.startswith('http') else urljoin(pdf_url, href)
                                        break
                        
                        # 如果找到了PDF链接，递归下载
                        if pdf_link and pdf_link != pdf_url:
                            logger.info(f"Found PDF link in HTML: {pdf_link}, attempting to download")
                            return download_pdf(pdf_link, save_folder, max_retries=max_retries-1)
                        else:
                            logger.warning(f"Could not find PDF link in HTML page: {pdf_url}")
                            # 如果是Semantic Scholar且有DOI，建议使用DOI方法
                            if 'semanticscholar.org' in pdf_url and 'paper_doi' in locals() and paper_doi:
                                logger.info(f"Suggestion: Try download_pdf_by_doi with DOI: {paper_doi}")
                except Exception as e:
                    logger.error(f"Error parsing HTML page: {e}")
            
            return None
        
        # 保存PDF文件
        with open(save_path, "wb") as file:
            file.write(content)
        
        logger.info(f"Successfully downloaded PDF to {save_path}")
        return save_path
        
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading PDF from {pdf_url}")
        return None
    except Exception as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {e}")
        return None

def download_pdf_by_doi(doi: str, download_dir: str = "downloaded_papers"):
    """
    Download a PDF using DOI.
    Tries multiple strategies including publisher pages, Unpaywall API, and Sci-Hub.
    
    Args:
        doi: DOI string (can include 'doi:' prefix or full URL)
        download_dir: Directory to save the PDF
    
    Returns:
        Path to downloaded PDF file, or None if download failed
    """
    # 清理DOI格式
    doi = doi.strip()
    if doi.lower().startswith('doi:'):
        doi = doi[4:].strip()
    if doi.lower().startswith('https://doi.org/'):
        doi = doi[16:].strip()
    elif doi.lower().startswith('http://doi.org/'):
        doi = doi[15:].strip()
    
    logger.info(f"Attempting to download PDF for DOI: {doi}")
    
    doi_url = f"https://doi.org/{doi}"
    os.makedirs(download_dir, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    # 策略1: 尝试使用Unpaywall API获取开放获取的PDF
    try:
        logger.info("Trying Unpaywall API for open access PDF")
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
        unpaywall_response = httpx.get(unpaywall_url, timeout=10)
        
        if unpaywall_response.status_code == 200:
            data = unpaywall_response.json()
            
            # 尝试best_oa_location
            if data.get('best_oa_location') and data['best_oa_location'].get('url_for_pdf'):
                pdf_url = data['best_oa_location']['url_for_pdf']
                logger.info(f"Found PDF via Unpaywall: {pdf_url}")
                
                # 尝试下载
                result = download_pdf(pdf_url, save_folder=download_dir, max_retries=1)
                if result:
                    logger.info(f"Successfully downloaded PDF via Unpaywall: {result}")
                    return result
            
            # 尝试oa_locations
            if data.get('oa_locations'):
                for location in data['oa_locations']:
                    if location.get('url_for_pdf'):
                        pdf_url = location['url_for_pdf']
                        logger.info(f"Trying alternative Unpaywall location: {pdf_url}")
                        result = download_pdf(pdf_url, save_folder=download_dir, max_retries=1)
                        if result:
                            logger.info(f"Successfully downloaded PDF via Unpaywall: {result}")
                            return result
    except Exception as e:
        logger.warning(f"Unpaywall API failed: {e}")
    
    # 策略2: 访问DOI重定向的出版商页面
    try:
        logger.info(f"Trying publisher page via DOI: {doi_url}")
        response = httpx.get(doi_url, headers=headers, timeout=30, follow_redirects=True)
        publisher_url = str(response.url)
        logger.info(f"Redirected to publisher page: {publisher_url}")
        
        if response.status_code == 200:
            content = response.content
            
            # 检查是否直接返回了PDF
            if content.startswith(b'%PDF'):
                filename = f"{doi.replace('/', '_')}.pdf"
                filepath = os.path.join(download_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(content)
                logger.info(f"PDF downloaded directly from DOI: {filepath}")
                return filepath
            
            # 解析HTML页面查找PDF链接
            soup = BeautifulSoup(content, 'html.parser')
            pdf_links = []
            
            # 方法1: 查找meta标签
            meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
            if meta_pdf and meta_pdf.get('content'):
                pdf_links.append(meta_pdf['content'])
                logger.info(f"Found PDF in meta tag: {meta_pdf['content']}")
            
            # 方法2: 查找所有可能的PDF链接
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text().lower().strip()
                
                # 使用_is_valid_pdf_link验证
                if _is_valid_pdf_link(href):
                    full_url = href if href.startswith('http') else urljoin(publisher_url, href)
                    if full_url not in pdf_links:
                        pdf_links.append(full_url)
                # 或者链接文本包含PDF相关关键词
                elif any(kw in link_text for kw in ['pdf', 'download', 'full text', 'view pdf']):
                    if href.endswith('.pdf') or 'pdf' in href.lower():
                        full_url = href if href.startswith('http') else urljoin(publisher_url, href)
                        if full_url not in pdf_links:
                            pdf_links.append(full_url)
            
            if pdf_links:
                logger.info(f"Found {len(pdf_links)} candidate PDF links from publisher page")
                
                # 按优先级排序
                def pdf_link_priority(url):
                    url_lower = url.lower()
                    if 'arxiv.org/pdf' in url_lower:
                        return 0
                    elif url_lower.endswith('.pdf') and '/ielx' in url_lower:
                        return 1
                    elif url_lower.endswith('.pdf'):
                        return 2
                    elif '/pdf/' in url_lower:
                        return 3
                    else:
                        return 4
                
                pdf_links.sort(key=pdf_link_priority)
                
                # 尝试下载每个候选链接
                for i, pdf_url in enumerate(pdf_links[:5]):  # 最多尝试前5个
                    logger.info(f"Trying candidate {i+1}/{min(5, len(pdf_links))}: {pdf_url}")
                    try:
                        result = download_pdf(pdf_url, save_folder=download_dir, max_retries=1)
                        if result:
                            logger.info(f"Successfully downloaded PDF from publisher: {result}")
                            return result
                    except Exception as e:
                        logger.warning(f"Failed to download from {pdf_url}: {e}")
                        continue
    except Exception as e:
        logger.error(f"Failed to access publisher page: {e}")
    
    # 策略3: 尝试使用Sci-Hub (作为最后手段)
    try:
        logger.info(f"Trying Sci-Hub as last resort for DOI: {doi}")
        scihub_mirrors = [
            'https://sci-hub.se',
            'https://sci-hub.st',
            'https://sci-hub.ru',
        ]
        
        for mirror in scihub_mirrors:
            try:
                scihub_url = f"{mirror}/{doi}"
                logger.info(f"Trying Sci-Hub mirror: {scihub_url}")
                
                response = httpx.get(scihub_url, headers=headers, timeout=30, follow_redirects=True)
                if response.status_code == 200:
                    content = response.content
                    
                    # 检查是否直接返回PDF
                    if content.startswith(b'%PDF'):
                        filename = f"{doi.replace('/', '_')}.pdf"
                        filepath = os.path.join(download_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        logger.info(f"PDF downloaded from Sci-Hub: {filepath}")
                        return filepath
                    
                    # 解析页面查找PDF链接
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Sci-Hub通常在iframe或embed中显示PDF
                    for tag in soup.find_all(['iframe', 'embed']):
                        src = tag.get('src', '')
                        if src and ('pdf' in src.lower() or src.startswith('//')):
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif not src.startswith('http'):
                                src = urljoin(mirror, src)
                            
                            logger.info(f"Found PDF in Sci-Hub iframe: {src}")
                            result = download_pdf(src, save_folder=download_dir, max_retries=1)
                            if result:
                                logger.info(f"Successfully downloaded PDF from Sci-Hub: {result}")
                                return result
            except Exception as e:
                logger.warning(f"Sci-Hub mirror {mirror} failed: {e}")
                continue
    except Exception as e:
        logger.error(f"Sci-Hub download failed: {e}")
    
    logger.error(f"All download strategies failed for DOI: {doi}")
    return None

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    
def replace_and_with_or(query, max_keep=1):
    parts = query.split(" AND ")
    
    if len(parts) <= max_keep + 1:
        return query
    
    if max_keep > 0:
        keep_positions = random.sample(range(len(parts) - 1), max_keep)
    else:
        keep_positions = []
    
    result = parts[0]
    for i in range(len(parts) - 1):
        if i in keep_positions:
            result += " AND " + parts[i + 1]  # 保留 AND
        else:
            result += " OR " + parts[i + 1]  # 将 AND 替换为 OR
    
    return result

if __name__ == "__main__":
    papers = search_kg_papers("machine learning", top_k=3)
    print(f"Found {len(papers)} papers")
    for paper in papers:
        print(paper)
