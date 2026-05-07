import requests
import re
import os
import json
import asyncio
import threading
import concurrent.futures
from bs4 import BeautifulSoup
from bs4.element import Comment
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List, Any
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.logger import get_logger

logger = get_logger(__name__)

import dotenv
dotenv.load_dotenv()

# Try to import aiohttp for async HTTP requests
try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "aiohttp"])
    import aiohttp


def extract_url_content(url: str, timeout: int = 120, retry_times: int = 3) -> Optional[str]:
    """
    Extract content from a URL, filtering out styles, scripts, and other non-content elements.
    Preserves all content-related information including text, links, tables, lists, etc.
    
    Args:
        url (str): The URL to extract content from
        timeout (int): Request timeout in seconds (default: 30)
        retry_times (int): Number of retry attempts (default: 3)
    
    Returns:
        Optional[str]: The extracted content as plain text, or None if extraction fails
    """
    # Define different request strategies
    strategies = [
        {
            "name": "简单请求（无headers）",
            "headers": {}
        },
        {
            "name": "最小headers",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        },
        {
            "name": "完整浏览器headers",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
        },
        {
            "name": "Firefox浏览器headers",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
                "Accept-Encoding": "gzip, deflate, br",
            }
        },
    ]
    
    last_error = None
    response = None
    
    # Try each strategy
    for strategy_idx, strategy in enumerate(strategies):
        try:
            if strategy_idx > 0:
                print(f"🔄 尝试策略 {strategy_idx + 1}/{len(strategies)}: {strategy['name']}")
            
            # Send GET request with current strategy
            response = requests.get(url, headers=strategy['headers'], timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Success!
            if strategy_idx > 0:
                print(f"✓ 成功使用策略: {strategy['name']}")
            break
            
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response.status_code == 403:
                if strategy_idx < len(strategies) - 1:
                    # Try next strategy
                    continue
                else:
                    # All strategies failed
                    print(f"❌ 所有请求策略均失败: 该网站有严格的反爬虫保护")
                    print(f"💡 建议: 1) 使用浏览器手动复制内容 2) 检查是否需要登录 3) 寻找API或RSS源")
                    return None
            else:
                # Other HTTP errors, raise immediately
                raise
                
        except requests.exceptions.RequestException as e:
            last_error = e
            if strategy_idx < len(strategies) - 1:
                # Try next strategy
                import time
                time.sleep(1)
                continue
            else:
                # All strategies failed
                print(f"❌ 网络请求失败: {e}")
                return None
    
    # If we get here without a response, something went wrong
    if response is None:
        if last_error:
            print(f"❌ 请求失败: {last_error}")
        return None
    
    try:
        
        # First, remove style and script tags from raw HTML using regex
        # This ensures they're completely removed before BeautifulSoup parsing
        html = response.text
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Parse cleaned HTML content with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove additional non-content elements
        # 1. Meta tags, links to stylesheets, and other metadata
        for element in soup(['meta', 'link', 'noscript']):
            element.decompose()
        
        # 2. Hidden elements and iframes
        for element in soup(['iframe', 'embed', 'object']):
            element.decompose()
        
        # 3. HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # 4. SVG and canvas elements (usually decorative)
        for element in soup(['svg', 'canvas']):
            element.decompose()
        
        # Get text content with newline separators to preserve structure
        text = soup.get_text(separator='\n')
        
        # Clean up the text while preserving content
        lines = []
        for line in text.splitlines():
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip lines that look like CSS/JS artifacts
            if re.match(r'^[\{\}\[\];,]+$', line):
                continue
            # Skip lines that still contain HTML-like tags (shouldn't happen, but just in case)
            if re.match(r'^<[^>]+>$', line):
                continue
            lines.append(line)
        
        # Join lines and remove excessive blank lines
        text = '\n'.join(lines)
        
        # Remove multiple consecutive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing content from {url}: {e}")
        return None


def extract_and_answer_query(
    url: str, 
    query: str, 
    model_name: str = "gpt-4o-mini",
    chunk_size: int = 16000,
    max_workers: int = 8,
    timeout: int = 120
) -> Tuple[bool, str]:
    """
    Extract content from URL and answer the query according to the content.
    
    This function:
    1. Extracts content from the URL
    2. Splits content into chunks (~16k characters each)
    3. Asks LLM in parallel which chunks are relevant to the query
    4. Generates final answer based on relevant chunks
    
    Args:
        url (str): The URL to extract content from
        query (str): The question to answer
        model_name (str): LLM model name (default: "gpt-4o-mini")
        chunk_size (int): Size of each chunk in characters (default: 16000)
        max_workers (int): Maximum parallel workers (default: 8)
        timeout (int): Request timeout in seconds (default: 120)
    
    Returns:
        Tuple[bool, str]: (success, answer or error message)
    """
    try:
        # Initialize LLM model
        model = get_model(model_name)
    except Exception as e:
        return False, f"Error initializing LLM model: {e}"
    
    # Step 1: Extract content from URL
    print(f"📥 正在提取URL内容: {url}")
    content = extract_url_content(url, timeout=timeout)
    
    if not content:
        return False, f"Error: Failed to extract content from {url}"
    
    print(f"✓ 提取成功，内容长度: {len(content):,} 字符")
    
    # Step 2: Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append({
            'index': len(chunks),
            'text': chunk,
            'start': i,
            'end': min(i + chunk_size, len(content))
        })
    
    print(f"📦 内容分割为 {len(chunks)} 个块 (每块约 {chunk_size:,} 字符)")
    
    # Step 3: Parallel check relevance and get answers from each chunk
    def check_and_answer_chunk(chunk_info: Dict) -> Tuple[int, Dict]:
        """Check if a chunk is relevant and get answer if relevant"""
        chunk_idx = chunk_info['index']
        chunk_text = chunk_info['text']
        
        prompt = f"""Based on the following text fragment extracted from a webpage, determine if it can answer the question. If yes, provide the answer.

Question: {query}

Text Fragment:
{chunk_text}

Please respond in JSON format with two fields:
- "related": true or false, indicating whether the text contains relevant information
- "answer": if related is true, provide the answer to the question; if false, leave it as an empty string

Example format:
{{"related": true, "answer": "Answer content here"}}
or
{{"related": false, "answer": ""}}

Please only return JSON, no other content."""
        
        try:
            response = model.generate(prompt, auto_fix_json=True, temperature=0.3, max_tokens=2000)
            
            # 解析JSON响应
            import json
            result = json.loads(response)
            
            return chunk_idx, result
        
        except Exception as e:
            print(f"⚠️  块 {chunk_idx} 处理失败: {e}")
            return chunk_idx, {"related": False, "answer": ""}
    
    print(f"🔍 并行检查相关性并获取答案 (最多 {max_workers} 个并发)...")
    
    chunk_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_and_answer_chunk, chunk) for chunk in chunks]
        
        for future in concurrent.futures.as_completed(futures):
            chunk_idx, result = future.result()
            if result.get("related", False):
                chunk_results.append((chunk_idx, result.get("answer", "")))
                print(f"  ✓ 块 {chunk_idx} 相关")
    
    # Sort by index to maintain order
    chunk_results.sort(key=lambda x: x[0])
    
    # Step 4: Generate final answer based on number of relevant chunks
    if len(chunk_results) == 0:
        print(f"✗ 未找到相关内容")
        return True, "No related answer"
    
    elif len(chunk_results) == 1:
        print(f"✓ 找到1个相关块，直接返回答案")
        return True, chunk_results[0][1]
    
    else:
        print(f"✓ 找到 {len(chunk_results)} 个相关块，合并生成最终答案...")
        
        # Combine all answers
        answers_list = [f"[Answer Fragment {idx+1}]\n{answer}" for idx, (_, answer) in enumerate(chunk_results)]
        combined_answers = "\n\n".join(answers_list)
        
        # Generate final answer by merging multiple answers
        final_prompt = f"""The following are multiple answer fragments extracted from different parts of the webpage, all relevant to the question. Please synthesize these answers to generate a complete and coherent final answer.

Question: {query}

Answer Fragments:
{combined_answers}

Please provide a comprehensive and complete answer that integrates all relevant information."""
        
        try:
            final_answer = model.generate(final_prompt, auto_fix_json=False, temperature=0.3, max_tokens=4000)
            print(f"✓ 最终答案生成完成")
            
            return True, final_answer
        
        except Exception as e:
            return False, f"Error generating final answer: {e}"


def extract_and_answer_query_from_url(url: str, query: str) -> Tuple[bool, str]:
    r"""Extract the content of a given url and answer the query according to the content.

    Args:
        url (str): The url of the webpage to be processed.
        query (str): The question to answer.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating whether the tool was executed successfully, and the answer to the query according to the content of the url (if success).
    """
    try:
        result = extract_and_answer_query(url, query)
        return result
    except Exception as e:
        logger.error(f"Extract and answer query from URL failed: {e}")
        raise


# ==============================================================================
# Literature Search Classes and Functions
# ==============================================================================

@dataclass
class PaperMetadata:
    """Paper metadata"""
    
    title: str
    authors: List[str]
    abstract: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    pdf_url: Optional[str] = None
    source: str = "unknown"  # Source: arxiv, semantic_scholar, crossref, core
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_citation(self, format_type: str = "apa") -> str:
        """Generate citation format"""
        if format_type == "bibtex":
            first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
            year = self.year or "Unknown"
            key = f"{first_author}{year}"
            authors = " and ".join(self.authors) if self.authors else "Unknown"
            
            return (
                f"@article{{{key},\n"
                f"  title = {{{self.title}}},\n"
                f"  author = {{{authors}}},\n"
                f"  year = {{{year}}},\n"
                f"  journal = {{{self.journal or 'Unknown'}}},\n"
                f"  doi = {{{self.doi or ''}}},\n"
                f"  url = {{{self.url or ''}}}\n"
                f"}}"
            )
        
        # APA format (default)
        authors_str = ""
        if self.authors:
            if len(self.authors) == 1:
                authors_str = self.authors[0]
            elif len(self.authors) == 2:
                authors_str = f"{self.authors[0]} & {self.authors[1]}"
            else:
                authors_str = f"{self.authors[0]} et al."
        
        year_str = f"({self.year})" if self.year else ""
        journal_str = f"{self.journal}." if self.journal else ""
        doi_str = f"https://doi.org/{self.doi}" if self.doi else ""
        
        return f"{authors_str} {year_str}. {self.title}. {journal_str} {doi_str}".strip()


class CitationManager:
    """Citation manager"""
    
    def __init__(self):
        self.papers: Dict[str, PaperMetadata] = {}
        
    def add_paper(self, paper: PaperMetadata) -> None:
        """Add paper"""
        key = paper.doi if paper.doi else paper.title.lower().strip()
        if key not in self.papers:
            self.papers[key] = paper
    
    def clear(self) -> None:
        """Clear all papers"""
        self.papers.clear()


class LiteratureSearch:
    """
    Multi-source academic literature search tool
    
    Supported sources:
    - arXiv: Preprints in physics, mathematics, computer science, etc.
    - Semantic Scholar: Comprehensive academic search engine
    - CrossRef: DOI lookup and metadata
    - CORE: Open access research papers
    """
    
    def __init__(self, 
                 email: Optional[str] = None,
                 api_keys: Optional[Dict[str, str]] = None,
                 citation_manager: Optional[CitationManager] = None,
                 timeout: int = 30):
        """
        Initialize literature search tool
        
        Args:
            email: Email for API access (optional)
            api_keys: API key dictionary (optional)
            citation_manager: Citation manager
            timeout: Request timeout in seconds
        """
        self.email = email or "user@example.com"
        self.api_keys = api_keys or {}
        self.citation_manager = citation_manager or CitationManager()
        self.timeout = timeout
        
        # Cache
        self._cache: Dict[str, List[PaperMetadata]] = {}
        
        # Default configuration
        self.default_max_results = 10
        
        # User-Agent for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
    
    async def search_arxiv(self,
                          query: str,
                          max_results: int = 10,
                          exact_title_first: bool = False,
                          **kwargs) -> List[PaperMetadata]:
        """
        Search arXiv preprints
        
        Args:
            query: Search query
            max_results: Maximum number of results
            exact_title_first: If True, prioritize exact title match (for known paper titles).
                              If False, use general relevance search (for topic/keyword search).
            
        Returns:
            List of paper metadata
        """
        cache_key = f"arxiv:{query}:{max_results}:{exact_title_first}"
        if cache_key in self._cache:
            logger.info(f"[arXiv] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.info(f"[arXiv] Searching: {query} (exact_title_first={exact_title_first})")
        
        url = "http://export.arxiv.org/api/query"
        
        # 策略1: 精确标题搜索 (适合已知论文标题)
        exact_params = {
            "search_query": f'ti:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # 策略2: 通用搜索 (适合主题/关键词搜索)
        general_params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                papers = []
                
                if exact_title_first:
                    # 模式1: 精确标题优先（用于查找特定论文）
                    logger.info(f"[arXiv] Exact title mode: trying ti:\"{query}\" first")
                    
                    # 先尝试精确标题搜索
                    async with session.get(url, params=exact_params, headers=self.headers) as response:
                        if response.status == 200:
                            xml_text = await response.text()
                            papers_exact = self._parse_arxiv_xml(xml_text)
                            logger.info(f"[arXiv] Exact title search found {len(papers_exact)} papers")
                            papers = papers_exact
                        else:
                            logger.warning(f"[arXiv] Exact title search failed: {response.status}")
                            papers = []
                    
                    # 如果精确搜索结果不够，用通用搜索补充
                    if len(papers) < max_results:
                        logger.info(f"[arXiv] Supplementing with general search: all:{query}")
                        async with session.get(url, params=general_params, headers=self.headers) as response:
                            if response.status == 200:
                                xml_text = await response.text()
                                papers_general = self._parse_arxiv_xml(xml_text)
                                logger.info(f"[arXiv] General search found {len(papers_general)} papers")
                                
                                # 合并结果，去重（基于标题）
                                seen_titles = {p.title.lower().strip() for p in papers}
                                for paper in papers_general:
                                    if paper.title.lower().strip() not in seen_titles:
                                        papers.append(paper)
                                        seen_titles.add(paper.title.lower().strip())
                                        if len(papers) >= max_results:
                                            break
                            else:
                                logger.warning(f"[arXiv] General search failed: {response.status}")
                else:
                    # 模式2: 通用搜索（用于主题/关键词搜索）
                    logger.info(f"[arXiv] General search mode: all:{query}")
                    async with session.get(url, params=general_params, headers=self.headers) as response:
                        if response.status == 200:
                            xml_text = await response.text()
                            papers = self._parse_arxiv_xml(xml_text)
                            logger.info(f"[arXiv] General search found {len(papers)} papers")
                        else:
                            logger.error(f"[arXiv] Request failed: {response.status}")
                            papers = []
                
                logger.info(f"[arXiv] Total found {len(papers)} papers")
                
                self._cache[cache_key] = papers
                for paper in papers:
                    self.citation_manager.add_paper(paper)
                
                return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[arXiv] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[arXiv] Search error: {e}")
            return []
    
    async def search_semantic_scholar(self,
                                      query: str,
                                      max_results: int = 10,
                                      **kwargs) -> List[PaperMetadata]:
        """
        Search Semantic Scholar
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"semantic:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[Semantic Scholar] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.info(f"[Semantic Scholar] Searching: {query}")
        
        # API configuration
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        api_key = os.getenv("S2_API_KEY") or self.api_keys.get("semantic_scholar")
        
        headers = dict(self.headers)
        if api_key:
            headers["x-api-key"] = api_key
        
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API limit
            "fields": "title,abstract,authors,year,venue,url,citationCount,externalIds,openAccessPdf"
        }
        
        try:
            await asyncio.sleep(1)  # Rate limiting
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 429:
                        logger.warning(f"[Semantic Scholar] Rate limited, waiting to retry...")
                        await asyncio.sleep(5)
                        return []
                    
                    if response.status != 200:
                        logger.error(f"[Semantic Scholar] Request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for item in data.get("data", []):
                        authors = [a.get("name", "") for a in item.get("authors", [])]
                        external_ids = item.get("externalIds", {})
                        pdf_info = item.get("openAccessPdf", {})
                        
                        paper = PaperMetadata(
                            title=item.get("title", ""),
                            authors=authors,
                            abstract=item.get("abstract", ""),
                            year=item.get("year"),
                            doi=external_ids.get("DOI"),
                            journal=item.get("venue"),
                            url=item.get("url"),
                            citations=item.get("citationCount"),
                            pdf_url=pdf_info.get("url") if pdf_info else None,
                            source="semantic_scholar"
                        )
                        papers.append(paper)
                    
                    logger.info(f"[Semantic Scholar] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[Semantic Scholar] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[Semantic Scholar] Search error: {e}")
            return []
    
    async def search_crossref(self,
                             query: str,
                             max_results: int = 10,
                             **kwargs) -> List[PaperMetadata]:
        """
        Search CrossRef (DOI database)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"crossref:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[CrossRef] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.info(f"[CrossRef] Searching: {query}")
        
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "select": "DOI,title,author,abstract,published,container-title,URL"
        }
        
        headers = dict(self.headers)
        headers["mailto"] = self.email
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"[CrossRef] Request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for item in data.get("message", {}).get("items", []):
                        # Extract authors
                        authors = []
                        for author in item.get("author", []):
                            given = author.get("given", "")
                            family = author.get("family", "")
                            if given and family:
                                authors.append(f"{given} {family}")
                            elif family:
                                authors.append(family)
                        
                        # Extract title
                        title_list = item.get("title", [])
                        title = title_list[0] if title_list else ""
                        
                        # Extract year
                        year = None
                        pub_date = item.get("published", {}).get("date-parts", [[]])[0]
                        if pub_date:
                            year = pub_date[0] if len(pub_date) > 0 else None
                        
                        # Extract journal
                        journal_list = item.get("container-title", [])
                        journal = journal_list[0] if journal_list else None
                        
                        paper = PaperMetadata(
                            title=title,
                            authors=authors,
                            abstract=item.get("abstract", ""),
                            year=year,
                            doi=item.get("DOI"),
                            journal=journal,
                            url=item.get("URL"),
                            source="crossref"
                        )
                        papers.append(paper)
                    
                    logger.info(f"[CrossRef] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[CrossRef] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[CrossRef] Search error: {e}")
            return []
    
    async def search_core(self,
                         query: str,
                         max_results: int = 10,
                         **kwargs) -> List[PaperMetadata]:
        """
        Search CORE (open access papers)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"core:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[CORE] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.info(f"[CORE] Searching: {query}")
        
        api_key = os.getenv("CORE_API_KEY") or self.api_keys.get("core")
        if not api_key:
            logger.warning("[CORE] No API key available, skipping search")
            return []
        
        url = "https://api.core.ac.uk/v3/search/works"
        headers = dict(self.headers)
        headers["Authorization"] = f"Bearer {api_key}"
        
        params = {
            "q": query,
            "limit": max_results
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"[CORE] Request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for item in data.get("results", []):
                        authors = [a.get("name", "") for a in item.get("authors", [])]
                        
                        paper = PaperMetadata(
                            title=item.get("title", ""),
                            authors=authors,
                            abstract=item.get("abstract", ""),
                            year=item.get("yearPublished"),
                            doi=item.get("doi"),
                            journal=item.get("publisher"),
                            url=item.get("downloadUrl"),
                            pdf_url=item.get("downloadUrl"),
                            source="core"
                        )
                        papers.append(paper)
                    
                    logger.info(f"[CORE] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[CORE] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[CORE] Search error: {e}")
            return []
    
    async def multi_source_search(self,
                                  query: str,
                                  sources: Optional[List[str]] = None,
                                  max_results: int = 10,
                                  exact_title_first: bool = False,
                                  **kwargs) -> Dict[str, List[PaperMetadata]]:
        """
        Multi-source parallel search
        
        Args:
            query: Search query
            sources: List of sources (arxiv, semantic_scholar, crossref, core)
            max_results: Maximum number of results per source
            exact_title_first: If True, prioritize exact title match in arXiv
            
        Returns:
            Mapping from source name to result list
        """
        if not sources:
            sources = ["arxiv", "semantic_scholar", "crossref"]
        
        logger.info(f"[Multi-source search] Query: {query}, Sources: {sources}, exact_title_first: {exact_title_first}")
        
        tasks = []
        task_sources = []
        
        for source in sources:
            if source == "arxiv":
                tasks.append(self.search_arxiv(query, max_results, exact_title_first=exact_title_first, **kwargs))
                task_sources.append("arxiv")
            elif source == "semantic_scholar":
                tasks.append(self.search_semantic_scholar(query, max_results, **kwargs))
                task_sources.append("semantic_scholar")
            elif source == "crossref":
                tasks.append(self.search_crossref(query, max_results, **kwargs))
                task_sources.append("crossref")
            elif source == "core":
                tasks.append(self.search_core(query, max_results, **kwargs))
                task_sources.append("core")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined = {}
        for source, result in zip(task_sources, results):
            if isinstance(result, Exception):
                logger.error(f"[{source}] Error: {result}")
                combined[source] = []
            else:
                combined[source] = result
        
        total = sum(len(papers) for papers in combined.values())
        logger.info(f"[Multi-source search] Found {total} papers in total")
        
        return combined
    
    async def search(self,
                    query: str,
                    max_results: int = 10,
                    sources: Optional[List[str]] = None,
                    exact_title_first: bool = False,
                    **kwargs) -> List[PaperMetadata]:
        """
        Simplified search interface - merges results from multiple sources
        
        Args:
            query: Search query
            max_results: Total maximum number of results to return after deduplication
            sources: List of sources
            exact_title_first: If True, prioritize exact title match in arXiv
            
        Returns:
            Merged and deduplicated list of papers (up to max_results)
            
        Note:
            To get max_results papers after deduplication, this method searches
            each source with max_results limit. The final list is deduplicated
            and trimmed to max_results.
        """
        results_dict = await self.multi_source_search(
            query, 
            sources=sources, 
            max_results=max_results,  # Each source returns up to max_results papers
            exact_title_first=exact_title_first,
            **kwargs
        )
        
        # Merge and deduplicate
        seen_titles = set()
        all_papers = []
        
        for source, papers in results_dict.items():
            for paper in papers:
                title_key = paper.title.lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(paper)
        
        # Return all deduplicated papers (up to max_results)
        # Since we searched each source with max_results limit,
        # we should have enough papers to return max_results after deduplication
        return all_papers[:max_results]
    
    def clear_cache(self) -> None:
        """Clear search cache"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse arXiv XML response
        
        Args:
            xml_data: arXiv XML response
            
        Returns:
            List of paper metadata
        """
        papers = []
        
        try:
            soup = BeautifulSoup(xml_data, "xml")
            
            for entry in soup.find_all("entry"):
                try:
                    # Title
                    title_elem = entry.find("title")
                    title = title_elem.text.strip().replace("\n", " ") if title_elem else ""
                    
                    # Abstract
                    summary_elem = entry.find("summary")
                    abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem else ""
                    
                    # Authors
                    authors = []
                    for author in entry.find_all("author"):
                        name_elem = author.find("name")
                        if name_elem:
                            authors.append(name_elem.text.strip())
                    
                    # Publication year
                    year = None
                    published_elem = entry.find("published")
                    if published_elem:
                        match = re.search(r"(\d{4})", published_elem.text)
                        if match:
                            year = int(match.group(1))
                    
                    # URL and PDF
                    url = None
                    pdf_url = None
                    for link in entry.find_all("link"):
                        href = link.get("href", "")
                        if link.get("rel") == "alternate":
                            url = href
                        elif link.get("title") == "pdf":
                            pdf_url = href
                    
                    # arXiv ID
                    id_elem = entry.find("id")
                    if id_elem and not url:
                        url = id_elem.text.strip()
                    
                    if not pdf_url and url:
                        # Generate PDF link from URL
                        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
                    
                    if title:
                        paper = PaperMetadata(
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            year=year,
                            journal="arXiv",
                            url=url,
                            pdf_url=pdf_url,
                            source="arxiv"
                        )
                        papers.append(paper)
                        
                except Exception as e:
                    logger.debug(f"Error parsing single arXiv entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers


# ---------- 文献搜索工具 ----------
class LiteratureSearchSingleton:
    """LiteratureSearch 单例管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LiteratureSearchSingleton, cls).__new__(cls)
                    cls._instance._literature_search = None
        return cls._instance
    
    def get_literature_search(self):
        """获取LiteratureSearch实例，线程安全"""
        if self._literature_search is None:
            with self._lock:
                if self._literature_search is None:
                    logger.info("初始化 LiteratureSearch 实例...")
                    # 从环境变量读取配置
                    api_keys = {
                        "semantic_scholar": os.getenv("S2_API_KEY"),
                        "core": os.getenv("CORE_API_KEY")
                    }
                    self._literature_search = LiteratureSearch(
                        email=os.getenv("LITERATURE_EMAIL", "user@example.com"),
                        api_keys=api_keys
                    )
                    logger.info("LiteratureSearch 实例初始化完成")
        return self._literature_search

# 全局实例
_literature_search_singleton = LiteratureSearchSingleton()


def search_academic_papers(
    query: str,
    max_results: int = 5,
    sources: Optional[List[str]] = None,
    exact_title_first: bool = False
) -> List[Dict[str, Any]]:
    """
    Search academic papers from multiple sources (arXiv, Semantic Scholar, CrossRef, CORE).
    
    This tool searches for academic papers using various free academic databases.
    It returns a JSON string with paper information including titles, authors,
    abstracts, and URLs.
    
    Args:
        query: Search query (e.g., "machine learning", "quantum computing")
        max_results: Maximum number of results to return (default: 5)
                    Note: This is the TOTAL number of results after deduplication.
                    The function will search each source with this limit to gather more papers.
        sources: List of sources to search. Options: ["arxiv", "semantic_scholar", "crossref", "core"]
                If None, defaults to ["arxiv", "semantic_scholar", "crossref"]
        exact_title_first: If True, prioritize exact title match in arXiv (for known paper titles).
                          If False, use general relevance search (for topic/keyword search).
                          Default: False
    
    Returns:
        List[Dict[str, Any]]: List of paper data with the following structure:
            {
                "query": str,
                "total_found": int,
                "papers": [
                    {
                        "title": str,
                        "authors": List[str],
                        "year": int,
                        "journal": str,
                        "abstract": str,
                        "pdf_url": str,
                        "doi": str,
                        "source": str
                    },
                    ...
                ]
            }
    
    Example:
        result = search_academic_papers("deep learning", max_results=5)
    """
    lit_search = _literature_search_singleton.get_literature_search()
    
    try:
        # 运行异步搜索
        # 为了获取更多结果,我们对每个源搜索max_results数量的论文
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            papers = loop.run_until_complete(
                lit_search.search(query, max_results=max_results, sources=sources, 
                                 exact_title_first=exact_title_first)
            )
        finally:
            loop.close()
        
        if not papers:
            return []
        
        # 转换为JSON格式
        papers_list = []
        for paper in papers:
            paper_dict = {
                "title": paper.title,
                "authors": paper.authors if paper.authors else [],
                "year": paper.year,
                "journal": paper.journal,
                "abstract": paper.abstract,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "source": paper.source
            }
            papers_list.append(paper_dict)
        
        return papers_list
        
    except Exception as e:
        logger.error(f"Literature search failed: {e}")
        return []


# ==============================================================================
# Paper and Webpage Content Extraction Functions
# ==============================================================================

def extract_paper_content_to_summary(paper_path: str) -> str:
    """
    Extract key content from a paper file (PDF or TXT).
    If the paper is too long, it will be split into 16k character chunks,
    each chunk will be processed in parallel to extract 5 key aspects,
    and then all chunks will be summarized into final 5 aspects.
    
    Args:
        paper_path: Path to the paper file (supports .pdf, .txt)
        
    Returns:
        JSON string with 5 key aspects:
        {
            "problem_and_background": str,
            "method_and_approach": str,
            "experiments_and_results": str,
            "conclusions_and_insights": str,
            "limitations_and_future_work": str
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Initialize model for extraction
    model_name = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
    model = get_model(model_name)
    
    # Read paper content based on file type
    try:
        if paper_path.endswith('.pdf'):
            # Extract text from PDF using PyPDF2
            try:
                from PyPDF2 import PdfReader
                with open(paper_path, 'rb') as f:
                    reader = PdfReader(f)
                    paper_text = ""
                    for page in reader.pages:
                        paper_text += page.extract_text()
                logger.info(f"Successfully extracted text from PDF: {len(paper_text)} characters")
            except ImportError:
                logger.error("PyPDF2 not installed. Please install it: pip install PyPDF2")
                return json.dumps({
                    "problem_and_background": None,
                    "method_and_approach": None,
                    "experiments_and_results": None,
                    "conclusions_and_insights": None,
                    "limitations_and_future_work": None
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to extract text from PDF: {e}")
                return json.dumps({
                    "problem_and_background": None,
                    "method_and_approach": None,
                    "experiments_and_results": None,
                    "conclusions_and_insights": None,
                    "limitations_and_future_work": None
                }, ensure_ascii=False, indent=2)
        else:
            # Read as text file
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
            logger.info(f"Successfully read text file: {len(paper_text)} characters")
    except Exception as e:
        logger.error(f"Failed to read paper file: {e}")
        return json.dumps({
            "problem_and_background": None,
            "method_and_approach": None,
            "experiments_and_results": None,
            "conclusions_and_insights": None,
            "limitations_and_future_work": None
        }, ensure_ascii=False, indent=2)
    
    # Define chunk size (16k characters)
    CHUNK_SIZE = 16000
    
    # Split text into chunks if needed
    chunks = []
    if len(paper_text) <= CHUNK_SIZE:
        chunks = [paper_text]
    else:
        # Split into chunks of CHUNK_SIZE
        for i in range(0, len(paper_text), CHUNK_SIZE):
            chunks.append(paper_text[i:i + CHUNK_SIZE])
    
    logger.info(f"Paper split into {len(chunks)} chunks")
    
    # Extraction prompt template for each chunk
    EXTRACTION_PROMPT = """You are a research assistant analyzing academic paper content. Extract detailed information from this paper chunk.

Paper chunk:
{chunk}

Please carefully extract and return in JSON format:
{{
    "problem_and_background": "Provide a comprehensive description (3-5 sentences) of the research problem, motivation, background context, related work, and research gaps being addressed. Include specific challenges and why this research is important. If not found, return None.",
    "method_and_approach": "Describe in detail all methods, approaches, algorithms, models, frameworks, or techniques used. Include technical details, mathematical formulations, architectures, and implementation specifics. Be thorough and comprehensive. If not found, return None.",
    "experiments_and_results": "Extract all experimental setups, datasets, evaluation metrics, quantitative results, performance comparisons, ablation studies, and key findings. Include specific numbers, percentages, improvements, and statistical significance. Preserve all data points and comparisons. If not found, return None.",
    "conclusions_and_insights": "Identify all conclusions, key insights, contributions, implications, and takeaways from the research. Include what was learned, what works well, and the significance of findings. If not found, return None.",
    "limitations_and_future_work": "Note all limitations, caveats, constraints, failure cases, open problems, and future research directions discussed. Include what didn't work, remaining challenges, and suggested improvements. If not found, return None."
}}

Important guidelines:
- Be detailed and comprehensive in your extraction
- Preserve specific technical terms, numbers, formulas, and metrics
- Extract complete information, not just summaries
- Include all quantitative results and comparisons
- If information exists, provide it in full detail
- Only return None if the aspect is truly not present in the chunk

Return ONLY the JSON object, no additional text."""
    
    # Function to extract from a single chunk
    def extract_from_chunk(chunk_idx: int, chunk_text: str) -> Dict[str, Any]:
        try:
            prompt = EXTRACTION_PROMPT.format(chunk=chunk_text)
            response = model.generate(prompt)
            
            # Check if response is None or empty
            if response is None or not response:
                logger.error(f"Failed to extract from chunk {chunk_idx + 1}: Empty or None response from model")
                return {
                    "problem_and_background": None,
                    "method_and_approach": None,
                    "experiments_and_results": None,
                    "conclusions_and_insights": None,
                    "limitations_and_future_work": None
                }
            
            # Try to parse JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # Clean up response to extract JSON
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            
            result = json.loads(response)
            logger.info(f"Successfully extracted from chunk {chunk_idx + 1}/{len(chunks)}")
            return result
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk_idx + 1}: {e}")
            return {
                "problem_and_background": None,
                "method_and_approach": None,
                "experiments_and_results": None,
                "conclusions_and_insights": None,
                "limitations_and_future_work": None
            }
    
    # Process chunks in parallel
    chunk_results = []
    with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
        futures = {executor.submit(extract_from_chunk, idx, chunk): idx 
                   for idx, chunk in enumerate(chunks)}
        
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                chunk_results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Chunk {chunk_idx + 1} processing failed: {e}")
                chunk_results.append((chunk_idx, {
                    "problem_and_background": None,
                    "method_and_approach": None,
                    "experiments_and_results": None,
                    "conclusions_and_insights": None,
                    "limitations_and_future_work": None
                }))
    
    # Sort results by chunk index
    chunk_results.sort(key=lambda x: x[0])
    chunk_extractions = [result for _, result in chunk_results]
    
    # If only one chunk, return directly
    if len(chunk_extractions) == 1:
        return json.dumps(chunk_extractions[0], ensure_ascii=False, indent=2)
    
    # Otherwise, summarize all chunks
    SUMMARY_PROMPT = """You are given multiple extractions from different chunks of the same academic paper. Please synthesize them into a single comprehensive and detailed summary.

Chunk extractions:
{chunk_extractions}

Please synthesize these into a final summary in JSON format:
{{
    "problem_and_background": "Comprehensive synthesis of the research problem, motivation, background, related work, and research gaps from all chunks. Include specific challenges and importance (3-5 sentences minimum).",
    "method_and_approach": "Complete and detailed synthesis of all methods, approaches, algorithms, models, frameworks, and techniques from all chunks. Include technical details, formulations, and implementation specifics. Be thorough and comprehensive.",
    "experiments_and_results": "Comprehensive compilation of all experimental setups, datasets, metrics, quantitative results, performance comparisons, and findings from all chunks. Preserve all specific numbers, percentages, improvements, and statistical data.",
    "conclusions_and_insights": "Complete synthesis of all conclusions, key insights, contributions, implications, and takeaways from all chunks. Include what was learned and the significance of findings.",
    "limitations_and_future_work": "Comprehensive summary of all limitations, constraints, failure cases, open problems, and future research directions from all chunks. Include challenges and suggested improvements."
}}

Guidelines:
- Combine ALL information from all chunks - be comprehensive and detailed
- Organize information logically and coherently
- Remove redundancy but preserve all unique information
- Maintain specific technical terms, numbers, formulas, and metrics
- If an aspect is None in all chunks, keep it as None in the final summary
- Prioritize completeness and technical detail over brevity

Return ONLY the JSON object, no additional text."""
    
    try:
        # Format chunk extractions for summary
        chunk_extractions_str = json.dumps(chunk_extractions, ensure_ascii=False, indent=2)
        summary_prompt = SUMMARY_PROMPT.format(chunk_extractions=chunk_extractions_str)
        
        # Generate summary
        summary_response = model.generate(summary_prompt)
        
        # Check if response is None or empty
        if summary_response is None or not summary_response:
            logger.error("Failed to synthesize final summary: Empty or None response from model")
            raise ValueError("Empty or None response from model")
        
        # Parse JSON from response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_response, re.DOTALL)
        if json_match:
            summary_response = json_match.group(1)
        
        # Clean up response
        summary_response = summary_response.strip()
        if summary_response.startswith('```'):
            summary_response = summary_response.split('```')[1]
            if summary_response.startswith('json'):
                summary_response = summary_response[4:]
        
        final_result = json.loads(summary_response)
        logger.info("Successfully synthesized final summary from all chunks")
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to synthesize final summary: {e}")
        # Fallback: merge all non-None values manually
        final_result = {
            "problem_and_background": None,
            "method_and_approach": None,
            "experiments_and_results": None,
            "conclusions_and_insights": None,
            "limitations_and_future_work": None
        }
        
        for key in final_result.keys():
            values = [chunk.get(key) for chunk in chunk_extractions if chunk.get(key) is not None]
            if values:
                final_result[key] = " ".join(values)
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)

def extract_webpage_content_to_summary(webpage_content: str) -> str:
    """
    Extract key content from webpage text.
    If the content is too long, it will be split into 16k character chunks,
    each chunk will be processed in parallel to extract 6 key aspects,
    and then all chunks will be summarized into final 6 aspects.
    
    Args:
        webpage_content: The text content of the webpage
        
    Returns:
        JSON string with 6 key aspects:
        {
            "page_overview": str,
            "main_points": str,
            "evidence_and_details": str,
            "conclusions_or_recommendations": str,
            "limitations_and_bias": str,
            "relevance_to_research_question": str
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Initialize model for extraction
    model_name = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
    model = get_model(model_name)
    
    # Define chunk size (16k characters)
    CHUNK_SIZE = 16000
    
    # Split text into chunks if needed
    chunks = []
    if len(webpage_content) <= CHUNK_SIZE:
        chunks = [webpage_content]
    else:
        # Split into chunks of CHUNK_SIZE
        for i in range(0, len(webpage_content), CHUNK_SIZE):
            chunks.append(webpage_content[i:i + CHUNK_SIZE])
    
    logger.info(f"Webpage content split into {len(chunks)} chunks")
    
    # Extraction prompt template for each chunk
    EXTRACTION_PROMPT = """You are a research assistant analyzing webpage content. Extract detailed information from this webpage chunk.

Webpage chunk:
{chunk}

Please carefully extract and return in JSON format:
{{
    "page_overview": "Provide a comprehensive overview (3-5 sentences) of what this page discusses, including the main topic, purpose, and context. If not found, return None.",
    "main_points": "List all key points, arguments, or findings in detail. Include specific claims, statements, and important information. Be thorough and comprehensive. If not found, return None.",
    "evidence_and_details": "Extract all supporting evidence, data, statistics, examples, case studies, experimental results, or specific details mentioned. Include numbers, dates, names, and concrete information. If not found, return None.",
    "conclusions_or_recommendations": "Identify any conclusions, recommendations, suggestions, implications, or actionable insights provided. Include any future directions or practical applications mentioned. If not found, return None.",
    "limitations_and_bias": "Note any limitations, caveats, disclaimers, potential biases, conflicting information, or uncertainties mentioned or apparent in the content. If not found, return None.",
    "relevance_to_research_question": "Analyze how this content could be relevant to research questions, what insights it provides, and what aspects might be useful for further investigation. If not found, return None."
}}

Important guidelines:
- Be detailed and comprehensive in your extraction
- Preserve specific facts, numbers, names, and dates
- Extract complete information, not just summaries
- If information exists, provide it in full detail
- Only return None if the aspect is truly not present in the chunk

Return ONLY the JSON object, no additional text."""
    
    # Function to extract from a single chunk
    def extract_from_chunk(chunk_idx: int, chunk_text: str) -> Dict[str, Any]:
        try:
            prompt = EXTRACTION_PROMPT.format(chunk=chunk_text)
            response = model.generate(prompt)
            
            # Check if response is None or empty
            if response is None or not response:
                logger.error(f"Failed to extract from chunk {chunk_idx + 1}: Empty or None response from model")
                return {
                    "page_overview": None,
                    "main_points": None,
                    "evidence_and_details": None,
                    "conclusions_or_recommendations": None,
                    "limitations_and_bias": None,
                    "relevance_to_research_question": None
                }
            
            # Try to parse JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # Clean up response to extract JSON
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            
            result = json.loads(response)
            logger.info(f"Successfully extracted from chunk {chunk_idx + 1}/{len(chunks)}")
            return result
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk_idx + 1}: {e}")
            return {
                "page_overview": None,
                "main_points": None,
                "evidence_and_details": None,
                "conclusions_or_recommendations": None,
                "limitations_and_bias": None,
                "relevance_to_research_question": None
            }
    
    # Process chunks in parallel
    chunk_results = []
    with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
        futures = {executor.submit(extract_from_chunk, idx, chunk): idx 
                   for idx, chunk in enumerate(chunks)}
        
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                chunk_results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Chunk {chunk_idx + 1} processing failed: {e}")
                chunk_results.append((chunk_idx, {
                    "page_overview": None,
                    "main_points": None,
                    "evidence_and_details": None,
                    "conclusions_or_recommendations": None,
                    "limitations_and_bias": None,
                    "relevance_to_research_question": None
                }))
    
    # Sort results by chunk index
    chunk_results.sort(key=lambda x: x[0])
    chunk_extractions = [result for _, result in chunk_results]
    
    # If only one chunk, return directly
    if len(chunk_extractions) == 1:
        return json.dumps(chunk_extractions[0], ensure_ascii=False, indent=2)
    
    # Otherwise, summarize all chunks
    SUMMARY_PROMPT = """You are given multiple extractions from different chunks of the same webpage. Please synthesize them into a single comprehensive and detailed summary.

Chunk extractions:
{chunk_extractions}

Please synthesize these into a final summary in JSON format:
{{
    "page_overview": "Comprehensive overview synthesizing information from all chunks. Include the main topic, purpose, context, and scope (3-5 sentences minimum).",
    "main_points": "Complete synthesis of all main points, arguments, and findings from all chunks. Preserve all important information, organize logically, and maintain detail. Be thorough.",
    "evidence_and_details": "Comprehensive compilation of all evidence, data, statistics, examples, and specific details from all chunks. Preserve all concrete information including numbers, dates, and names.",
    "conclusions_or_recommendations": "Complete synthesis of all conclusions, recommendations, suggestions, and implications from all chunks. Include all actionable insights and future directions.",
    "limitations_and_bias": "Comprehensive summary of all limitations, caveats, disclaimers, and potential biases from all chunks.",
    "relevance_to_research_question": "Detailed analysis of how the entire page content is relevant to research questions, what insights it provides, and what aspects are useful for investigation."
}}

Guidelines:
- Combine ALL information from all chunks - be comprehensive and detailed
- Organize information logically and coherently
- Remove redundancy but preserve all unique information
- Maintain specific facts, numbers, names, and dates
- If an aspect is None in all chunks, keep it as None in the final summary
- Prioritize completeness and detail over brevity

Return ONLY the JSON object, no additional text."""
    
    try:
        # Format chunk extractions for summary
        chunk_extractions_str = json.dumps(chunk_extractions, ensure_ascii=False, indent=2)
        summary_prompt = SUMMARY_PROMPT.format(chunk_extractions=chunk_extractions_str)
        
        # Generate summary
        summary_response = model.generate(summary_prompt)
        
        # Check if response is None or empty
        if summary_response is None or not summary_response:
            logger.error("Failed to synthesize final summary: Empty or None response from model")
            raise ValueError("Empty or None response from model")
        
        # Parse JSON from response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_response, re.DOTALL)
        if json_match:
            summary_response = json_match.group(1)
        
        # Clean up response
        summary_response = summary_response.strip()
        if summary_response.startswith('```'):
            summary_response = summary_response.split('```')[1]
            if summary_response.startswith('json'):
                summary_response = summary_response[4:]
        
        final_result = json.loads(summary_response)
        logger.info("Successfully synthesized final summary from all chunks")
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to synthesize final summary: {e}")
        # Fallback: merge all non-None values manually
        final_result = {
            "page_overview": None,
            "main_points": None,
            "evidence_and_details": None,
            "conclusions_or_recommendations": None,
            "limitations_and_bias": None,
            "relevance_to_research_question": None
        }
        
        for key in final_result.keys():
            values = [chunk.get(key) for chunk in chunk_extractions if chunk.get(key) is not None]
            if values:
                final_result[key] = " ".join(values)
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)


def summarize_webpage(webpage_url: str) -> Dict[str, Any]:
    """
    Extract and summarize content from a specified webpage URL.
    
    This function will:
    1. Extract content from the webpage URL
    2. Check content length (skip if > 160k characters)
    3. Analyze and extract 6 key aspects from the content
    
    Args:
        webpage_url: URL of the webpage to extract and summarize
        
    Returns:
        Dict[str, Any]: Webpage summary with the following structure:
        {
            "url": str,  # Original URL
            "success": bool,  # Whether extraction was successful
            "content_length": int,  # Length of extracted content (if successful)
            "summary": {  # Extracted content (if successful)
                "page_overview": str,
                "main_points": str,
                "evidence_and_details": str,
                "conclusions_or_recommendations": str,
                "limitations_and_bias": str,
                "relevance_to_research_question": str
            },
            "error": str  # Error message (if failed)
        }
    
    Example:
        result = summarize_webpage("https://example.com/article")
    """
    # Content length limit (160k characters)
    MAX_CONTENT_LENGTH = 160000
    
    result = {
        "url": webpage_url,
        "success": False,
        "content_length": None,
        "summary": None,
        "error": None
    }
    
    try:
        # Check if URL should be skipped (Wikipedia/YouTube)
        url_lower = webpage_url.lower()
        skip_domains = ['wikipedia.org', 'youtube.com', 'youtu.be', 'm.youtube.com']
        if any(domain in url_lower for domain in skip_domains):
            result["error"] = "Skipped: Wikipedia or YouTube URL"
            logger.warning(f"Skipping URL: {webpage_url} (Wikipedia/YouTube)")
            return result
        
        # Extract webpage content
        logger.info(f"Extracting content from URL: {webpage_url}")
        content = extract_url_content(webpage_url)
        
        if not content:
            result["error"] = f"Failed to extract content from {webpage_url}"
            logger.error(result["error"])
            return result
        
        # Check content length
        content_length = len(content)
        result["content_length"] = content_length
        
        if content_length > MAX_CONTENT_LENGTH:
            result["error"] = f"Content too long: {content_length} characters (max: {MAX_CONTENT_LENGTH})"
            logger.warning(result["error"])
            return result
        
        logger.info(f"Extracted {content_length} characters from webpage")
        
        # Analyze webpage content
        logger.info(f"Analyzing webpage content...")
        summary_json = extract_webpage_content_to_summary(content)
        summary = json.loads(summary_json)
        
        result["success"] = True
        result["summary"] = summary
        logger.info(f"Successfully extracted and summarized webpage content")
        
        return result
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        logger.error(f"Failed to summarize webpage: {e}")
        return result

def search_and_summarize_papers(query: str, max_number: int = 3) -> List[Dict[str, Any]]:
    """
    Search academic papers, download PDFs, and extract content summaries.
    
    This function will:
    1. Search 2*max_number papers from each of 3 sources (total 6*max_number papers)
    2. Filter papers that have PDF links
    3. Sort by year in descending order (newest first)
    4. Auto-download papers until max_number successful downloads or all papers exhausted
    5. Extract 5 key aspects from all successfully downloaded papers
    6. Return max_number papers with summaries
    
    Args:
        query: Search query string
        max_number: Final number of papers to return, do not set more than 3(default: 3)
        
    Returns:
        List[Dict[str, Any]]: List of papers, each containing:
        {
            "title": str,
            "authors": List[str],
            "year": int,
            "journal": str,
            "abstract": str,
            "pdf_url": str,
            "doi": str,
            "source": str,
            "downloaded": bool,  # Whether successfully downloaded
            "local_path": str,   # Local file path (if downloaded successfully)
            "summary": {         # Extracted content summary (if downloaded successfully)
                "problem_and_background": str,
                "method_and_approach": str,
                "experiments_and_results": str,
                "conclusions_and_insights": str,
                "limitations_and_future_work": str
            }
        }
    """
    # Import from our_tools to avoid circular imports
    from tools.our_tools import download_media_from_url
    
    # 计算搜索数量：每个源搜索2*max_number，共3个源
    search_per_source = max_number * 2
    total_search = search_per_source * 3  # 6*max_number
    
    logger.info(f"开始搜索论文: query='{query}', 每个源搜索={search_per_source}篇, 总搜索={total_search}篇, 目标下载={max_number}篇")
    
    # 1. 搜索论文（从3个源各搜索2*max_number篇）
    try:
        papers = search_academic_papers(query, max_results=search_per_source)
        logger.info(f"搜索到 {len(papers)} 篇论文")
    except Exception as e:
        logger.error(f"搜索论文失败: {e}")
        return []
    
    if not papers:
        logger.warning("没有搜索到任何论文")
        return []

    # print("papers: ", papers)
    
    # 2. 先筛选出所有有PDF链接的论文
    papers_with_pdf = []
    papers_without_pdf_count = 0
    
    for i, paper in enumerate(papers):
        if paper.get("pdf_url"):
            papers_with_pdf.append(paper)
        else:
            papers_without_pdf_count += 1
    
    logger.info(f"筛选结果: {len(papers_with_pdf)} 篇有PDF链接, {papers_without_pdf_count} 篇无PDF链接（已过滤）")
    
    if not papers_with_pdf:
        logger.warning("没有找到任何有PDF链接的论文")
        return []
    
    # 3. 按年份倒序排序（越新的论文排越前面）
    papers_sorted = sorted(papers_with_pdf, key=lambda p: p.get('year') or 0, reverse=True)
    logger.info(f"论文已按年份倒序排序，最新年份: {papers_sorted[0].get('year', 'Unknown')}")
    
    # 4. 为排序后的论文添加索引
    papers_with_pdf = [(i, paper) for i, paper in enumerate(papers_sorted)]
    
    logger.info(f"准备下载 {len(papers_with_pdf)} 篇有PDF链接的论文，目标: {max_number} 篇")
    
    # 5. 定义下载和提取的任务函数
    def download_and_extract(paper_info: tuple) -> tuple:
        """
        下载单篇论文并提取内容
        注意：这个函数会被并行调用，但内部的extract_paper_content_to_summary已经是并行的
        为避免并行套并行问题，我们在这里串行调用下载和提取
        """
        idx, paper = paper_info
        paper_result = paper.copy()
        paper_result["downloaded"] = False
        paper_result["local_path"] = None
        paper_result["summary"] = None
        
        pdf_url = paper.get("pdf_url")
        title = paper.get('title', 'Unknown')[:50]
        
        logger.info(f"正在下载论文 (索引 {idx}): {title}...")
        
        try:
            # 下载
            download_result = download_media_from_url(pdf_url)
            
            if download_result.get("success"):
                local_path = download_result.get("path")
                paper_result["downloaded"] = True
                paper_result["local_path"] = local_path
                logger.info(f"✓ 下载成功 (索引 {idx}): {local_path}")
                
                # 提取论文内容（注意：extract_paper_content_to_summary内部已经是并行的）
                logger.info(f"正在提取论文内容 (索引 {idx})...")
                try:
                    summary_json = extract_paper_content_to_summary(local_path)
                    summary = json.loads(summary_json)
                    paper_result["summary"] = summary
                    logger.info(f"✓ 内容提取成功 (索引 {idx})")
                except Exception as e:
                    logger.error(f"✗ 提取内容失败 (索引 {idx}): {e}")
                    paper_result["summary"] = {
                        "problem_and_background": None,
                        "method_and_approach": None,
                        "experiments_and_results": None,
                        "conclusions_and_insights": None,
                        "limitations_and_future_work": None
                    }
            else:
                logger.warning(f"✗ 下载失败 (索引 {idx}): {download_result.get('path', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"✗ 下载过程出错 (索引 {idx}): {e}")
        
        return (idx, paper_result)
    
    # 6. 自动顺延下载机制：尝试下载直到成功下载max_number篇或用完所有论文
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    successfully_downloaded = []
    attempted_indices = set()
    current_idx = 0
    max_attempts = len(papers_with_pdf)  # 最多尝试所有有PDF的论文
    
    logger.info(f"开始并行下载，目标：成功下载 {max_number} 篇")
    
    # 循环下载直到达到目标数量或用完所有论文
    attempt_round = 1
    while len(successfully_downloaded) < max_number and current_idx < len(papers_with_pdf):
        # 确定本轮要尝试的论文
        papers_to_try = []
        batch_size = min(3, max_number - len(successfully_downloaded))  # 每次并行下载3篇
        
        while len(papers_to_try) < batch_size and current_idx < len(papers_with_pdf):
            if current_idx not in attempted_indices:
                papers_to_try.append(papers_with_pdf[current_idx])
                attempted_indices.add(current_idx)
            current_idx += 1
        
        if not papers_to_try:
            break
        
        logger.info(f"第 {attempt_round} 轮下载：尝试 {len(papers_to_try)} 篇论文 (已成功: {len(successfully_downloaded)}/{max_number})")
        
        # 并行下载本轮论文
        with ThreadPoolExecutor(max_workers=min(3, len(papers_to_try))) as executor:
            futures = {executor.submit(download_and_extract, paper_info): paper_info 
                       for paper_info in papers_to_try}
            
            for future in as_completed(futures):
                try:
                    idx, paper_result = future.result()
                    if paper_result.get("downloaded"):
                        successfully_downloaded.append((idx, paper_result))
                        logger.info(f"✓ 成功下载第 {len(successfully_downloaded)}/{max_number} 篇")
                    else:
                        logger.warning(f"✗ 论文下载失败，将尝试下一篇")
                except Exception as e:
                    paper_info = futures[future]
                    idx, paper = paper_info
                    logger.error(f"处理论文时发生异常: {e}")
        
        attempt_round += 1
    
    # 7. 容错机制：如果一篇都没下载成功，重试一次前max_number篇
    if len(successfully_downloaded) == 0 and len(papers_with_pdf) > 0:
        logger.warning("⚠️ 第一次尝试未成功下载任何论文，开始重试...")
        
        # 重置并重试前max_number篇
        retry_papers = papers_with_pdf[:min(max_number, len(papers_with_pdf))]
        
        with ThreadPoolExecutor(max_workers=min(3, len(retry_papers))) as executor:
            futures = {executor.submit(download_and_extract, paper_info): paper_info 
                       for paper_info in retry_papers}
            
            for future in as_completed(futures):
                try:
                    idx, paper_result = future.result()
                    if paper_result.get("downloaded"):
                        successfully_downloaded.append((idx, paper_result))
                        logger.info(f"✓ 重试成功！下载第 {len(successfully_downloaded)}/{max_number} 篇")
                except Exception as e:
                    logger.error(f"重试时发生异常: {e}")
        
        if len(successfully_downloaded) == 0:
            logger.error("❌ 重试后仍未成功下载任何论文")
    
    # 8. 按索引排序并返回结果
    successfully_downloaded.sort(key=lambda x: x[0])
    results = [paper_result for _, paper_result in successfully_downloaded[:max_number]]
    
    logger.info(f"完成！总搜索 {len(papers)} 篇，有PDF {len(papers_with_pdf)} 篇，成功下载并提取 {len(results)} 篇")
    return results


def extract_webpage_content_to_summary(webpage_content: str) -> str:
    """
    Extract key content from webpage text.
    If the content is too long, it will be split into 16k character chunks,
    each chunk will be processed in parallel to extract 6 key aspects,
    and then all chunks will be summarized into final 6 aspects.
    
    Args:
        webpage_content: The text content of the webpage
        
    Returns:
        JSON string with 6 key aspects:
        {
            "page_overview": str,
            "main_points": str,
            "evidence_and_details": str,
            "conclusions_or_recommendations": str,
            "limitations_and_bias": str,
            "relevance_to_research_question": str
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from models import get_model
    
    # Initialize model for extraction
    model_name = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
    model = get_model(model_name)
    
    # Define chunk size (16k characters)
    CHUNK_SIZE = 16000
    
    # Split text into chunks if needed
    chunks = []
    if len(webpage_content) <= CHUNK_SIZE:
        chunks = [webpage_content]
    else:
        # Split into chunks of CHUNK_SIZE
        for i in range(0, len(webpage_content), CHUNK_SIZE):
            chunks.append(webpage_content[i:i + CHUNK_SIZE])
    
    logger.info(f"Webpage content split into {len(chunks)} chunks")
    
    # Extraction prompt template for each chunk
    EXTRACTION_PROMPT = """You are a research assistant analyzing webpage content. Extract detailed information from this webpage chunk.

Webpage chunk:
{chunk}

Please carefully extract and return in JSON format:
{{
    "page_overview": "Provide a comprehensive overview (3-5 sentences) of what this page discusses, including the main topic, purpose, and context. If not found, return None.",
    "main_points": "List all key points, arguments, or findings in detail. Include specific claims, statements, and important information. Be thorough and comprehensive. If not found, return None.",
    "evidence_and_details": "Extract all supporting evidence, data, statistics, examples, case studies, experimental results, or specific details mentioned. Include numbers, dates, names, and concrete information. If not found, return None.",
    "conclusions_or_recommendations": "Identify any conclusions, recommendations, suggestions, implications, or actionable insights provided. Include any future directions or practical applications mentioned. If not found, return None.",
    "limitations_and_bias": "Note any limitations, caveats, disclaimers, potential biases, conflicting information, or uncertainties mentioned or apparent in the content. If not found, return None.",
    "relevance_to_research_question": "Analyze how this content could be relevant to research questions, what insights it provides, and what aspects might be useful for further investigation. If not found, return None."
}}

Important guidelines:
- Be detailed and comprehensive in your extraction
- Preserve specific facts, numbers, names, and dates
- Extract complete information, not just summaries
- If information exists, provide it in full detail
- Only return None if the aspect is truly not present in the chunk

Return ONLY the JSON object, no additional text."""
    
    # Function to extract from a single chunk
    def extract_from_chunk(chunk_idx: int, chunk_text: str) -> Dict[str, Any]:
        try:
            prompt = EXTRACTION_PROMPT.format(chunk=chunk_text)
            response = model.generate(prompt)
            
            # Check if response is None or empty
            if response is None or not response:
                logger.error(f"Failed to extract from chunk {chunk_idx + 1}: Empty or None response from model")
                return {
                    "page_overview": None,
                    "main_points": None,
                    "evidence_and_details": None,
                    "conclusions_or_recommendations": None,
                    "limitations_and_bias": None,
                    "relevance_to_research_question": None
                }
            
            # Try to parse JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # Clean up response to extract JSON
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            
            result = json.loads(response)
            logger.info(f"Successfully extracted from chunk {chunk_idx + 1}/{len(chunks)}")
            return result
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk_idx + 1}: {e}")
            return {
                "page_overview": None,
                "main_points": None,
                "evidence_and_details": None,
                "conclusions_or_recommendations": None,
                "limitations_and_bias": None,
                "relevance_to_research_question": None
            }
    
    # Process chunks in parallel
    chunk_results = []
    with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
        futures = {executor.submit(extract_from_chunk, idx, chunk): idx 
                   for idx, chunk in enumerate(chunks)}
        
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                chunk_results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Chunk {chunk_idx + 1} processing failed: {e}")
                chunk_results.append((chunk_idx, {
                    "page_overview": None,
                    "main_points": None,
                    "evidence_and_details": None,
                    "conclusions_or_recommendations": None,
                    "limitations_and_bias": None,
                    "relevance_to_research_question": None
                }))
    
    # Sort results by chunk index
    chunk_results.sort(key=lambda x: x[0])
    chunk_extractions = [result for _, result in chunk_results]
    
    # If only one chunk, return directly
    if len(chunk_extractions) == 1:
        return json.dumps(chunk_extractions[0], ensure_ascii=False, indent=2)
    
    # Otherwise, summarize all chunks
    SUMMARY_PROMPT = """You are given multiple extractions from different chunks of the same webpage. Please synthesize them into a single comprehensive and detailed summary.

Chunk extractions:
{chunk_extractions}

Please synthesize these into a final summary in JSON format:
{{
    "page_overview": "Comprehensive overview synthesizing information from all chunks. Include the main topic, purpose, context, and scope (3-5 sentences minimum).",
    "main_points": "Complete synthesis of all main points, arguments, and findings from all chunks. Preserve all important information, organize logically, and maintain detail. Be thorough.",
    "evidence_and_details": "Comprehensive compilation of all evidence, data, statistics, examples, and specific details from all chunks. Preserve all concrete information including numbers, dates, and names.",
    "conclusions_or_recommendations": "Complete synthesis of all conclusions, recommendations, suggestions, and implications from all chunks. Include all actionable insights and future directions.",
    "limitations_and_bias": "Comprehensive summary of all limitations, caveats, disclaimers, and potential biases from all chunks.",
    "relevance_to_research_question": "Detailed analysis of how the entire page content is relevant to research questions, what insights it provides, and what aspects are useful for investigation."
}}

Guidelines:
- Combine ALL information from all chunks - be comprehensive and detailed
- Organize information logically and coherently
- Remove redundancy but preserve all unique information
- Maintain specific facts, numbers, names, and dates
- If an aspect is None in all chunks, keep it as None in the final summary
- Prioritize completeness and detail over brevity

Return ONLY the JSON object, no additional text."""
    
    try:
        # Format chunk extractions for summary
        chunk_extractions_str = json.dumps(chunk_extractions, ensure_ascii=False, indent=2)
        summary_prompt = SUMMARY_PROMPT.format(chunk_extractions=chunk_extractions_str)
        
        # Generate summary
        summary_response = model.generate(summary_prompt)
        
        # Check if response is None or empty
        if summary_response is None or not summary_response:
            logger.error("Failed to synthesize final summary: Empty or None response from model")
            raise ValueError("Empty or None response from model")
        
        # Parse JSON from response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_response, re.DOTALL)
        if json_match:
            summary_response = json_match.group(1)
        
        # Clean up response
        summary_response = summary_response.strip()
        if summary_response.startswith('```'):
            summary_response = summary_response.split('```')[1]
            if summary_response.startswith('json'):
                summary_response = summary_response[4:]
        
        final_result = json.loads(summary_response)
        logger.info("Successfully synthesized final summary from all chunks")
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to synthesize final summary: {e}")
        # Fallback: merge all non-None values manually
        final_result = {
            "page_overview": None,
            "main_points": None,
            "evidence_and_details": None,
            "conclusions_or_recommendations": None,
            "limitations_and_bias": None,
            "relevance_to_research_question": None
        }
        
        for key in final_result.keys():
            values = [chunk.get(key) for chunk in chunk_extractions if chunk.get(key) is not None]
            if values:
                final_result[key] = " ".join(values)
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)


def summarize_paper(paper_source: str) -> Dict[str, Any]:
    """
    Extract and summarize content from a specified paper (local file, URL, or paper name).
    
    This function can handle:
    - Local file paths (.pdf or .txt files)
    - Paper URLs (will download first, then extract)
    - Paper names/titles (will search first, then download and extract)
    
    Args:
        paper_source: Path to local paper file, URL to download paper, OR paper name/title to search
        
    Returns:
        Dict[str, Any]: Paper summary with the following structure:
        {
            "source": str,  # Original input (file path, URL, or paper name)
            "source_type": str,  # "local_file", "downloaded_url", or "searched_paper"
            "paper_info": dict,  # Paper metadata (if searched)
            "local_path": str,  # Path to the paper file (original or downloaded)
            "success": bool,  # Whether extraction was successful
            "summary": {  # Extracted content (if successful)
                "problem_and_background": str,
                "method_and_approach": str,
                "experiments_and_results": str,
                "conclusions_and_insights": str,
                "limitations_and_future_work": str
            },
            "error": str  # Error message (if failed)
        }
    
    Example:
        # Local file
        result = summarize_paper("/path/to/paper.pdf")
        
        # URL
        result = summarize_paper("https://arxiv.org/pdf/2301.12345.pdf")
        
        # Paper name/title
        result = summarize_paper("Attention Is All You Need")
    """
    # Import from our_tools to avoid circular imports
    from tools.our_tools import download_media_from_url, search_academic_papers
    
    result = {
        "source": paper_source,
        "source_type": None,
        "paper_info": None,
        "local_path": None,
        "success": False,
        "summary": None,
        "error": None
    }
    
    try:
        # Determine input type: URL, local file, or paper name
        is_url = paper_source.startswith("http://") or paper_source.startswith("https://")
        is_local_file = os.path.exists(paper_source) if not is_url else False
        
        if is_url:
            # Case 1: URL - download directly
            logger.info(f"Input detected as URL: {paper_source}")
            download_result = download_media_from_url(paper_source)
            
            if not download_result.get("success"):
                result["error"] = f"Failed to download paper: {download_result.get('path', 'Unknown error')}"
                logger.error(result["error"])
                return result
            
            local_path = download_result.get("path")
            result["source_type"] = "downloaded_url"
            result["local_path"] = local_path
            logger.info(f"Paper downloaded to: {local_path}")
            
        elif is_local_file:
            # Case 2: Local file - use directly
            logger.info(f"Input detected as local file: {paper_source}")
            local_path = paper_source
            result["source_type"] = "local_file"
            result["local_path"] = local_path
            logger.info(f"Using local paper file: {local_path}")
            
        else:
            # Case 3: Paper name/title - search first
            logger.info(f"Input detected as paper name/title: {paper_source}")
            logger.info(f"Searching for paper: {paper_source}")
            
            # Search for the paper (use exact_title_first=True for better matching)
            papers = search_academic_papers(paper_source, max_results=10, exact_title_first=True)
            
            if not papers:
                result["error"] = f"No papers found for query: {paper_source}"
                logger.error(result["error"])
                return result
            
            # Filter papers with PDF URLs
            papers_with_pdf = [p for p in papers if p.get("pdf_url")]
            
            if not papers_with_pdf:
                result["error"] = f"Found {len(papers)} papers but none have PDF links available"
                logger.error(result["error"])
                return result
            
            # Find the best matching paper by title similarity
            def calculate_similarity(query: str, title: str) -> float:
                """Calculate similarity score between query and title"""
                query_lower = query.lower().strip()
                title_lower = title.lower().strip()
                
                # Exact match - highest score
                if query_lower == title_lower:
                    return 1.0
                
                # Query is substring of title or vice versa
                if query_lower in title_lower or title_lower in query_lower:
                    return 0.9
                
                # Calculate word overlap score
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                
                # Remove common stop words for better matching
                stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
                query_words -= stop_words
                title_words -= stop_words
                
                if not query_words or not title_words:
                    return 0.0
                
                # Jaccard similarity
                intersection = len(query_words & title_words)
                union = len(query_words | title_words)
                
                return intersection / union if union > 0 else 0.0
            
            # Score each paper
            paper_scores = []
            for paper in papers_with_pdf:
                title = paper.get("title", "")
                score = calculate_similarity(paper_source, title)
                paper_scores.append((score, paper))
                logger.debug(f"Paper: {title[:60]}... | Score: {score:.3f}")
            
            # Sort by score (descending) and select the best match
            paper_scores.sort(key=lambda x: x[0], reverse=True)
            best_score, paper_with_pdf = paper_scores[0]
            
            logger.info(f"Best match (score={best_score:.3f}): {paper_with_pdf.get('title', 'Unknown')[:80]}")
            
            # Warn if the best match score is low
            if best_score < 0.3:
                logger.warning(f"Low similarity score ({best_score:.3f}). The found paper may not be what you're looking for.")
                logger.warning(f"Found: {paper_with_pdf.get('title')}")
                logger.warning(f"Query: {paper_source}")
            
            # Store paper info
            result["paper_info"] = {
                "title": paper_with_pdf.get("title"),
                "authors": paper_with_pdf.get("authors"),
                "year": paper_with_pdf.get("year"),
                "journal": paper_with_pdf.get("journal"),
                "doi": paper_with_pdf.get("doi"),
                "source": paper_with_pdf.get("source"),
                "pdf_url": paper_with_pdf.get("pdf_url")
            }
            
            logger.info(f"Found paper: {paper_with_pdf.get('title', 'Unknown')[:80]}")
            logger.info(f"Downloading from: {paper_with_pdf.get('pdf_url')}")
            
            # Download the paper
            download_result = download_media_from_url(paper_with_pdf.get("pdf_url"))
            
            if not download_result.get("success"):
                result["error"] = f"Failed to download paper: {download_result.get('path', 'Unknown error')}"
                logger.error(result["error"])
                return result
            
            local_path = download_result.get("path")
            result["source_type"] = "searched_paper"
            result["local_path"] = local_path
            logger.info(f"Paper downloaded to: {local_path}")
        
        # Extract content from the paper
        logger.info(f"Extracting content from paper: {local_path}")
        summary_json = extract_paper_content_to_summary(local_path)
        summary = json.loads(summary_json)
        
        result["success"] = True
        result["summary"] = summary
        logger.info(f"Successfully extracted and summarized paper content")
        
        return result
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        logger.error(f"Failed to summarize paper: {e}")
        return result




def search_and_summarize_webpages(query: str, max_number: int = 3, time_range: str = "d3",
                                   region: str = "None") -> List[Dict[str, Any]]:
    """
    Search webpages, extract content, and analyze summaries.
    
    This function will:
    1. Search webpages (default: 8)
    2. Attempt to extract content from the first N webpages (default: 3)
    3. If extraction fails or content exceeds 160k characters, automatically move to the next
    4. Extract 6 key aspects from all successfully extracted webpage content
    5. Return webpage data with summaries
    
    Args:
        query: Search query string
        max_number: Maximum number of webpages to extract, do not set it more than 3 (default: 3)
        time_range (str): Time range filter (default: 'd3' = past 3 days). Supported values:
            - 'h' or 'd1': Past 24 hours
            - 'd3': Past 3 days
            - 'w' or 'd7': Past week
            - 'm' or 'm1': Past month
            - 'y' or 'y1': Past year
        region (str): Optional region/country code (ISO 3166-1 alpha-2), e.g. 'cn', 'us' (default: 'None')
        
    Returns:
        List[Dict[str, Any]]: List of webpages, each containing:
        {
            "title": str,
            "description": str,
            "url": str,
            "extracted": bool,  # Whether content was successfully extracted
            "summary": {        # Extracted content summary (if extraction successful)
                "page_overview": str,
                "main_points": str,
                "evidence_and_details": str,
                "conclusions_or_recommendations": str,
                "limitations_and_bias": str,
                "relevance_to_research_question": str
            }
        }
        Note: Original content field is not included to reduce return data size
    """
    from tools.tool_integration import search_google
    
    # 内容长度限制（160k字符）
    MAX_CONTENT_LENGTH = 160000
    
    logger.info(f"开始搜索网页: query='{query}', max_number={max_number}, time_range={time_range}, region={region}")

    # 1. 搜索网页
    try:
        webpages = search_google(query, num_result_pages=2 * max_number, time_range=time_range, region=region)
        logger.info(f"搜索到 {len(webpages)} 个网页")
    except Exception as e:
        logger.error(f"搜索网页失败: {e}")
        return []
    
    if not webpages:
        logger.warning("没有搜索到任何网页")
        return []
    
    # 2. 过滤掉不需要提取的URL（Wikipedia和YouTube）
    def should_skip_url(url: str) -> bool:
        """判断URL是否应该跳过"""
        if not url:
            return True
        url_lower = url.lower()
        # 跳过Wikipedia和YouTube
        skip_domains = [
            'wikipedia.org',
            'youtube.com',
            'youtu.be',
            'm.youtube.com'
        ]
        return any(domain in url_lower for domain in skip_domains)
    
    # 分类网页：可提取的和需要跳过的
    extractable_webpages = []
    skipped_webpages = []
    
    for i, webpage in enumerate(webpages):
        url = webpage.get("url", "")
        if should_skip_url(url):
            webpage_result = webpage.copy()
            webpage_result["extracted"] = False
            webpage_result["content"] = None
            webpage_result["summary"] = None
            webpage_result["skip_reason"] = "Skipped: Wikipedia or YouTube URL"
            skipped_webpages.append((i, webpage_result))
            logger.info(f"跳过网页 {i+1}/{len(webpages)} (Wikipedia/YouTube): {webpage.get('title', 'Unknown')[:50]}")
        else:
            extractable_webpages.append((i, webpage))
    
    logger.info(f"找到 {len(extractable_webpages)} 个可提取的网页，{len(skipped_webpages)} 个被跳过")
    
    if not extractable_webpages:
        logger.warning("没有可提取的网页")
        return []
    
    # 3. 定义提取和分析的任务函数
    def extract_and_analyze(webpage_info: tuple) -> tuple:
        """
        提取单个网页内容并分析
        返回: (idx, webpage_result, success)
        """
        idx, webpage = webpage_info
        webpage_result = webpage.copy()
        webpage_result["extracted"] = False
        webpage_result["content"] = None
        webpage_result["summary"] = None
        
        url = webpage.get("url")
        title = webpage.get('title', 'Unknown')[:50]
        
        logger.info(f"正在提取网页 (索引 {idx}): {title}...")
        
        try:
            # 提取网页内容
            content = extract_url_content(url)
            
            if content:
                # 检查内容长度
                content_length = len(content)
                if content_length > MAX_CONTENT_LENGTH:
                    logger.warning(f"✗ 内容过长 (索引 {idx}): {content_length} 字符 > {MAX_CONTENT_LENGTH} 字符，跳过")
                    webpage_result["skip_reason"] = f"Content too long: {content_length} characters"
                    return (idx, webpage_result, False)
                
                webpage_result["extracted"] = True
                webpage_result["content"] = content
                logger.info(f"✓ 提取成功 (索引 {idx}): {content_length} 字符")
                
                # 分析网页内容
                logger.info(f"正在分析网页内容 (索引 {idx})...")
                try:
                    summary_json = extract_webpage_content_to_summary(content)
                    summary = json.loads(summary_json)
                    webpage_result["summary"] = summary
                    logger.info(f"✓ 内容分析成功 (索引 {idx})")
                    return (idx, webpage_result, True)
                except Exception as e:
                    logger.error(f"✗ 分析内容失败 (索引 {idx}): {e}")
                    webpage_result["summary"] = {
                        "page_overview": None,
                        "main_points": None,
                        "evidence_and_details": None,
                        "conclusions_or_recommendations": None,
                        "limitations_and_bias": None,
                        "relevance_to_research_question": None
                    }
                    return (idx, webpage_result, True)  # 提取成功但分析失败，仍视为成功
            else:
                logger.warning(f"✗ 提取失败 (索引 {idx}): 未能获取内容")
                return (idx, webpage_result, False)
        
        except Exception as e:
            logger.error(f"✗ 提取过程出错 (索引 {idx}): {e}")
            return (idx, webpage_result, False)
    
    # 4. 自动顺延提取机制：尝试提取直到成功提取max_number个或用完所有网页
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    successfully_extracted = []
    attempted_indices = set()
    current_idx = 0
    
    logger.info(f"开始并行提取，目标：成功提取 {max_number} 个网页")
    
    # 循环提取直到达到目标数量或用完所有网页
    attempt_round = 1
    while len(successfully_extracted) < max_number and current_idx < len(extractable_webpages):
        # 确定本轮要尝试的网页
        webpages_to_try = []
        batch_size = min(3, max_number - len(successfully_extracted))  # 每次并行提取3个
        
        while len(webpages_to_try) < batch_size and current_idx < len(extractable_webpages):
            if current_idx not in attempted_indices:
                webpages_to_try.append(extractable_webpages[current_idx])
                attempted_indices.add(current_idx)
            current_idx += 1
        
        if not webpages_to_try:
            break
        
        logger.info(f"第 {attempt_round} 轮提取：尝试 {len(webpages_to_try)} 个网页 (已成功: {len(successfully_extracted)}/{max_number})")
        
        # 并行提取本轮网页
        with ThreadPoolExecutor(max_workers=min(3, len(webpages_to_try))) as executor:
            futures = {executor.submit(extract_and_analyze, webpage_info): webpage_info 
                       for webpage_info in webpages_to_try}
            
            for future in as_completed(futures):
                try:
                    idx, webpage_result, success = future.result()
                    if success:
                        successfully_extracted.append((idx, webpage_result))
                        logger.info(f"✓ 成功提取第 {len(successfully_extracted)}/{max_number} 个网页")
                    else:
                        logger.warning(f"✗ 网页提取失败，将尝试下一个")
                except Exception as e:
                    webpage_info = futures[future]
                    idx, webpage = webpage_info
                    logger.error(f"处理网页时发生异常: {e}")
        
        attempt_round += 1
    
    # 5. 按索引排序并返回结果（移除content字段以减少返回数据量）
    successfully_extracted.sort(key=lambda x: x[0])
    results = []
    for _, webpage_result in successfully_extracted[:max_number]:
        # 创建副本并移除content字段
        result_copy = webpage_result.copy()
        if "content" in result_copy:
            del result_copy["content"]
        results.append(result_copy)
    
    # 统计成功提取的数量
    extracted_count = len(results)
    
    logger.info(f"完成！共搜索 {len(webpages)} 个网页（跳过 {len(skipped_webpages)} 个Wikipedia/YouTube），成功提取并分析 {extracted_count} 个")
    return results




if __name__ == "__main__":

    # ## 1. extract and answer query from url
    # url = "https://export.shobserver.com/baijiahao/html/1050377.html"
    # query = "受害者姓名和年龄"
    # result = extract_and_answer_query_from_url(url, query)
    # print(result)

    ## 2. search and summarize webpages
    query = "明尼苏达ICE事件"
    result = search_and_summarize_webpages("uncertainties in historical tropical cyclone records review")
    print(result)

    # ## 3. search and summarize papers
    # query = "AI scientist"
    # result = search_and_summarize_papers(query)
    # print(result)

