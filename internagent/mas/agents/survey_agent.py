"""
Survey Agent for InternAgent

This module implements the Survey Agent, which performs comprehensive literature
surveys on research topics. The agent generates intelligent search queries, retrieves
relevant academic papers from multiple sources, scores papers based on relevance,
and performs deep reading analysis to extract methodological details from top papers.
This agent supports automated, iterative literature review with query refinement.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import asyncio
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.literature_search import LiteratureSearch, PaperMetadata
from ..tools.web_search import WebSearch
from ..tools.utils import parse_io_description, format_papers_for_printing_next_query,\
    download_pdf, extract_text_from_pdf, download_pdf_by_doi, select_papers

logger = logging.getLogger(__name__)


class SurveyAgent(BaseAgent):
    """
    Survey Agent conducts comprehensive literature surveys for research topics.

    This agent performs intelligent literature search by:
    - Generating context-aware search queries based on research topics
    - Retrieving papers from multiple academic sources (Semantic Scholar, arXiv, CrossRef, CORE)
    - Iteratively refining search queries to expand paper coverage
    - Scoring papers based on relevance, novelty, and methodological quality
    - Performing deep reading analysis on top-ranked papers to extract methodological details

    The agent employs an iterative search strategy that starts with keyword queries
    and progressively diversifies using paper similarity and reference-based queries
    to build a comprehensive literature bank.
    
    Supported Search Tools:
    - literature_search: Searches academic papers from arXiv, Semantic Scholar, CrossRef, CORE, KG Papers
    - web_search: Searches web pages (general web content) via Google Serper API
    
    Note: literature_search is for academic papers, web_search is for general web content.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the survey agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.max_papers = config.get("max_papers", 5) # max paper for download and deep read
        self.search_depth = config.get("search_depth", "moderate")  # shallow, moderate, deep
        self.sources = config.get("sources", ["arxiv", "crossref"])
        self.max_concurrent_tasks = 10
        # Initialize tools
        tools_config = config.get("_global_config", {}).get("tools", {})
        self.literature_search = None
        self.web_search = None
        
        self._init_literature_search(tools_config.get("literature_search", {}))
        self._init_web_search(tools_config.get("web_search", {}))
    
    def _init_literature_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the literature search tool (multi-source academic search).
        
        Args:
            config: Literature search configuration
        """
        try:
            self.literature_search = LiteratureSearch(
                config=config
            )
            logger.info("Literature search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize literature search: {str(e)}")
    
    def _init_web_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the web search tool.
        """
        try:
            self.web_search = WebSearch(config=config)
            logger.info("Web search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize web search: {str(e)}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute survey agent with configurable search sources.
        
        Args:
            context: Context dictionary containing research topic information
            params: Parameters including optional 'search_sources' to specify which search tools to use
                   Available sources: 'literature_search', 'web_search'
        
        Returns:
            Dict containing:
                - 'papers': List of academic papers (from literature_search)
                - 'web_results': List of web pages (from web_search)
        """
        results = {
            "papers": [],
            "web_results": []
        }
        
        # Execute literature search for academic papers
            
        # Execute web search for web pages (separate from papers)
        if 'web_search' in self.sources:
            web_results, _ = await self.web_search_query(context=context)
            results["web_results"] = web_results
            # remove 'web_search' from sources to avoid duplication
            self.sources = [src for src in self.sources if src != 'web_search']
            
        papers, _ = await self.advanced_query_paper(context=context)
        results["papers"] = papers
        
        return results
    
    async def literature_search_query(self, 
                                     query: str, 
                                     max_results: int = 10) -> Dict[str, Any]:
        """
        Search academic literature only (NOT web pages).
        
        Args:
            query: Search query string
            max_results: Maximum number of results per source
        
        Returns:
            Dict with source names as keys and paper lists as values
            Example: {'arxiv': [...], 'semantic_scholar': [...], 'kg_papers': [...]}
        """
        all_results = {}
        
        try:
            logger.info(f"[Literature Search] Searching academic papers: {query} from sources {self.sources}")
            # Use configured sources or defaults (exclude kg_papers from multi_source_search)

            if self.sources:
                lit_results = await self.literature_search.multi_source_search(
                    query=query,
                    sources=self.sources,
                    max_results=max_results
                )
                
                # Convert PaperMetadata objects to dict format
                for source, papers in lit_results.items():
                    if papers:
                        formatted_papers = []
                        for paper in papers:
                            paper_dict = {
                                'title': paper.title,
                                'authors': paper.authors,
                                'abstract': paper.abstract,
                                'content': paper.content or '',
                                'year': paper.year,
                                'doi': paper.doi,
                                'url': paper.url,
                                'source': paper.source,
                                'citations': paper.citations,
                                'pdf_url': paper.pdf_url
                            }
                            formatted_papers.append(paper_dict)
                        
                        all_results[source] = formatted_papers
                    
        except Exception as e:
            logger.error(f"[Literature Search] Search error: {e}")
        
        return all_results
    
    async def web_search_query(self, context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Perform web search based on research context (NOT academic papers).
        
        Args:
            context: Context containing research topic information
        
        Returns:
            Tuple of (web_results, search_queries)
        """
        web_results = []
        search_queries = []
        
        if not self.web_search:
            logger.warning("Web search tool not initialized")
            return web_results, search_queries
        
        goal_description = context.get("description", "")
        
        try:
            raw_results = self.web_search.search_serper(goal_description, 10)
            
            if raw_results:
                for result in raw_results:
                    if 'error' not in result:
                        web_results.append({
                            'title': result.get('title', ''),
                            'description': result.get('long_description', ''),
                            'url': result.get('url', ''),
                            'source': 'web_search'
                        })
            logger.info(f"[Web search] found {len(web_results)} results")
        except Exception as e:
            logger.error(f"Web search error: {e}")
            raise AgentExecutionError(f"Failed to perform web search: {str(e)}")
        
        return web_results, search_queries


    async def advanced_query_paper(self, context) -> Tuple[List[Dict[str, Any]], List[str]]:
        # ------------------------------------------------------------
        # Helper: scoring single batch
        # ------------------------------------------------------------
        async def score_batch(batch_index, batch, semaphore):
            async with semaphore:
                # Prepare papers (content fallback to abstract)
                abs_batch = [
                    {
                        'id': paper['id'],
                        'title': paper['title'],
                        'content': paper['content'] if paper['content'] else paper['abstract']
                    }
                    for paper in batch
                ]

                # Build prompt
                if io_description is not None:
                    prompt = (
                        "You are a precise and reliable literature-review scoring assistant.\n\n"
                        "Read each paper in the list below and assign a score to each one individually.\n"
                        "Each paper is a dictionary containing: id, title, and content.\n\n"
                        "Scoring criteria:\n"
                        f"1. Relevance to the domain: {domain}\n"
                        f"2. Input/Output match:\n"
                        f"   - Input: {io_description[0]}\n"
                        f"   - Output: {io_description[1]}\n"
                        "3. Empirical novelty of the method\n"
                        "4. Interestingness and meaningfulness\n\n"
                        "Instructions:\n"
                        "- Score each paper independently.\n"
                        "- Use a scoring scale from 1 to 10 (10 = excellent match).\n"
                        "- Do NOT add new papers. Do NOT modify IDs.\n"
                        "- Return only a JSON object where:\n"
                        "    keys = paper.id\n"
                        "    values = numeric scores\n\n"
                        f"The papers to score are:\n{abs_batch}\n\n"
                        "Return JSON only."
                    )
                else:
                    prompt = (
                        "You are a precise and reliable literature-review scoring assistant.\n\n"
                        "Read each paper in the list below and assign a score to each one individually.\n"
                        "Each paper is a dictionary containing: id, title, and content.\n\n"
                        "Scoring criteria:\n"
                        "1. Relevance to the target topic\n"
                        "2. Novelty\n"
                        "3. Empirical strength\n"
                        "4. Meaningfulness\n\n"
                        "Instructions:\n"
                        "- Score each paper independently.\n"
                        "- Use a scoring scale from 1 to 10.\n"
                        "- Do NOT add new papers. Do NOT modify IDs.\n"
                        "- Return only a JSON object where:\n"
                        "    keys = paper.id\n"
                        "    values = numeric scores\n\n"
                        f"The papers to score are:\n{abs_batch}\n\n"
                        "Return JSON only."
                    )

                # Call model
                try:
                    response = await self._call_model(
                        prompt=prompt,
                        schema=output_schema_paper_score
                    )
                    return batch_index, response
                except Exception as e:
                    logger.error(f"Failed scoring batch {batch_index}: {e}")
                    return batch_index, {}


        # ------------------------------------------------------------
        # Helper: literature search single query
        # ------------------------------------------------------------
        async def run_lit_query(q, semaphore):
            async with semaphore:
                try:
                    out = await self.literature_search_query(q, 10)
                    return q, out
                except Exception as e:
                    logger.error(f"Query {q} failed: {e}")
                    return q, None

        # ------------------------------------------------------------
        # Extract context
        # ------------------------------------------------------------
        search_queries = []
        goal_description = context.get("description", {})
        domain = context.get("domain", "")

        # Schemas
        output_schema_paper_score = {
            "type": "object",
            "Properties": {
                "^[a-zA-Z0-9_]+$": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10
                }
            }
        }

        output_schema_paper_details = {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "contributions": {"type": "string"},
                "methods": {"type": "string"},
                "challenges": {"type": "string"}
            },
            "required": ["background", "contributions", "methods", "challenges"]
        }

        # ------------------------------------------------------------
        # Step 1: define task attribute
        # ------------------------------------------------------------
        define_task_attribute_prompt = (
            f"You are a researcher working on: {domain}. "
            f"Define task attributes using Input(...) Output(...). "
            f"Just return the attribute itself."
        )
        try:
            response = await self._call_model(prompt=define_task_attribute_prompt)
            io_description = parse_io_description(response)
        except Exception as e:
            logger.error(f"Error defining task attribute: {e}")
            raise AgentExecutionError("Failed to define task attribute")

        # ------------------------------------------------------------
        # Step 2: initial query (KeywordQuery)
        # ------------------------------------------------------------
        init_keyword_query_prompt = (
            f"You are a researcher doing literature review on {goal_description}. "
            f"Propose keywords for Semantic Scholar. "
            f"Return KeywordQuery('...') only."
        )
        try:
            response = await self._call_model(prompt=init_keyword_query_prompt)
            init_query = response
        except Exception as e:
            logger.error("Initial keyword query generation failed", exc_info=e)
            raise AgentExecutionError("Failed to generate initial keyword query")

        # Run initial query
        init_paper_lst = await self.literature_search_query(init_query, 10)
        search_queries.append(init_query)

        # Flatten initial papers
        if init_paper_lst:
            flattened = []
            for src, papers in init_paper_lst.items():
                if isinstance(papers, list):
                    flattened.extend(papers)
                elif isinstance(papers, dict) and "data" in papers:
                    flattened.extend(papers["data"])
            paper_bank = {str(i): p for i, p in enumerate(flattened)}
        else:
            paper_bank = {}

        # ------------------------------------------------------------
        # Step 3: generate remaining queries (串行生成，但暂不执行)
        # ------------------------------------------------------------
        all_queries = []
        iteration = 0

        while len(paper_bank) < self.max_papers and iteration < 10:
            data_list = [
                {'id': id, **info}
                for id, info in paper_bank.items()
            ]
            grounding_k = 10
            grounding_papers = data_list[:grounding_k]
            grounding_str = format_papers_for_printing_next_query(grounding_papers)

            if io_description:
                new_query_prompt = (
                    f"You are a researcher studying {domain}. "
                    f"Generate a new query (PaperQuery / KeywordQuery / GetReferences) "
                    f"based on current results. "
                    f"Input={io_description[0]} Output={io_description[1]}. "
                    f"Papers so far: {grounding_str}. "
                    f"Previous queries: {search_queries}. "
                    f"Return ONLY the new query."
                )
            else:
                new_query_prompt = (
                    f"You are a researcher studying {domain}. "
                    f"Generate new query based on: {grounding_str}. "
                    f"Previous queries: {search_queries}. "
                    f"Return ONLY the query."
                )

            try:
                response = await self._call_model(prompt=new_query_prompt)
                new_query = response
                all_queries.append(new_query)
                search_queries.append(new_query)
            except Exception as e:
                logger.error(f"Error generating new query: {e}")
                break

            iteration += 1

        # ------------------------------------------------------------
        # Step 4: run all literature queries concurrently
        # ------------------------------------------------------------
        semaphore_lit = asyncio.Semaphore(self.max_concurrent_tasks)

        lit_tasks = [
            asyncio.create_task(run_lit_query(q, semaphore_lit))
            for q in all_queries
        ]
        lit_results = await asyncio.gather(*lit_tasks)

        # process results
        for q, new_paper_lst in lit_results:
            if not new_paper_lst:
                continue

            flattened = []
            for source, papers in new_paper_lst.items():
                if isinstance(papers, list):
                    flattened.extend(papers)
                elif isinstance(papers, dict) and "data" in papers:
                    flattened.extend(papers["data"])

            existing_titles = {p['title'] for p in paper_bank.values()}
            new_papers = [p for p in flattened if p['title'] not in existing_titles]

            if new_papers:
                start = len(paper_bank)
                for i, p in enumerate(new_papers):
                    paper_bank[str(start+i)] = p

        # ------------------------------------------------------------
        # Step 5: scoring (并发 + 写回修复)
        # ------------------------------------------------------------
        data_list = [{'id': id, **info} for id, info in paper_bank.items()]
        paper_bank = data_list[:]  # convert to list

        BATCH_SIZE = 10
        batches = []
        for batch_index in range(0, len(paper_bank), BATCH_SIZE):
            batch = paper_bank[batch_index:batch_index + BATCH_SIZE]
            batches.append((batch_index, batch))

        semaphore_score = asyncio.Semaphore(self.max_concurrent_tasks)

        score_tasks = [
            asyncio.create_task(score_batch(bi, batch, semaphore_score))
            for bi, batch in batches
        ]
        score_results = await asyncio.gather(*score_tasks)
        logger.info(f"Completed scoring all batches: {score_results}")
        # ------------------------------------------------------------
        # FIX: scoring 写回逻辑（绝不再出现 KeyError: 'score'）
        # ------------------------------------------------------------
        for batch_index, score_dict in score_results:
            batch_start = batch_index
            batch_end = min(batch_index + BATCH_SIZE, len(paper_bank))
            batch_size = batch_end - batch_start

            # 1. 全部先初始化默认 score（避免缺失值导致 KeyError）
            for global_id in range(batch_start, batch_end):
                paper_bank[global_id]['score'] = 1  # 默认最低分

            # 2. 用 LLM 返回结果覆盖
            if isinstance(score_dict, dict):
                for key, score in score_dict.items():
                    try:
                        local_id = int(key)
                        if 0 <= local_id < batch_size:
                            global_id = batch_start + local_id
                            paper_bank[global_id]['score'] = score
                    except:
                        continue

        logger.info(f"Number of papers in paper_bank: {len(paper_bank)}")

        # ------------------------------------------------------------
        # Step 6: deep read (与你原版本一致)
        # ------------------------------------------------------------
        rag_read_depth = 3
        selected_for_deep_read = select_papers(paper_bank, self.max_papers, rag_read_depth)

        base_dir = 'tmp'
        pdf_dir = os.path.join(base_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        for paper in selected_for_deep_read:
            url = paper.get('url')
            doi = paper.get('doi')

            # -----------------------------------------------
            # 1. 优先使用已有的 paper.content，如果存在且非空
            # -----------------------------------------------
            content_text = paper.get("content")
            if isinstance(content_text, str) and content_text.strip():
                text = content_text.strip()
                logger.info(f"Using existing content from kg for paper ID {paper['id']}")
            else:
                # -----------------------------------------------
                # 2. 否则从 PDF 下载或通过 DOI 下载
                # -----------------------------------------------
                pdf_path = None
                if url:
                    pdf_path = download_pdf(url, save_folder=pdf_dir)
                if doi and not pdf_path:
                    pdf_path = download_pdf_by_doi(doi=doi, download_dir=pdf_dir)

                text = None
                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)

            # -----------------------------------------------
            # 3. 只有当 text 存在时才进行 LLM 分析
            # -----------------------------------------------
            if text:
                detail_prompt = (
                    f"Analyze the following paper text: {text}\n"
                    f"Extract background, contributions, methods, challenges. Return JSON."
                )
                try:
                    response = await self._call_model(
                        prompt=detail_prompt,
                        schema=output_schema_paper_details
                    )
                    paper["background"] = response.get("background", "")
                    paper["contributions"] = response.get("contributions", "")
                    paper["methods"] = response.get("methods", "")
                    paper["challenges"] = response.get("challenges", "")
                except Exception:
                    pass

        selected_ids = [p['id'] for p in selected_for_deep_read]
        for p in paper_bank:
            p['is_deep_read'] = (p['id'] in selected_ids)

        return paper_bank, search_queries
