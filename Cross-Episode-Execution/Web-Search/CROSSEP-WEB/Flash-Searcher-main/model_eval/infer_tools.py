#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional, Tuple
import os
import requests
import json
import time
import asyncio
from utils import openai_service

def read_page_jina(url: str) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown',
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        'X-Timeout': '10',
        'X-Token-Budget': '200000',
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error reading page: {str(e)}"

def read_page_crawl4ai(url: str) -> str:
    """Read and return the content of a webpage using crawl4ai."""
    try:
        from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

        async def _crawl():
            # CacheMode.DISABLED bypasses the SQLite cache entirely, preventing
            # "disk I/O error" when multiple threads initialize the db simultaneously.
            config = CrawlerRunConfig(cache_mode=CacheMode.DISABLED)
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)
                return result.markdown or ""

        markdown = asyncio.run(_crawl())

        if not markdown:
            return f"Error reading page: No content extracted from {url}"
        return markdown
    except ImportError:
        return "Error: crawl4ai is not installed. Please install it with: pip install crawl4ai>=0.7.4"
    except Exception as e:
        return f"Error reading page with crawl4ai: {str(e)}"

def read_page(url: str) -> str:
    """
    Read and return the content of a webpage.
    
    Provider selection via WEB_ACCESS_PROVIDER environment variable:
    - 'jina' (default): Use Jina Reader API (requires JINA_API_KEY)
    - 'crawl4ai': Use crawl4ai library (free, open-source, no API key needed)
    
    Note: If both JINA_API_KEY and crawl4ai are configured, the provider
    specified in WEB_ACCESS_PROVIDER will be used. No automatic fallback.
    """
    provider = os.getenv("WEB_ACCESS_PROVIDER", "jina").lower()
    
    if provider == "crawl4ai":
        return read_page_crawl4ai(url)
    elif provider == "jina":
        return read_page_jina(url)
    else:
        # Invalid provider, default to jina with warning
        import warnings
        warnings.warn(
            f"Invalid WEB_ACCESS_PROVIDER='{provider}'. "
            f"Valid values are 'jina' or 'crawl4ai'. Using 'jina' as default.",
            UserWarning
        )
        return read_page_jina(url)

def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "United States",
        "num": serp_num
    })
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json()

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. Try a more general query."
            
            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"\nDate published: {page['date']}" if "date" in page else "",
                    "snippet": f"\n{page.get('snippet', 'No snippet')}",
                    "source": f"\nSource: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#')
                })
            
            return search_results, ""
        
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
    
    return [], "Unexpected error in web search"

def search_tool(query: str) -> str:
    """Execute web search and return formatted results."""
    search_results, error_msg = web_search_google_serper(query, serp_num=5)
    
    if error_msg:
        return error_msg
    
    formatted_results = []
    for result in search_results:
        formatted_results.append(
            f"{result['idx']}. [{result['title']}]({result['link']})"
            f"{result['date']}{result['source']}\n"
            f"   {result['snippet'].strip()}"
        )
    
    return "\n\n".join(formatted_results) if formatted_results else "No search results found"


def truncate_text(text: str, max_length: int = 60000) -> str:
    return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

def get_summary_prompt(query: str, url: str, content: str) -> str:
    return (
        f"Task: Extract all content from the web page that matches the search query.\n"
        f"Search Query: {query}\n\n"
        f"Web Page Content [url:{url}]:\n{content}\n\n"
        "Instructions:\n"
        "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
        "- If no relevant information exists, please straightly output 'No relevant information'\n"
        "- Keep the summary under 500 words"
    )

def retry_predict(prompt: str, max_retries: int = 3) -> str:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            response = openai_service(messages, os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"), os.getenv("SUMMARY_MODEL"))
            if hasattr(response, 'content'):
                content = response.content
                return content.strip() if isinstance(content, str) else str(content)
            return str(response)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Content extraction failed: {str(e)}"
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    return "Content extraction failed after multiple attempts"

def crawl_tool(url: str, query: str) -> str:

    if not url.startswith(('http://', 'https://')):
        return "Invalid URL format. Must start with http:// or https://"
    
    page_content = read_page(url)
    if page_content.startswith("Error"):
        return page_content
    
    truncated_content = truncate_text(page_content)
    prompt = get_summary_prompt(query, url, truncated_content)
    
    return retry_predict(prompt)
