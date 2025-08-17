"""Tools for the retrieval graph.

This module contains tool definitions that can be used by the agent
to perform specialized tasks like industry-wide analysis and web search.
"""

import os
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults

from retrieval_graph import retrieval
from retrieval_graph.utils import format_docs


@tool
async def industry_analysis_tool(
    query: str,
    config: RunnableConfig = None
) -> str:
    """Retrieve documents from all major semiconductor companies for industry-wide comparative analysis.
    
    This tool retrieves 2 documents from each of the major semiconductor companies 
    (NVIDIA, AMD, Intel, Broadcom) to provide comprehensive industry perspective.
    
    Args:
        query: The original user query to search across all companies
        config: Configuration for the retrieval process
        
    Returns:
        str: Formatted documents from all companies for comparative analysis
    """
    
    print(f"🏭 Industry Analysis Tool: Retrieving from all companies for '{query}'")
    
    # Define all company files
    company_files = ["nvidia_10k.pdf", "amd_10k.pdf", "intel_10k.pdf", "broadcom_10k.pdf"]
    all_results = []
    
    try:
        with retrieval.make_retriever(config) as retriever:
            for company_file in company_files:
                try:
                    print(f"🔍 Retrieving 2 chunks from {company_file}")
                    
                    # Create filter for this specific company
                    company_filter = {"source_file": company_file}
                    
                    # Apply the filter and limit to 2 chunks per company
                    if hasattr(retriever, 'search_kwargs'):
                        original_kwargs = getattr(retriever, 'search_kwargs', {})
                        retriever.search_kwargs = {
                            **original_kwargs,
                            "filter": company_filter,
                            "k": 2  # Exactly 2 chunks from each company
                        }
                    
                    company_results = await retriever.ainvoke(query, config)
                    
                    # Restore original search kwargs
                    if hasattr(retriever, 'search_kwargs'):
                        retriever.search_kwargs = original_kwargs
                    
                    all_results.extend(company_results)
                    print(f"✅ Retrieved {len(company_results)} chunks from {company_file}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to retrieve from {company_file}: {e}")
                    continue
            
            print(f"📊 Total industry analysis chunks: {len(all_results)} from {len(company_files)} companies")
            
            if not all_results:
                return "No industry-wide documents found for this query."
            
            # Format the results for the agent
            formatted_docs = format_docs(all_results)
            
            # Add analysis context
            analysis_context = f"""
Industry-Wide Analysis Documents for: "{query}"

Retrieved {len(all_results)} documents ({len(all_results)//len(company_files)} per company) from across the semiconductor industry:
- NVIDIA, AMD, Intel, and Broadcom

Use these documents to provide comparative analysis and industry perspective:

{formatted_docs}
"""
            
            return analysis_context
            
    except Exception as e:
        error_msg = f"⚠️ Industry analysis failed: {e}"
        print(error_msg)
        return f"Error performing industry analysis: {str(e)}"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information about semiconductor industry topics.
    
    This tool searches for real-time information, recent news, current market data,
    and latest developments in the semiconductor industry that may not be available
    in the static 10-K filings.
    
    Args:
        query: Search query for current information (e.g., "NVIDIA recent earnings", 
               "semiconductor industry news 2024", "AI chip market trends")
        
    Returns:
        str: Formatted search results with recent information
        
    Example usage:
        - Recent earnings announcements and market reactions
        - Current stock prices and analyst opinions  
        - Breaking news about regulatory changes
        - New product launches and partnerships
        - Latest industry trends and forecasts
    """
    
    print(f"🔍 Web Search: Searching for current information about '{query}'")
    
    try:
        # Initialize Tavily search with API key from environment
        search = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )
        
        # Perform the search
        results = search.invoke({"query": query})
        
        if not results:
            return f"No current web information found for: {query}"
        
        # Format the results
        formatted_results = f"Web Search Results for: '{query}'\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content available')
            url = result.get('url', 'No URL')
            
            formatted_results += f"Result {i}:\n"
            formatted_results += f"Title: {title}\n"
            formatted_results += f"Content: {content}\n"
            formatted_results += f"Source: {url}\n\n"
        
        print(f"✅ Found {len(results)} web search results")
        return formatted_results
        
    except Exception as e:
        error_msg = f"⚠️ Web search failed: {e}"
        print(error_msg)
        return f"Error performing web search: {str(e)}"


# List of available tools for the agent
AVAILABLE_TOOLS = [industry_analysis_tool, web_search_tool]