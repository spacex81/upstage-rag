"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast
import re

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model

# Company Detection and Filtering Functions

def detect_companies(query: str) -> list[str]:
    """Detect company names in user queries and return corresponding file patterns.
    
    Args:
        query: User's query text
        
    Returns:
        List of source file patterns to filter by, or empty list for all companies
    """
    # Company name variations and their corresponding file patterns
    company_patterns = {
        "nvidia": ["nvidia_10k.pdf"],
        "nvda": ["nvidia_10k.pdf"],
        "amd": ["amd_10k.pdf"], 
        "intel": ["intel_10k.pdf"],
        "intc": ["intel_10k.pdf"],
        "broadcom": ["broadcom_10k.pdf"],
        "avgo": ["broadcom_10k.pdf"],
    }
    
    query_lower = query.lower()
    detected_files = []
    
    for company_name, file_patterns in company_patterns.items():
        # Use word boundaries to match whole words only
        if re.search(r'\b' + re.escape(company_name) + r'\b', query_lower):
            detected_files.extend(file_patterns)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(detected_files))


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.
    Company detection happens in the retrieve function for filtering.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        
        # Log detected companies for debugging
        detected_companies = detect_companies(human_input)
        if detected_companies:
            print(f"🔍 Query contains companies: {detected_companies}")
        else:
            print("🔍 Industry-wide query detected")
            
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query with company-aware filtering.

    This function takes the current state and configuration, detects any company
    names in the query, applies appropriate metadata filtering, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    query = state.queries[-1]
    
    # Detect companies mentioned in the query
    company_files = detect_companies(query)
    
    with retrieval.make_retriever(config) as retriever:
        if len(company_files) > 1:
            # Multi-company query: retrieve 2 chunks per company for balanced results
            print(f"🏢 Multi-company query detected: {company_files}")
            print(f"📊 Retrieving 2 chunks per company for balanced representation")
            
            all_results = []
            
            for company_file in company_files:
                try:
                    print(f"🔍 Retrieving from {company_file}")
                    
                    # Create filter for this specific company
                    company_filter = {"source_file": company_file}
                    
                    # Apply the filter if the retriever supports it
                    if hasattr(retriever, 'search_kwargs'):
                        # Update search kwargs with company filter and k=2
                        original_kwargs = getattr(retriever, 'search_kwargs', {})
                        retriever.search_kwargs = {
                            **original_kwargs,
                            "filter": company_filter,
                            "k": 2  # Get exactly 2 chunks from this company
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
            
            print(f"📈 Total chunks retrieved: {len(all_results)} from {len(company_files)} companies")
            response = all_results
            
        elif len(company_files) == 1:
            # Single company query: use normal retrieval with filtering
            print(f"🏢 Single company query: {company_files[0]}")
            
            metadata_filter = {"source_file": company_files[0]}
            
            try:
                if hasattr(retriever, 'search_kwargs'):
                    # Update search kwargs with metadata filter
                    original_kwargs = getattr(retriever, 'search_kwargs', {})
                    retriever.search_kwargs = {
                        **original_kwargs,
                        "filter": metadata_filter
                    }
                
                response = await retriever.ainvoke(query, config)
                
                # Restore original search kwargs
                if hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs = original_kwargs
                    
            except Exception as e:
                print(f"⚠️ Company filtering failed, falling back to unfiltered search: {e}")
                response = await retriever.ainvoke(query, config)
        else:
            # No specific companies detected, search all documents
            print("🌐 Industry-wide query: searching across all companies")
            response = await retriever.ainvoke(query, config)
        
        return {"retrieved_docs": response}


async def agent_reasoning(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """ReAct agent that reasons about the query and decides whether to use tools."""
    configuration = Configuration.from_runnable_config(config)
    
    # Import tools
    from retrieval_graph.tools import AVAILABLE_TOOLS
    
    # ReAct prompt for reasoning and tool usage
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    
    # Load model and bind tools for ReAct pattern
    model = load_chat_model(configuration.response_model)
    model_with_tools = model.bind_tools(AVAILABLE_TOOLS)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    
    print("🤔 Agent reasoning about the query and available tools...")
    response = await model_with_tools.ainvoke(message_value, config)
    
    return {"messages": [response]}


def should_continue_react(state: State) -> str:
    """Determine if the ReAct agent should continue with tool execution or provide final response."""
    
    last_message = state.messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"🔧 Agent decided to use {len(last_message.tool_calls)} tool(s)")
        return "execute_tools"
    else:
        print("✅ Agent provided final response without tools")
        return "__end__"








# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Import tools for ToolNode
from retrieval_graph.tools import AVAILABLE_TOOLS

# Add nodes for ReAct pattern
builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(agent_reasoning)
builder.add_node("execute_tools", ToolNode(AVAILABLE_TOOLS))

# Define the ReAct flow
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "agent_reasoning")

# ReAct loop: agent reasons, then either uses tools or provides final response
builder.add_conditional_edges(
    "agent_reasoning",
    should_continue_react,
    {
        "execute_tools": "execute_tools",
        "__end__": "__end__"
    }
)

# After tool execution, go back to agent reasoning for continued ReAct loop
builder.add_edge("execute_tools", "agent_reasoning")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
