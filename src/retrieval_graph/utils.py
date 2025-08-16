"""Utility functions for the retrieval graph.

This module contains utility functions for handling messages, documents,
and other common operations in project.

Functions:
    get_message_text: Extract text content from various message formats.
    format_docs: Convert documents to an xml-formatted string.
"""
import os 
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_upstage import ChatUpstage

def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML with citation information.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string with citation info when available.
    """
    metadata = doc.metadata or {}
    
    # Extract citation information
    page_number = metadata.get('page_number')
    hierarchical_section = metadata.get('hierarchical_section')
    source_file = metadata.get('source_file', '')
    
    # Create citation text based on available metadata
    citation_parts = []
    if page_number:
        citation_parts.append(f"Page {page_number}")
    if hierarchical_section:
        citation_parts.append(hierarchical_section)
    
    # Format XML attributes (all metadata)
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"
    
    # Add citation text at the end of content if we have citation info
    content = doc.page_content
    if citation_parts:
        citation_text = ", ".join(citation_parts)
        # Extract just the filename from source_file path
        file_name = source_file.split('/')[-1] if source_file else "document"
        content = f"{doc.page_content}\n\n[Citation: {citation_text} from {file_name}]"

    return f"<document{meta}>\n{content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""



def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    if provider == "upstage":
        return ChatUpstage(
            model=model,
            reasoning_effort="high"
        )
    
    return init_chat_model(model, model_provider=provider)