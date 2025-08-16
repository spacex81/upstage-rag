"""Simple Section Building Graph - Bare minimum functionality."""

from typing import Optional, Sequence
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState


async def load_docs_for_sections(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, Sequence[Document]]:
    """Load PDF documents using UpstageDocumentParseLoader API for section building."""
    from langchain_upstage import UpstageDocumentParseLoader
    
    # Load the first PDF file from documents folder
    docs_folder = Path("documents")
    pdf_files = list(docs_folder.glob("*.pdf"))
    
    if not pdf_files:
        return {"docs": []}
    
    # Use the first PDF file found
    pdf_file = pdf_files[0]
    
    print(f"📋 Parsing PDF with Upstage API for sections: {pdf_file.name}")
    
    # Use UpstageDocumentParseLoader with auto format (no ocr or split params)
    loader = UpstageDocumentParseLoader(str(pdf_file))
    documents = await loader.aload()
    
    # Add source file metadata
    for doc in documents:
        doc.metadata.update({
            "source_file": pdf_file.name
        })
    
    print(f"✅ Loaded {len(documents)} documents from API for section building")
    
    return {"docs": documents}


async def create_sections(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, Sequence[Document]]:
    """Extract sections using LLM and save to file."""
    from pydantic import BaseModel, Field
    from langchain_upstage import ChatUpstage
    
    # Define section schema
    class SectionOutput(BaseModel):
        section_name: str = Field(description="Section identifier")
        section_title: str = Field(description="Full section title")
        level: int = Field(description="Hierarchy level: 1 (top), 2 (sub)")
        parent_section: Optional[str] = Field(description="Parent section name")
        description: Optional[str] = Field(description="Section description")
    
    class DocumentSectionsOutput(BaseModel):
        sections: list[SectionOutput]
    
    # Initialize LLM
    llm = ChatUpstage(model="solar-pro2")
    structured_llm = llm.with_structured_output(DocumentSectionsOutput)
    
    # Check if sections already exist
    source_file = "Nvidia_10k.pdf"
    sections_file = f"nosql/{source_file}_sections.json"
    
    try:
        with open(sections_file, "r", encoding="utf-8") as f:
            existing_sections = json.load(f)
            print(f"✅ Found existing sections file with {len(existing_sections)} sections")
            sections = existing_sections
    except (FileNotFoundError, json.JSONDecodeError):
        print("📄 No existing sections found, extracting from document...")
        
        # Process document in chunks of 45 pages
        chunk_size = 45
        all_sections = []
        
        # Check for partial progress cache
        cache_file = f"cache/{source_file}_partial_sections.json"
        processed_chunks = set()
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                all_sections = cache_data.get("sections", [])
                processed_chunks = set(cache_data.get("processed_chunks", []))
                print(f"📄 Resuming from cache: {len(all_sections)} sections, {len(processed_chunks)} chunks done")
        except (FileNotFoundError, json.JSONDecodeError):
            print("📄 No cache found, starting fresh")
        
        for i in range(0, len(state.docs), chunk_size):
            chunk_docs = state.docs[i:i + chunk_size]
            chunk_id = f"{i}-{i+chunk_size-1}"
            
            # Skip if this chunk was already processed
            if chunk_id in processed_chunks:
                print(f"⏭️  Skipping already processed chunk {chunk_id}")
                continue
            
            # Combine chunk content
            chunk_content = ""
            
            for i, doc in enumerate(chunk_docs):
                doc_num = i + 1
                chunk_content += f"\n\n=== DOCUMENT {doc_num} ===\n{doc.page_content}"
            
            # Extract sections from this chunk
            prompt = f"""
            Analyze this SEC 10-K document chunk and extract the major sections in a 2-tier hierarchy.
            
            GOAL: Create exactly 2 levels:
            - Level 1: Major document Parts (Part I, Part II, Part III, Part IV)
            - Level 2: Major SEC Items within each Part
            
            RULES:
            1. Only extract major, meaningful sections that START within this chunk
            2. Look for ACTUAL section headers in the document content
            3. Create proper parent-child relationships
            4. Use standard SEC 10-K section names
            5. Only extract sections that actually BEGIN in this chunk
            
            Document chunk content:
            {chunk_content}
            """
            
            print(f"🔍 Extracting sections from chunk {chunk_id}...")
            result = await structured_llm.ainvoke(prompt)
            chunk_sections = [s.model_dump() for s in result.sections]
            all_sections.extend(chunk_sections)
            processed_chunks.add(chunk_id)
            print(f"✅ Found {len(chunk_sections)} sections in chunk {chunk_id}")
            
            # Save progress after each chunk
            Path("cache").mkdir(exist_ok=True)
            cache_data = {
                "sections": all_sections,
                "processed_chunks": list(processed_chunks)
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            print(f"💾 Saved progress: {len(all_sections)} sections, chunk {chunk_id} completed")
        
        sections = all_sections
        print(f"✅ Total extracted {len(sections)} sections from all chunks")
        
        # Save final sections to nosql
        Path("nosql").mkdir(exist_ok=True)
        with open(sections_file, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2)
        print(f"💾 Saved final sections to {sections_file}")
        
        # Clean up cache
        try:
            import os
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print("🧹 Cleaned up cache file")
        except Exception:
            pass
    
    # Ensure nosql directory exists
    Path("nosql").mkdir(exist_ok=True)
    
    # Save sections to file
    with open(sections_file, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    
    print(f"✅ Created {len(sections)} sections and saved to {sections_file}")
    
    # Update document metadata
    processed_docs = []
    for doc in state.docs:
        doc.metadata.update({
            "has_sections": True,
            "num_sections": len(sections),
            "sections_file": sections_file
        })
        processed_docs.append(doc)
    
    return {"docs": processed_docs}


# Build the simple section graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)

# Add nodes
builder.add_node("load_docs_for_sections", load_docs_for_sections)
builder.add_node("create_sections", create_sections)

# Create simple pipeline
builder.add_edge("__start__", "load_docs_for_sections")
builder.add_edge("load_docs_for_sections", "create_sections")

# Compile the section graph
section_graph = builder.compile()
section_graph.name = "SimpleSectionGraph"