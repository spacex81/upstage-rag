"""
Simple two-step section processing script:
1. Extract raw sections from document chunks (no hierarchy)
2. Build 2-layer hierarchy from raw sections using LLM
"""
import asyncio
import pickle
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Raw section extraction (no hierarchy yet)
class RawSectionOutput(BaseModel):
    section_name: str = Field(description="Section identifier")
    section_title: str = Field(description="Full section title")
    page_number: int = Field(description="Page number where section starts")
    description: Optional[str] = Field(description="Section description")

class RawDocumentSectionsOutput(BaseModel):
    sections: list[RawSectionOutput]

# Hierarchical sections (with level and parent)
class HierarchicalSectionOutput(BaseModel):
    section_name: str = Field(description="Section identifier")
    section_title: str = Field(description="Full section title")
    page_number: int = Field(description="Page number where section starts")
    level: int = Field(description="Hierarchy level: 1 (top), 2 (sub)")
    parent_section: Optional[str] = Field(description="Parent section name")
    description: Optional[str] = Field(description="Section description")

class HierarchicalDocumentSectionsOutput(BaseModel):
    sections: list[HierarchicalSectionOutput]

async def extract_raw_sections(documents, source_file):
    """Step 1: Extract raw sections from document chunks"""
    print("🔍 Step 1: Extracting raw sections from document...")
    
    # Check if raw sections already exist
    raw_sections_file = f"nosql/{source_file}_raw_sections.json"
    
    try:
        with open(raw_sections_file, "r", encoding="utf-8") as f:
            existing_raw_sections = json.load(f)
            print(f"✅ Found existing raw sections file with {len(existing_raw_sections)} sections")
            return existing_raw_sections
    except (FileNotFoundError, json.JSONDecodeError):
        print("📄 No existing raw sections found, extracting from document...")
    
    # Initialize LLM for raw extraction
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    structured_llm = llm.with_structured_output(RawDocumentSectionsOutput)
    
    # Process document in chunks of 45 pages
    chunk_size = 45
    all_sections = []
    
    for i in range(0, len(documents), chunk_size):
        chunk_docs = documents[i:i + chunk_size]
        start_page = chunk_docs[0].metadata.get('page_number', 0)
        end_page = chunk_docs[-1].metadata.get('page_number', 0)
        
        # Combine chunk content
        chunk_content = ""
        for doc in chunk_docs:
            page_num = doc.metadata.get('page_number', 0)
            chunk_content += f"\n\n=== PAGE {page_num} ===\n{doc.page_content}"
        
        # Extract sections from this chunk
        prompt = f"""
        Analyze this SEC 10-K document chunk (pages {start_page}-{end_page}) and extract ALL major sections.
        
        IMPORTANT: Extract sections as they appear - DO NOT organize into hierarchy yet.
        
        RULES:
        1. Only extract sections that START within pages {start_page}-{end_page}
        2. Look for ACTUAL section headers in the document content
        3. Identify the correct page number where each section starts
        4. Extract major SEC sections like "Part I", "Item 1. Business", etc.
        5. Include section name, title, page number, and brief description
        6. DO NOT assign levels or parent relationships yet
        
        Document chunk content:
        {chunk_content}
        """
        
        print(f"🔍 Extracting raw sections from pages {start_page}-{end_page}...")
        result = await structured_llm.ainvoke(prompt)
        chunk_sections = [s.model_dump() for s in result.sections]
        all_sections.extend(chunk_sections)
        print(f"✅ Found {len(chunk_sections)} sections in chunk {start_page}-{end_page}")
    
    print(f"✅ Total extracted {len(all_sections)} raw sections from all chunks")
    
    # Save raw sections
    Path("nosql").mkdir(exist_ok=True)
    with open(raw_sections_file, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=2)
    print(f"💾 Saved raw sections to {raw_sections_file}")
    
    return all_sections

async def build_hierarchy(raw_sections, source_file):
    """Step 2: Build 2-layer hierarchy from raw sections"""
    print("🏗️ Step 2: Building 2-layer hierarchy from raw sections...")
    
    # Check if hierarchical sections already exist
    sections_file = f"nosql/{source_file}_sections.json"
    
    try:
        with open(sections_file, "r", encoding="utf-8") as f:
            existing_sections = json.load(f)
            print(f"✅ Found existing hierarchical sections file with {len(existing_sections)} sections")
            return existing_sections
    except (FileNotFoundError, json.JSONDecodeError):
        print("📄 No existing hierarchical sections found, building hierarchy...")
    
    # Initialize LLM for hierarchy building
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    structured_llm = llm.with_structured_output(HierarchicalDocumentSectionsOutput)
    
    # Create summary of raw sections for LLM
    sections_summary = ""
    for i, section in enumerate(raw_sections):
        sections_summary += f"{i+1}. Page {section.get('page_number', 0)}: {section.get('section_name', '')} - {section.get('section_title', '')}\n"
    
    # Build hierarchy using LLM
    prompt = f"""
    Organize these raw sections into a clean 2-tier hierarchy for a SEC 10-K document.
    
    GOAL: Create exactly 2 levels:
    - Level 1: Major document Parts (Part I, Part II, Part III, Part IV)
    - Level 2: Major SEC Items within each Part
    
    RULES:
    1. Assign level 1 to major Parts (Part I, Part II, etc.)
    2. Assign level 2 to Items under each Part
    3. Set parent_section for level 2 items to their corresponding Part
    4. Keep the same page numbers and descriptions
    5. Remove any duplicate or redundant sections
    6. Only include major, meaningful sections
    
    Raw sections to organize:
    {sections_summary}
    
    Return the organized sections with proper level and parent_section assignments.
    """
    
    print("🏗️ Organizing sections into hierarchy...")
    result = await structured_llm.ainvoke(prompt)
    hierarchical_sections = [s.model_dump() for s in result.sections]
    print(f"✅ Organized {len(hierarchical_sections)} sections into hierarchy")
    
    # Save hierarchical sections
    with open(sections_file, "w", encoding="utf-8") as f:
        json.dump(hierarchical_sections, f, indent=2)
    print(f"💾 Saved hierarchical sections to {sections_file}")
    
    return hierarchical_sections

async def main():
    """Main function that runs both steps"""
    # Load the first PDF file from documents folder
    docs_folder = Path("documents")
    pdf_files = list(docs_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in documents/ folder")
        return
    
    pdf_file = pdf_files[0]
    source_file = pdf_file.name
    
    # Load parsed documents from cache
    cache_file = f"cache/{pdf_file.stem}_parsed_docs.pkl"
    
    if not os.path.exists(cache_file):
        print(f"❌ Parsed documents not found: {cache_file}")
        print("Run scripts/save_parsed_docs.py first!")
        return
    
    print(f"📋 Loading documents from cache: {cache_file}")
    with open(cache_file, "rb") as f:
        documents = pickle.load(f)
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Step 1: Extract raw sections
    raw_sections = await extract_raw_sections(documents, source_file)
    
    # Step 2: Build hierarchy
    hierarchical_sections = await build_hierarchy(raw_sections, source_file)
    
    print(f"🎉 Complete! Final result: {len(hierarchical_sections)} hierarchical sections")
    return hierarchical_sections

if __name__ == "__main__":
    asyncio.run(main())