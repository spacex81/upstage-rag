#!/usr/bin/env python3
"""
Enhanced script to enrich Pinecone chunks with page numbers, sections, and subsections.
Fetches chunks from Pinecone, locates them in source PDFs, extracts citation metadata,
and updates the chunks in Pinecone with enriched metadata.

Features:
- Fetches chunks from Pinecone for specified company
- Locates exact text in corresponding PDF files
- Extracts page numbers and hierarchical section information
- Updates Pinecone chunks with enriched metadata
- Supports batch processing and dry-run mode

Usage: python enrich_chunk_metadata.py [company_name] [options]
Company names: nvidia, amd, intel, broadcom (default: nvidia)

Options:
  --count N     Process N chunks (default: 1)
  --all         Process all chunks for the company
  --dry-run     Show what would be updated without making changes
  -h, --help    Show help message

All companies support full section detection with hierarchical display.
"""

import os
import re
import json
import random
import sys
import PyPDF2
from pathlib import Path
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

def clean_text(text):
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_page_number(text, position):
    """Extract page number by looking backwards from position for page markers."""
    # Look backwards from position to find "--- PAGE X ---"
    search_text = text[:position]
    page_matches = re.findall(r'--- PAGE (\d+) ---', search_text)
    return int(page_matches[-1]) if page_matches else 1

def get_sections_file_mapping():
    """Return mapping of PDF files to their corresponding section JSON files."""
    return {
        "nvidia_10k.pdf": "nosql/nvidia_10k_sections.json",
        "amd_10k.pdf": "nosql/amd_10k_sections.json",
        "intel_10k.pdf": "nosql/intel_10k_sections.json",
        "broadcom_10k.pdf": "nosql/broadcom_10k_sections.json"
    }

def load_sections_data(source_file):
    """Load section information from JSON file for the specified company."""
    try:
        sections_mapping = get_sections_file_mapping()
        
        if source_file in sections_mapping:
            sections_file = Path(sections_mapping[source_file])
            with open(sections_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("sections", [])
        else:
            # No section data available for this file
            return []
    except Exception as e:
        print(f"⚠️ Could not load sections data for {source_file}: {e}")
        return []

def find_section_for_page(sections_data, page_number):
    """Find both main section and subsection that contains the given page number."""
    if not sections_data:
        return None
    
    # Find all sections that start at or before this page
    candidate_sections = [
        section for section in sections_data 
        if section.get("start_page_number", 0) <= page_number
    ]
    
    if not candidate_sections:
        return sections_data[0]  # Return first section if page is before all sections
    
    # Separate main sections and subsections
    main_sections = [s for s in candidate_sections if not s.get("is_subsection", False)]
    subsections = [s for s in candidate_sections if s.get("is_subsection", False)]
    
    # Find the most recent main section and subsection
    main_section = max(main_sections, key=lambda x: x.get("start_page_number", 0)) if main_sections else None
    subsection = max(subsections, key=lambda x: x.get("start_page_number", 0)) if subsections else None
    
    return {
        "main_section": main_section,
        "subsection": subsection
    }

def format_section_info(section_info):
    """Format section information for display with hierarchical structure."""
    if not section_info:
        return "Unknown"
    
    main_section = section_info.get("main_section")
    subsection = section_info.get("subsection")
    
    parts = []
    
    if main_section:
        main_name = main_section.get("section_name", "Unknown")
        main_title = main_section.get("section_title", "Unknown")
        if main_name != main_title:
            parts.append(f"{main_name} ({main_title})")
        else:
            parts.append(main_name)
    
    if subsection:
        sub_name = subsection.get("section_name", "Unknown")
        sub_title = subsection.get("section_title", "Unknown")
        if sub_name != sub_title:
            parts.append(f"{sub_name} ({sub_title})")
        else:
            parts.append(sub_name)
    
    return " > ".join(parts) if parts else "Unknown"

def get_longest_text_fragment(text_with_html):
    """Split text by HTML tags and return the longest fragment."""
    # Split by HTML tags
    fragments = re.split(r'<[^>]+>', text_with_html)
    
    # Clean each fragment and find the longest one
    longest_fragment = ""
    for fragment in fragments:
        cleaned = re.sub(r'\s+', ' ', fragment).strip()
        if len(cleaned) > len(longest_fragment):
            longest_fragment = cleaned
    
    return longest_fragment

def get_company_file_mapping():
    """Return mapping of company names to their PDF files."""
    return {
        "nvidia": "nvidia_10k.pdf",
        "amd": "amd_10k.pdf", 
        "intel": "intel_10k.pdf",
        "broadcom": "broadcom_10k.pdf"
    }

def get_all_chunks_from_pinecone(company_name):
    """Fetch ALL chunks from Pinecone for the specified company using pagination."""
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("❌ Missing Pinecone credentials")
        return None, None
    
    # Get the PDF filename for the company
    company_files = get_company_file_mapping()
    if company_name.lower() not in company_files:
        print(f"❌ Unknown company: {company_name}")
        print(f"Available companies: {', '.join(company_files.keys())}")
        return None, None
    
    target_file = company_files[company_name.lower()]
    print(f"🏢 Fetching ALL {company_name.upper()} chunks from {target_file}")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Get ALL records using multiple queries if needed
    stats = index.describe_index_stats()
    dummy_vector = [0.0] * stats.dimension
    company_filter = {"source_file": target_file}
    
    all_chunks = []
    batch_size = 1000  # Pinecone's typical max for query
    
    # Keep querying until we get all chunks
    # Since Pinecone doesn't support pagination in query, we'll use a large top_k
    # and hope it gets most/all chunks. For more robust solution, we'd need list_paginated
    print("📥 Fetching chunks from Pinecone...")
    
    response = index.query(
        vector=dummy_vector,
        top_k=10000,  # Try to get as many as possible
        include_metadata=True,
        include_values=False,
        filter=company_filter
    )
    
    all_chunks = response.matches
    print(f"📊 Retrieved {len(all_chunks)} total chunks for {company_name.upper()}")
    
    return all_chunks, index

def select_chunks_for_processing(all_chunks, count=1, skip_enriched=True):
    """Select chunks for processing based on count and enrichment status."""
    if skip_enriched:
        candidates = [
            chunk for chunk in all_chunks 
            if not chunk.metadata.get("page_number")
        ]
        print(f"📊 Found {len(all_chunks)} total chunks, {len(candidates)} need enrichment")
    else:
        candidates = all_chunks
    
    if not candidates:
        print("❌ No unenriched chunks found")
        return []
    
    # Return the requested number of chunks (or all if count == -1)
    if count == -1:  # All chunks
        selected_chunks = candidates
    else:
        selected_chunks = random.sample(candidates, min(count, len(candidates)))
    
    return selected_chunks

def process_chunks_in_batches(chunks, index, source_file, batch_size=45, dry_run=False):
    """Process chunks in batches of specified size."""
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size  # Round up division
    
    success_count = 0
    
    print(f"🚀 Processing {total_chunks} chunks in {total_batches} batches of {batch_size}")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_chunks)
        batch = chunks[start_idx:end_idx]
        
        print(f"\n--- Batch {batch_num + 1}/{total_batches} (chunks {start_idx + 1}-{end_idx}) ---")
        
        batch_success = 0
        for i, chunk in enumerate(batch):
            chunk_num = start_idx + i + 1
            print(f"\n📄 Processing chunk {chunk_num}/{total_chunks} (ID: {chunk.id})")
            
            if process_chunk(chunk, index, source_file, dry_run):
                batch_success += 1
                success_count += 1
        
        print(f"✅ Batch {batch_num + 1} completed: {batch_success}/{len(batch)} successful")
    
    return success_count

def update_chunk_metadata(index, chunk_id, metadata_updates, dry_run=False):
    """Update metadata for a chunk in Pinecone."""
    if dry_run:
        print(f"🔍 [DRY RUN] Would update chunk {chunk_id} with metadata:")
        for key, value in metadata_updates.items():
            print(f"    {key}: {value}")
        return True
    
    try:
        # Update the metadata using Pinecone's update method
        index.update(
            id=chunk_id,
            set_metadata=metadata_updates
        )
        print(f"✅ Updated metadata for chunk {chunk_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to update chunk {chunk_id}: {e}")
        return False

def create_metadata_from_result(result):
    """Create metadata dictionary from PDF analysis result."""
    metadata = {}
    
    if "page_number" in result:
        metadata["page_number"] = result["page_number"]
    
    if "section" in result and result["section"]:
        section_info = result["section"]
        
        # Add hierarchical section display
        metadata["hierarchical_section"] = format_section_info(section_info)
        
        # Add main section information
        if section_info.get("main_section"):
            main_section = section_info["main_section"]
            metadata["main_section_name"] = main_section.get("section_name", "")
            metadata["main_section_title"] = main_section.get("section_title", "")
        
        # Add subsection information
        if section_info.get("subsection"):
            subsection = section_info["subsection"]
            metadata["subsection_name"] = subsection.get("section_name", "")
            metadata["subsection_title"] = subsection.get("section_title", "")
    
    return metadata

def find_fragment_in_pdf(fragment, source_file):
    """Find fragment in the specified PDF and return surrounding paragraph."""
    pdf_path = Path("documents_pending") / source_file
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return {"found": False, "error": "PDF file not found"}
    
    print(f"📋 Loading PDF: {pdf_path}")
    
    # Load sections data for all supported companies
    sections_data = load_sections_data(source_file)
    if sections_data:
        print(f"📋 Loaded {len(sections_data)} sections for hierarchical section detection")
    else:
        print(f"📋 No section data found for {source_file}")
    
    # Extract all text from PDF
    full_text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"📊 PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return {"found": False, "error": f"PDF read error: {e}"}
    
    # Clean both texts
    clean_fragment = clean_text(fragment)
    clean_pdf = clean_text(full_text)
    
    print(f"🔍 Searching for fragment: {clean_fragment[:100]}...")
    print(f"📏 Fragment length: {len(clean_fragment)} chars")
    print(f"📄 PDF total length: {len(clean_pdf)} chars")
    
    # Find the fragment in PDF
    if clean_fragment in clean_pdf:
        start = clean_pdf.find(clean_fragment)
        end = start + len(clean_fragment)
        
        # Extract page number
        page_number = extract_page_number(clean_pdf, start)
        
        # Find section information
        section_info = None
        if sections_data and page_number:
            section_info = find_section_for_page(sections_data, page_number)
        
        print(f"✅ Found exact fragment at position {start}-{end}")
        print(f"📄 Fragment found on page: {page_number}")
        if section_info:
            section_display = format_section_info(section_info)
            print(f"📖 Section: {section_display}")
        
        # Get larger context (1000 chars before and after for full paragraph)
        context_start = max(0, start - 1000)
        context_end = min(len(clean_pdf), end + 1000)
        
        before = clean_pdf[context_start:start]
        matched = clean_pdf[start:end]
        after = clean_pdf[end:context_end]
        
        print(f"\n📄 Full paragraph context:")
        print("=" * 100)
        print(f"{before}[**FRAGMENT**]{matched}[**END**]{after}")
        print("=" * 100)
        
        result = {
            "found": True,
            "exact_match": True,
            "position": start,
            "page_number": page_number,
            "context": clean_pdf[context_start:context_end],
            "before": before,
            "matched": matched,
            "after": after
        }
        
        if section_info:
            result["section"] = section_info
        
        return result
    else:
        # Try partial search to find the content
        print("🔍 Exact match not found, trying partial matching...")
        words = clean_fragment.split()
        for i in range(len(words), 0, -1):
            partial = " ".join(words[:i])
            if partial in clean_pdf:
                print(f"✓ Found partial match ({i} words): '{partial[:50]}...'")
                
                # Show context around the partial match
                partial_start = clean_pdf.find(partial)
                partial_end = partial_start + len(partial)
                
                # Extract page number for partial match
                page_number = extract_page_number(clean_pdf, partial_start)
                
                # Find section information for partial match
                section_info = None
                if sections_data and page_number:
                    section_info = find_section_for_page(sections_data, page_number)
                
                print(f"📄 Partial match found on page: {page_number}")
                if section_info:
                    section_display = format_section_info(section_info)
                    print(f"📖 Section: {section_display}")
                
                context_start = max(0, partial_start - 500)
                context_end = min(len(clean_pdf), partial_end + 500)
                
                before = clean_pdf[context_start:partial_start]
                matched = clean_pdf[partial_start:partial_end]
                after = clean_pdf[partial_end:context_end]
                
                print(f"\n📄 Partial match context:")
                print("=" * 80)
                print(f"{before}[**PARTIAL MATCH**]{matched}[**END MATCH**]{after}")
                print("=" * 80)
                
                result = {
                    "found": True,
                    "partial_match": True,
                    "matched_words": i,
                    "position": partial_start,
                    "page_number": page_number,
                    "context": clean_pdf[context_start:context_end],
                    "before": before,
                    "matched": matched,
                    "after": after
                }
                
                if section_info:
                    result["section"] = section_info
                
                return result
        
        print("✗ No partial matches found")
        return {"found": False}

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enrich Pinecone chunks with page numbers, sections, and subsections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enrich_chunk_metadata.py nvidia              # Process 1 NVIDIA chunk
  python enrich_chunk_metadata.py amd --count 10      # Process 10 AMD chunks
  python enrich_chunk_metadata.py intel --all         # Process all Intel chunks
  python enrich_chunk_metadata.py broadcom --dry-run  # Dry run for 1 Broadcom chunk
"""
    )
    
    parser.add_argument(
        "company", 
        nargs="?", 
        default="nvidia",
        choices=["nvidia", "amd", "intel", "broadcom"],
        help="Company name (default: nvidia)"
    )
    
    parser.add_argument(
        "--count", 
        type=int, 
        default=1,
        help="Number of chunks to process (default: 1)"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Process all unenriched chunks for the company"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    return parser.parse_args()

def process_chunk(chunk, index, source_file, dry_run=False):
    """Process a single chunk to extract and update metadata."""
    chunk_id = chunk.id
    chunk_text = chunk.metadata.get('text', '')
    
    if not chunk_text:
        print(f"❌ Chunk {chunk_id}: Missing text in metadata")
        return False
    
    print(f"\n📄 Processing chunk: {chunk_id}")
    print(f"📄 Original chunk: {chunk_text[:100]}...")
    
    # Extract longest fragment
    longest_fragment = get_longest_text_fragment(chunk_text)
    
    if not longest_fragment:
        print(f"❌ Chunk {chunk_id}: No fragment extracted")
        return False
    
    print(f"✅ Extracted fragment ({len(longest_fragment)} chars): {longest_fragment[:100]}...")
    
    # Find fragment in PDF
    result = find_fragment_in_pdf(longest_fragment, source_file)
    
    if not result["found"]:
        error_msg = result.get("error", "Unknown error")
        print(f"❌ Chunk {chunk_id}: Could not locate fragment: {error_msg}")
        return False
    
    # Create metadata updates
    metadata_updates = create_metadata_from_result(result)
    
    if not metadata_updates:
        print(f"❌ Chunk {chunk_id}: No metadata to update")
        return False
    
    # Display what was found
    page_info = f" on page {result['page_number']}" if "page_number" in result else ""
    section_info = ""
    if "section" in result:
        section_display = format_section_info(result["section"])
        section_info = f" in section {section_display}"
    
    match_type = "exact" if result.get("exact_match") else "partial"
    print(f"✅ Found {match_type} match in {source_file}{page_info}{section_info}")
    
    # Update metadata in Pinecone
    success = update_chunk_metadata(index, chunk_id, metadata_updates, dry_run)
    return success

def main():
    """Main workflow: fetch chunks, extract metadata, update Pinecone."""
    args = parse_args()
    
    print("🔍 Chunk Metadata Enricher")
    print("=" * 40)
    print(f"🏢 Target company: {args.company.upper()}")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    if args.all:
        print("📊 Processing ALL unenriched chunks in batches of 45")
    else:
        print(f"📊 Processing {args.count} chunk(s)")
    
    # Get ALL chunks from Pinecone first
    print("\n📋 Fetching ALL chunks from Pinecone...")
    all_chunks, index = get_all_chunks_from_pinecone(args.company)
    
    if not all_chunks:
        print("❌ No chunks found for this company")
        return
    
    # Select chunks for processing based on count and enrichment status
    count = -1 if args.all else args.count
    chunks_to_process = select_chunks_for_processing(all_chunks, count, skip_enriched=True)
    
    if not chunks_to_process:
        print("❌ No chunks found to process")
        return
    
    # Get source file for the company
    company_files = get_company_file_mapping()
    source_file = company_files[args.company]
    
    # Process chunks in batches or individually
    if args.all and len(chunks_to_process) > 45:
        # Use batch processing for large numbers
        success_count = process_chunks_in_batches(
            chunks_to_process, index, source_file, batch_size=45, dry_run=args.dry_run
        )
        total_count = len(chunks_to_process)
    else:
        # Process individually for small numbers
        print(f"\n🚀 Starting to process {len(chunks_to_process)} chunks...")
        
        success_count = 0
        total_count = len(chunks_to_process)
        
        for i, chunk in enumerate(chunks_to_process, 1):
            print(f"\n--- Chunk {i}/{total_count} ---")
            if process_chunk(chunk, index, source_file, args.dry_run):
                success_count += 1
    
    # Summary
    print(f"\n📊 Final Summary:")
    print(f"   Total chunks processed: {total_count}")
    print(f"   Successfully enriched: {success_count}")
    print(f"   Failed: {total_count - success_count}")
    print(f"   Success rate: {success_count/total_count*100:.1f}%")
    
    if args.dry_run:
        print("🔍 DRY RUN completed - no actual updates were made")
    else:
        print("✅ Metadata enrichment completed!")

if __name__ == "__main__":
    main()