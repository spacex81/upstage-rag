#!/usr/bin/env python3
"""
Simple script to check how many chunks have 'hierarchical_section' metadata.
Shows total chunks, chunks with hierarchical_section, and chunks without.

Usage: python check_hierarchical_sections.py [company_name]
Company names: nvidia, amd, intel, broadcom, all (default: all)
"""

import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

def get_company_file_mapping():
    """Return mapping of company names to their PDF files."""
    return {
        "nvidia": "nvidia_10k.pdf",
        "amd": "amd_10k.pdf", 
        "intel": "intel_10k.pdf",
        "broadcom": "broadcom_10k.pdf"
    }

def get_all_chunks_for_company(company_name):
    """Fetch all chunks for a specific company from Pinecone."""
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("❌ Missing Pinecone credentials")
        return []
    
    company_files = get_company_file_mapping()
    if company_name not in company_files:
        print(f"❌ Unknown company: {company_name}")
        return []
    
    target_file = company_files[company_name]
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Get all records for this company
    stats = index.describe_index_stats()
    dummy_vector = [0.0] * stats.dimension
    
    company_filter = {"source_file": target_file}
    
    # Get chunks (using large top_k to get all)
    response = index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        include_values=False,
        filter=company_filter
    )
    
    return response.matches

def check_hierarchical_sections(company_name):
    """Check hierarchical_section metadata status for a company."""
    print(f"🔍 Checking {company_name.upper()} chunks...")
    
    chunks = get_all_chunks_for_company(company_name)
    
    if not chunks:
        print(f"❌ No chunks found for {company_name}")
        return None
    
    total_chunks = len(chunks)
    with_hierarchical = 0
    without_hierarchical = 0
    
    for chunk in chunks:
        metadata = chunk.metadata
        
        if "hierarchical_section" in metadata and metadata["hierarchical_section"]:
            with_hierarchical += 1
        else:
            without_hierarchical += 1
    
    # Results
    section_rate = (with_hierarchical / total_chunks * 100) if total_chunks > 0 else 0
    
    print(f"📊 {company_name.upper()} Results:")
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   With hierarchical_section: {with_hierarchical:,} ({section_rate:.1f}%)")
    print(f"   Without hierarchical_section: {without_hierarchical:,} ({100-section_rate:.1f}%)")
    
    return {
        "company": company_name,
        "total": total_chunks,
        "with_hierarchical": with_hierarchical,
        "without_hierarchical": without_hierarchical,
        "rate": section_rate
    }

def main():
    """Main function."""
    import sys
    
    # Parse command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print("Usage: python check_hierarchical_sections.py [company_name]")
            print("Companies: nvidia, amd, intel, broadcom, all (default: all)")
            print("Example: python check_hierarchical_sections.py nvidia")
            return
        company = sys.argv[1].lower()
    else:
        company = "all"
    
    print("📊 Hierarchical Section Check")
    print("=" * 35)
    
    # Check companies
    if company == "all":
        companies = ["nvidia", "amd", "intel", "broadcom"]
        all_results = []
        
        for comp in companies:
            result = check_hierarchical_sections(comp)
            if result:
                all_results.append(result)
            print()  # Empty line between companies
        
        # Overall summary
        if len(all_results) > 1:
            total_all = sum(r["total"] for r in all_results)
            with_hierarchical_all = sum(r["with_hierarchical"] for r in all_results)
            without_hierarchical_all = sum(r["without_hierarchical"] for r in all_results)
            overall_rate = (with_hierarchical_all / total_all * 100) if total_all > 0 else 0
            
            print("📈 OVERALL SUMMARY:")
            print(f"   Total chunks: {total_all:,}")
            print(f"   With hierarchical_section: {with_hierarchical_all:,} ({overall_rate:.1f}%)")
            print(f"   Without hierarchical_section: {without_hierarchical_all:,} ({100-overall_rate:.1f}%)")
    
    else:
        if company not in ["nvidia", "amd", "intel", "broadcom"]:
            print(f"❌ Invalid company: {company}")
            print("Available: nvidia, amd, intel, broadcom, all")
            return
        
        check_hierarchical_sections(company)

if __name__ == "__main__":
    main()