#!/usr/bin/env python3
"""
Simple script to check chunk enrichment status.
Shows total chunks, chunks with page_number metadata, and chunks without page_number.

Usage: python check_enrichment_simple.py [company_name]
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
    
    # Get chunks (Pinecone limit is usually 10,000)
    response = index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        include_values=False,
        filter=company_filter
    )
    
    return response.matches

def check_enrichment(company_name):
    """Check enrichment status for a company."""
    print(f"🔍 Checking {company_name.upper()} chunks...")
    
    chunks = get_all_chunks_for_company(company_name)
    
    if not chunks:
        print(f"❌ No chunks found for {company_name}")
        return None
    
    total_chunks = len(chunks)
    enriched_chunks = 0
    not_enriched_chunks = 0
    
    for chunk in chunks:
        metadata = chunk.metadata
        
        if "page_number" in metadata and metadata["page_number"] is not None:
            enriched_chunks += 1
        else:
            not_enriched_chunks += 1
    
    # Results
    enrichment_rate = (enriched_chunks / total_chunks * 100) if total_chunks > 0 else 0
    
    print(f"📊 {company_name.upper()} Results:")
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   With page_number: {enriched_chunks:,} ({enrichment_rate:.1f}%)")
    print(f"   Without page_number: {not_enriched_chunks:,} ({100-enrichment_rate:.1f}%)")
    
    return {
        "company": company_name,
        "total": total_chunks,
        "enriched": enriched_chunks,
        "not_enriched": not_enriched_chunks,
        "rate": enrichment_rate
    }

def main():
    """Main function."""
    import sys
    
    # Parse command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print("Usage: python check_enrichment_simple.py [company_name]")
            print("Companies: nvidia, amd, intel, broadcom, all (default: all)")
            print("Example: python check_enrichment_simple.py nvidia")
            return
        company = sys.argv[1].lower()
    else:
        company = "all"
    
    print("📊 Simple Enrichment Check")
    print("=" * 30)
    
    # Check companies
    if company == "all":
        companies = ["nvidia", "amd", "intel", "broadcom"]
        all_results = []
        
        for comp in companies:
            result = check_enrichment(comp)
            if result:
                all_results.append(result)
            print()  # Empty line between companies
        
        # Overall summary
        if len(all_results) > 1:
            total_all = sum(r["total"] for r in all_results)
            enriched_all = sum(r["enriched"] for r in all_results)
            not_enriched_all = sum(r["not_enriched"] for r in all_results)
            overall_rate = (enriched_all / total_all * 100) if total_all > 0 else 0
            
            print("📈 OVERALL SUMMARY:")
            print(f"   Total chunks: {total_all:,}")
            print(f"   With page_number: {enriched_all:,} ({overall_rate:.1f}%)")
            print(f"   Without page_number: {not_enriched_all:,} ({100-overall_rate:.1f}%)")
    
    else:
        if company not in ["nvidia", "amd", "intel", "broadcom"]:
            print(f"❌ Invalid company: {company}")
            print("Available: nvidia, amd, intel, broadcom, all")
            return
        
        check_enrichment(company)

if __name__ == "__main__":
    main()