#!/usr/bin/env python3
"""
Simple script to extract longest fragment from Pinecone chunk.
"""

import os
import re
import json
import random
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

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

def get_chunk_from_pinecone():
    """Fetch a random chunk from Pinecone."""
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("❌ Missing Pinecone credentials")
        return None
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Get multiple records and pick one randomly
    stats = index.describe_index_stats()
    dummy_vector = [0.0] * stats.dimension
    
    response = index.query(
        vector=dummy_vector,
        top_k=min(50, stats.total_vector_count),  # Get up to 50 records
        include_metadata=True,
        include_values=False
    )
    
    if response.matches:
        # Pick a random record from the results
        random_record = random.choice(response.matches)
        return random_record
    return None

print("🔍 Extract Fragment from Chunk")
print("=" * 35)

# Get chunk from Pinecone
print("📋 Fetching chunk from Pinecone...")
record = get_chunk_from_pinecone()

if record:
    print(f"✅ Got record: {record.id}")
    chunk_text = record.metadata.get('text', '')
    source_file = record.metadata.get('source_file', '')
    
    print(f"📁 Source: {source_file}")
    print(f"📄 Original chunk: {chunk_text}")
    
    # Extract longest fragment
    longest_fragment = get_longest_text_fragment(chunk_text)
    print(f"\n🔍 Longest fragment: {longest_fragment}")
    
    # Save to JSON
    result = {
        "record_id": record.id,
        "source_file": source_file,
        "original_chunk": chunk_text,
        "longest_fragment": longest_fragment,
        "fragment_length": len(longest_fragment)
    }
    
    with open("fragment_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved result to fragment_result.json")
    
else:
    print("❌ No record found")