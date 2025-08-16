#!/usr/bin/env python3
"""
Script to list Pinecone records with metadata filter.
This will show vectors that have source_file metadata containing a keyword (case insensitive).
"""

import os
import sys
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

def list_filtered_pinecone_records(keyword):
    # Load environment variables
    load_dotenv()
    
    # Get Pinecone credentials from environment
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("❌ Error: Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env file")
        return False
    
    try:
        # Initialize Pinecone client
        print(f"🔗 Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        
        # Get the index
        print(f"📋 Getting index: {index_name}")
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        print(f"📊 Total vectors in index: {total_vectors}")
        print(f"📊 Dimension: {stats.dimension}")
        print(f"📊 Index fullness: {stats.index_fullness}")
        
        if total_vectors == 0:
            print("📝 Index is empty - no records to display")
            return True
        
        # List records and filter in Python (since Pinecone doesn't support regex)
        print(f"\n🔍 Listing records with source_file containing '{keyword}' (case insensitive)...")
        print("=" * 80)
        
        # Query to get all records - using a dummy vector of zeros
        dummy_vector = [0.0] * stats.dimension
        query_response = index.query(
            vector=dummy_vector,
            top_k=min(1000, total_vectors),  # Get up to 1000 records to filter
            include_metadata=True,
            include_values=False
        )
        
        if not query_response.matches:
            print("📝 No records found in index")
            return True
        
        # Filter records in Python
        filtered_matches = []
        keyword_lower = keyword.lower()
        
        for match in query_response.matches:
            if match.metadata and "source_file" in match.metadata:
                source_file = str(match.metadata["source_file"]).lower()
                if keyword_lower in source_file:
                    filtered_matches.append(match)
        
        if not filtered_matches:
            print(f"📝 No records found with source_file containing '{keyword}'")
            return True
        
        # Display filtered records
        for i, match in enumerate(filtered_matches, 1):
            print(f"\n📄 Record {i}:")
            print(f"   ID: {match.id}")
            print(f"   Score: {match.score:.4f}")
            if match.metadata:
                print(f"   Metadata:")
                for key, value in match.metadata.items():
                    # Truncate long values for readability
                    display_value = str(value)
                    if len(display_value) > 100:
                        display_value = display_value[:100] + "..."
                    # Highlight source_file field
                    if key == "source_file":
                        print(f"     {key}: 🎯 {display_value}")
                    else:
                        print(f"     {key}: {display_value}")
            else:
                print(f"   Metadata: None")
        
        print(f"\n✅ Found {len(filtered_matches)} records with source_file containing '{keyword}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing filtered Pinecone records: {str(e)}")
        return False

if __name__ == "__main__":
    print("📋 Pinecone Index Filtered Lister")
    print("=" * 40)
    
    # Get keyword from command line argument or user input
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    else:
        keyword = input("Enter keyword to search in source_file metadata: ").strip()
        if not keyword:
            print("❌ No keyword provided. Exiting.")
            sys.exit(1)
    
    print(f"🔍 Searching for records with source_file containing: '{keyword}'")
    success = list_filtered_pinecone_records(keyword)
    
    if success:
        print("\n✅ Successfully listed filtered Pinecone records!")
    else:
        print("\n❌ Failed to list filtered Pinecone records.")