#!/usr/bin/env python3
"""
Script to list all records in Pinecone index.
This will show vectors, metadata, and stats from the configured Pinecone index.
"""

import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

def list_pinecone_records():
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
        
        # List records by querying with dummy vector
        print(f"\n🔍 Listing records (showing first 20)...")
        print("=" * 60)
        
        # Query to get some records - using a dummy vector of zeros
        dummy_vector = [0.0] * stats.dimension
        query_response = index.query(
            vector=dummy_vector,
            top_k=min(20, total_vectors),  # Get up to 20 records
            include_metadata=True,
            include_values=False
        )
        
        if not query_response.matches:
            print("📝 No matches found in query")
            return True
        
        # Display records
        for i, match in enumerate(query_response.matches, 1):
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
                    print(f"     {key}: {display_value}")
            else:
                print(f"   Metadata: None")
        
        print(f"\n✅ Listed {len(query_response.matches)} records")
        if total_vectors > 20:
            print(f"   (Showing first 20 of {total_vectors} total records)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing Pinecone records: {str(e)}")
        return False

if __name__ == "__main__":
    print("📋 Pinecone Index Lister")
    print("=" * 40)
    success = list_pinecone_records()
    
    if success:
        print("\n✅ Successfully listed Pinecone records!")
    else:
        print("\n❌ Failed to list Pinecone records.")