#!/usr/bin/env python3
"""
Script to fetch one random record from Pinecone vector database.
This will show a single vector with its metadata for testing purposes.
"""

import os
import random
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

def fetch_random_record():
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
        
        if total_vectors == 0:
            print("📝 Index is empty - no records to fetch")
            return True
        
        # Fetch a random record using dummy vector query
        print(f"\n🎲 Fetching one random record...")
        print("=" * 60)
        
        # Query to get random records - using a dummy vector of zeros
        dummy_vector = [0.0] * stats.dimension
        
        # Get more records than we need, then pick one randomly
        query_response = index.query(
            vector=dummy_vector,
            top_k=min(50, total_vectors),  # Get up to 50 records to choose from
            include_metadata=True,
            include_values=False
        )
        
        if not query_response.matches:
            print("📝 No records found in query")
            return True
        
        # Pick a random record from the results
        random_record = random.choice(query_response.matches)
        
        # Display the random record
        print(f"📄 Random Record:")
        print(f"   ID: {random_record.id}")
        print(f"   Score: {random_record.score:.4f}")
        print(f"   Content Preview: {random_record.id[:50]}...")
        
        if random_record.metadata:
            print(f"   Metadata:")
            for key, value in random_record.metadata.items():
                # Truncate very long values for readability
                display_value = str(value)
                if len(display_value) > 200:
                    display_value = display_value[:200] + "..."
                print(f"     {key}: {display_value}")
        else:
            print(f"   Metadata: None")
        
        print(f"\n✅ Successfully fetched random record from {total_vectors} total records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching random record: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎲 Pinecone Random Record Fetcher")
    print("=" * 40)
    success = fetch_random_record()
    
    if success:
        print("\n✅ Successfully fetched random record!")
    else:
        print("\n❌ Failed to fetch random record.")