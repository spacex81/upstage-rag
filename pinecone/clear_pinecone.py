#!/usr/bin/env python3
"""
Script to clear all data from Pinecone index.
This will delete all vectors and metadata from the configured Pinecone index.
"""

import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

def clear_pinecone_index():
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
        
        # Get index stats before deletion
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        print(f"📊 Current vectors in index: {total_vectors}")
        
        if total_vectors == 0:
            print("✅ Index is already empty!")
            return True
        
        # Confirm deletion
        confirm = input(f"\n⚠️  Are you sure you want to delete ALL {total_vectors} vectors from '{index_name}'? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Operation cancelled.")
            return False
        
        # Delete all vectors (delete_all is simpler than listing and deleting by IDs)
        print("🗑️  Deleting all vectors...")
        index.delete(delete_all=True)
        
        # Verify deletion
        print("🔍 Verifying deletion...")
        stats_after = index.describe_index_stats()
        remaining_vectors = stats_after.total_vector_count
        
        if remaining_vectors == 0:
            print("✅ Successfully cleared all data from Pinecone index!")
            return True
        else:
            print(f"⚠️  Warning: {remaining_vectors} vectors still remain in index")
            return False
            
    except Exception as e:
        print(f"❌ Error clearing Pinecone index: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧹 Pinecone Index Cleaner")
    print("=" * 40)
    success = clear_pinecone_index()
    
    if success:
        print("\n✅ Pinecone index cleared successfully!")
    else:
        print("\n❌ Failed to clear Pinecone index.")