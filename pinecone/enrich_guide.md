  Usage Guide

  Basic Commands:

  # 1. Process 1 NVIDIA chunk (default company, default count)
  python pinecone/enrich_chunk_metadata.py

  # 2. Process 1 chunk from a specific company
  python pinecone/enrich_chunk_metadata.py nvidia
  python pinecone/enrich_chunk_metadata.py amd
  python pinecone/enrich_chunk_metadata.py intel
  python pinecone/enrich_chunk_metadata.py broadcom

  # 3. Process multiple chunks
  python pinecone/enrich_chunk_metadata.py nvidia --count 5    # Process 5 NVIDIA chunks
  python pinecone/enrich_chunk_metadata.py amd --count 10      # Process 10 AMD chunks

  # 4. Process ALL chunks for a company (be careful!)
  python pinecone/enrich_chunk_metadata.py intel --all

  # 5. DRY RUN - See what would happen without making changes
  python pinecone/enrich_chunk_metadata.py nvidia --dry-run
  python pinecone/enrich_chunk_metadata.py amd --count 10 --dry-run
  python pinecone/enrich_chunk_metadata.py intel --all --dry-run

  # 6. Get help
  python pinecone/enrich_chunk_metadata.py --help