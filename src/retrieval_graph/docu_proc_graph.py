"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Optional, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState


# Load PDF Node Start
async def load_pdf_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, Sequence[Document]]:
    """Load PDF documents using UpstageDocumentParseLoader API with pickle caching.
    
    This function will:
    - Check for cached parsed documents first
    - Load PDF files from documents folder if no cache
    - Parse documents using Upstage API with auto format
    - Cache parsed documents to pickle file
    - Return documents for further processing
    
    Args:
        state (IndexState): Current state (may contain file paths or be empty)
        config (Optional[RunnableConfig]): Configuration for loading process
        
    Returns:
        dict[str, Sequence[Document]]: Updated state with loaded documents
    """
    from langchain_upstage import UpstageDocumentParseLoader
    from pathlib import Path
    import pickle
    
    # Load the first PDF file from documents folder
    docs_folder = Path("documents")
    pdf_files = list(docs_folder.glob("*.pdf"))
    
    if not pdf_files:
        return {"docs": []}
    
    # Use the first PDF file found
    pdf_file = pdf_files[0]
    
    # Create cache directory and file path
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{pdf_file.stem}_parsed_docs.pkl"
    
    # Check if cached version exists
    if cache_file.exists():
        print(f"📋 Loading cached documents: {cache_file.name}")
        try:
            with open(cache_file, "rb") as f:
                documents = pickle.load(f)
            print(f"✅ Loaded {len(documents)} documents from cache")
            return {"docs": documents}
        except Exception as e:
            print(f"⚠️ Cache read failed ({e}), falling back to API parsing")
    
    print(f"📋 Parsing PDF with Upstage API: {pdf_file.name}")
    
    # Use UpstageDocumentParseLoader with auto format (no ocr or split params)
    loader = UpstageDocumentParseLoader(str(pdf_file))
    documents = await loader.aload()
    
    # Add source file metadata
    for doc in documents:
        doc.metadata.update({
            "source_file": pdf_file.name
        })
    
    print(f"✅ Loaded {len(documents)} documents from API")
    
    # Cache the parsed documents
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(documents, f)
        print(f"💾 Cached documents to {cache_file.name}")
    except Exception as e:
        print(f"⚠️ Failed to cache documents: {e}")
    
    return {"docs": documents}
# Load PDF Node End


# Split Documents Node Start  
async def split_documents(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, Sequence[Document]]:
    """Split large documents using semantic chunking for better retrieval.
    
    This function will:
    - Take documents from state
    - Pre-split large documents to avoid token limits
    - Use semantic chunking to split based on content meaning
    - Split documents into coherent semantic chunks
    - Preserve metadata and add chunk information
    
    Args:
        state (IndexState): Current state containing documents to split
        config (Optional[RunnableConfig]): Configuration for splitting
        
    Returns:
        dict[str, Sequence[Document]]: Updated state with split documents
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_upstage import UpstageEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Initialize semantic chunker with Upstage embeddings
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    semantic_splitter = SemanticChunker(embeddings)
    
    # Pre-splitter to handle very large documents that exceed token limits
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # Stay well below 4000 token limit
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split all documents using semantic chunking
    split_docs = []
    for doc in state.docs:
        # First, pre-split if document is too large
        if len(doc.page_content) > 3000:
            pre_chunks = pre_splitter.split_documents([doc])
            print(f"📄 Pre-split large document into {len(pre_chunks)} parts")
        else:
            pre_chunks = [doc]
        
        # Then apply semantic chunking to each pre-chunk
        for pre_chunk in pre_chunks:
            try:
                chunks = semantic_splitter.split_documents([pre_chunk])
                
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_type": "semantic"
                    })
                split_docs.extend(chunks)
            except Exception as e:
                print(f"⚠️ Semantic chunking failed for a document part: {e}")
                # Fallback: use the pre-chunk as-is
                pre_chunk.metadata.update({
                    "chunk_id": 0,
                    "total_chunks": 1,
                    "chunk_type": "fallback"
                })
                split_docs.append(pre_chunk)
    
    print(f"✅ Split {len(state.docs)} documents into {len(split_docs)} semantic chunks")
    
    return {"docs": split_docs}
# Split Documents Node End


# Enrich Metadata Node Start
async def enrich_metadata(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, Sequence[Document]]:
    """Enrich document metadata with additional information.
    
    This function will:
    - Add source file information
    - Add processing timestamps
    - Add document type and size information
    - Prepare documents for indexing
    
    Args:
        state (IndexState): Current state containing documents to enrich
        config (Optional[RunnableConfig]): Configuration for metadata enrichment
        
    Returns:
        dict[str, Sequence[Document]]: Updated state with enriched documents
    """
    from datetime import datetime
    
    # Enrich each document with additional metadata
    enriched_docs = []
    for doc in state.docs:
        # Clean metadata - only keep simple types (string, number, boolean)
        clean_metadata = {}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                clean_metadata[key] = value
            # Skip complex objects like coordinates, nested dicts, etc.
        
        # Add processing timestamp and document stats
        clean_metadata.update({
            "processed_at": datetime.now().isoformat(),
            "doc_length": len(doc.page_content),
            "doc_type": "pdf_chunk"
        })
        
        doc.metadata = clean_metadata
        enriched_docs.append(doc)
    
    print(f"✅ Enriched metadata for {len(enriched_docs)} documents")
    
    return {"docs": enriched_docs}  
# Enrich Metadata Node End




# Index Node Start
def ensure_docs_have_user_id(
    docs: Sequence[Document], config: RunnableConfig
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        docs (Sequence[Document]): A sequence of Document objects to process.
        config (RunnableConfig): A configuration object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    user_id = config["configurable"]["user_id"]
    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in docs
    ]

async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Index documents in the vector store using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")
    
    with retrieval.make_retriever(config) as retriever:
        stamped_docs = ensure_docs_have_user_id(state.docs, config)
        await retriever.aadd_documents(stamped_docs)
    
    print(f"✅ Successfully indexed {len(state.docs)} documents to vector store")
    
    return {"docs": "delete"}
# Index Node End



builder = StateGraph(IndexState, config_schema=IndexConfiguration)

# Add all nodes to the graph
builder.add_node("load_pdf_docs", load_pdf_docs)
builder.add_node("split_documents", split_documents)
builder.add_node("enrich_metadata", enrich_metadata)
builder.add_node("index_docs", index_docs)

# Create the simplified document processing pipeline flow
builder.add_edge("__start__", "load_pdf_docs")
builder.add_edge("load_pdf_docs", "split_documents")
builder.add_edge("split_documents", "enrich_metadata")
builder.add_edge("enrich_metadata", "index_docs")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "DocumentProcessingGraph"
