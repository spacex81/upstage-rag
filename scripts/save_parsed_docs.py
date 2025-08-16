"""
Simple script to run UpstageDocumentParseLoader once and save results to local file.
This avoids repeated API calls during development/testing.
"""
import asyncio
import pickle
import os
from pathlib import Path
from langchain_upstage import UpstageDocumentParseLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def save_parsed_docs():
    # Load the first PDF file from documents folder
    docs_folder = Path("documents")
    pdf_files = list(docs_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in documents/ folder")
        return
    
    # Use the first PDF file found
    pdf_file = pdf_files[0]
    print(f"Parsing PDF: {pdf_file.name}")
    
    # Parse with UpstageDocumentParseLoader
    loader = UpstageDocumentParseLoader(
        str(pdf_file), 
        ocr="force",
        split="page"
    )
    documents = await loader.aload()
    
    # Add page numbers to each document
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "page_number": i + 1,
            "source_file": pdf_file.name
        })
    
    # Create cache directory
    Path("cache").mkdir(exist_ok=True)
    
    # Save parsed documents
    cache_file = f"cache/{pdf_file.stem}_parsed_docs.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(documents, f)
    
    print(f"✅ Saved {len(documents)} parsed documents to {cache_file}")
    print(f"📄 Document has {len(documents)} pages")

if __name__ == "__main__":
    asyncio.run(save_parsed_docs())