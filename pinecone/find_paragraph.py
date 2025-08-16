#!/usr/bin/env python3
"""
Script to find paragraph in PDF based on fragment from JSON.
"""

import PyPDF2
import json
import re
from pathlib import Path

def clean_text(text):
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()

def find_fragment_in_pdf(fragment):
    """Find fragment in PDF and return surrounding paragraph."""
    pdf_path = Path("documents/nvidia_10k.pdf")
    
    print(f"📋 Loading PDF: {pdf_path}")
    
    # Extract all text from PDF
    full_text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        print(f"📊 PDF has {len(pdf_reader.pages)} pages")
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
    
    # Clean both texts
    clean_fragment = clean_text(fragment)
    clean_pdf = clean_text(full_text)
    
    print(f"🔍 Searching for fragment: {clean_fragment}")
    print(f"📏 Fragment length: {len(clean_fragment)} chars")
    print(f"📄 PDF total length: {len(clean_pdf)} chars")
    
    # Find the fragment in PDF
    if clean_fragment in clean_pdf:
        start = clean_pdf.find(clean_fragment)
        end = start + len(clean_fragment)
        
        print(f"✅ Found fragment at position {start}-{end}")
        
        # Get larger context (1000 chars before and after for full paragraph)
        context_start = max(0, start - 1000)
        context_end = min(len(clean_pdf), end + 1000)
        
        before = clean_pdf[context_start:start]
        matched = clean_pdf[start:end]
        after = clean_pdf[end:context_end]
        
        print(f"\n📄 Full paragraph context:")
        print("=" * 100)
        print(f"{before}[**FRAGMENT**]{matched}[**END**]{after}")
        print("=" * 100)
        
        return {
            "found": True,
            "position": start,
            "context": clean_pdf[context_start:context_end],
            "before": before,
            "matched": matched,
            "after": after
        }
    else:
        # Try partial search to find the content
        words = clean_fragment.split()
        for i in range(len(words), 0, -1):
            partial = " ".join(words[:i])
            if partial in clean_pdf:
                print(f"✓ Found partial match ({i} words): '{partial}'")
                
                # Show context around the partial match
                partial_start = clean_pdf.find(partial)
                partial_end = partial_start + len(partial)
                
                context_start = max(0, partial_start - 500)
                context_end = min(len(clean_pdf), partial_end + 500)
                
                before = clean_pdf[context_start:partial_start]
                matched = clean_pdf[partial_start:partial_end]
                after = clean_pdf[partial_end:context_end]
                
                print(f"\n📄 Partial match context:")
                print("=" * 80)
                print(f"{before}[**PARTIAL MATCH**]{matched}[**END MATCH**]{after}")
                print("=" * 80)
                
                return {
                    "found": True,
                    "partial_match": True,
                    "matched_words": i,
                    "position": partial_start,
                    "context": clean_pdf[context_start:context_end],
                    "before": before,
                    "matched": matched,
                    "after": after
                }
        
        print("✗ No partial matches found")
        return {"found": False}

print("🔍 Find Paragraph in PDF")
print("=" * 30)

# Load fragment from JSON
print("📋 Loading fragment from JSON...")
try:
    with open("fragment_result.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    fragment = data["longest_fragment"]
    record_id = data["record_id"]
    source_file = data["source_file"]
    
    print(f"✅ Loaded fragment from record: {record_id}")
    print(f"📁 Source: {source_file}")
    print(f"📄 Fragment: {fragment}")
    
    # Find in PDF
    result = find_fragment_in_pdf(fragment)
    
    if result["found"]:
        if result.get("partial_match"):
            print(f"\n✅ Successfully found paragraph using partial match ({result['matched_words']} words)!")
        else:
            print("\n✅ Successfully found paragraph with exact match!")
    else:
        print("\n❌ Could not locate paragraph")
        
except FileNotFoundError:
    print("❌ fragment_result.json not found. Run extract_fragment.py first.")
except Exception as e:
    print(f"❌ Error: {e}")