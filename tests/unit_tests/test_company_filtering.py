#!/usr/bin/env python3
"""
Test script for company-aware retrieval functionality.
"""

import sys
sys.path.append('src')

from retrieval_graph.graph import detect_companies

def test_company_detection():
    """Test the company detection function with various queries."""
    
    test_cases = [
        # Single company queries
        ("What is NVIDIA's revenue?", ["nvidia_10k.pdf"]),
        ("Tell me about AMD's strategy", ["amd_10k.pdf"]),
        ("Intel earnings report", ["intel_10k.pdf"]),
        ("Broadcom financial performance", ["broadcom_10k.pdf"]),
        
        # Ticker symbol queries
        ("NVDA stock analysis", ["nvidia_10k.pdf"]),
        ("INTC vs AMD comparison", ["intel_10k.pdf", "amd_10k.pdf"]),
        ("AVGO quarterly results", ["broadcom_10k.pdf"]),
        
        # Multi-company queries
        ("Compare NVIDIA and AMD", ["nvidia_10k.pdf", "amd_10k.pdf"]),
        ("Intel vs Broadcom vs NVIDIA", ["intel_10k.pdf", "broadcom_10k.pdf", "nvidia_10k.pdf"]),
        
        # Industry-wide queries (no specific companies)
        ("Semiconductor industry trends", []),
        ("AI chip market analysis", []),
        ("What are the main risks in chip industry?", []),
        
        # Edge cases
        ("nvidia vs amd vs intel comparison", ["nvidia_10k.pdf", "amd_10k.pdf", "intel_10k.pdf"]),
        ("How is NVIDIA performing compared to Intel?", ["nvidia_10k.pdf", "intel_10k.pdf"]),
    ]
    
    print("🧪 Testing Company Detection Function")
    print("=" * 50)
    
    all_passed = True
    
    for query, expected in test_cases:
        result = detect_companies(query)
        passed = set(result) == set(expected)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} Query: '{query}'")
        print(f"     Expected: {expected}")
        print(f"     Got:      {result}")
        print()
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_company_detection()