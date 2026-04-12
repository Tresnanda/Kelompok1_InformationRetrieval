import os
import re
import math
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import PyPDF2
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import json
from indexer import PDFCorpusIndexer
from vsm import HybridSearchEngine

# Main execution
if __name__ == "__main__":
    # Configuration
    CORPUS_PATH = "dataset"  # Using downloads directory
    CONTENT_INDEX_PATH = "content_index.pkl"
    TITLE_INDEX_PATH = "title_index.pkl"
    FILTER_SECTIONS = True  # Set to False to index everything
    MAX_DOCS = 342  # Limit to 100 documents as requested

    # Build or load hybrid index
    indexer = PDFCorpusIndexer(CORPUS_PATH)

    # Check if both indices exist
    if os.path.exists(CONTENT_INDEX_PATH) and os.path.exists(TITLE_INDEX_PATH):
        print("Loading existing hybrid indices...")
        indexer.load_index(CONTENT_INDEX_PATH, TITLE_INDEX_PATH)
    else:
        print("Building new hybrid indices...")
        indexer.build_index(filter_sections=FILTER_SECTIONS, max_docs=MAX_DOCS)
        indexer.save_index(CONTENT_INDEX_PATH, TITLE_INDEX_PATH)

    # Display sample terms from both indices
    content_terms = list(indexer.content_index.df.keys())
    title_terms = list(indexer.title_index.df.keys())
    print("Sample content terms:", content_terms[:30])
    print("Sample title terms:", title_terms[:30])

    # Create hybrid search engine
    hybrid_search = HybridSearchEngine(indexer.content_index, indexer.title_index)

    # Interactive search
    print("\n" + "="*60)
    print("Hybrid Indonesian Thesis Search Engine - Ready!")
    print("Combining Abstract-based (30%) and Title-based (70%) retrieval")
    print("="*60)

    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        print(f"\nSearching for: '{query}'")
        print("-" * 60)

        results = hybrid_search.search(query, top_k=10)

        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results:\n")
            for rank, (doc_id, final_score, title, filename, content_score, title_score) in enumerate(results, 1):
                print(f"{rank}. [Final Score: {final_score:.4f}]")
                print(f"   Title: {title}")
                print(f"   File: {filename}")
                print(f"   Abstract Score: {content_score:.4f} | Title Score: {title_score:.4f}")
                print()