from typing import List, Dict, Tuple, Set
from compressor import GapEncoder, VBEncoder
from collections import defaultdict, Counter
import math

class InvertedIndex:
    """Standard inverted index with compression"""

    def __init__(self):
        # Hash-based term dictionary: term -> compressed posting list
        self.index: Dict[str, bytes] = {}
        # Term -> document frequency
        self.df: Dict[str, int] = {}
        # Document metadata
        self.doc_metadata: Dict[int, Dict] = {}
        # Document lengths for normalization
        self.doc_lengths: Dict[int, float] = {}
        # Total number of documents
        self.num_docs = 0
        # Average document length
        self.avg_doc_length = 0
        self.doc_vectors: Dict[int, Dict[str, float]] = {}   # doc_id -> {term: tfidf}
        self.doc_norms: Dict[int, float] = {} 
    
    def add_posting(self, term: str, doc_id: int, frequency: int):
        """Add a posting to the index (before compression)"""
        if term not in self.index:
            self.index[term] = []
        self.index[term].append((doc_id, frequency))
    
    def compress_index(self):
        """Compress the index using VB + Gap encoding"""
        compressed_index = {}
        
        for term, postings in self.index.items():
            # Sort by docID
            postings = sorted(postings, key=lambda x: x[0])
            
            # Separate docIDs and frequencies
            doc_ids = [p[0] for p in postings]
            frequencies = [p[1] for p in postings]
        
            # Apply gap encoding to docIDs
            gaps = GapEncoder.encode(doc_ids)
            
            # Apply VB encoding
            encoded_docids = VBEncoder.encode_list(gaps)
            encoded_freqs = VBEncoder.encode_list(frequencies)
            
            # Store compressed data
            compressed_index[term] = {
                'docids': encoded_docids,
                'freqs': encoded_freqs,
                'df': len(postings)
            }
            
            self.df[term] = len(postings)
        
        self.index = compressed_index
    
    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """Retrieve and decompress postings for a term"""
        if term not in self.index:
            return []
        
        compressed_data = self.index[term]
        
        # Decode
        gaps = VBEncoder.decode(compressed_data['docids'])
        frequencies = VBEncoder.decode(compressed_data['freqs'])
        
        # Convert gaps back to docIDs
        doc_ids = GapEncoder.decode(gaps)
        
        return list(zip(doc_ids, frequencies))
    
    def build_tfidf_doc_vectors(self):
        """
        Build TF-IDF vectors for all documents and compute norms.
        This should be called BEFORE compressing the index (because self.index
        currently holds term -> postings lists).
        """
        # Reset
        self.doc_vectors = defaultdict(dict)
        self.doc_norms = {}

        # self.index currently: term -> list of (doc_id, freq)
        N = self.num_docs if self.num_docs > 0 else 1

        for term, postings in list(self.index.items()):
            df = len(postings)
            if df == 0:
                continue
            # idf: use log10(N / df) (consistent with your TFIDF class)
            idf = math.log10(N / df) if df > 0 else 0.0

            for doc_id, freq in postings:
                doc_len = self.doc_lengths.get(doc_id, 1)
                tf = freq / doc_len if doc_len > 0 else 0.0
                weight = tf * idf
                if weight != 0.0:
                    self.doc_vectors[doc_id][term] = weight

        # compute norms
        for doc_id, vec in self.doc_vectors.items():
            self.doc_norms[doc_id] = math.sqrt(sum(w * w for w in vec.values()))