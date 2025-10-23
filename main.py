import os
import re
import math
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import PyPDF2
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class VBEncoder:
    """Variable Byte Encoding for integer compression"""
    
    @staticmethod
    def encode(number: int) -> bytes:
        """Encode a single integer using VB encoding"""
        bytes_list = []
        while True:
            bytes_list.insert(0, number % 128)
            if number < 128:
                break
            number //= 128
        bytes_list[-1] += 128  # Set continuation bit on last byte
        return bytes(bytes_list)
    
    @staticmethod
    def encode_list(numbers: List[int]) -> bytes:
        """Encode a list of integers"""
        return b''.join(VBEncoder.encode(n) for n in numbers)
    
    @staticmethod
    def decode(byte_stream: bytes) -> List[int]:
        """Decode VB encoded bytes back to integers"""
        numbers = []
        current = 0
        for byte in byte_stream:
            if byte < 128:
                current = 128 * current + byte
            else:
                current = 128 * current + (byte - 128)
                numbers.append(current)
                current = 0
        return numbers


class GapEncoder:
    """Gap encoding for docID sequences"""
    
    @staticmethod
    def encode(doc_ids: List[int]) -> List[int]:
        """Convert absolute docIDs to gaps"""
        if not doc_ids:
            return []
        gaps = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            gaps.append(doc_ids[i] - doc_ids[i-1])
        return gaps
    
    @staticmethod
    def decode(gaps: List[int]) -> List[int]:
        """Convert gaps back to absolute docIDs"""
        if not gaps:
            return []
        doc_ids = [gaps[0]]
        for i in range(1, len(gaps)):
            doc_ids.append(doc_ids[i-1] + gaps[i])
        return doc_ids


class IndonesianPreprocessor:
    """Text preprocessing for Indonesian documents"""
    
    def __init__(self):
        # Initialize Sastrawi stemmer and stopword remover
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        # Additional Indonesian stopwords
        self.custom_stopwords = {
            'yang', 'dan', 'di', 'dari', 'untuk', 'pada', 'dengan', 'adalah',
            'ini', 'itu', 'ke', 'dalam', 'atau', 'oleh', 'akan', 'telah',
            'dapat', 'juga', 'sebagai', 'tidak', 'ada', 'tersebut', 'sehingga'
        }
    
    def remove_noise(self, text: str) -> str:
        """Remove noise: URLs, emails, special characters, numbers"""
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers, keep Indonesian letters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess(self, text: str) -> List[str]:
        """Complete preprocessing pipeline"""
        # 1. Casefold (lowercase)
        text = text.lower()
        
        # 2. Noise removal
        text = self.remove_noise(text)
        
        # 3. Stopword removal using Sastrawi
        text = self.stopword_remover.remove(text)
        
        # 4. Tokenization
        tokens = text.split()
        
        # 5. Additional stopword filtering
        tokens = [t for t in tokens if t not in self.custom_stopwords]
        
        # 6. Filter short tokens (< 3 characters)
        tokens = [t for t in tokens if len(t) >= 3]
        
        # 7. Stemming using Sastrawi
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens


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



class TFIDFVectorSpaceModel:
    """Vector Space Model with TF-IDF weighting"""
    
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.preprocessor = IndonesianPreprocessor()
    
    def compute_tf(self, freq: int, doc_length: int) -> float:
        """Compute normalized term frequency"""
        if doc_length == 0:
            return 0
        # Normalized TF
        return freq / doc_length
    
    def compute_idf(self, term: str) -> float:
        """Compute inverse document frequency"""
        df = self.index.df.get(term, 0)
        if df == 0:
            return 0
        return math.log10(self.index.num_docs / df)
    
    def compute_tfidf(self, term: str, doc_id: int, freq: int) -> float:
        """Compute TF-IDF weight"""
        doc_length = self.index.doc_lengths.get(doc_id, 1)
        tf = self.compute_tf(freq, doc_length)
        idf = self.compute_idf(term)
        return tf * idf
    
    def get_document_vector(self, doc_id: int) -> Dict[str, float]:
        """Return the precomputed TF-IDF vector for doc_id (fast)."""
        return self.index.doc_vectors.get(doc_id, {})

    
    def get_query_vector(self, query_terms: List[str]) -> Dict[str, float]:
        """Get TF-IDF vector for a query"""
        term_freqs = Counter(query_terms)
        query_length = len(query_terms)
        
        vector = {}
        for term, freq in term_freqs.items():
            if term in self.index.df:
                tf = freq / query_length
                idf = self.compute_idf(term)
                vector[term] = tf * idf
        
        return vector
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two vectors"""
        # Compute dot product
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(w ** 2 for w in vec1.values()))
        mag2 = math.sqrt(sum(w ** 2 for w in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)
    
    def search(self, query: str, top_k: int = 10):
        query_terms = self.preprocessor.preprocess(query)
        print("Query terms:", query_terms)
        
        if not query_terms:
            return []

        query_vector = self.get_query_vector(query_terms)
        print("Query vector:", query_vector)

        candidate_docs = set()
        for term in query_terms:
            postings = self.index.get_postings(term)
            print(f"{term} postings:", postings)
            candidate_docs.update(doc_id for doc_id, _ in postings)

        print("Candidate docs:", candidate_docs)
        scores = []
        for doc_id in candidate_docs:
            doc_vector = self.get_document_vector(doc_id)
            print(f"Doc vector {doc_id}:", doc_vector)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            print(f"Sim({doc_id}) =", similarity)
            if similarity > 0:
                title = self.index.doc_metadata.get(doc_id, {}).get("title", "Unknown")
                scores.append((doc_id, similarity, title))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]



class PDFCorpusIndexer:
    """Index PDF corpus and build inverted index"""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.preprocessor = IndonesianPreprocessor()
        self.index = InvertedIndex()
        self.doc_id_counter = 0
    
    def extract_text_from_pdf(self, pdf_path: str, filter_sections: bool = True) -> str:
        """Extract text from PDF file with optional section filtering"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    
                    if filter_sections:
                        # Skip pages with these sections (case-insensitive)
                        skip_keywords = [
                            'daftar pustaka', 'references', 'bibliography',
                            'kata pengantar', 'foreword', 'preface',
                            'daftar isi', 'table of contents',
                            'daftar tabel', 'daftar gambar',
                            'lembar persetujuan', 'lembar pengesahan',
                            'pernyataan orisinalitas'
                        ]
                        
                        page_lower = page_text.lower()
                        
                        # Check if page starts with skip keywords (first 200 chars)
                        if any(keyword in page_lower[:200] for keyword in skip_keywords):
                            continue
                        
                        # Additional filtering: skip if page is mostly references
                        # (contains many years in brackets like [2018], [2019])
                        year_pattern_count = len(re.findall(r'\[\d{4}\]|\(\d{4}\)', page_text))
                        if year_pattern_count > 10:  # Likely a reference page
                            continue
                    
                    text += page_text + " "
                    
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

    
    def extract_title(self, text: str, filename: str) -> str:
        """Extract title from document (heuristic)"""
        # Try to get title from first non-empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Take first substantial line as title
            for line in lines[:10]:
                if len(line) > 10 and len(line) < 200:
                    return line
        return filename.replace('.pdf', '')
    
    def build_index(self, filter_sections: bool = True):
        """Build inverted index from PDF corpus
        
        Args:
            filter_sections: If True, skip common non-content sections
        """
        print("Starting indexing process...")
        print(f"Section filtering: {'ENABLED' if filter_sections else 'DISABLED'}")
        
        pdf_files = [f for f in os.listdir(self.corpus_path) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        
        # Temporary posting lists
        temp_index = defaultdict(list)
        
        for idx, filename in enumerate(pdf_files):
            print(f"Processing [{idx+1}/{total_files}]: {filename}")
            
            pdf_path = os.path.join(self.corpus_path, filename)
            text = self.extract_text_from_pdf(pdf_path, filter_sections=filter_sections)
            
            if not text.strip():
                continue
            
            # Extract title
            title = self.extract_title(text, filename)
            
            # Preprocess
            tokens = self.preprocessor.preprocess(text)
            
            if not tokens:
                continue
            
            # Store metadata
            self.index.doc_metadata[self.doc_id_counter] = {
                'filename': filename,
                'title': title,
                'path': pdf_path
            }
            
            # Compute term frequencies
            term_freqs = Counter(tokens)
            doc_length = len(tokens)
            self.index.doc_lengths[self.doc_id_counter] = doc_length
            
            # Add to temporary index
            for term, freq in term_freqs.items():
                temp_index[term].append((self.doc_id_counter, freq))
            
            self.doc_id_counter += 1
        
        # Transfer to main index
        # ... after populating temp_index and setting self.index.index = postings ...
        self.index.num_docs = self.doc_id_counter
        for term, postings in temp_index.items():
            self.index.index[term] = postings
            self.index.df[term] = len(postings)

        # Compute average document length
        if self.index.doc_lengths:
            self.index.avg_doc_length = sum(self.index.doc_lengths.values()) / len(self.index.doc_lengths)

        # Build TF-IDF document vectors BEFORE compression for fast search
        print("\nBuilding TF-IDF document vectors...")
        self.index.build_tfidf_doc_vectors()

        # Compress index (we can still keep compressed postings for storage/search decompression)
        print("\nCompressing index...")
        self.index.compress_index()

        
        print(f"\nIndexing complete!")
        print(f"Total documents: {self.index.num_docs}")
        print(f"Total terms: {len(self.index.index)}")
        print(f"Average document length: {self.index.avg_doc_length:.2f}")
    
    def save_index(self, index_path: str):
        """Save index to disk"""
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load index from disk"""
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
        print(f"Index loaded from {index_path}")


# Main execution
if __name__ == "__main__":
    # Configuration
    CORPUS_PATH = "dataset"  # Change this to your PDF folder
    INDEX_PATH = "thesis_index.pkl"
    FILTER_SECTIONS = True  # Set to False to index everything
    
    # Build or load index
    indexer = PDFCorpusIndexer(CORPUS_PATH)
    
    # Check if index exists
    if os.path.exists(INDEX_PATH):
        print("Loading existing index...")
        indexer.load_index(INDEX_PATH)
    else:
        print("Building new index...")
        indexer.build_index(filter_sections=FILTER_SECTIONS)
        indexer.save_index(INDEX_PATH)
        
    terms = list(indexer.index.df.keys())
    print("Sample terms:", terms[:30])

    
    # Create search engine
    search_engine = TFIDFVectorSpaceModel(indexer.index)
    
    # Interactive search
    print("\n" + "="*60)
    print("Indonesian Thesis Search Engine - Ready!")
    print("="*60)
    
    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\nSearching for: '{query}'")
        print("-" * 60)
        
        results = search_engine.search(query, top_k=10)
        
        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results:\n")
            for rank, (doc_id, score, title) in enumerate(results, 1):
                filename = indexer.index.doc_metadata[doc_id]['filename']
                print(f"{rank}. [Score: {score:.4f}]")
                print(f"   Title: {title}")
                print(f"   File: {filename}")
                print()