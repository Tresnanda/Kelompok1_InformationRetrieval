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
        stopwords_path='resources/stopwords-id.txt'
        self.custom_stopwords = {
            'yang', 'dan', 'di', 'dari', 'untuk', 'pada', 'dengan', 'adalah',
            'ini', 'itu', 'ke', 'dalam', 'atau', 'oleh', 'akan', 'telah',
            'dapat', 'juga', 'sebagai', 'tidak', 'ada', 'tersebut', 'sehingga'
        }
        
        self.slang_dict = self._load_slang_dict('resouces/merged_slang_dict.json')
        
    def _load_slang_dict(self, path) -> Dict[str, str]:
         try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
         except FileNotFoundError:
             print(f'Slang dictionary {path} not found')
             return {}
        
    
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
    
    def normalize_slang(self, text) -> str:
        if not self.slang_dict:
            return text
        
        words = text.split()
        normalized = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized)
    
    def preprocess(self, text: str) -> List[str]:
        """Complete preprocessing pipeline"""
        # 1. Casefold (lowercase)
        text = text.lower()
        
        # 2. Noise removal
        text = self.remove_noise(text)
        
        text = self.normalize_slang(text)
        
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


class HybridSearchEngine:
    """Hybrid search engine combining content-based and title-based retrieval"""

    def __init__(self, content_index: InvertedIndex, title_index: InvertedIndex):
        self.content_index = content_index
        self.title_index = title_index
        self.preprocessor = IndonesianPreprocessor()

        # Create separate TF-IDF models for content and title
        self.content_model = TFIDFVectorSpaceModel(content_index)
        self.title_model = TFIDFVectorSpaceModel(title_index)

        # Weight parameters for hybrid scoring (tunable) - title gets higher weight
        self.content_weight = 0.3
        self.title_weight = 0.7

    def set_weights(self, content_weight: float, title_weight: float):
        """Set weights for content and title components"""
        if content_weight + title_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.content_weight = content_weight
        self.title_weight = title_weight

    def search(self, query: str, top_k: int = 10):
        """Perform hybrid search combining content and title scores"""
        query_terms = self.preprocessor.preprocess(query)
        print("Query terms:", query_terms)

        if not query_terms:
            return []

        # Get content-based scores
        content_results = self.content_model.search(query, top_k=top_k*2)  # Get more for better combination
        content_scores = {doc_id: score for doc_id, score, _ in content_results}

        # Get title-based scores
        title_results = self.title_model.search(query, top_k=top_k*2)
        title_scores = {doc_id: score for doc_id, score, _ in title_results}

        # Combine scores from both indices
        all_docs = set(content_scores.keys()) | set(title_scores.keys())
        combined_scores = []

        for doc_id in all_docs:
            content_score = content_scores.get(doc_id, 0.0)
            title_score = title_scores.get(doc_id, 0.0)

            # Weighted combination
            final_score = (self.content_weight * content_score +
                          self.title_weight * title_score)

            if final_score > 0:
                # Get title from content index metadata
                title = self.content_index.doc_metadata.get(doc_id, {}).get("title", "Unknown")
                filename = self.content_index.doc_metadata.get(doc_id, {}).get("filename", "Unknown")
                combined_scores.append((doc_id, final_score, title, filename,
                                      content_score, title_score))

        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]



class PDFCorpusIndexer:
    """Index PDF corpus and build inverted index"""

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.preprocessor = IndonesianPreprocessor()
        self.content_index = InvertedIndex()
        self.title_index = InvertedIndex()
        self.doc_id_counter = 0
    
    def extract_abstract_from_pdf(self, pdf_path: str) -> str:
        """Extract only abstract section from PDF file"""
        full_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    full_text += page_text + " "

        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

        # Extract abstract using multiple patterns
        abstract = self._extract_abstract_section(full_text)
        return abstract

    def _extract_abstract_section(self, text: str) -> str:
        """Extract abstract section from full text using various patterns"""
        text_lower = text.lower()

        # Common abstract keywords in Indonesian/English
        abstract_starters = [
            'abstrak', 'abstract', 'abstrak—', 'abstract—',
            '\nabstrak', '\nabstract', '\nabstrak ', '\nabstract '
        ]

        # Common abstract enders
        abstract_enders = [
            'kata kunci:', 'keywords:', 'keyword:',
            'kata kunci', 'keywords', 'keyword',
            'abstrak', 'abstract',  # next section starts
            'pendahuluan', 'introduction', 'bab i',
            'latar belakang', 'background'
        ]

        # Find abstract start
        start_pos = len(text)
        for starter in abstract_starters:
            pos = text_lower.find(starter)
            if pos != -1 and pos < start_pos:
                start_pos = pos

        # If no abstract found, return empty string
        if start_pos == len(text):
            return ""

        # Find abstract end
        end_pos = len(text)
        for ender in abstract_enders:
            pos = text_lower.find(ender, start_pos + 50)  # search after start
            if pos != -1 and pos < end_pos:
                end_pos = pos

        # Extract abstract text
        abstract_text = text[start_pos:end_pos].strip()

        # Clean up: remove the "ABSTRAK" or "ABSTRACT" header and clean formatting
        lines = abstract_text.split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Skip if it's just the abstract header
            if line.lower() in ['abstrak', 'abstract']:
                continue

            # Remove common formatting artifacts
            line = re.sub(r'^[A-Z][A-Z\s]+$', '', line)  # Remove all-caps headers
            line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering

            if line and len(line) > 10:  # Keep substantial lines
                cleaned_lines.append(line)

        # Join and clean final abstract
        final_abstract = ' '.join(cleaned_lines)

        # Remove multiple spaces and common artifacts
        final_abstract = re.sub(r'\s+', ' ', final_abstract)
        final_abstract = re.sub(r'^[A-Z][A-Z\s]*', '', final_abstract)  # Remove initial all-caps

        return final_abstract.strip()
    def extract_core_sections(self, pdf_path: str) -> str:
        """Extract Abstract, Methodology (Bab III), and Conclusion (Bab V) sections from a thesis PDF"""
        full_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                    page_lower = page_text.lower()

                    # Skip pages that are clearly non-content
                    skip_keywords = [
                        'daftar isi', 'table of contents', 'lembar pengesahan',
                        'daftar pustaka', 'references', 'bibliography',
                        'kata pengantar', 'lembar persetujuan'
                    ]
                    if any(keyword in page_lower[:300] for keyword in skip_keywords):
                        continue

                    full_text += page_text + " "
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

        text_lower = full_text.lower()

        # --- ABSTRACT ---
        abstract = self._extract_abstract_section(full_text)

        # --- METHODOLOGY (Bab III) ---
        methodology = ""
        bab3_patterns = [
            r'\bbab\s*iii\b', r'\bbab\s*3\b',
            r'metode penelitian', r'metodologi penelitian', r'methodology'
        ]
        bab4_5_boundaries = [
            r'\bbab\s*iv\b', r'\bbab\s*4\b',
            r'\bbab\s*v\b', r'\bbab\s*5\b',
            r'hasil dan pembahasan', r'results and discussion'
        ]

        start_m = min([text_lower.find(pat) for pat in bab3_patterns if text_lower.find(pat) != -1] or [len(text_lower)])
        end_m = min([text_lower.find(pat, start_m + 50) for pat in bab4_5_boundaries if text_lower.find(pat, start_m + 50) != -1] or [len(text_lower)])
        if start_m != len(text_lower):
            methodology = full_text[start_m:end_m]

        # --- CONCLUSION (Bab V) ---
        conclusion = ""
        bab5_patterns = [
            r'\bbab\s*v\b', r'\bbab\s*5\b',
            r'kesimpulan', r'conclusion', r'conclusions'
        ]
        end_keywords = [
            'daftar pustaka', 'references', 'bibliography'
        ]
        start_c = min([text_lower.find(pat) for pat in bab5_patterns if text_lower.find(pat) != -1] or [len(text_lower)])
        end_c = min([text_lower.find(pat, start_c + 50) for pat in end_keywords if text_lower.find(pat, start_c + 50) != -1] or [len(text_lower)])
        if start_c != len(text_lower):
            conclusion = full_text[start_c:end_c]

        # Combine all
        combined = " ".join([abstract, methodology, conclusion])
        combined = re.sub(r'\s+', ' ', combined)
        return combined.strip()


    def extract_text_from_pdf(self, pdf_path: str, filter_sections: bool = True) -> str:
        """Extract text from PDF file with optional section filtering (legacy method)"""
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
        """Extract title from document filename (without .pdf extension)"""
        # Extract title from filename by removing .pdf extension
        title = filename.replace('.pdf', '')
        return title
    
    def build_index(self, filter_sections: bool = True, max_docs: int = None):
        """Build inverted index from PDF corpus with content and title separation

        Args:
            filter_sections: If True, skip common non-content sections
            max_docs: Maximum number of documents to process (None for all)
        """
        print("Starting hybrid indexing process (Abstract-based + Title-based)...")
        print(f"Max documents: {max_docs if max_docs else 'ALL'}")

        pdf_files = [f for f in os.listdir(self.corpus_path) if f.endswith('.pdf')]

        # Limit number of documents if specified
        if max_docs:
            pdf_files = pdf_files[:max_docs]

        total_files = len(pdf_files)

        # Temporary posting lists for content and title
        temp_content_index = defaultdict(list)
        temp_title_index = defaultdict(list)

        for idx, filename in enumerate(pdf_files):
            print(f"Processing [{idx+1}/{total_files}]: {filename}")

            pdf_path = os.path.join(self.corpus_path, filename)
            abstract = self.extract_abstract_from_pdf(pdf_path)

            combined_text = self.extract_core_sections(pdf_path)
            if not combined_text.strip():
                print(f"  No main sections found, skipping...")
                continue


            # Extract title from filename
            title = self.extract_title(abstract, filename)

            # Preprocess abstract as content
            content_tokens = self.preprocessor.preprocess(combined_text)

            # Preprocess title separately
            title_tokens = self.preprocessor.preprocess(title)

            if not content_tokens and not title_tokens:
                continue

            # Store metadata in both indices
            self.content_index.doc_metadata[self.doc_id_counter] = {
                'filename': filename,
                'title': title,
                'path': pdf_path
            }
            self.title_index.doc_metadata[self.doc_id_counter] = {
                'filename': filename,
                'title': title,
                'path': pdf_path
            }

            # Process content tokens
            if content_tokens:
                content_term_freqs = Counter(content_tokens)
                content_length = len(content_tokens)
                self.content_index.doc_lengths[self.doc_id_counter] = content_length

                # Add to content index
                for term, freq in content_term_freqs.items():
                    temp_content_index[term].append((self.doc_id_counter, freq))

            # Process title tokens
            if title_tokens:
                title_term_freqs = Counter(title_tokens)
                title_length = len(title_tokens)
                self.title_index.doc_lengths[self.doc_id_counter] = title_length

                # Add to title index
                for term, freq in title_term_freqs.items():
                    temp_title_index[term].append((self.doc_id_counter, freq))

            self.doc_id_counter += 1

        # Transfer content index
        self.content_index.num_docs = self.doc_id_counter
        for term, postings in temp_content_index.items():
            self.content_index.index[term] = postings
            self.content_index.df[term] = len(postings)

        # Transfer title index
        self.title_index.num_docs = self.doc_id_counter
        for term, postings in temp_title_index.items():
            self.title_index.index[term] = postings
            self.title_index.df[term] = len(postings)

        # Compute average document lengths
        if self.content_index.doc_lengths:
            self.content_index.avg_doc_length = sum(self.content_index.doc_lengths.values()) / len(self.content_index.doc_lengths)
        if self.title_index.doc_lengths:
            self.title_index.avg_doc_length = sum(self.title_index.doc_lengths.values()) / len(self.title_index.doc_lengths)

        # Build TF-IDF document vectors BEFORE compression
        print("\nBuilding TF-IDF document vectors for content...")
        self.content_index.build_tfidf_doc_vectors()

        print("\nBuilding TF-IDF document vectors for title...")
        self.title_index.build_tfidf_doc_vectors()

        # Compress indices
        print("\nCompressing content index...")
        self.content_index.compress_index()

        print("\nCompressing title index...")
        self.title_index.compress_index()


        print(f"\nHybrid indexing complete!")
        print(f"Total documents: {self.content_index.num_docs}")
        print(f"Abstract terms: {len(self.content_index.index)}")
        print(f"Title terms: {len(self.title_index.index)}")
        print(f"Average abstract length: {self.content_index.avg_doc_length:.2f}")
        print(f"Average title length: {self.title_index.avg_doc_length:.2f}")
    
    def save_index(self, content_index_path: str, title_index_path: str):
        """Save both content and title indices to disk"""
        with open(content_index_path, 'wb') as f:
            pickle.dump(self.content_index, f)
        print(f"Content index saved to {content_index_path}")

        with open(title_index_path, 'wb') as f:
            pickle.dump(self.title_index, f)
        print(f"Title index saved to {title_index_path}")

    def load_index(self, content_index_path: str, title_index_path: str):
        """Load both content and title indices from disk"""
        with open(content_index_path, 'rb') as f:
            self.content_index = pickle.load(f)
        print(f"Content index loaded from {content_index_path}")

        with open(title_index_path, 'rb') as f:
            self.title_index = pickle.load(f)
        print(f"Title index loaded from {title_index_path}")


# Main execution
if __name__ == "__main__":
    # Configuration
    CORPUS_PATH = "downloads"  # Using downloads directory
    CONTENT_INDEX_PATH = "content_index.pkl"
    TITLE_INDEX_PATH = "title_index.pkl"
    FILTER_SECTIONS = True  # Set to False to index everything
    MAX_DOCS = 100  # Limit to 100 documents as requested

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