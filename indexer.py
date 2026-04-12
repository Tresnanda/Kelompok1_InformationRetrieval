import PyPDF2
import re
from preprocessor import IndonesianPreprocessor
from invertedindex import InvertedIndex
import os
from collections import defaultdict, Counter
import pickle

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
            r'metode penelitian', r'metodologi penelitian', r'methodology', r'bab iii'
        ]
        bab4_5_boundaries = [
            r'\bbab\s*iv\b', r'\bbab\s*4\b',
            r'\bbab\s*v\b', r'\bbab\s*5\b',
            r'hasil dan pembahasan', r'results and discussion', r'bab iv'
        ]

        start_m = min([text_lower.find(pat) for pat in bab3_patterns if text_lower.find(pat) != -1] or [len(text_lower)])
        end_m = min([text_lower.find(pat, start_m + 50) for pat in bab4_5_boundaries if text_lower.find(pat, start_m + 50) != -1] or [len(text_lower)])
        if start_m != len(text_lower):
            methodology = full_text[start_m:end_m]

        # --- CONCLUSION (Bab V) ---
        conclusion = ""
        bab5_patterns = [
            r'\bbab\s*v\b', r'\bbab\s*5\b',
            r'kesimpulan', r'conclusion', r'conclusions', r'bab v', r'hasil dan pembahasan'
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
