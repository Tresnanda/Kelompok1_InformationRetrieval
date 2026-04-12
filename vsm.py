from invertedindex import InvertedIndex
from preprocessor import IndonesianPreprocessor
import math
from typing import List, Dict, Tuple, Set
from spellcorrector import SpellingCorrector
from collections import defaultdict, Counter

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
            # print(f"Doc vector {doc_id}:", doc_vector)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            # print(f"Sim({doc_id}) =", similarity)
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

        # Initialize spelling corrector
        self.spelling_corrector = SpellingCorrector(set())
        self.spelling_corrector.build_vocabulary_from_indices(content_index, title_index)

    def set_weights(self, content_weight: float, title_weight: float):
        """Set weights for content and title components"""
        if content_weight + title_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.content_weight = content_weight
        self.title_weight = title_weight

    def search(self, query: str, top_k: int = 10):
        """Perform hybrid search combining content and title scores with spelling correction"""
        # Check for spelling corrections
        corrected_query, was_corrected, corrections = self.spelling_corrector.correct_query_spelling(query)

        final_query = query

        # If spelling corrections were suggested, ask for user confirmation
        if was_corrected:
            print(f"\nSpelling corrections found:")

            # Show each correction clearly
            for original, suggestions in corrections.items():
                if suggestions:
                    print(f"  '{original}' -> '{suggestions[0]}'")

            print(f"\n  Original query: {query}")
            print(f"  Corrected query: {corrected_query}")

            response = input("\nApakah yang dimaksud (y/n)? ").strip().lower()

            if response == 'y' or response == 'yes':
                final_query = corrected_query
                print(f"✓ Menggunakan query yang dikoreksi: '{final_query}'")
            else:
                final_query = query
                print(f"✓ Menggunakan query asli: '{query}'")

        # Perform search with the final query
        query_terms = self.preprocessor.preprocess(final_query)
        print("Query terms (preprocesed):", query_terms)

        if not query_terms:
            return []

        # Get content-based scores
        content_results = self.content_model.search(final_query, top_k=top_k*2)  # Get more for better combination
        content_scores = {doc_id: score for doc_id, score, _ in content_results}

        # Get title-based scores
        title_results = self.title_model.search(final_query, top_k=top_k*2)
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