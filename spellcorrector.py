from typing import List, Dict, Tuple, Set
from preprocessor import IndonesianPreprocessor
import re
from collections import defaultdict, Counter

class SpellingCorrector:
    """Contextual spelling correction using n-gram language model and Levenshtein edit distance"""

    def __init__(self, vocabulary: Set[str]):
        self.vocabulary = vocabulary
        self.preprocessor = IndonesianPreprocessor()
        # n-gram models for context
        self.bigram_model = defaultdict(int)  # (word1, word2) -> count
        self.trigram_model = defaultdict(int)  # (word1, word2, word3) -> count
        self.unigram_counts = defaultdict(int)  # word -> frequency

    def build_vocabulary_from_indices(self, content_index, title_index):
        """Build vocabulary from both content and title indices"""
        vocab = set()

        # Add terms from content index (preprocessed terms)
        if hasattr(content_index, 'df'):
            vocab.update(content_index.df.keys())

        # Add terms from title index (preprocessed terms)
        if hasattr(title_index, 'df'):
            vocab.update(title_index.df.keys())

        # Also add original words from document titles and metadata
        # This helps with spelling correction of common Indonesian words
        if hasattr(content_index, 'doc_metadata'):
            for doc_id, metadata in content_index.doc_metadata.items():
                # Extract words from title
                title = metadata.get('title', '')
                if title:
                    # Simple tokenization and lowercase
                    words = re.findall(r'\b[a-zA-Z]+\b', title.lower())
                    vocab.update(words)

                # Extract words from filename
                filename = metadata.get('filename', '')
                if filename:
                    # Remove .pdf extension and split
                    base_name = filename.replace('.pdf', '').lower()
                    words = re.findall(r'\b[a-zA-Z]+\b', base_name)
                    vocab.update(words)

        self.vocabulary = vocab
        print(f"Built vocabulary with {len(vocab)} unique terms")

        # Build n-gram models from document content
        self._build_ngram_models(content_index)

    def _build_ngram_models(self, index):
        """Build n-gram language models from indexed documents"""
        print("Building n-gram models for contextual spelling...")

        # Extract text from documents
        all_sentences = []

        # Get text from document metadata and content
        if hasattr(index, 'doc_metadata'):
            for doc_id, metadata in index.doc_metadata.items():
                # Extract words from title
                title = metadata.get('title', '')
                if title:
                    # Preprocess to get tokens
                    title_tokens = self.preprocessor.preprocess(title)
                    if title_tokens:
                        all_sentences.append(title_tokens)

                # Extract words from filename
                filename = metadata.get('filename', '')
                if filename:
                    # Remove .pdf extension and preprocess
                    base_name = filename.replace('.pdf', '').lower()
                    # Simple tokenization for filename
                    words = base_name.replace('-', ' ').replace('_', ' ').split()
                    filename_tokens = [w for w in words if len(w) >= 3]
                    if filename_tokens:
                        all_sentences.append(filename_tokens)

        # Also use the indexed terms to build common patterns
        if hasattr(index, 'df'):
            # Create artificial sentences from common term pairs
            common_terms = sorted(index.df.keys(), key=index.df.get, reverse=True)[:100]

            # Add individual terms as unigrams
            for term in common_terms:
                self.unigram_counts[term] = index.df[term]

        # Build n-grams from all sentences
        for sentence in all_sentences:
            # Unigrams
            for word in sentence:
                self.unigram_counts[word] += 1

            # Bigrams
            for i in range(len(sentence) - 1):
                self.bigram_model[(sentence[i], sentence[i+1])] += 1

            # Trigrams
            for i in range(len(sentence) - 2):
                self.trigram_model[(sentence[i], sentence[i+1], sentence[i+2])] += 1

        # If still no bigrams, create some based on common combinations
        if len(self.bigram_model) < 10:
            print("Creating synthetic bigrams from vocabulary...")
            common_words = [w for w in list(self.vocabulary)[:50] if len(w) >= 3]
            # Create some common combinations
            common_pairs = [
                ('sistem', 'informasi'),
                ('analisis', 'data'),
                ('user', 'interface'),
                ('perancangan', 'sistem'),
                ('implementasi', 'algoritma'),
                ('pengujian', 'sistem'),
                ('evaluasi', 'kinerja'),
                ('metode', 'penelitian'),
                ('pengembangan', 'aplikasi'),
                ('desain', 'database')
            ]

            for w1, w2 in common_pairs:
                if w1 in self.vocabulary and w2 in self.vocabulary:
                    self.bigram_model[(w1, w2)] = 5
                    self.bigram_model[(w2, w1)] = 3  # Add reverse with lower weight

        print(f"Built models with {len(self.unigram_counts)} unigrams, {len(self.bigram_model)} bigrams")

    def get_contextual_score(self, candidate: str, prev_word: str = None, next_word: str = None) -> float:
        """
        Calculate contextual probability score for a candidate word
        Using n-gram language model
        """
        # Fallback: if no n-gram data, return base score
        if not self.unigram_counts:
            return 0.1  # Small default score

        score = 0.0
        total_contexts = 0.0

        # Get unigram frequency (base probability)
        unigram_count = self.unigram_counts.get(candidate, 0)
        total_unigrams = sum(self.unigram_counts.values()) if self.unigram_counts else 1
        base_prob = unigram_count / total_unigrams
        score += base_prob * 0.3  # 30% weight for base probability

        # Check bigram context: prev_word -> candidate
        if prev_word and prev_word in self.vocabulary:
            bigram_count = self.bigram_model.get((prev_word, candidate), 0)
            # Calculate total bigrams with prev_word
            prev_total = 0
            for w in self.vocabulary:
                if self.bigram_model.get((prev_word, w), 0) > 0:
                    prev_total += self.bigram_model.get((prev_word, w), 0)

            if prev_total > 0:
                bigram_prob = bigram_count / prev_total
                score += bigram_prob * 0.35  # 35% weight for left context
                total_contexts += 1

        # Check bigram context: candidate -> next_word
        if next_word and next_word in self.vocabulary:
            bigram_count = self.bigram_model.get((candidate, next_word), 0)
            # Calculate total bigrams with next_word
            next_total = 0
            for w in self.vocabulary:
                if self.bigram_model.get((w, next_word), 0) > 0:
                    next_total += self.bigram_model.get((w, next_word), 0)

            if next_total > 0:
                bigram_prob = bigram_count / next_total
                score += bigram_prob * 0.35  # 35% weight for right context
                total_contexts += 1

        # Normalize if we have context
        if total_contexts > 0:
            # Add small bonus for having context matches
            score *= (1 + 0.1 * total_contexts)

        return score

    def suggest_correction(self, word: str, max_suggestions: int = 3,
                         prev_word: str = None, next_word: str = None) -> List[Tuple[str, float]]:
        """
        Suggest spelling corrections with context awareness
        Combines edit distance and contextual probability
        """
        if not word or len(word) < 3:
            return []

        # PERBAIKAN 1: Cek frekuensi kata.
        # Jika kata ada di vocabulary TAPI tidak ada di unigram_counts (artinya jarang muncul/typo di corpus),
        # maka kita tetap lanjutkan proses koreksi.
        # Kita hanya return [] jika kata tersebut benar-benar valid (frekuensi > 0).
        if word in self.vocabulary and self.unigram_counts.get(word, 0) > 0:
            return []

        suggestions = []
        candidates = []

        # First, filter candidates by length difference for efficiency
        max_length_diff = 2

        for vocab_word in self.vocabulary:
            length_diff = abs(len(vocab_word) - len(word))
            if length_diff <= max_length_diff:
                candidates.append(vocab_word)

        # Calculate scores for each candidate
        for candidate in candidates:
            # PERBAIKAN 2: Abaikan kandidat yang sama persis dengan input JIKA kandidat tersebut frekuensinya 0/rendah.
            # Ini mencegah 'sentime' (skor similarity 1.0) mengalahkan 'sentimen' (skor similarity 0.8) 
            # hanya karena 'sentime' ada di index sebagai typo.
            if candidate == word and self.unigram_counts.get(candidate, 0) == 0:
                continue
            
            distance = self.levenshtein_distance(word, candidate)
            max_length = max(len(word), len(candidate))

            # Only consider candidates with reasonable edit distance
            if distance <= max_length * 0.4:
                # Calculate similarity based on edit distance
                similarity = 1.0 - (distance / max_length)

                if similarity >= 0.6:
                    # Get contextual score
                    contextual_score = self.get_contextual_score(candidate, prev_word, next_word)

                    # Combine edit distance similarity with contextual score
                    # Edit distance: 50%, Context: 50%
                    final_score = (similarity * 0.5) + (contextual_score * 0.5)

                    suggestions.append((candidate, final_score, distance, contextual_score))

        # Sort by final score (descending)
        suggestions.sort(key=lambda x: x[1], reverse=True)

        # Return only candidate and final score
        return [(s[0], s[1]) for s in suggestions[:max_suggestions]]

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings
        Returns minimum number of single-character edits (insertions, deletions, substitutions)
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        # s1 is now longer than or equal to s2
        if len(s2) == 0:
            return len(s1)

        # Initialize previous row
        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                # Calculate costs for insertion, deletion, substitution
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)

                # Take minimum of the three operations
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity score based on edit distance
        Returns value between 0 and 1, where 1 means identical
        """
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 1.0

        distance = self.levenshtein_distance(word1, word2)
        similarity = 1.0 - (distance / max_len)

        return similarity

    def correct_query_spelling(self, query: str) -> Tuple[str, bool, Dict[str, List[str]]]:
        """
        Correct spelling in a query WITH context awareness
        Returns: (corrected_query, was_corrected, corrections_dict)
        """
        # Simple tokenization (split by whitespace, punctuation)
        import re
        raw_tokens = re.findall(r'\b\w+\b', query.lower())

        if not raw_tokens:
            return query, False, {}

        corrected_tokens = []
        corrections = {}
        was_corrected = False

        # Check each raw token with context
        for i, token in enumerate(raw_tokens):
            # Get context words
            prev_word = raw_tokens[i-1] if i > 0 else None
            next_word = raw_tokens[i+1] if i < len(raw_tokens) - 1 else None

            # Only check spelling for tokens longer than 2 characters
            if len(token) >= 3:
                # Get suggestions with context
                suggestions = self.suggest_correction(token, prev_word=prev_word, next_word=next_word)
                
                # Check if we have a valid correction that is DIFFERENT from the original token
                if suggestions and suggestions[0][0] != token:
                    corrected_tokens.append(suggestions[0][0])
                    corrections[token] = [s[0] for s in suggestions[:3]]
                    was_corrected = True
                else:
                    corrected_tokens.append(token)

        # Reconstruct the query with corrected words
        corrected_query = ' '.join(corrected_tokens)
        return corrected_query, was_corrected, corrections