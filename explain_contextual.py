# Contextual Spelling Correction Explanation

print("="*70)
print("Contextual Spelling Correction with N-gram")
print("="*70)

print("\nFeatures Added:")
print("1. Context-aware spelling correction")
print("2. N-gram language model (unigram, bigram, trigram)")
print("3. Considers previous and next words")
print("4. Combines edit distance + contextual probability")

print("\n" + "="*70)
print("How It Works:")
print("="*70)

print("\nExample Query: 'Analsiis dan peranacngan user'")
print("\nStep 1: Tokenization")
print("Tokens: ['analsiis', 'dan', 'peranacngan', 'user']")

print("\nStep 2: Check with context")
print("Word 'analsiis':")
print("  - Previous word: None")
print("  - Next word: 'dan'")
print("  - Candidates: ['analisis', 'analisa']")

print("\nWord 'peranacngan':")
print("  - Previous word: 'dan'")
print("  - Next word: 'user'")
print("  - Candidates: ['perancangan', 'perangkatan']")

print("\nStep 3: Score calculation")
print("For each candidate:")
print("  Final Score = (EditSimilarity * 0.5) + (ContextScore * 0.5)")
print("  ContextScore includes:")
print("    - Unigram probability (word frequency)")
print("    - Bigram: P(word|previous)")
print("    - Bigram: P(next|word)")

print("\n" + "="*70)
print("N-gram Models Built from Corpus:")
print("="*70)

print("\n1. Unigram: word -> frequency")
print("   Example: 'analisis': 50, 'interface': 45")

print("\n2. Bigram: (word1, word2) -> frequency")
print("   Example: ('user', 'interface'): 25, ('analisis', 'data'): 10")

print("\n3. Contextual Probability:")
print("   P(analisis|dan) = frequency('dan', 'analisis') / frequency('dan', *)")

print("\n" + "="*70)
print("Benefits of Contextual Spelling:")
print("="*70)

print("\n1. Better for homonyms:")
print("   - 'bank' (financial vs river)")
print("   - Context determines the correct meaning")

print("\n2. Selects best candidate from multiple options:")
print("   - 'perancangan' vs 'perangkatan'")
print("   - Chooses based on surrounding words")

print("\n3. Domain-specific:")
print("   - Model built from academic papers")
print("   - Understands technical terminology")

print("\n4. Handles ambiguity:")
print("   - Multiple possible corrections")
print("   - Selects most contextually appropriate")

print("\n" + "="*70)
print("Implementation Details:")
print("="*70)

print("\nThe system now:")
print("1. Builds n-gram models from indexed documents")
print("2. Uses context when suggesting corrections")
print("3. Combines edit distance with n-gram probability")
print("4. Provides more accurate suggestions")

print("\nTo test:")
print("1. Delete old index files if needed")
print("2. Run: python main.py")
print("3. Try queries with typos")

print("\nExample:")
print("Query: 'analsiis data'")
print("Expected: 'analisis data' (context-aware)")

print("\n" + "="*70)
print("Contextual spelling correction is now active!")
print("="*70)