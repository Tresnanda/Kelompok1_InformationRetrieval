from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from typing import List, Dict, Tuple, Set
import json
import re

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
        
        self.slang_dict = self._load_slang_dict('resources/merged_slang_dict.json')
        
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
        
        # # 7. Stemming using Sastrawi
        # tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens