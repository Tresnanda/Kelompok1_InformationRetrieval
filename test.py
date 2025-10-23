from main import PDFCorpusIndexer
from main import InvertedIndex  # <-- make sure this line is BEFORE pickle.load
import pickle

with open("thesis_index.pkl", "rb") as f:
    index = pickle.load(f)

for term in ["data", "uin", "aplikasi", "android"]:
    postings = index.get_postings(term)
    print(term, "→", postings)
    
print("num_docs =", index.num_docs)
print("df['data'] =", index.df.get("data"))
print("df['uin'] =", index.df.get("uin"))

