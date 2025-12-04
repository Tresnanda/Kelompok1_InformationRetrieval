import pickle
from main import InvertedIndex

# Path to your saved indices

with open("content_index.pkl", "rb") as f:
    raw_data = f.read()

print(raw_data[:2000])  # print first 2000 bytes
