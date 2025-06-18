import json
import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cross_decomposition import CCA


LEXICON_PATH = r"data\Lexicons\dsfsi-en-za-bilex_11-off-ZA-lang-updated.v.1.0 (1).json"
ZULU_MODEL_PATH = "data/Models/word2vec_zulu_normalized.model"
XHOSA_MODEL_PATH = "data/Models/word2vec_xhosa_normalized.model"
ZULU_ALIGNED_MODEL_PATH = "data/Models/word2vec_zulu_aligned.model"
XHOSA_ALIGNED_MODEL_PATH = "data/Models/word2vec_xhosa_aligned.model"

# load bilingual lexicon from lexicons 
with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
    lexicon = json.load(f)
en_zul = lexicon.get("en-zul", [])
en_xho = lexicon.get("en-xho", []) # get all zulu and xhosa entries 

#make disctionaries for zulu and xhosa
zul_dict = {list(entry.keys())[0]: entry[list(entry.keys())[0]][0] for entry in en_zul if entry[list(entry.keys())[0]]}
xho_dict = {list(entry.keys())[0]: entry[list(entry.keys())[0]][0] for entry in en_xho if entry[list(entry.keys())[0]]}

# get common english words in zulu and xhosa dictionaries
common_english = set(zul_dict.keys()) & set(xho_dict.keys())
# build the zulu xhosa bilingual pairs
bilingual_pairs = [(zul_dict[eng], xho_dict[eng]) for eng in common_english]
print(f"Found {len(bilingual_pairs)} bilingual pairs between Zulu and Xhosa.")

#load the previously allinged models
zulu_model = Word2Vec.load(ZULU_MODEL_PATH)
xhosa_model = Word2Vec.load(XHOSA_MODEL_PATH)

#filter valid pairs based on the vocabulary of the models
valid_pairs = [
    (zul, xho)
    for zul, xho in bilingual_pairs
    if zul in zulu_model.wv.key_to_index and xho in xhosa_model.wv.key_to_index
]

print(f"[INFO] {len(valid_pairs)} word pairs found in both vocabularies.")

#create the CCA input matrices
# X will contain zulu embeddings, Y will contain xhosa embeddings
X = np.array([zulu_model.wv[zul] for zul, xho in valid_pairs])
Y = np.array([xhosa_model.wv[xho] for zul, xho in valid_pairs])

#run the cca and fit model
cca = CCA(n_components=min(X.shape[1], Y.shape[1]))
X_c, Y_c = cca.fit_transform(X, Y)

print("CCA completed and embeddings aligned.")

#Create final aligned dictionaries
aligned_zulu_vecs = {}
aligned_xhosa_vecs = {}

for idx, (zul, xho) in enumerate(valid_pairs):
    aligned_zulu_vecs[zul] = X_c[idx]
    aligned_xhosa_vecs[xho] = Y_c[idx]


#zulu aligned KeyedVectors
aligned_zulu_kv = KeyedVectors(vector_size=X_c.shape[1])
aligned_zulu_kv.add_vectors(list(aligned_zulu_vecs.keys()), list(aligned_zulu_vecs.values()))
#xhosa aligned KeyedVectors
aligned_xhosa_kv = KeyedVectors(vector_size=Y_c.shape[1])
aligned_xhosa_kv.add_vectors(list(aligned_xhosa_vecs.keys()), list(aligned_xhosa_vecs.values()))

#save models for evaluation usage
aligned_zulu_kv.save(ZULU_ALIGNED_MODEL_PATH)
aligned_xhosa_kv.save(XHOSA_ALIGNED_MODEL_PATH)


print(f"[INFO] Aligned models saved to:")
print(f" - Zulu: {ZULU_ALIGNED_MODEL_PATH}")
print(f" - Xhosa: {XHOSA_ALIGNED_MODEL_PATH}")
