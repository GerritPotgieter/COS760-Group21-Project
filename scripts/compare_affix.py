from gensim.models import Word2Vec
import pickle
import numpy as np
# lado models 
standard_model = Word2Vec.load("data/Models/word2vec_xhosa.model")
affix_model = Word2Vec.load("data/Models/word2vec_xhosa_affixes.model")

#load the dictionary from embedding_affix (affixdictionary)
with open("data/Models/affix_dictionary.pkl", "rb") as f:
    affix_dict = pickle.load(f)

def get_affix_vector(word, model, dictionary):
    #finds the affixes that make up the word and averages out the vector value 
    if word not in dictionary:
        print(f"Word '{word}' not found in dictionary.")
        return None
    
    affix_tokens = dictionary[word]
    vectors = []
    
    for token in affix_tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
        else:
            print(f"Token '{token}' not in model vocabulary.")
    
    if not vectors:
        return None
    
    return np.mean(vectors, axis=0)

query_words = ["umntu", "zabahlali", "fundisa", "abantwana","ukufuneka", "abantu", "lungiselelo"]

# affix_mapped = { 
#     "umntu": ["u_NPrePre1", "m_BPre1", "ntu_NStem"],
#     "zabahlali": ["za_SC6", "ba_BPre2", "hlali_NStem"],
#     "fundisa": ["fund_VRoot", "is_CausExt", "a_VerbTerm"],
#     "abantwana": ["a_NPrePre2", "ba_BPre2", "ntwana_NStem"],
# }

print("Comparing Similar Words")
for word in query_words:
    print(f"\n{word}")
    if word in standard_model.wv:
        print("Standard (non-affix):")
        for w, sim in standard_model.wv.most_similar(word, topn=5):
            print(f"  {w}: {sim:.3f}")
    else:
        print("Standard: Not in vocab.")

    vec = get_affix_vector(word, affix_model, affix_dict)
    if vec is not None:
        print("Affix (morphemes):")
        for w, sim in affix_model.wv.similar_by_vector(vec, topn=5):
            print(f"  {w}: {sim:.3f}")
    else:
        print("Affix: Not found or no affix tokens in vocab.")

# print(affix_model.wv.index_to_key[:50])
print("Ukufuneka" in affix_dict)  # Should be True
print(affix_dict.get("Ukufuneka"))  # Should print something like ['ku_BPre15', 'funek_VRoot', 'a_VerbTerm']
