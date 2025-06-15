from gensim.models import Word2Vec
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

AFFIX_FILE_PATH = 'data/Nguni/TRAIN/SADII.XH.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt'
AFFIX_ZULU_FILE_PATH = 'data/Nguni/TRAIN/SADII.ZU.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt'
AFFIX_FILE_PATH_TEST = 'data/Nguni/TEST/SADII.XH.Morph_Lemma_POS.1.0.0.TEST.CTexT.TG.2021-09-30.txt'
AFFIX_ZULU_FILE_PATH_TEST = 'data/Nguni/TEST/SADII.ZU.Morph_Lemma_POS.1.0.0.TEST.CTexT.TG.2021-09-30.txt'

# lado models 
standard_model = Word2Vec.load("data/Models/word2vec_xhosa.model")
affix_model = Word2Vec.load("data/Models/word2vec_xhosa_affixes.model")
standard_model_zulu = Word2Vec.load("data/Models/word2vec_zulu.model")
affix_model_zulu = Word2Vec.load("data/Models/word2vec_zulu_affixes.model")

#load the dictionary from embedding_affix (affixdictionary)
with open("data/Models/affix_dictionary.pkl", "rb") as f:
    affix_dict = pickle.load(f)

with open("data/Models/affix_test_dictionary.pkl", "rb") as f:
    affix_test_dict = pickle.load(f)

with open("data/Models/affix_test_dictionary_zulu.pkl", "rb") as f:
    affix_test_dict_zulu = pickle.load(f)

with open("data/Models/affix_dictionary_zulu.pkl", "rb") as f:
    affix_dict_zulu = pickle.load(f)

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


def get_standard_vector(word, model):
     if word in model.wv:
          return model.wv[word]
     else: 
          print(f"Word '{word}' not found in surface model")
          return None

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query_words = ["umntu", "zabahlali", "fundisa", "abantwana","ukufuneka", "abantu", "lungiselelo"]

# affix_mapped = { 
#     "umntu": ["u_NPrePre1", "m_BPre1", "ntu_NStem"],
#     "zabahlali": ["za_SC6", "ba_BPre2", "hlali_NStem"],
#     "fundisa": ["fund_VRoot", "is_CausExt", "a_VerbTerm"],
#     "abantwana": ["a_NPrePre2", "ba_BPre2", "ntwana_NStem"],
# }

def load_pos_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue  # skip malformed lines
            surface_word, affix_str, pos_tag = parts
            affix_tokens = affix_str.split('+') if affix_str else []
            data.append((surface_word, affix_tokens, pos_tag))
    return data



def load_pos_data_with_morphemes(filepath):
    surface_pos = []
    affix_pos = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                word = parts[0]
                raw_morphs = parts[1]
                pos = parts[3]

                # Clean morphemes (e.g., enz[VRoot] → enz_VRoot)
                morphs = [
                    m.replace('[', '_').replace(']', '')
                    for m in raw_morphs.split('-')
                ]

                surface_pos.append((word, pos))
                affix_pos.append((morphs, pos))
                
    return surface_pos, affix_pos



def build_dataset(model_type, model, affix_test_dict, pos_data):
    X, y = [], []

    for surface_word, affix_tokens, pos_tag in pos_data:
        if model_type == 'surface':
            if surface_word in model.wv:
                X.append(model.wv[surface_word]) 
                y.append(pos_tag)
        elif model_type == 'affix':
            morphemes = affix_dict[surface_word]
            vectors = [model.wv[m] for m in morphemes if m in model.wv]
            if len(vectors) == len(morphemes) and vectors:
                X.append(np.mean(vectors, axis=0))
                y.append(pos_tag)
    
    return np.array(X), np.array(y)



def train_and_evaluate(X, y, label="Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nResults for {label} Embeddings:\n")
    print(classification_report(y_test, y_pred))




def embed_surface_data(data, model):
    X, y = [], []
    for word, label in data:
        if word in model.wv:
            X.append(model.wv[word])
            y.append(label)
    return X, y

def embed_affix_data(data, model):
    X, y = [], []
    for morphemes, label in data:
        vectors = [model.wv[m] for m in morphemes if m in model.wv]
        if vectors:
            X.append(np.mean(vectors, axis=0))
            y.append(label)
    return X, y



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
print("--------------------")
print("Testing overlap")
surface_vocab = set(standard_model.wv.index_to_key)
affix_vocab = set(affix_model.wv.index_to_key)

shared = surface_vocab & affix_vocab
only_surface = surface_vocab - affix_vocab
only_affix = affix_vocab - surface_vocab

print(f"Surface vocab size: {len(surface_vocab)}")
print(f"Affix vocab size: {len(affix_vocab)}")
print(f"Shared tokens: {len(shared)}")
print(f"Only in surface: {len(only_surface)}")
print(f"Only in affix: {len(only_affix)}")
print("--------------------")
print("Affix")
print("Word1\t\tWord2\t\tSimilarity Score")

word_pairs = [
    ("umfundi", "abafundi"),
    ("ukufunda", "funda"),
    ("ukukhusela", "kukhuselo"),
    ("abantu", "umntu"),
    ("abantwana", "umntwana"),
    ("ukusebenza", "isebenza"),
]

for word1, word2 in word_pairs:
    vec1 = get_affix_vector(word1, affix_model, affix_dict)
    vec2 = get_affix_vector(word2, affix_model, affix_dict)

    if vec1 is not None and vec2 is not None:
            sim = cosine_similarity([vec1], [vec2])[0][0]
            print(f"{word1:<15} {word2:<15} {sim:.3f}")
    else:
            print(f"{word1:<15} {word2:<15} {'N/A (missing)'}")



print("--------------------")
print("Surface")
print("Word1\t\tWord2\t\tSimilarity Score")
for word1, word2 in word_pairs:
    vec1 = get_standard_vector(word1, standard_model)
    vec2 = get_standard_vector(word2, standard_model)

    if vec1 is not None and vec2 is not None:
            sim = cosine_similarity([vec1], [vec2])[0][0]
            print(f"{word1:<15} {word2:<15} {sim:.3f}")
    else:
            print(f"{word1:<15} {word2:<15} {'N/A (missing)'}")




print("--------------------")
print("Coverage Testing (Out of vocabulary)")
totalsurface = len(affix_test_dict)
surfacecov = sum(1 for word in affix_test_dict if word in surface_vocab)
print(f'{round(surfacecov/totalsurface*100, 2)}'"%")
total_affix_words = 0
affix_covered = 0

for word in affix_test_dict:
    morphemes = affix_test_dict[word]
    if morphemes:
        total_affix_words += 1
        if all(m in affix_vocab for m in morphemes):
            affix_covered += 1

if total_affix_words > 0:
    affix_coverage = round(affix_covered / total_affix_words * 100, 2)
    print(f"Affix coverage: {affix_covered}/{total_affix_words} = {affix_coverage}%")
else:
    print("No test words found in the word-to-morpheme mapping.")

print("--------------------")
#pos loading 
surface_data, affix_data = load_pos_data_with_morphemes(AFFIX_FILE_PATH)
print("Surface samples:", surface_data[:3])
print("Affix samples:", affix_data[:3])
# X_surface, y_surface = build_dataset("surface", standard_model, affix_test_dict, posdata)
# X_affix, y_affix = build_dataset("affix", affix_model, affix_test_dict, posdata)

# train_and_evaluate(X_surface, y_surface, label="Surface")
# train_and_evaluate(X_affix, y_affix, label="Affix")

# if vec1 is not None and vec2 is not None:
#     similarity = cosine_similarity([vec1], [vec2])[0][0]
#     print("Affix similarity:", similarity)
# else:
#     print("One of the vectors not found in affix model.")
X_surface, y_surface = embed_surface_data(surface_data, standard_model)
X_affix, y_affix = embed_affix_data(affix_data, affix_model)

all_labels = y_surface + y_affix
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

y_surface_encoded = label_encoder.transform(y_surface)
y_affix_encoded = label_encoder.transform(y_affix)




def train_and_evaluate(X, y, label=""):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n=== {label} POS Classification Report ===")
    print(classification_report(y_test, y_pred, labels=label_encoder.transform(label_encoder.classes_),target_names=label_encoder.classes_, zero_division=0))





# all_labels = [label for (_, label) in surface_data]
# label_encoder = LabelEncoder()
# label_encoder.fit(all_labels)
# X = [vector for (vector, _) in surface_data]
# y = [label for (_, label) in surface_data]
# X_train, X_test, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
# y_train = label_encoder.transform(y_train_labels)
# y_test = label_encoder.transform(y_test_labels)




# these need to be uncommented for pos classif within xhosa (shows better for affix due to oov)
# train_and_evaluate(X_surface, y_surface_encoded, "Surface Embeddings")
# train_and_evaluate(X_affix, y_affix_encoded, "Affix Embeddings")


translation_pairs = [
    #zulu to xhosa word pairs 
    ("umfana", "umfana"),         # boy
    ("intombazane", "intombazana"), # girl
    ("abantu", "abantu"),         # people
    ("ubaba", "utata"),           # father
    ("umama", "umama"),           # mother
    ("umntwana", "umntwana"),     # child
    ("umfazi", "umfazi"),         # woman
    ("indoda", "indoda"),         # man
    ("odade", "oodade"),          # sisters

   
    ("ikhanda", "intloko"),       # head
    ("iso", "iliso"),             # eye
    ("indlebe", "indlebe"),       # ear
    ("umlomo", "umlomo"),         # mouth
    ("izandla", "izandla"),       # hands
    ("unyawo", "unyawo"),         # foot

  
    ("amanzi", "amanzi"),         # water
    ("ilanga", "ilanga"),         # sun
    ("inkanyezi", "inkwenkwezi"),# star
    ("umoya", "umoya"),           # wind/spirit
    ("umhlaba", "umhlaba"),       # earth/land

 
    ("inja", "inja"),             # dog
    ("ikati", "ikati"),           # cat
    ("inkomo", "inkomo"),         # cow
    ("ihhashi", "ihashe"),        # horse
    ("inyoni", "inyoni"),         # bird
    ("ingwe", "ingwe"),           # leopard

    
    ("hamba", "hamba"),           # walk/go
    ("funda", "funda"),           # learn/read
    ("phuza", "sela"),            # drink
    ("dla", "tshiza"),            # eat
    ("lala", "lala"),             # sleep
    ("bona", "bona"),             # see
    ("thanda", "thanda"),         # love
    ("zama", "zama"),             # try
    ("cela", "cela"),             # ask

  
    ("indlu", "indlu"),           # house
    ("imoto", "imoto"),           # car
    ("isitulo", "isitulo"),       # chair
    ("ithebhulethi", "ithebhulethi"), # tablet
    ("incwadi", "incwadi"),       # book

   
    ("mnyama", "mnyama"),         # black
    ("mhlophe", "mhlophe"),       # white
    ("luhlaza", "luhlaza"),       # green
    ("bomvu", "bomvu"),           # red
    ("nsundu", "nsundu"),         # brown

    
    ("ukukhanya", "ukukhanya"),   # light
    ("umnyango", "umnyango"),     # door
    ("isikhumbuzo", "inkumbulo"), # memory
    ("ubomi", "impilo"),          # life
]


# transres = []; 
# for zu, xh in translation_pairs: 
#     entry = {
#         "zulu": zu, 
#         "xhosa": xh, 
#         "surface_in_vocab": zu in standard_model_zulu.wv and xh in standard_model.wv,
#         "affix_in_vocab": zu in affix_model_zulu.wv and xh in affix_model.wv,
#         "surface_cosine": None,
#         "affix_cosine": None,
#     }
#     if entry["surface_in_vocab"]:
#         entry["surface_cosine"] = cosine_similarity([standard_model_zulu.wv[zu]], [standard_model.wv[xh]])[0][0]

#     # Affix model similarity
#     if entry["affix_in_vocab"]:
#         entry["affix_cosine"] = cosine_similarity([affix_model_zulu.wv[zu]], [affix_model.wv[xh]])[0][0]


#     transres.append(entry)

#     df = pd.DataFrame(transres)

#     print("=== Summary ===")
#     print("Total translation pairs:", len(df))
#     print("Covered by surface model:", df['surface_in_vocab'].sum())
#     print("Covered by affix model:", df['affix_in_vocab'].sum())

#     print("\n--- Cosine Similarity Stats ---")
#     print("Surface mean similarity:", df['surface_cosine'].dropna().mean())
#     print("Affix mean similarity:", df['affix_cosine'].dropna().mean())
#     print("Surface std:", df['surface_cosine'].dropna().std())
#     print("Affix std:", df['affix_cosine'].dropna().std())

#     # Save results
#     df.to_csv("results/alignment_xhzu.csv", index=False)

#     # Optional: histogram plot
#     plt.figure(figsize=(8, 5))
#     plt.hist(df['surface_cosine'].dropna(), bins=20, alpha=0.6, label='Surface', color='blue')
#     plt.hist(df['affix_cosine'].dropna(), bins=20, alpha=0.6, label='Affix', color='green')
#     plt.title("Cosine Similarity Distribution")
#     plt.xlabel("Cosine Similarity")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("results/cosine_similarity_hist_xhzu.png")
#     plt.show()


def findsharedvocab(minfreq = 1, exclude_short = True, min_length = 3): 
     zulu_vocab = set(standard_model_zulu.wv.key_to_index.keys())
     xhosa_vocab = set(standard_model.wv.key_to_index.keys())
     shared_raw = zulu_vocab.intersection(xhosa_vocab)
     shared_filtered = []
     for word in shared_raw:
        # Check frequency in both models
        zulu_freq = standard_model_zulu.wv.get_vecattr(word, "count")
        xhosa_freq = standard_model.wv.get_vecattr(word, "count")
        
        # Apply filters
        if zulu_freq >= minfreq and xhosa_freq >= minfreq:
            if not exclude_short or len(word) >= min_length:
                shared_filtered.append(word)
    
     shared_vocab = shared_filtered
    
     print(f"Total Zulu vocabulary: {len(zulu_vocab):,}")
     print(f"Total Xhosa vocabulary: {len(xhosa_vocab):,}")
     print(f"Raw shared vocabulary: {len(shared_raw):,}")
     print(f"Filtered shared vocabulary: {len(shared_filtered):,}")
     print(f"Shared vocab as % of Zulu: {len(shared_filtered)/len(zulu_vocab)*100:.2f}%")
     print(f"Shared vocab as % of Xhosa: {len(shared_filtered)/len(xhosa_vocab)*100:.2f}%")
    
     return shared_filtered



def analyze_shared_vocabulary_patterns(shared_vocab):
        
        # Length distribution
        lengths = [len(word) for word in shared_vocab]
        
        # Morphological patterns (common prefixes/suffixes)
        prefixes = Counter()
        suffixes = Counter()
        
        for word in shared_vocab:
            if len(word) >= 4:
                prefixes[word[:2]] += 1
                suffixes[word[-2:]] += 1
        
        # Most common shared words by frequency
        word_freqs = []
        for word in shared_vocab:
            zulu_freq = standard_model_zulu.wv.get_vecattr(word, "count")
            xhosa_freq = standard_model.wv.get_vecattr(word, "count")
            word_freqs.append((word, zulu_freq, xhosa_freq, zulu_freq + xhosa_freq))
        
        word_freqs.sort(key=lambda x: x[3], reverse=True)
        
        results = {
            'length_stats': {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'min': min(lengths),
                'max': max(lengths)
            },
            'top_prefixes': prefixes.most_common(10),
            'top_suffixes': suffixes.most_common(10),
            'most_frequent': word_freqs[:20]
        }
        
       
        return results


sharedvocab = findsharedvocab(); 
lenmiki = analyze_shared_vocabulary_patterns(sharedvocab); 
print(lenmiki); 


def compute_cross_lingual_similarities(top_n=1020, shared_vocab=sharedvocab):
        
        #Compute cosine similarities for shared vocabulary across languages
        
        
        
       
        word_freqs = []
        for word in shared_vocab:
            zulu_freq = standard_model_zulu.wv.get_vecattr(word, "count")
            xhosa_freq = standard_model.wv.get_vecattr(word, "count")
            word_freqs.append((word, zulu_freq + xhosa_freq))
        
        word_freqs.sort(key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in word_freqs[:top_n]]
        
        
        similarities = []
        for word in top_words:
            zulu_vec = standard_model_zulu.wv[word].reshape(1, -1)
            xhosa_vec = standard_model.wv[word].reshape(1, -1)
            sim = cosine_similarity(zulu_vec, xhosa_vec)[0][0]
            similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        sim_values = [sim for _, sim in similarities]
        stats = {
            'mean_similarity': np.mean(sim_values),
            'median_similarity': np.median(sim_values),
            'std_similarity': np.std(sim_values),
            'min_similarity': min(sim_values),
            'max_similarity': max(sim_values)
        }
        
        
        
        return similarities, stats
    

crosslingualsim, crosslingualstats = compute_cross_lingual_similarities(shared_vocab=sharedvocab); 
print(crosslingualsim); 
print(crosslingualstats); 



def find_shared_morphemes(affix_model_zulu, affix_model_xhosa, min_length=2):
    """
    Find shared morphemes between affix-based models
    """
    zulu_morphemes = set(affix_model_zulu.wv.key_to_index.keys())
    print("Printing morphemes...")
    zulu_list = list(zulu_morphemes)
    print(zulu_list[:5])
    xhosa_morphemes = set(affix_model_xhosa.wv.key_to_index.keys())
    xhosa_list = list(xhosa_morphemes)
    print(xhosa_list[:5])
    shared_morphemes = list(zulu_morphemes.intersection(xhosa_morphemes))
    
    # Filter by length if needed
    if min_length:
        shared_morphemes = [morph for morph in shared_morphemes if len(morph) >= min_length]
    
    print(f"Zulu morphemes: {len(zulu_morphemes):,}")
    print(f"Xhosa morphemes: {len(xhosa_morphemes):,}")
    print(f"Shared morphemes: {len(shared_morphemes):,}")
    print(f"Shared as % of Zulu: {len(shared_morphemes)/len(zulu_morphemes)*100:.2f}%")
    print(f"Shared as % of Xhosa: {len(shared_morphemes)/len(xhosa_morphemes)*100:.2f}%")
    
    return shared_morphemes

sharedmorphs = find_shared_morphemes(affix_model_zulu=affix_model_zulu, affix_model_xhosa=affix_model); 



def compute_affix_similarities(word, topn = 791): 
    #will compute cosine similarities on shared vocab based on affix embeddings 

    if affix_dict_zulu.get(word) is None or affix_dict.get(word) is None:  
        return "Word was missing from a dictionary."; 
    X, y = [], []
    zulu_morphemes = affix_dict_zulu[word]; 
    xhosa_morphemes = affix_dict[word]; 
    # print("Printing dicitonary format....")
    # print(zulu_morphemes)
    # print(xhosa_morphemes)
    xhosavectors = [affix_model.wv[m] for m in xhosa_morphemes if m in affix_model.wv]  
    if len(xhosavectors) == len(xhosa_morphemes) and xhosavectors:
        # print("appended x...")
        X.append(np.mean(xhosavectors, axis=0))

    zuluvectors = [affix_model_zulu.wv[m] for m in zulu_morphemes if m in affix_model_zulu.wv]
    if len(zuluvectors) == len(zulu_morphemes) and zuluvectors:
        # print("appended y...")
        y.append(np.mean(zuluvectors, axis=0))
        
    return X, y; 




# print("Type of sharedvocab:", type(sharedvocab))
# print("First 5 items in sharedvocab:", sharedvocab[:5])
# print("Type of first item:", type(sharedvocab[0]) if len(sharedvocab) > 0 else "Empty")
# print("Type of affix_dict_zulu:", type(affix_dict_zulu))
# if len(affix_dict_zulu) > 0:
#     first_key = list(affix_dict_zulu.keys())[0]
#     print("First key in affix_dict_zulu:", first_key, type(first_key))
#     print("First value in affix_dict_zulu:", affix_dict_zulu[first_key])

print(sharedvocab[0])
xhd, zud = compute_affix_similarities(sharedvocab[0])
print(xhd, zud); 
print(sharedmorphs[:5])


xa_affix_sim = []
zu_affix_sim = []
for word in sharedvocab: 
    x, z = compute_affix_similarities(word); 
    if isinstance(x, list) and isinstance(z, list) and x and z:
        xa_affix_sim.extend(x)
        zu_affix_sim.extend(z)

xa_affix_sim = np.array(xa_affix_sim)
zu_affix_sim = np.array(zu_affix_sim)

if len(xa_affix_sim) > 0 and len(zu_affix_sim) > 0:
    affix_similarities = []
    for i in range(len(xa_affix_sim)):
    
        sim = cosine_similarity([xa_affix_sim[i]], [zu_affix_sim[i]])[0][0]
        affix_similarities.append(sim)
    
    print(f"Affix-based similarities calculated for {len(affix_similarities)} words")
    print(f"Mean affix similarity: {np.mean(affix_similarities):.4f}")



# sample_words = sharedvocab[:10]
# for word in sample_words:
#     if affix_dict_zulu.get(word) and affix_dict.get(word):
#         print("-"*45)
#         print(f"Word: {word}")
#         print(f"Zulu morphemes: {affix_dict_zulu[word]}")
#         print(f"Xhosa morphemes: {affix_dict[word]}")
#         print("---")

def collapse_pos_label(label):
    #attempt to balance results - too few occurences in some finer tags
    if label.startswith("ADJ"):
        return "ADJ"
    elif label.startswith("POSS"):
        return "POSS"
    elif label.startswith("PRON"):
        return "PRON"
    elif label.startswith("NLOC"):
        return "NLOC"
    elif label.startswith("LOC"):
        return "LOC"
    elif label.startswith("V"):
        return "V"
    elif label.startswith("PUNC") or label in {"PUNCT", "PUNCTUATION"}:
        return "PUNC"
    elif label.startswith("CONJ"):
        return "CONJ"
    elif label.startswith("NEG"):
        return "NEG"
    elif label.startswith("REL"):
        return "REL"
    elif label.startswith("ADV"):
        return "ADV"
    elif label.startswith("N"):
        return "N"
    elif label.startswith("NUM"):
        return "NUM"
    elif label.startswith("DET"):
        return "DET"
    elif label.startswith("AUX"):
        return "AUX"
    elif label.startswith("COP"):
        return "COP"
    elif label.startswith("TENSE"):
        return "TENSE"
    elif label == "P":
        return "PREP"
    elif label == "CL":
        return "CL"
    elif label == "ABBR":
        return "ABBR"
    else:
        return "OTHER"  




zulu_surface, zulu_affix = load_pos_data_with_morphemes(AFFIX_ZULU_FILE_PATH) #train file for zulu
xhosa_surface, xhosa_affix = load_pos_data_with_morphemes(AFFIX_FILE_PATH_TEST) #test file for xhosa 

#zu
X_train_s_raw, y_train_s_raw = embed_surface_data(zulu_surface, standard_model_zulu)
X_train_a_raw, y_train_a_raw = embed_affix_data(zulu_affix, affix_model_zulu)
#xhosa embeds
X_test_s_raw, y_test_s_raw = embed_surface_data(xhosa_surface, standard_model)
X_test_a_raw, y_test_a_raw = embed_affix_data(xhosa_affix, affix_model)

# 3. Filter for common tags
common_tags = set(y_train_s_raw).intersection(set(y_test_s_raw))

# 4. Filter all four datasets to only contain common tags
def filter_by_common_tags(X, y, tags):
    return zip(*[(x, label) for x, label in zip(X, y) if label in tags])

X_train_s, y_train_s = filter_by_common_tags(X_train_s_raw, y_train_s_raw, common_tags)
X_test_s, y_test_s   = filter_by_common_tags(X_test_s_raw,  y_test_s_raw,  common_tags)
X_train_a, y_train_a = filter_by_common_tags(X_train_a_raw, y_train_a_raw, common_tags)
X_test_a, y_test_a   = filter_by_common_tags(X_test_a_raw,  y_test_a_raw,  common_tags)

y_train_s = [collapse_pos_label(lbl) for lbl in y_train_s]
y_test_s  = [collapse_pos_label(lbl) for lbl in y_test_s]
y_train_a = [collapse_pos_label(lbl) for lbl in y_train_a]
y_test_a  = [collapse_pos_label(lbl) for lbl in y_test_a]


labelencoder = LabelEncoder(); 
label_encoder.fit(y_train_s)  

y_train_s = label_encoder.transform(y_train_s)
y_test_s  = label_encoder.transform(y_test_s)

y_train_a = label_encoder.transform(y_train_a)
y_test_a  = label_encoder.transform(y_test_a)

clf_s = LogisticRegression(max_iter=1000)
clf_s.fit(X_train_s, y_train_s)
y_pred_s = clf_s.predict(X_test_s)
print("\n================ Surface Embeddings POS Classification Report ===============")
print(classification_report(y_test_s, y_pred_s, labels=label_encoder.transform(label_encoder.classes_), target_names=label_encoder.classes_, zero_division=0))

# affix based classif
# remember line 318 commented for faster run (xhosa pos)**
clf_a = LogisticRegression(max_iter=1000)
clf_a.fit(X_train_a, y_train_a)
y_pred_a = clf_a.predict(X_test_a)
print("\n================= Affix Embeddings POS Classification Report ================")
print(classification_report(y_test_a, y_pred_a, labels=label_encoder.transform(label_encoder.classes_), target_names=label_encoder.classes_, zero_division=0))






