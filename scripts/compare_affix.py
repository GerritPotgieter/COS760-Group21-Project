from gensim.models import Word2Vec
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


AFFIX_FILE_PATH = 'data/Nguni/TRAIN/SADII.XH.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt'

# lado models 
standard_model = Word2Vec.load("data/Models/word2vec_xhosa.model")
affix_model = Word2Vec.load("data/Models/word2vec_xhosa_affixes.model")

#load the dictionary from embedding_affix (affixdictionary)
with open("data/Models/affix_dictionary.pkl", "rb") as f:
    affix_dict = pickle.load(f)

with open("data/Models/affix_test_dictionary.pkl", "rb") as f:
    affix_test_dict = pickle.load(f)

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

train_and_evaluate(X_surface, y_surface_encoded, "Surface Embeddings")
train_and_evaluate(X_affix, y_affix_encoded, "Affix Embeddings")