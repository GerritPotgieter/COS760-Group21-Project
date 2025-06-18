from gensim.models import Word2Vec
import os
import re
import numpy as np
import pickle

#remember to switch vars for xhosa, zulu and then also keep in mind the pickle files toward the eof *********
AFFIX_FILE_PATH = 'data/Nguni/TRAIN/SADII.ZU.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt'  
OUTPUT_MODEL_PATH = 'data/Models/word2vec_zulu_affixes.model' 
AFFIX_TEST_FILE_PATH = 'data/Nguni/TEST/SADII.ZU.Morph_Lemma_POS.1.0.0.TEST.CTexT.TG.2021-09-30.txt'
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
SG = 1  # 1 = Skip-Gram, 0 = CBOW
EPOCHS = 10



    
def process_affix_line(line):
    tokens = []
    parts = line.strip().split('-')
    for part in parts:
        match = re.match(r"(.+?)\[(.+?)\]", part)
        if match:
            morpheme, tag = match.groups()
            tokens.append(f"{morpheme}_{tag}")
        else:
            tokens.append(part)  # fallback if no match
    return tokens

#each sent per <#line> break 
import re

def load_affix_sentences(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # check for line delimiter
            if line.startswith("<LINE#"):
                if current_sentence:
                    sentences.append(current_sentence)
                    # print("successfully appended sentence...")
                    current_sentence = []
                continue

            # morphological info in second col
            parts = line.split('\t')
            if len(parts) < 2:
                continue

            affix_form = parts[1]
            morphemes = []
            for part in affix_form.split('-'):
                match = re.match(r"(.+?)\[(.+?)\]", part)
                if match:
                    morpheme, tag = match.groups()
                    morphemes.append(f"{morpheme}_{tag}")
                else:
                    morphemes.append(part)

            current_sentence.extend(morphemes)

        # append last sentence
        if current_sentence:
            sentences.append(current_sentence)

    return sentences


    


def build_affix_dictionary(filepath):
    word_to_affix_map = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '' or line.startswith("<LINE#"):
                continue

            parts = line.strip().split('\t')
            if len(parts) >= 2:
                original_word = parts[0]
                affix_form = parts[1]

                # Convert something like "u[NPrePre15]-ku[BPre15]-hamb[VRoot]-is[CausExt]-a[VerbTerm]"
                # to ["u_NPrePre15", "ku_BPre15", ...]
                affix_tokens = affix_form.split('-')
                affix_tokens = [token.replace('[', '_').replace(']', '') for token in affix_tokens]

                word_to_affix_map[original_word] = affix_tokens

    return word_to_affix_map


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

print(f"Loading affix-tokenized sentences from {AFFIX_FILE_PATH}")
sentences = load_affix_sentences(AFFIX_FILE_PATH)
print(f"Loaded {len(sentences)} sentences")
for sent in sentences[:1]:
    print(sent)


print("Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=4,
    sg=SG,
    epochs=EPOCHS
)

#saving modle 
w2v_model.save(OUTPUT_MODEL_PATH)
print(f"Affix-based Word2Vec model saved to {OUTPUT_MODEL_PATH}")

#creating the dictionary which maps the surface level words to the tokens with affixes 
affixdictionary = build_affix_dictionary(AFFIX_FILE_PATH)
with open("data/Models/affix_dictionary_zulu.pkl", "wb") as f:
    pickle.dump(affixdictionary, f)
    print("Saved affix dictionary to pkl...")

affixtestdictionary = build_affix_dictionary(AFFIX_TEST_FILE_PATH)
with open("data/Models/affix_test_dictionary_zulu.pkl", "wb") as f:
    pickle.dump(affixtestdictionary, f)
    print("Saved affix TEST dictionary to pkl...")

#testing model 
query_word = "aba"  # can be affix or root from your dataset (ba fu ni)
if query_word in w2v_model.wv:
    print(f"\nMost similar to '{query_word}':")
    for word, score in w2v_model.wv.most_similar(query_word, topn=5):
        print(f"{word}: {score:.3f}")
else:
    print(f"\nWord '{query_word}' not in vocabulary.")

    






