from gensim.models import Word2Vec
import os


AFFIX_FILE_PATH = 'data/Processed/affixes_sentences_XHOSA.txt'  
OUTPUT_MODEL_PATH = 'data/Models/word2vec_xhosa_affixes.model' 
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
SG = 1  # 1 = Skip-Gram, 0 = CBOW
EPOCHS = 10

#loading sentences
def load_affix_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

print(f"Loading affix-tokenized sentences from {AFFIX_FILE_PATH}")
sentences = load_affix_sentences(AFFIX_FILE_PATH)
print(f"Loaded {len(sentences)} sentences")


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

#testing model 
query_word = "ba"  # can be affix or root from your dataset (ba fu ni)
if query_word in w2v_model.wv:
    print(f"\nMost similar to '{query_word}':")
    for word, score in w2v_model.wv.most_similar(query_word, topn=5):
        print(f"{word}: {score:.3f}")
else:
    print(f"\nWord '{query_word}' not in vocabulary.")
