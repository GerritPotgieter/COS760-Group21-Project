from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import string
import os


INPUT_MODEL_PATH = 'data/Models/word2vec_xhosa.model'  #Take the model you want to normalize
OUTPUT_MODEL_PATH = 'data/Models/word2vec_xhosa_normalized.model' #output file name and path
RAW_SENTENCES_PATH = 'data/Processed/sentences_XHOSA.txt' #take OG sentences
MIN_COUNT = 2  # optional: remove very rare words

def preprocess_sentences(sentences):
    processed = []
    for sentence in sentences:
        clean_tokens = []
        for token in sentence.strip().split(): #take away whitespace 
            token = token.lower() #lowercase the token
            token = ''.join([c for c in token if c.isalpha()])  # remove punctuation/digits
            if token:
                clean_tokens.append(token)
        if clean_tokens:
            processed.append(clean_tokens)
    return processed


with open(RAW_SENTENCES_PATH, 'r', encoding='utf-8') as f: #load the raw sentences
    raw_sentences = f.readlines() #just reads the content

#call the function to process the sentences
sentences = preprocess_sentences(raw_sentences)

#call Word2Vec and train model
normalized_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=MIN_COUNT,
    workers=4,
    sg=1  # Skip-gram
)

#Save normalized model to the specified path
normalized_model.save(OUTPUT_MODEL_PATH)
print(f"Normalized model saved to {OUTPUT_MODEL_PATH}")
