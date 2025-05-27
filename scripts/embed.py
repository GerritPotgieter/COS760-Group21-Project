from gensim.models import Word2Vec

# Load the saved sentences
def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

sentences = load_sentences('data/Processed/sentences_ZULU.txt') #change to whatever processed text you wan tto embed

# Train Word2Vec
w2v_model = Word2Vec(
    sentences,
    vector_size=100,   # Dimensionality of the word vectors
    window=5,          # Context window size
    min_count=2,       # Ignores words with total frequency lower than this
    workers=4,         # Number of threads
    sg=1,              # Use skip-gram (sg=1), CBOW if 0
    epochs=10
)

# Save the model
w2v_model.save("data/Models/word2vec_zulu.model") # Change the path as needed
print("Word2Vec model saved.")

# Load the model
model = Word2Vec.load("data/Models/word2vec_zulu.model")

# Example query
print(model.wv.most_similar("umphakathi", topn=5))

#Results : [('aqinisekise', 0.939693033695221), ('nomphakathi', 0.9326698184013367), ('kungenzeka', 0.9323709607124329), ('akwazi', 0.9298385381698608), ('abe', 0.9297009706497192)]
# 1. They ensure/certify -> indicates repsonbility or action taken in relation to the community
# 2. And the community -> indicates a connection or relationship with the community
# 3. It is possible -> indicates potential or capability in relation to the community
# 4. Can -> indicates ability or possibility in relation to the community
# 5. They are -> indicates existence or presence in relation to the community
# This shows that the model captures semantic relationships and context around the word "umphakathi" (community).