from gensim.models import Word2Vec

FILE_PATH = 'data/Processed/sentences_XHOSA.txt' # Change this to the path of your processed text file
OUTPUT_FILE = 'data/Models/word2vec_xhosa.model' # Change this to the desired output filepath

# Load the saved sentences
def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

sentences = load_sentences(FILE_PATH) #change to whatever processed text you wan to embed

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
w2v_model.save(OUTPUT_FILE) # Change the path as needed
print("Word2Vec model saved.")

# Load the model
model = Word2Vec.load(OUTPUT_FILE)

# Example query
print(model.wv.most_similar("zabahlali", topn=5))

#Results : [('aqinisekise', 0.939693033695221), ('nomphakathi', 0.9326698184013367), ('kungenzeka', 0.9323709607124329), ('akwazi', 0.9298385381698608), ('abe', 0.9297009706497192)]
# 1. They ensure/certify -> indicates responsibility or action taken in relation to the community
# 2. And the community -> indicates a connection or relationship with the community
# 3. It is possible -> indicates potential or capability in relation to the community
# 4. Can -> indicates ability or possibility in relation to the community
# 5. They are -> indicates existence or presence in relation to the community
# This shows that the model captures semantic relationships and context around the word "umphakathi" (community).