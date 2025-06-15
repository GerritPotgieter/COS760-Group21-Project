import numpy as np
from gensim.models import Word2Vec
from numpy.linalg import norm

def export_normalized_vec(model_path, output_path):
    model = Word2Vec.load(model_path)
    vectors = model.wv.vectors
    words = model.wv.index_to_key
    
    # Normalize vectors (L2)
    normalized_vectors = vectors / np.expand_dims(norm(vectors, axis=1), axis=1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {model.vector_size}\n")
        for word, vector in zip(words, normalized_vectors):
            vector_str = " ".join(map(str, vector))
            f.write(f"{word} {vector_str}\n")

    print(f"Normalized vectors exported to {output_path}")
if __name__ == "__main__":
    # Example usage
    ZULU_MODEL_PATH = "data/Models/word2vec_zulu.model"
    XHOSA_MODEL_PATH = "data/Models/word2vec_xhosa.model"
    
    ZULU_OUTPUT_PATH = "data/Models/word2vec_zulu_normalized.vec"
    XHOSA_OUTPUT_PATH = "data/Models/word2vec_xhosa_normalized.vec"
    
    export_normalized_vec(ZULU_MODEL_PATH, ZULU_OUTPUT_PATH)
    export_normalized_vec(XHOSA_MODEL_PATH, XHOSA_OUTPUT_PATH)