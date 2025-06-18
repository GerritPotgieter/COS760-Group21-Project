from gensim.models import KeyedVectors
import json
import random
from random import sample

# This script evaluates the alignment of Zulu and Xhosa word embeddings
# by checking how well Zulu words can be predicted from Xhosa embeddings.

# Load aligned models
aligned_zul = KeyedVectors.load("data/Models/word2vec_zulu_aligned.model")
aligned_xho = KeyedVectors.load("data/Models/word2vec_xhosa_aligned.model")

# Load lexicon
with open("data/Lexicons/dsfsi-en-za-bilex_11-off-ZA-lang-updated.v.1.0 (1).json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

zul_xho = {list(d.keys())[0]: d[list(d.keys())[0]][0] for d in lexicon["en-zul"] if list(d.values())[0]}
xho_en = {list(d.keys())[0]: d[list(d.keys())[0]][0] for d in lexicon["en-xho"] if list(d.values())[0]}

# Pivot on English
zul_xho_pairs = [(zul, xho_en[eng]) for eng, zul in zul_xho.items() if eng in xho_en]



# Filter to only words in both vocabularies
eval_pairs = [(z, x) for (z, x) in zul_xho_pairs
              if z in aligned_zul and x in aligned_xho]

# Evaluation
top1, top5, total = 0, 0, len(eval_pairs)

for zul_word, xho_word in eval_pairs:
    try:
        sims = aligned_xho.most_similar([aligned_zul[zul_word]], topn=5)
        preds = [w for w, _ in sims]
        if xho_word == preds[0]:
            top1 += 1
        if xho_word in preds:
            top5 += 1
    except KeyError:
        continue  # skip OOV words

print(f"Total eval pairs: {total}")
print(f"Top-1 accuracy: {top1 / total:.2%}")
print(f"Top-5 accuracy: {top5 / total:.2%}")

#Total eval pairs: 183
#Top-1 accuracy: 74.86%
#Top-5 accuracy: 85.79%
#Latest output from model run on 5/29/2025

print("\nSample word alignments (Zulu → Xhosa):\n")

# Randomly select 10 pairs for inspection
# The output produced here is for manual inspection and validation
# Would recommend actually the words and their meanings between the two langs to evaluate accuracy properly
for zul_word, xho_gold in sample(eval_pairs, 10):
    if zul_word not in aligned_zul:
        continue
    sims = aligned_xho.most_similar([aligned_zul[zul_word]], topn=5)
    print(f"Zulu: {zul_word}")
    print(f"  Gold Xhosa: {xho_gold}")
    for rank, (word, score) in enumerate(sims, start=1):
        indicator = "Best " if word == xho_gold else ""
        print(f"    {rank}. {word} (score: {score:.4f}) {indicator}")
    print("-" * 40)


# Sample output:

# Sample word alignments (Zulu → Xhosa):
#Zulu: imbangela
#  Gold Xhosa: unobangela
#    1. iziphumo (score: 0.3992) 
#    2. ingxelo (score: 0.3093) 
#    3. unobangela (score: 0.2924) Best 
#    4. usapho (score: 0.2550) 
#    5. funda (score: 0.2473) 