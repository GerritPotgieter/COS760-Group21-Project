# This script is essentially the same as Eval.py , but the key difference is that instead of Zulu -> Xhosa, it evaluates Xhosa -> Zulu.
from gensim.models import KeyedVectors
import json
from random import sample

# Load aligned models (source: Xhosa, target: Zulu)
aligned_xho = KeyedVectors.load("data/Models/word2vec_xhosa_aligned.model")
aligned_zul = KeyedVectors.load("data/Models/word2vec_zulu_aligned.model")

# Load lexicon
with open("data/Lexicons/dsfsi-en-za-bilex_11-off-ZA-lang-updated.v.1.0 (1).json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

# Extract en→xho and en→zul lexicons
en_xho = {list(d.keys())[0]: d[list(d.keys())[0]][0] for d in lexicon["en-xho"] if list(d.values())[0]}
en_zul = {list(d.keys())[0]: d[list(d.keys())[0]][0] for d in lexicon["en-zul"] if list(d.values())[0]}

# Pivot on English: English → (Xho, Zul)
xho_zul_pairs = [(xho, en_zul[eng]) for eng, xho in en_xho.items() if eng in en_zul]

# Filter to only words present in both vocabularies
eval_pairs = [(x, z) for (x, z) in xho_zul_pairs if x in aligned_xho and z in aligned_zul]

# Evaluation
top1, top5, total = 0, 0, len(eval_pairs)

for xho_word, zul_word in eval_pairs:
    try:
        sims = aligned_zul.most_similar([aligned_xho[xho_word]], topn=5)
        preds = [w for w, _ in sims]
        if zul_word == preds[0]:
            top1 += 1
        if zul_word in preds:
            top5 += 1
    except KeyError:
        continue  # skip OOV words

print(f"Total eval pairs: {total}")
print(f"Top-1 accuracy: {top1 / total:.2%}")
print(f"Top-5 accuracy: {top5 / total:.2%}")

#Total eval pairs: 183
#Top-1 accuracy: 74.86%
#Top-5 accuracy: 85.79%
# Latest output from model run on 5/29/2025

print("\nSample word alignments (Xhosa → Zulu):\n")

# Sample 10 random alignments for manual inspection
for xho_word, zul_gold in sample(eval_pairs, 10):
    if xho_word not in aligned_xho:
        continue
    sims = aligned_zul.most_similar([aligned_xho[xho_word]], topn=5)
    print(f"Xhosa: {xho_word}")
    print(f"  Gold Zulu: {zul_gold}")
    for rank, (word, score) in enumerate(sims, start=1):
        indicator = "Best " if word == zul_gold else ""
        print(f"    {rank}. {word} (score: {score:.4f}) {indicator}")
    print("-" * 40)

#Sample word alignments (Xhosa → Zulu):

#Xhosa: ephakathi
#  Gold Zulu: kahle
#    1. kahle (score: 0.4743) Best
#    2. phakathi (score: 0.2758)
#    3. eduze (score: 0.2174)
#    4. ikhono (score: 0.1909)
#    5. isiyingi (score: 0.1871)