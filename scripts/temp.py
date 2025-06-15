import json

with open("data/Lexicons/dsfsi-en-za-bilex_11-off-ZA-lang-updated.v.1.0 (1).json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

zul_xho = {list(d.keys())[0]: d[list(d.keys())[0]][0] for d in lexicon["en-zul"] if list(d.values())[0]}
xho_en = {list(d.keys())[0]: d[list(d.keys())[0]][0] for d in lexicon["en-xho"] if list(d.values())[0]}

zul_xho_pairs = []
for eng, zul in zul_xho.items():
    if eng in xho_en:
        xho_word = xho_en[eng]

        # Split by space and take only the first token in case of phrases
        zul_word_clean = zul.split()[0]
        xho_word_clean = xho_word.split()[0]

        print(f"Zulu: {zul_word_clean}, Xhosa: {xho_word_clean}")  # Debug print

        zul_xho_pairs.append((zul_word_clean, xho_word_clean))

with open('zulu_xhosa_cleaned.txt', 'w', encoding='utf-8') as f_out:
    for zul_word, xho_word in zul_xho_pairs:
        f_out.write(f"{zul_word} {xho_word}\n")
