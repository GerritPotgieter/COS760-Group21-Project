import os

# This script extracts morpheme (affix-based) tokens from the Nguni datasets.
# It reads the TRAIN file, splits the second column (morpheme decomposition),
# and saves each sentence to a new file with morphemes tokenized.

TRAIN_FILE = 'data/Nguni/TRAIN/SADII.ZU.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt'  # set for xhosa
OUTPUT_FILE = 'data/Processed/affix_sentences_ZULU.txt'  # Desired output file

def extract_affix_tokens(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("<LINE#"):  # Sentence boundary
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            columns = line.split('\t')
            if len(columns) < 2:
                continue

            morphemes = columns[1]  # 2nd column = morphological decomposition
            morpheme_tokens = morphemes.split('+')  # e.g. 'ba+fund+i' -> ['ba', 'fund', 'i'] ;  splits into prefix root suffix 

            current_sentence.extend(morpheme_tokens)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

# Test the extraction
sentences = extract_affix_tokens(TRAIN_FILE)
print(sentences[0])  # Preview first sentence

# Save to file
def save_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(' '.join(sentence) + '\n')

save_sentences(sentences, OUTPUT_FILE)
print(f"Affix-based sentences saved to {OUTPUT_FILE}")
