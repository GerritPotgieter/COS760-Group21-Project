import os
#This script extracts surface tokens from a the Nguni datasets given
#As far as I know the formats are similiar and this script should work for all Nguni languages
#The script reads the TRAIN file, extracts the first word token from each line,
#and saves the sentences to a new file in a simple format

TRAIN_FILE = 'data/Nguni/TRAIN/SADII.XH.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt' #Change this to the filepath of the TRAIN file you want to process
OUTPUT_FILE = 'data/Processed/sentences_XHOSA.txt' #Change this to the desired output filepath
def extract_surface_tokens(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("<LINE#"): #Skip empty lines and headers
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            #Extract first word token (far left)
            surface_token = line.split('\t')[0]
            current_sentence.append(surface_token)

    #get the actual sentences
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

#test if the function works and actually pulls tokens
sentences = extract_surface_tokens(TRAIN_FILE)
print(sentences[0]) 

# Function to save sentences to a file
def save_sentences(sentences, OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(' '.join(sentence) + '\n')

# Save the extracted sentences to the output file
save_sentences(sentences, OUTPUT_FILE)
print(f"Sentences saved to {OUTPUT_FILE}")