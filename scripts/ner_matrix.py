from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import time

# MasakhaNER2-supported languages
languages = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]

# Evaluate pipeline on a test dataset
def evaluate_ner(dataset, pipeline):
    true_labels = []
    pred_labels = []

    for item in dataset:
        tokens = item["tokens"]
        labels = item["ner_tags"]

        try:
            preds = pipeline(" ".join(tokens))
            pred = [p['entity_group'] for p in preds]
        except Exception:
            pred = []

        true = labels[:len(pred)]
        true_labels.extend(true)
        pred_labels.extend(pred)

    if not pred_labels:
        return 0.0, 0.0, 0.0

    return precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

# Main function to evaluate using one large multilingual model across all languages
def run_multilingual_pivot_transfer(languages):
    results = pd.DataFrame(index=languages, columns=languages)

    # Load the single multilingual model once
    model_name = "Davlan/xlm-roberta-large-masakhaner"
    print(f"\nLoading multilingual model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    for source_lang in languages:
        for target_lang in languages:
            try:
                print(f"Evaluating {source_lang} → {target_lang}...")
                dataset = load_dataset("masakhane/masakhaner2", target_lang)['test']
                precision, recall, f1 = evaluate_ner(dataset, ner_pipe)
                results.loc[source_lang, target_lang] = round(f1, 3)
            except Exception as e:
                print(f"⚠️ Skipping {source_lang} → {target_lang}: {e}")
                results.loc[source_lang, target_lang] = None
            time.sleep(1)  # Respect Hugging Face API

    return results

# Run evaluation
matrix = run_multilingual_pivot_transfer(languages)

# Display and save
print("\nF1 Score Pivot Transfer Matrix (single multilingual model):")
print(matrix)

matrix.to_csv("pivot_transfer_matrix_multilingual.csv")

