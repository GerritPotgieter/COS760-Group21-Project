from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support
import torch
# Amharic (amh)                               these are the known languages in the masakhaner dataset
# Hausa (hau)
# Igbo (ibo)
# Kinyarwanda (kin)
# Luganda (lug)
# Luo (luo)
# Nigerian-Pidgin (pcm)
# Swahili (swa)
# Wolof (wol)
# Yoruba (yor)

#my task for research
#  Pivot Language Selection: Why do mutual intelligibility metrics fail to predict
# embedding quality (e.g., isiXhosa→isiZulu outperforms the reverse)? Data
# Efficiency: How do conjunctive languages (e.g., isiZulu) scale with data size
# compared to disjunctive languages (e.g., Sesotho sa Leboa) in downstream tasks
# like NER?
languages = ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul']
conjunctive_languages = ['kin', 'lug', 'swa']
disjunctive_languages = ['amh', 'hau', 'ibo', 'luo', 'pcm', 'wol', 'yor']

def convert_ner_tags_to_strings(dataset_split):
    """Convert integer NER tags to string labels using ClassLabel mapping."""
    features = dataset_split.features["ner_tags"]
    if hasattr(features, "feature") and hasattr(features.feature, "int2str"):
        label_class = features.feature
        return dataset_split.map(
            lambda ex: {
                "ner_tags": [label_class.int2str(i) if isinstance(i, int) else str(i) for i in ex["ner_tags"]]
            },
            desc="Converting NER tags to string labels"
        )
    return dataset_split

def evaluate_ner(dataset, ner_pipeline):
    true_labels = []
    pred_labels = []

    for item in dataset:
        tokens = item["tokens"]
        # Convert to string and strip BIO prefixes
        labels = [
            label.split("-")[-1] if isinstance(label, str) and label != "O" else "O"
            for label in item["ner_tags"]
        ]

        try:
            preds = ner_pipeline(" ".join(tokens))
            pred = [p['entity_group'] for p in preds]
        except:
            pred = []

        true = labels[:len(pred)]
        true_labels.extend(true)
        pred_labels.extend(pred)

    return precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)


def run_pivot_transfer_with_large_model(source_lang, target_lang, ner_pipeline):
    print(f"\nEvaluating {source_lang.upper()} model on {target_lang.upper()} test data (large multilingual model)...")
    try:
        dataset = load_dataset("masakhane/masakhaner2", target_lang)
        test_data = convert_ner_tags_to_strings(dataset['test'])

        scores = evaluate_ner(test_data, ner_pipeline)
        print(f"{source_lang.upper()} → {target_lang.upper()} | Precision: {scores[0]:.4f}, Recall: {scores[1]:.4f}, F1: {scores[2]:.4f}")
    except Exception as e:
        print(f"Error during evaluation for {source_lang}→{target_lang}: {e}")

# Load large multilingual model once
model_name = "Davlan/xlm-roberta-large-masakhaner"
print(f"\nLoading multilingual model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example evaluation: Swahili → Luganda and Luganda → Swahili
source = "swa"
target = "lug"

run_pivot_transfer_with_large_model(source, target, ner_pipeline)
run_pivot_transfer_with_large_model(target, source, ner_pipeline)
# languages = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]
# conjunctive_languages = ['kin', 'lug', 'swa']
# disjunctive_languages = ['amh', 'hau', 'ibo', 'luo', 'pcm', 'wol', 'yor']

# def evaluate_ner(dataset, ner_pipeline):
#     true_labels = []
#     pred_labels = []

#     for item in dataset:
#         tokens = item["tokens"]
#         labels = item["ner_tags"]

#         try:
#             preds = ner_pipeline(" ".join(tokens))
#             pred = [p['entity_group'] for p in preds]
#         except:
#             pred = []

#         true = labels[:len(pred)]  # crude alignment
#         true_labels.extend(true)
#         pred_labels.extend(pred)

#     return precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

# def run_pivot_transfer_with_large_model(source_lang, target_lang, ner_pipeline):
#     print(f"\nEvaluating {source_lang.upper()} model on {target_lang.upper()} test data (large multilingual model)...")
#     try:
#         # Load target language dataset
#         test_data = load_dataset("masakhane/masakhaner2", target_lang)['test']
#         # Evaluate
#         scores = evaluate_ner(test_data, ner_pipeline)
#         print(f"{source_lang.upper()} → {target_lang.upper()} | Precision: {scores[0]:.4f}, Recall: {scores[1]:.4f}, F1: {scores[2]:.4f}")
#     except Exception as e:
#         print(f"Error during evaluation for {source_lang}→{target_lang}: {e}")

# # Load large multilingual model once
# model_name = "Davlan/xlm-roberta-large-masakhaner"
# print(f"\nLoading multilingual model: {model_name}")
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# # Example: Swahili → Luganda and Luganda → Swahili
# source = "swa"
# target = "lug"

# run_pivot_transfer_with_large_model(source, target, ner_pipeline)
# run_pivot_transfer_with_large_model(target, source, ner_pipeline)