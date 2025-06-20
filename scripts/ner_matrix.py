
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline , XLMRobertaTokenizer, XLMRobertaForTokenClassification
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import torch
import time


# ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul']
languages = ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul']
# conjunctive_languages = ['kin', 'lug', 'swa']
# disjunctive_languages = ['amh', 'hau', 'ibo', 'luo', 'pcm', 'wol', 'yor']

# def convert_ner_tags_to_strings(dataset_split):
#     features = dataset_split.features["ner_tags"]
#     if hasattr(features, "feature") and hasattr(features.feature, "int2str"):
#         label_class = features.feature
#         return dataset_split.map(
#             lambda ex: {"ner_tags": [label_class.int2str(i) if isinstance(i, int) else i for i in ex["ner_tags"]]},
#             desc="Converting NER tags to string labels"
#         )
#     return dataset_split
def convert_ner_tags_to_strings(dataset_split):
    label_list = dataset_split.features["ner_tags"].feature.names
    return dataset_split.map(
        lambda ex: {"ner_tags": [label_list[i] for i in ex["ner_tags"]]},
        desc="Converting NER tags to string labels"
    )

def evaluate_ner(dataset, ner_pipeline, batch_size=16):
    true_labels = []
    pred_labels = []
    texts = [" ".join(item["tokens"]) for item in dataset]
    all_true = [item["ner_tags"] for item in dataset]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_true = all_true[i:i + batch_size]

        try:
            preds = ner_pipeline(batch_texts)
            if isinstance(preds[0], list):
                batch_pred = [[str(p['entity_group']) for p in sentence] for sentence in preds]
            else:
                batch_pred = [[str(p['entity_group']) for p in preds]]
        except:
            batch_pred = [[] for _ in batch_texts]

        for true_seq, pred_seq in zip(batch_true, batch_pred):
            # Convert all true labels to strings just in case
            true_seq = [str(label) for label in true_seq[:len(pred_seq)]]
            true_labels.extend(true_seq)
            pred_labels.extend(pred_seq)

        # print("Sample true:", true_labels[:5])
        # print("Sample pred:", pred_labels[:5])
        # print("Types:", type(true_labels[0]), type(pred_labels[0]))

    return precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)


def run_pivot_transfer_with_large_model(source_lang, target_lang, ner_pipeline):
    print(f"\nEvaluating {source_lang.upper()} model on {target_lang.upper()} test data (with GPU + batching)...")
    try:
        dataset = load_dataset("masakhane/masakhaner2", target_lang)
        test_data = convert_ner_tags_to_strings(dataset['test'])
        start = time.time()
        scores = evaluate_ner(test_data, ner_pipeline)
        end = time.time()
        print(f"{source_lang.upper()} → {target_lang.upper()} | Precision: {scores[0]:.4f}, Recall: {scores[1]:.4f}, F1: {scores[2]:.4f} | Time: {end - start:.2f}s")
        return scores[2]  # Return F1 for matrix
    except Exception as e:
        print(f"Error during evaluation for {source_lang}→{target_lang}: {e}")
        return 0.0

# Load large multilingual model once with GPU support
model_name = "Davlan/xlm-roberta-large-masakhaner"    #change models here
#model_name = "AfroXLM-R-large"
device = 0 if torch.cuda.is_available() else -1 
print(f"\nLoading multilingual model on GPU: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

# Matrix for all language pairs
import numpy as np
f1_matrix = np.zeros((len(languages), len(languages)))

for i, src in enumerate(languages):
    for j, tgt in enumerate(languages):
        if src != tgt:
            f1 = run_pivot_transfer_with_large_model(src, tgt, ner_pipeline)
            f1_matrix[i, j] = f1
        else:
            f1_matrix[i, j] = np.nan  # skip self-transfer

# Display results as a 10x10 matrix
import pandas as pd
f1_df = pd.DataFrame(f1_matrix, index=languages, columns=languages)
print("\nF1-score Matrix (rows = source, cols = target):")
print(f1_df.round(3))













# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# from datasets import load_dataset
# from sklearn.metrics import precision_recall_fscore_support
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define language groups
# languages = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]

# # isiZulu to Sesotho may perform poorly due to lack of shared morphology or script divergence.

# # Convert NER tag ints to strings
# def convert_ner_tags_to_strings(dataset_split):
#     features = dataset_split.features["ner_tags"]
#     if hasattr(features, "feature") and hasattr(features.feature, "int2str"):
#         label_class = features.feature
#         return dataset_split.map(
#             lambda ex: {"ner_tags": [label_class.int2str(i) if isinstance(i, int) else i for i in ex["ner_tags"]]},
#             desc="Converting NER tags to string labels"
#         )
#     return dataset_split

# # Evaluate model
# def evaluate_ner(dataset, ner_pipeline):
#     true_labels = []
#     pred_labels = []

#     for item in dataset:
#         tokens = item["tokens"]
#         labels = [
#             label.split("-")[-1] if isinstance(label, str) and label != "O" else "O"
#             for label in item["ner_tags"]
#         ]

#         try:
#             preds = ner_pipeline(" ".join(tokens))
#             pred = [p['entity_group'] for p in preds]
#         except:
#             pred = []

#         true = labels[:len(pred)]  # crude alignment
#         true_labels.extend(true)
#         pred_labels.extend(pred)

#     return precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

# # Load multilingual model once
# model_name = "Davlan/xlm-roberta-large-masakhaner"
# print(f"\nLoading multilingual model: {model_name}")
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# # Initialize score matrices
# precision_matrix = pd.DataFrame(np.zeros((10, 10)), index=languages, columns=languages)
# recall_matrix = pd.DataFrame(np.zeros((10, 10)), index=languages, columns=languages)
# f1_matrix = pd.DataFrame(np.zeros((10, 10)), index=languages, columns=languages)

# # Evaluate all pairs
# for source in languages:
#     for target in languages:
#         if source == target:
#             continue
#         print(f"\nEvaluating {source.upper()} model on {target.upper()} test data...")
#         try:
#             dataset = load_dataset("masakhane/masakhaner2", target)
#             test_data = convert_ner_tags_to_strings(dataset['test'])
#             scores = evaluate_ner(test_data, ner_pipeline)
#             precision_matrix.loc[source, target] = scores[0]
#             recall_matrix.loc[source, target] = scores[1]
#             f1_matrix.loc[source, target] = scores[2]
#             print(f"{source.upper()} → {target.upper()} | Precision: {scores[0]:.4f}, Recall: {scores[1]:.4f}, F1: {scores[2]:.4f}")
#         except Exception as e:
#             print(f"Error for {source}→{target}: {e}")
#             precision_matrix.loc[source, target] = recall_matrix.loc[source, target] = f1_matrix.loc[source, target] = np.nan

# # Display matrices
# print("\n=== Precision Matrix ===")
# print(precision_matrix.round(3))

# print("\n=== Recall Matrix ===")
# print(recall_matrix.round(3))

# print("\n=== F1-score Matrix ===")
# print(f1_matrix.round(3))

# # Optional: Heatmap visualization
# def plot_heatmap(matrix, title):
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, cbar_kws={"label": title})
#     plt.title(title)
#     plt.xlabel("Target Language")
#     plt.ylabel("Source Language")
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show()

# plot_heatmap(f1_matrix, "F1-score Heatmap")
# # plot_heatmap(precision_matrix, "Precision Heatmap")
# # plot_heatmap(recall_matrix, "Recall Heatmap")
