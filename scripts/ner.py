from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# Amharic (amh)                               these are the known languages in the masakhaner dataset
# Hausa (hau)
# Igbo (ibo)
# Kinyarwanda (kin)
# Luganda (kin)
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

languages = ["amh", 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol',  'yor' ] #zulu, xhosa, setswana 
conjunctive_languages = ['kin', 'lug', 'swa']  # languages with conjunctive properties
disjunctive_languages = ['amh', 'hau', 'ibo', 'luo', 'pcm', 'wol', 'yor']  # languages with disjunctive properties


datasets = {}
for lang in languages:
    datasets[lang] = load_dataset("masakhane/masakhaner2", lang)
    print(f"Loaded dataset for {lang}")