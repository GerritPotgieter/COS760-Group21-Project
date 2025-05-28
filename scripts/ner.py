from datasets import load_dataset

languages = ['zul', 'xho', 'tsn'] #zulu, xhosa, setswana 


datasets = {}
for lang in languages:
    datasets[lang] = load_dataset("masakhane/masakhaner2", lang)
    print(f"Loaded dataset for {lang}")