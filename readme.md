# Project: Optimizing Cross-Lingual Embeddings for isiZulu, Sepedi, and Setswana: Alignment Strategies, Pivot Selection, and Morphological Challenges.

In this project, we will attempt to address some concerns regarding morphological complexity
and suboptimal language pivot selection. 

Group Members:
Gerrit Potgieter (u22508041)
Mihail Dicoski (u22495292)
Heinrich Niebuhr (u22555855)



## Setup:
```
python -m venv crosslingual-env
crosslingual-env\scripts\activate

pip install requirements.txt # this installs all dependencies used for the project
```

If you are having trouble installing dependencies try precompiled binary: 
```
pip install numpy pandas nltk gensim --only-binary :all:
```


## Running the code:
There are specific scripts that you can run to get our results, namely the eval.py, eval_reverse.py , ner_matrix.py , compare_affix.py

### Script to run code
```
python3 script_name.py
```

