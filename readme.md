# Project: Optimizing Cross-Lingual Embeddings for isiZulu, Sepedi, and Setswana: Alignment Strategies, Pivot Selection, and Morphological Challenges.
![GitHub repo size](https://img.shields.io/github/repo-size/GerritPotgieter/COS760-Group21-Project)




In this project, we will attempt to address some concerns regarding morphological complexity
and suboptimal language pivot selection. 

To view our presentation to get a better overview you can find our presentation [here](https://drive.google.com/file/d/1TTT4m09AyaUwzokhb4xRswOxsDD5bpH7/view?usp=sharing) 

Group Members:
Gerrit Potgieter (u22508041)
Mihail Dicoski (u22495292)
Heinrich Niebuhr (u22555855)



## Setup:
```
python -m venv crosslingual-env
crosslingual-env\scripts\activate # windows
source venv/bin/activate #linux

pip install requirements.txt # this installs all dependencies used for the project
```

If you are having trouble installing dependencies try precompiled binary: 
```
pip install numpy pandas nltk gensim --only-binary :all:
```


## Running the code:
There are specific scripts that you can run to get our results, namely the eval.py, eval_reverse.py , ner_matrix.py , compare_affix.py

for ner_matrix, please ensure that you have downloaded the CUDA graphics processing from nvidia devoper sight and run the installer. Please ensure that you are using a pyhton 3.10 version

### Script to run code
From the root of the project 
```
python scripts/script_name.py

#for ner_matrix please use the following:
py -3.10 ner_matrix
```

#### Additional Note regarding allignment strategy
To obtain our VecMap results you have to use the provided files in [this](https://github.com/artetxem/vecmap) repo.

Use our zulu_xhosa_cleaned.txt as the dictionary and normalized models as the source and target embeddings.

Run this with the supervised command as provided in the VecMap repo.

#### Final note regarding NER task
 Care when running thi sfile as it uses alot of processing power and takes a while to run  and extremy long to finish ( over 2 hours )
