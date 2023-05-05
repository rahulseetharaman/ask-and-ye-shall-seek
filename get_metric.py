import pandas as pd
from evaluate import load
import pickle
import numpy as np
import sys

cache_dir="/work/pi_hzamani_umass_edu/rseetharaman"

bertscore = load("bertscore", cache_dir=cache_dir)

generated = open("./answers.txt").readlines()

df=pd.read_csv("./val_bart.txt", sep='\t', header=None)
refs = df.iloc[:, 1].values.tolist()
refs = [[r]*5 for r in refs]
refs = [sent for r in refs for sent in r]
metric = bertscore.compute(predictions=generated, references=refs, model_type="microsoft/deberta-xlarge-mnli")
pickle.dump(metric, open("bertscore.pkl", "wb"))


refs = df.iloc[:, 1].values
generated = np.array(generated).reshape(-1,10).tolist()
refs = np.array(refs).reshape(-1,2).tolist()

x = []
y = []

for i, g in enumerate(generated): 
    gens = list(g)
    for g in gens:
        x.append(g)
        y.append(refs[i])

bleu = load("bleu", cache_dir=cache_dir)
metric = bleu.compute(predictions=x, references=y)
pickle.dump(metric, open("bleu.pkl", "wb"))


rouge = load("rouge", cache_dir=cache_dir)
metric = rouge.compute(predictions=x, references=y)
pickle.dump(metric, open("rouge.pkl", "wb"))


rouge = load("sacrebleu", cache_dir=cache_dir)
metric = rouge.compute(predictions=x, references=y)
pickle.dump(metric, open("sacrebleu.pkl", "wb"))
