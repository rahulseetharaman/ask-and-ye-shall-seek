import pandas as pd
from evaluate import load
import pickle
import numpy as np
import sys

bleu = load("bleu")
# bertscore = load("bertscore")
# rouge = load("rouge")
# sacrebleu = load("sacrebleu")


for split in ['test', 'val', 'train']:

    generated = open(f"./{split}_answers.txt").readlines()
    df=pd.read_csv(f"./{split}_bart.txt", sep='\t', header=None)
    refs = df.iloc[:, 1].values.tolist()
    refs = [[r]*5 for r in refs]
    refs = [sent for r in refs for sent in r]

    # metric = bertscore.compute(predictions=generated, references=refs, model_type="microsoft/deberta-xlarge-mnli")
    # pickle.dump(metric, open(f"{split}_bertscore.pkl", "wb"))

    refs = df.iloc[:, 1].values
    generated = np.array(generated).reshape(-1,5).tolist()

    x = []
    y = []

    for i, g in enumerate(generated): 
        gens = list(g)
        for g in gens:
            x.append(g)
            y.append(refs[i])

    
    metric = bleu.compute(predictions=x, references=y)
    print(metric)
    pickle.dump(metric, open(f"{split}_bleu.pkl", "wb"))

    # metric = rouge.compute(predictions=x, references=y)
    # print(metric)
    # pickle.dump(metric, open(f"{split}_rouge.pkl", "wb"))

    # metric = sacrebleu.compute(predictions=x, references=y)
    # print(metric)
    # pickle.dump(metric, open(f"{split}_sacrebleu.pkl", "wb"))

