import pandas as pd
from evaluate import load
import pickle
import numpy as np
import json

bleu = load("bleu")
bertscore = load("bertscore")
rouge = load("rouge")
sacrebleu = load("sacrebleu")


def evaluate_metrics(model_name):
    with open(f"../preds/{model_name}_preds.json") as f:
        pred_samples = json.load(f)

    generated = [y_pred for sample in pred_samples for y_pred in sample['y_preds'] ]
    refs = [[y_true] for sample in pred_samples for y_true in [sample['y_true'], sample['y_true']] ]

    metrics = []

    metric = bleu.compute(predictions=generated, references=refs)
    print(metric)
    pickle.dump(metric, open(f"{model_name}_test_bleu.pkl", "wb"))
    metrics.append(metric)

    metric = rouge.compute(predictions=generated, references=refs)
    print(metric)
    pickle.dump(metric, open(f"{model_name}_test_rouge.pkl", "wb"))
    metrics.append(metric)

    metric = sacrebleu.compute(predictions=generated, references=refs)
    print(metric)
    pickle.dump(metric, open(f"{model_name}_test_sacrebleu.pkl", "wb"))
    metrics.append(metric)

    # flatten refs for bertscore
    refs = [r[0] for r in refs]

    metric = bertscore.compute(predictions=generated, references=refs, model_type="microsoft/deberta-xlarge-mnli")
    pickle.dump(metric, open(f"{model_name}_test_bertscore.pkl", "wb"))
    metrics.append(metric)

    with open(f'{model_name}_metrics', 'w') as f:
        json.dump(metrics, f)

if __name__== '__main__':
    evaluate_metrics('ft5-base')