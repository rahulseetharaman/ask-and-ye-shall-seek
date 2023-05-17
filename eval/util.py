# This file contains utility functions to help with the evaluation process.

import pandas as pd
import json

def aggregate_scores():
    scored_dict = { 'ft5-base': [0, 0, 0, 0],
                    'bart': [0, 0, 0, 0],
                    't5_kldiv': [0, 0, 0, 0],
                    'ft5-large': [0, 0, 0, 0],
                    'ft5-xl': [0, 0, 0, 0], 
                    'ft5-small': [0, 0, 0, 0],
                    't5_canard': [0, 0, 0, 0]}
    # iterate over all annotation files
    for i in range(1, 5):
        with open(f"human_eval/annotator_{i}.csv") as f:
            df = pd.read_csv(f)
            annotation_df = df.dropna() 
            for index, row in annotation_df.iterrows():
                scored_dict[row['model']][0] += row['grammatical_correctness']
                scored_dict[row['model']][1] += row['diversity']
                scored_dict[row['model']][2] += row['clariscore']
                scored_dict[row['model']][3] += row['relevance']

    # average out the scores
    for model in scored_dict:
        for i in range(4):
            scored_dict[model][i] /= 80

    return scored_dict

if __name__ == "__main__":
    scored_dict = aggregate_scores()
    
    # store the results
    with open('human_eval/aggregated_results.json', 'w') as f:
        json.dump(scored_dict, f)