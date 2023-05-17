# This file generates a CSV file with the following columns:
# id, model, cq, grammatical_correctness, diversity, clariscore, relevance

import csv
import json
import random

def generate_template():
    with open('../data/coqa_abg_test.json') as f:
        data = json.load(f)
    with open('../preds/ft5-small_preds.json') as f:
        ft5_small_preds = json.load(f)
    with open('../preds/ft5-base_preds.json') as f:
        ft5_base_preds = json.load(f)
    with open('../preds/ft5-large_preds.json') as f:
        ft5_large_preds = json.load(f)
    with open('../preds/ft5-xl_preds.json') as f:
        ft5_xl_preds = json.load(f)
    with open('../preds/bart_data_aug.json') as f:
        bart_preds = json.load(f)
    with open('../preds/t5_canard.json') as f:
        t5_canard_preds = json.load(f)
    with open('../preds/t5_kldiv_data_aug.json') as f:
        t5_kldiv_preds = json.load(f)

    pred_dict = {'ft5-small': ft5_small_preds, 'ft5-base': ft5_base_preds, 'ft5-large': ft5_large_preds, 'ft5-xl': ft5_xl_preds, 'bart': bart_preds, 't5_canard': t5_canard_preds, 't5_kldiv': t5_kldiv_preds}

    model_list = ['ft5-small', 'ft5-base', 'ft5-large', 'ft5-xl', 'bart', 't5_canard', 't5_kldiv']

    data_index = 0
    # write to reference file
    with open('reference-rohan.txt', 'w') as ref:
        with open('annotate-rohan.csv', 'w') as ann:
            csv_writer = csv.writer(ann)
            csv_writer.writerow(['id', 'model', 'cq', 'grammatical_correctness', 'diversity', 'clariscore', 'relevance'])
            for i in range(90, 123):
                while(data['data'][data_index]['ambiguity'] != 'ambiguous'):
                    data_index += 1
                
                # write to reference file
                ref.write(f"{data['data'][data_index]['id']}\n")
                ref.write(f"{ft5_xl_preds[i]['x']}\n")
                ref.write(f"Ground Truth: {ft5_xl_preds[i]['y_true']}\n\n\n")

                # shuffle model list
                random.shuffle(model_list)

                # write to annotation file
                for model in model_list:
                    csv_writer.writerow([data['data'][data_index]['id'], model, pred_dict[model][i]['y_preds'][0], '', '', '', ''])
                    csv_writer.writerow([data['data'][data_index]['id'], model, pred_dict[model][i]['y_preds'][1], '', '', '', ''])
                    csv_writer.writerow([])

                csv_writer.writerow([])
                data_index += 1

if __name__ == "__main__":
    generate_template()