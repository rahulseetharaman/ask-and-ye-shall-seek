import sys


from utils import convert_to_json
from metric.evaluator import get_evaluator
import pandas as pd
import json
import sys
import numpy as np

task = 'dialogue'


data = json.load(open(f"../abg-coqa/coqa_abg_val.json"))

all_conversations = data['data']
context_list = []
src_list = []
for conv in all_conversations:
    if conv['ambiguity'] == 'ambiguous':
        h = list(sorted(conv['history_turns'], key=lambda x: x['turn_id']))
        history = '\n'.join([c['question'] + '\n' + c['answer'] for c in h]) + '\n' + conv['target_turn']['question']
        for k,v in conv.items():
            if 'clarification' in k:
                src_list.append(history+v['question'])
                context_list.append(history)

print(len(src_list))
print(len(context_list))
generated = open("./answers.txt").readlines()

src_list = [q for ques in src_list for q in [ques] * 5]
context_list = [q for context in context_list for q in [context] * 5]
output_list = generated
print(len(src_list), len(context_list), len(output_list))

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, 
                       src_list=src_list, context_list=context_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get multi-dimensional evaluation scores
eval_scores = evaluator.evaluate(data, print_result=True)