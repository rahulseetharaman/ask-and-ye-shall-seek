import pandas as pd
import numpy as np
import json
from pprint import pprint
import sys
import random

'''

USAGE:
python preproc.py {train, val, test}
This script is used to preprocess data for instruction tuning a T5 model.
'''

data = json.load(open(f"../abg-coqa/coqa_abg_{sys.argv[1]}.json"))

all_conversations = data['data']
amb_conversations = []
for conv in all_conversations:
    if conv['ambiguity'] == 'ambiguous':
        amb_conversations.append(conv)

conv_text = []


def construct_template(story, user_aq, history_turns):
    prev_utterances = ''
    for h in history_turns: 
        prev_utterances += "{} [ENDOFTURN] {} [ENDOFTURN]".format(h['question'], h['answer'])
    template = f'''Input: [CONTEXT] {story} [ENDOFTURN] {prev_utterances} {user_aq} [ENDOFDIALOGUE] [QUESTION]'''
    return template

for conv in amb_conversations: 
    bot_story = conv['story']
    user_aq = conv['target_turn']['question']
    history_turns = sorted(conv['history_turns'], key=lambda x: x['turn_id'])
    for c in conv.keys():
        if 'clarification' in c:
            bot_cq = conv[c]['question']
            context = construct_template(bot_story, user_aq, history_turns)
            conv_text.append([context, bot_cq])

f=open(f"{sys.argv[1]}_bart.txt", "w")

for c in conv_text:
    context = c[0].replace('\n', '').strip()
    cq = c[1].replace('\n', '').strip()
    f.write(f"{context}\t{cq}\n")

f.close()




