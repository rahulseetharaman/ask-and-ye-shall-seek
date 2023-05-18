from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
import pandas as pd

class ABGCoQADataset(Dataset):
    def __init__(self, data_file, tokenizer):
        super().__init__()
        self.dataset = self._build_data(data_file)
        self.contexts = [d[0] for d in self.dataset]
        self.cqs = [d[1] for d in self.dataset]
        encodings = tokenizer.prepare_seq2seq_batch(self.contexts, self.cqs, max_length=512, padding=True, truncation=True, return_tensors="pt")
        self.encoder_input_ids = encodings.input_ids
        self.encoder_attention_mask = encodings.attention_mask
        self.decoder_input_ids = encodings.labels[:, :-1].clone()  # skip last
        self.labels = encodings.labels[:, 1:].clone()              # skip first

    def _build_data(self, data_file):
        dataset = []
        with open(f"data/coqa_abg/{data_file}") as f:
            coqa_data = json.load(f)

        # create a dict of dicts by id
        coqa_data_dict = {d['id']: d for d in coqa_data['data']}

        if 'train' in data_file:
            for i in range(15):
                with open(f"data/data_aug/output_{i}.json") as f:
                    output_data = json.load(f)
                    for output in output_data:
                        story = coqa_data_dict[output['id']]['story']
                        ambiguous_question = coqa_data_dict[output['id']]['target_turn']['question']
                        history_turns = '\n'.join([ f"{turn['question']}\n{turn['answer']}" for turn in coqa_data_dict[output['id']]['history_turns']])
                        clarification_question_list = [coqa_data_dict[output['id']]['clarification_turn']['question']]

                        if output['new_questions'].find('Clarifying Questions') != -1:
                            output['new_questions'] = output['new_questions'][output['new_questions'].find('Clarifying Questions') + len('Clarifying Questions'):]

                        start = -1
                        for index, letter in enumerate(output['new_questions']):
                            if letter.isupper() and start == -1:
                                start = index
                            elif letter == '?':
                                end = index
                                clarification_question_list.append(output['new_questions'][start: end + 1])
                                start = -1

                        for cq in clarification_question_list:
                            dataset.append([f"Story:\n{story}\nHistory:\n{history_turns}\nAmbiguous Question:\n{ambiguous_question}", cq])

        else:

            for data in coqa_data['data']:
                if data['ambiguity'] == 'ambiguous':
                    story = data['story']
                    ambiguous_question = data['target_turn']['question']
                    history_turns = '\n'.join([ f"{turn['question']}\n{turn['answer']}" for turn in data['history_turns']])
                    clarification_question = data['clarification_turn']['question']

                    dataset.append([f"Story:\n{story}\nHistory:\n{history_turns}\nAmbiguous Question:\n{ambiguous_question}", clarification_question])

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encoder_input_ids[item], self.encoder_attention_mask[item], self.decoder_input_ids[item], self.labels[item] 
    
if __name__ == "__main__":
    data_obj = ABGCoQADataset('coqa_abg_test.json', 'test')
    print(len(data_obj.dataset))