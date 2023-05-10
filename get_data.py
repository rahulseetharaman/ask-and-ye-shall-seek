# from torch.utils.data import Dataset
# from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
import pandas as pd

class ABGCoQADataset():
    def __init__(self):
        super().__init__()
        # self.all_texts = pd.read_csv(data_path, sep='\t', header=None)
        # self.contexts = self.all_texts.iloc[:, 0].values.tolist()
        # self.clques = self.all_texts.iloc[:, 1].values.tolist()
        # encodings = tokenizer.prepare_seq2seq_batch(self.contexts, self.clques, max_length=512, padding=True, truncation=True, return_tensors="pt")
        # self.encoder_input_ids = encodings.input_ids
        # self.encoder_attention_mask = encodings.attention_mask
        # self.decoder_input_ids = encodings.labels[:, :-1].clone()  # skip last
        # self.labels = encodings.labels[:, 1:].clone()              # skip first

    def _build_data(self, data_dir):
        dataset = []
        with open(f"{data_dir}/coqa_abg_train.json") as f:
            coqa_data = json.load(f)

        # create a dict of dicts by id
        coqa_data_dict = {d['id']: d for d in coqa_data['data']}
        for i in range(15):
            with open(f"{data_dir}/output_{i}.json") as f:
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
                        dataset.append({'context': f"Story:\n{story}\nHistory:\n{history_turns}\nAmbiguous Question:\n{ambiguous_question}", 'cq': cq})

        return dataset

    def __len__(self):
        return len(self.all_texts)

    def __getitem__(self, item):
        return self.encoder_input_ids[item], self.encoder_attention_mask[item], self.decoder_input_ids[item], self.labels[item] 
    
if __name__ == "__main__":
    data_obj = ABGCoQADataset()
    dataset = data_obj._build_data('data')
    print(dataset[41])