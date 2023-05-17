import json
import numpy as np

filename = "output_ft5-base_preds.json.json"

data = json.load(open(filename))
tot = np.array([0,0,0,0])
for element in data:
    # print(element)
    scores = element["Scores"]
    scores1 = scores[0].split(":")[1][2:-1].split(',')
    scores2 = scores[1].split(":")[1][2:-1].split(',')
    s1 = [eval(i) for i in scores1]
    s2 = [eval(i) for i in scores2]
    # print(s1, s2)
    tot += np.array(s1)
    tot += np.array(s2)
    # break
print(tot/(2*len(data)))