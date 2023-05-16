
filename = "ft5-xl_preds.json"
import json
import requests
from tqdm import tqdm

instructions = '''Assume you are an expert in english language. Given a story, an ambiguous question based on the story and two clarifying questions for the ambiguous questions. Evaluate the quality of each clarifying question and give a score from 1 to 5 with 1 being the lowest or worst and 5 being the highest or best for each of the following criterias:
1. How much sense does the clarifying question make according to the English language?
2. How different are the two questions from each other?
3. How effective is the clarifying question in removing the ambiguity of the ambiguous question?
5. How likely is it that the clarifying question is answerable based on the story and ambiguous question.
Give a score for each of the clarifying questions strictly in the following format without any explanation:
clarifyingquestion1: [1,2,3,4]\nclarifyingquestion2: [5,1,1,2]'''


def getResponse(prompt):

    api_endpoint = "https://api.openai.com/v1/completions"
    # api_key = "sk-"
    api_key = "sk-"
    api_key = "sk-"


    request_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    request_data = {
        "model": "text-davinci-003",
        "prompt": f"Generate three clarifying questions for the answers to the ambiguous question based on the given passage {prompt}.",
        "max_tokens": 50,
        "temperature": 0.5
    }

    response = requests.post(api_endpoint, headers=request_headers, json=request_data)

    if response.status_code == 200:
        # response_text = response.json()
        response_text = response.json()["choices"][0]["text"]

        return response_text
    else:
        print(f"Request failed with status code: {str(response.status_code)}")



def format_text(entry):
    ans = "Story: " +entry["story"] + "\n"
    ans += "Ambiguous Question: " +(entry["target_turn"]['question']) + "\n"
    ans += "Possible Answers to the clarifying Question: "
    for answer in entry["clarification_turn"]["answers"]:
        ans += answer['clr_ans'] + ", "
    ans = ans[:-2]+"\n"
    return ans

f = open(filename)
jsonDump = []
trainData = json.load(f)
outCount = 0
ctr = 0
for i,element in enumerate(tqdm(trainData)):
    storyamb = element['x'].split('Ambiguous Question:')
    story = storyamb[0].split('History')[0]
    prompt = instructions + "\n" + story + "\nAmbiguous Question:" + storyamb[1]
    prompt += "\nClarifying Question 1: " + element["y_pred"][0]
    prompt += "\nClarifying Question 2: " + element["y_pred"][1]
    # print(prompt)

    # break

    # if(ctr%30 == 0):
    #     print(ctr)
    # ctr+=1
    # print(entry)
    # formatted = (format_text(entry))

    resp = getResponse(prompt)
    
    # print(formatted)
    # print(resp)
    jsonDump.append({"prompt":prompt, "Scores":[resp.split('\n')[0],resp.split('\n')[1]]})
    # if len(jsonDump) >= 50:
    #     outfile = "output_"+str(outCount)+".json"
    #     with open(outfile, "w") as outfile:
    #         json.dump(jsonDump, outfile)
    #     jsonDump=[]
    #     outCount += 1
        # break
outfile = "output_"+str(filename)+".json"
with open(outfile, "w") as outfile:
    json.dump(jsonDump, outfile)

f.close()