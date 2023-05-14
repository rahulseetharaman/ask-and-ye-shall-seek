import json
import requests

def getResponse(prompt):

    api_endpoint = "https://api.openai.com/v1/completions"
    # api_key = "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"
    api_key = "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"
    api_key = "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"


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

f = open('coqa_abg_train.json')
jsonDump = []
trainData = json.load(f)
outCount = 0
ctr = 0
for entry in trainData["data"]:
    if(entry["ambiguity"]=="ambiguous"):
        if(ctr%30 == 0):
            print(ctr)
        ctr+=1
        # print(entry)
        formatted = (format_text(entry))
        resp = getResponse(formatted)
        # print(formatted)
        # print(resp)
        jsonDump.append({"id":entry["id"],"new_questions":resp})
        if len(jsonDump) >= 50:
            outfile = "output_"+str(outCount)+".json"
            with open(outfile, "w") as outfile:
                json.dump(jsonDump, outfile)
            jsonDump=[]
            outCount += 1
        # break
outfile = "output_"+str(outCount)+".json"
with open(outfile, "w") as outfile:
    json.dump(jsonDump, outfile)
jsonDump=[]
outCount += 1
f.close()