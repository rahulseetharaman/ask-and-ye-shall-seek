import requests
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("result_dicts_json", help="Name of the JSON file containing the key-value pairs")
args = parser.parse_args()

api_endpoint = "https://api.openai.com/v1/completions"
api_key = "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"

request_headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + api_key
}

# Load the key-value pairs from the JSON file
with open(args.result_dicts_json, 'r') as f:
    result_dict = json.load(f)

# Create a dictionary to store the clarifying questions
clarifying_questions = {}

# Iterate through the key-value pairs
for key, values in result_dict.items():
    # Create a list to store the clarifying questions for the current key
    key_questions = []
    for value, score in values:
        # Generate clarifying question for the key using ChatGPT
        prompt = f"Value 1: {value[0]}\nValue 2: {value[1]}\nGenerate a clarifying question for the key: {key}"
        request_data = {
            "model": "text-davinci-003",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.5
        }

        response = requests.post(api_endpoint, headers=request_headers, json=request_data)

        if response.status_code == 200:
            response_text = response.json()["choices"][0]["text"]
            key_questions.append(response_text)

    # Add the list of clarifying questions to the dictionary
    clarifying_questions[key] = key_questions

# Write the clarifying questions to a JSON file
output_file = "clarifying_questions.json"
with open(output_file, 'w') as f:
    json.dump(clarifying_questions, f)

print("Clarifying questions saved to:", output_file)
