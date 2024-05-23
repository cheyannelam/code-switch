import pandas as pd
import openai
import os
import re
from tqdm import tqdm

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("open_ai_key")

# Load the prompts DataFrame
df = pd.read_csv("generated_prompts.csv")

# Function to generate code-switched utterances for a given prompt
def generate_code_switched_utterance(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

# Select a random subset of prompts from the DataFrame
selected_prompts = df.sample(n=100, replace=True, random_state=42)["Prompt"]

# Generate code-switched utterances for each selected prompt
code_switched_utterances = []
for prompt in tqdm(selected_prompts):
    utterance = generate_code_switched_utterance(prompt)
    code_switched_utterances.append(utterance)
    print(utterance)  # Optional: Print each utterance for debugging

# Create a DataFrame with the code-switched utterances
df_output = pd.DataFrame(code_switched_utterances, columns=["Code-Switched Utterance"])

# Write DataFrame to CSV
df_output.to_csv("switch_utter_list_17.csv", index=False)

# Display the generated code-switched utterances
print(df_output)

# Read the contents of the previously generated file
with open('switch_utter_list_17.csv', 'r', encoding='utf-8') as file:
    file_contents = file.read()

# Remove backticks, square brackets, and double quotes
cleaned_contents = file_contents.replace("`", "").replace("[", "").replace("]", "").replace('"', '')

# Split by comma and trim whitespace
sentences = [sentence.strip() for sentence in cleaned_contents.split(",")]

# Print the list without "Sentence X:" prefix
for index, sentence in enumerate(sentences, start=1):
    sentence_without_prefix = re.sub(r'^Sentence \d+:', '', sentence)
    print(sentence_without_prefix.strip())

# Use regular expressions to find the sentences from the file content
sentences_regex = re.findall(r'""([^"]+)""', file_contents)

# Write sentences to a TXT file
output_file_path = 'utterance.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for index, sentence in enumerate(sentences_regex, start=1):
        output_file.write(f"{sentence}\n")

print(f"Sentences saved to {output_file_path}")


