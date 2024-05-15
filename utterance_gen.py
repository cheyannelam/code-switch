import pandas as pd
import openai
import os
import random
import re

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("open_ai_key")

# Load the prompts DataFrame
df = pd.read_csv("code_switched_prompts.csv")

# Function to generate code-switched utterances for a given prompt
def generate_code_switched_utterance(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",  # You can use any suitable GPT-3 model here
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Select a random subset of prompts from the DataFrame
selected_prompts = df.sample(n=100,replace=True, random_state=42)["Prompt"]

# Generate code-switched utterances for each selected prompt
code_switched_utterances = []
for prompt in selected_prompts:
    utterance = generate_code_switched_utterance(prompt)
    code_switched_utterances.append(utterance)

# Clean the code-switched utterances using regular expressions
cleaned_utterances = []
for utterance in code_switched_utterances:
    cleaned_utterance = re.sub(r'^\d+\.\s+', '', utterance)  # Remove numbers at the beginning
    cleaned_utterance = re.sub(r'^\[\w+\s+Sentence:\s*', '', cleaned_utterance)
    cleaned_utterance = re.sub(r'\[.*?\]:\s+', '', cleaned_utterance)  # Remove prompt description
    cleaned_utterance = re.sub(r'\[.*?\]\s+', '', cleaned_utterance) 
    cleaned_utterance = re.sub(r'\[.*?\\s+', '', cleaned_utterance) 
    cleaned_utterances.append(cleaned_utterance.strip())

# Create a DataFrame with the cleaned code-switched utterances
df_output = pd.DataFrame({"Code-Switched Utterance": cleaned_utterances})

# Write DataFrame to CSV
df_output.to_csv("switch_utter.csv", index=False)

# Display the generated code-switched utterances
print(df_output)
