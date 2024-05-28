import pandas as pd
import openai
import os
import re
from tqdm import tqdm
from collections import Counter
import time
import random

openai.api_key = os.getenv("open_ai_key")

# Function to classify the topic of an utterance
def classify_topic(utterance, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies utterances into topics."},
                    {"role": "user", "content": f"Extract and return only the main topic from the utterance give : avoid any comments apart from the classified topic : '{utterance}'."}
                ]
            )
            topic = response.choices[0].message['content'].strip()
            return topic
        except openai.error.APIError as e:
            print(f"APIError: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"Failed to classify topic after {retries} retries.")

# Function to get top topics from the CSV file
def get_top_topics(file_path, column_name='context', sample_size=1000):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise ValueError(f"The CSV file must contain a '{column_name}' column.")
    sample_df = df.sample(n=sample_size, random_state=1)
    sample_df['topic'] = sample_df[column_name].apply(classify_topic)
    topic_counts = Counter(sample_df['topic'])
    top_5_contexts = topic_counts.most_common(5)
    top_5_topics = list({topic for topic, count in top_5_contexts})  # Ensure topics are unique
    return top_5_topics

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

# Define a single template with placeholders for context-specific information
template = """
[{context}]:
- Use a mix of formal and informal language to reflect different communication settings.
- Add code-switching examples in other languages depending on the target audience and purpose of the dataset.
[Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
[Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
[Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
[Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
[Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
[Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
Please output as a list of sentences. Nothing else but the list.
example
        ""I need to check el código before continuing.""
"""

# Get topics from topic_gen.py
file_path = '/home/public/data/dialpad/dev_anonymized.csv'
top_topics = get_top_topics(file_path)

# Variable to specify the total number of instances to generate
total_instances = 50  # Change this to the desired number of total instances

# Distribute total instances evenly across the contexts
num_contexts = len(top_topics)
instances_per_context = total_instances // num_contexts

# Generate code-switched prompts for different contexts
def generate_prompts(contexts, instances_per_context):
    prompts = []
    for context in contexts:
        for i in range(instances_per_context):
            prompt = template.format(context=context)
            prompts.append(prompt)
    return prompts

prompts = generate_prompts(top_topics, instances_per_context)

# Generate code-switched utterances for each selected prompt
generated_utterances = set()  # To store unique utterances
for prompt in tqdm(prompts):
    utterance = generate_code_switched_utterance(prompt)
    if utterance not in generated_utterances:
        generated_utterances.add(utterance)
        print(utterance)  # Optional: Print each utterance for debugging

# Create a DataFrame with the unique code-switched utterances
df_output = pd.DataFrame(list(generated_utterances), columns=["Code-Switched Utterance"])

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

# Append unique sentences to a TXT file (if file exists, read and deduplicate)
output_file_path = 'utterance.txt'
if os.path.exists(output_file_path):
    with open(output_file_path, 'r', encoding='utf-8') as output_file:
        existing_sentences = set(line.strip() for line in output_file)
else:
    existing_sentences = set()

# Add new unique sentences
new_sentences = set(sentences_regex)
all_sentences = existing_sentences.union(new_sentences)

# Shuffle the sentences before writing them back to the file
shuffled_sentences = list(all_sentences)
random.shuffle(shuffled_sentences)

# Write deduplicated and shuffled sentences to the file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for sentence in shuffled_sentences:
        output_file.write(f"{sentence}\n")

print(f"Sentences appended to {output_file_path}, deduplicated, and shuffled.")
