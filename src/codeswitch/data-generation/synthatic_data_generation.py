import pandas as pd
import openai
import os
import re
from tqdm import tqdm
from collections import Counter
import time
import random

openai.api_key = os.getenv("open_ai_key")


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


def get_top_topics(file_path, column_name='context', sample_size=1000):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise ValueError(f"The CSV file must contain a '{column_name}' column.")
    sample_df = df.sample(n=sample_size, random_state=1)
    sample_df['topic'] = sample_df[column_name].apply(classify_topic)
    topic_counts = Counter(sample_df['topic'])
    top_5_contexts = topic_counts.most_common(5)
    top_5_topics = list({topic for topic, count in top_5_contexts})
    return top_5_topics


def generate_code_switched_utterance(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()


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

file_path = '/home/public/data/dialpad/dev_anonymized.csv'
top_topics = get_top_topics(file_path)

total_instances = 50
num_contexts = len(top_topics)
instances_per_context = total_instances // num_contexts


def generate_prompts(contexts, instances_per_context):
    prompts = []
    for context in contexts:
        for i in range(instances_per_context):
            prompt = template.format(context=context)
            prompts.append(prompt)
    return prompts


prompts = generate_prompts(top_topics, instances_per_context)

generated_utterances = set()
for prompt in tqdm(prompts):
    utterance = generate_code_switched_utterance(prompt)
    if utterance not in generated_utterances:
        generated_utterances.add(utterance)
        print(utterance)

df_output = pd.DataFrame(list(generated_utterances), columns=["Code-Switched Utterance"])
df_output.to_csv("switch_utter_list_17.csv", index=False)

print(df_output)

with open('switch_utter_list_17.csv', 'r', encoding='utf-8') as file:
    file_contents = file.read()

cleaned_contents = file_contents.replace("`", "").replace("[", "").replace("]", "").replace('"', '')

sentences = [sentence.strip() for sentence in cleaned_contents.split(",")]

for index, sentence in enumerate(sentences, start=1):
    sentence_without_prefix = re.sub(r'^Sentence \d+:', '', sentence)
    print(sentence_without_prefix.strip())

sentences_regex = re.findall(r'""([^"]+)""', file_contents)

output_file_path = 'utterance.txt'
if os.path.exists(output_file_path):
    with open(output_file_path, 'r', encoding='utf-8') as output_file:
        existing_sentences = set(line.strip() for line in output_file)
else:
    existing_sentences = set()

new_sentences = set(sentences_regex)
all_sentences = existing_sentences.union(new_sentences)

shuffled_sentences = list(all_sentences)
random.shuffle(shuffled_sentences)

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for sentence in shuffled_sentences:
        output_file.write(f"{sentence}\n")

print(f"Sentences appended to {output_file_path}, deduplicated, and shuffled.")
