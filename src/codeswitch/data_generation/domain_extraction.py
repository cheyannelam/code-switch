import json
import time

import openai
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

tqdm.pandas()


# Function to classify the topic of an utterance
def classify_topic(client, utterance, retries=3, delay=2):

    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that classifies utterances into topics.",
                    },
                    {
                        "role": "user",
                        "content": f"Extract and return only the main topic from the utterance given: avoid any comments apart from the classified topic: '{utterance}'.",
                    },
                ],
            )
            topic = response.choices[0].message.content
            return topic.strip()
        except openai.APIError as e:
            print(f"OpenAIError: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to classify topic after {retries} retries.")


def group_topics(client, topics, num_topics=6):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarize utterances into topics. You are designed to output JSON.",
            },
            {
                "role": "user",
                "content": f"Return the most common {num_topics} topics from the list: {topics}. The output should be a list",
            },
        ],
    )
    topic = response.choices[0].message.content
    return topic.strip()


# Function to get top topics from a CSV file
def get_top_topics(client, csv_file_path, column_name="context", sample_size=100):
    dataframe = pd.read_csv(csv_file_path)
    if column_name not in dataframe.columns:
        raise ValueError(f"The CSV file must contain a '{column_name}' column.")
    sample_dataframe = dataframe.sample(n=sample_size, random_state=1)
    sample_dataframe["topic"] = sample_dataframe[column_name].progress_apply(
        lambda x: classify_topic(client, x)
    )
    topics = sample_dataframe["topic"].to_list()
    with open("topics.txt", "w", encoding="utf-8") as file:
        for topic in topics:
            file.write(f"{topic}\n")
    top_topics = group_topics(client, topics, num_topics=6)
    print(top_topics)
    top_topics = json.loads(top_topics)
    top_topics = list(top_topics.values())[0]
    return top_topics


def main():
    client = OpenAI()
    csv_file_path = "/home/public/data/dialpad/dev_anonymized.csv"
    top_topics = get_top_topics(client, csv_file_path, sample_size=100)
    with open("top_topics.txt", "w", encoding="utf-8") as file:
        for topic in top_topics:
            file.write(f"{topic}\n")


if __name__ == "__main__":
    main()
