# flake8: ignore: W605, W291
import json

import numpy as np
from openai import OpenAI


def read_topic():
    with open("top_topics.txt", "r", encoding="utf-8") as file:
        topics = file.read().splitlines()
        return topics


def utterance_generation(pos, topic, lang, client):

    prompt = f"""
            <s> [INST] Please provide an English sentence that includes a single noun, which is switched to Spanish, within the context of a financial inquiry. Return the JSON in the following format:
            {{"base_language": "English",
                "syn_cat": "single noun",
                "topic": "Financial inquirie",
                "text:": insert sentence here}} [\\INST]
            {{"base_language": "English",
                "syn_cat": "single noun",
                "topic": "Financial inquiries",
                "text": "Could you please provide me with the saldo of my account?"}} <\\s>
            <s> [INST] Please provide a Spanish sentence that includes an adjective, which is switched to English, within the context of insurance matters. Return the JSON in the following format:
            {{"base_language": "Spanish",
                "syn_cat": "adjective",
                "topic": "Insurance Matters",
                "text:": insert sentence here}} [\\INST]
            {{"base_language": "Spanish",
                "syn_cat": "adjective",
                "topic": "Insurance Matters",
                "text": "Es importante tener un comprehensive seguro de auto para estar protegido en caso de un accidente."}} <\\s>

            <s> [INST]
            Please insert a {lang[0]} sentence with {pos} as a syntatic catergory (syn_cat) that is code-switched to {lang[1]} in {topic} context. Return the JSON as the following format:
            {{"base_language": "{lang[0]}",
                "syn_cat": "{pos}",
                "topic": "{topic}",
                "text:": insert sentence here}} [INST]
            """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generate code-switch utterances. You are designed to output JSON",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    output = response.choices[0].message.content
    return output


def utterances_generation(client, pos, pos_prob, topics, num_utterance=20):
    langs = ["English", "Spanish"]

    pos_lst = np.random.choice(pos, num_utterance, replace=True, p=pos_prob)
    topics_lst = np.random.choice(topics, num_utterance, replace=True)
    lang_lst = np.random.choice([0, 1], num_utterance, replace=True)
    lang_lst = [
        (langs[0], langs[1]) if x == 0 else (langs[1], langs[0]) for x in lang_lst
    ]
    all_outputs = []

    for pos_item, topic, lang in zip(pos_lst, topics_lst, lang_lst):
        while True:
            output = utterance_generation(pos_item, topic, lang, client)
            output = json.loads(output)
            if "text" in list(output.keys()):
                break

        output = {
            "base_language": lang[0],
            "syn_cat": pos,
            "topic": topic,
            "text:": output["text"],
        }

        all_outputs.append(output)
        # print(prompt)
        print(output)
    return all_outputs


def main():
    syn_cat = {
        "determiner": 3,
        "single noun": 175,
        "subject noun phrase": 69,
        "object noun phrase": 140,
        "verb": 19,
        "verb phrase": 40,
        "independent clause": 79,
        "subordinate clause": 38,
        "relative clause": 38,
        "adjective": 15,
        "predicate adjective": 43,
        "adverb": 47,
        "preposition": 2,
        "prepositional phrase": 23.5,
        "adj phrase": 23.5,
        "adverbial phrase": 23.5,
        "infinitive phrase": 23.5,
        "subordinate conjunction": 16.33,
        "coordinate conjunction": 16.33,
        "relative pronoun": 16.3,
    }
    client = OpenAI()
    total = sum(syn_cat.values())
    syn_cat = {k: v / total for k, v in syn_cat.items()}
    pos = list(syn_cat.keys())
    pos_prob = list(syn_cat.values())
    topics = read_topic()
    utterances = utterances_generation(client, pos, pos_prob, topics, num_utterance=100)
    # print(utterances)
    data = []
    file_path = "output_json.json"
    for utterance in utterances:
        data.append(utterance)
    with open(file_path, "w", encoding="utf-8") as file:
        for line in data:
            j = json.dumps(line, ensure_ascii=False)
            file.write(f"{j}\n")

    print(f"JSON data successfully written to {'file_path'}")
    # print(data)
    # df_utterances = pd.DataFrame(data)


if __name__ == "__main__":
    main()
