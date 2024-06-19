import json

import click
import numpy as np
from openai import OpenAI
from p_tqdm import p_map


def read_topic():
    with open("top_topics.txt", "r", encoding="utf-8") as file:
        topics = file.read().splitlines()
        return topics


def utterance_generation(pos, topic, lang, client):
    template = "Please insert a {source_lang} sentence with a {target_lang} code-switched {pos} syntatic catergory item in {topic} context. Return the JSON in the following format: {output_struct}"

    def output_struct(lang, pos, topic, text):
        return json.dumps(
            {"base_language": lang, "syn_cat": pos, "topic": topic, "text": text},
            ensure_ascii=False,
            indent=4,
        )

    prompts = [
        [
            template.format(
                source_lang="English",
                pos="single noun",
                target_lang="Spanish",
                topic="financial inquiry",
                output_struct=output_struct(
                    "English",
                    "single noun",
                    "financial inquiry",
                    "Insert sentence here",
                ),
            ),
            output_struct(
                "English",
                "single noun",
                "financial inquiry",
                "Could you please provide me with the saldo of my account?",
            ),
        ],
        [
            template.format(
                source_lang="Spanish",
                pos="adjective",
                target_lang="English",
                topic="insurance matters",
                output_struct=output_struct(
                    "Spanish",
                    "adjective",
                    "Insurance Matters",
                    "Insert sentence here",
                ),
            ),
            output_struct(
                "Spanish",
                "adjective",
                "Insurance Matters",
                "Es importante tener un comprehensive seguro de auto para estar protegido en caso de un accidente.",
            ),
        ],
        [
            template.format(
                source_lang=lang[0],
                pos=pos,
                target_lang=lang[1],
                topic=topic,
                output_struct=output_struct(
                    lang[0], pos, topic, "Insert sentence here"
                ),
            )
        ],
    ]

    messages = []
    for item in prompts:
        for i, content in enumerate(item):
            messages.append({"role": ["system", "user"][i], "content": content})

    # pylint: disable-next=duplicate-code
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and experienced linguist that give code-switch utterance examples. You only output JSON",
            },
        ]
        + messages,
    )
    output = response.choices[0].message.content
    return output


def utterances_generation(pos, pos_prob, topics, num_utterance=20, workers=8):
    langs = ["English", "Spanish"]

    pos_lst = np.random.choice(pos, num_utterance, replace=True, p=pos_prob)
    topics_lst = np.random.choice(topics, num_utterance, replace=True)
    lang_lst = np.random.choice([0, 1], num_utterance, replace=True)
    lang_lst = [
        (langs[0], langs[1]) if x == 0 else (langs[1], langs[0]) for x in lang_lst
    ]
    all_outputs = []

    def task(pos_item, topic, lang):
        client = OpenAI()
        while True:
            output = utterance_generation(pos_item, topic, lang, client)
            output = json.loads(output)
            if "text" in output.keys():
                break

        return {
            "base_language": lang[0],
            "syn_cat": pos_item,
            "topic": topic,
            "text:": output["text"],
        }

    all_outputs = p_map(task, pos_lst, topics_lst, lang_lst, num_cpus=workers)
    return all_outputs


@click.command()
@click.option("--num-utterance", default=100, help="Number of utterances to generate")
@click.option("--output-fpath", default="output.json", help="Output file path")
@click.option("--workers", default=8, help="Number of parallel workers")
def main(num_utterance, output_fpath, workers):
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
    total = sum(syn_cat.values())
    syn_cat = {k: v / total for k, v in syn_cat.items()}
    pos = list(syn_cat.keys())
    pos_prob = list(syn_cat.values())
    topics = read_topic()
    utterances = utterances_generation(
        pos, pos_prob, topics, num_utterance=num_utterance, workers=workers
    )
    # print(utterances)

    with open(output_fpath, "w", encoding="utf-8") as file:
        for line in utterances:
            j = json.dumps(line, ensure_ascii=False)
            file.write(f"{j}\n")

    print(f"JSON data successfully written to {'file_path'}")
    # print(data)
    # df_utterances = pd.DataFrame(data)


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    main()
