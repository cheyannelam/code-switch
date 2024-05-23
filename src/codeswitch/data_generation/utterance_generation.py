import json
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

from codeswitch.data_generation.generate_prompt_template import CONTEXTS, get_template

# Load OpenAI API key from environment variable
# openai.api_key = os.getenv("open_ai_key")
# Load the prompts DataFrame
df = pd.read_csv("generated_prompts.csv")


# Function to generate code-switched utterances for a given prompt
def generate_code_switched_utterance(prompt):
    prompt += """
    Give me a list of generated sentences in the form of {'sentences': ['sentence 1', 'sentence 2']}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message["content"].strip()


# Generate code-switched utterances for each selected prompt
def main():
    selected_prompts = [get_template(x) for x in CONTEXTS]
    code_switched_utterances = []

    for prompt in tqdm(selected_prompts):
        utterance = json.loads(generate_code_switched_utterance(prompt))
        utterance = utterance["sentences"]
        code_switched_utterances += utterance
        print(utterance)
    output_file_path = "utterance.txt"

    Path(output_file_path).write_text(
        "\n".join(code_switched_utterances), encoding="utf-8"
    )
    print(f"Sentences saved to {output_file_path}")


if __name__ == "__main__":
    main()
