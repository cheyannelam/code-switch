# pylint: skip-file

import os
import random

import openai
import pandas as pd

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("open_ai_key")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Set up OpenAI API client
openai.api_key = OPENAI_API_KEY

# Dictionary of prompts for different domains
domain_prompts = {
    "business meetings": {
        "English": "Generate a random sentence suitable for a business meeting.",
        "Spanish": "Generar una oración aleatoria adecuada para una reunión de negocios.",
        "Code-Switched": "Generate a random code-switched sentence suitable for a business meeting.",
    },
    "contact center calls": {
        "English": "Generate a random sentence suitable for a contact center call.",
        "Spanish": "Generar una oración aleatoria adecuada para una llamada al centro de contacto.",
        "Code-Switched": "Generate a random code-switched sentence suitable for a contact center call.",
    },
    "business calls": {
        "English": "Generate a random sentence suitable for a business call.",
        "Spanish": "Generar una oración aleatoria adecuada para una llamada de negocios.",
        "Code-Switched": "Generate a random code-switched sentence suitable for a business call.",
    },
    "sales calls": {
        "English": "Generate a random sentence suitable for a sales call.",
        "Spanish": "Generar una oración aleatoria adecuada para una llamada de ventas.",
        "Code-Switched": "Generate a random code-switched sentence suitable for a sales call.",
    },
}


# Function to generate text based on domain prompt and language
def generate_text(domain, language):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=domain_prompts[domain][language],
        temperature=0.7,
        max_tokens=100,
    )
    return response.choices[0].text.strip()


# Function to generate sentences of specified number for a given domain and language
def generate_sentences(domain, language, num_sentences):
    sentences = []
    for _ in range(num_sentences):
        sentence = generate_text(domain, language)
        sentences.append({"Sentence": sentence, "Language": language})
    return sentences


if __name__ == "__main__":
    num_sentences = 150  # Adjust the number of sentences to generate
    domains = [
        "business meetings",
        "contact center calls",
        "business calls",
        "sales calls",
    ]
    languages = ["English", "Spanish", "Code-Switched"]

    # Initialize an empty set to keep track of generated sentences
    generated_sentences = set()

    # Generate sentences for each domain and language
    all_sentences = []
    for domain in domains:
        for language in languages:
            sentences = generate_sentences(domain, language, num_sentences)
            all_sentences.extend(sentences)

    # Shuffle the sentences
    random.shuffle(all_sentences)

    # Create DataFrame
    df = pd.DataFrame(all_sentences)

    # Write to CSV
    df.to_csv("training_utterences.csv", index=False)

    # Display DataFrame
    print("Data written to 'training_utterences.csv':")
    print(df)
