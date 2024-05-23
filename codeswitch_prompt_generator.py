import os

import openai
import pandas as pd

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("open_ai_key")


# Function to generate code-switched prompts using the fine-tuned model
def generate_code_switched_prompts(contexts, num_instances=10):
    prompts = []
    for context in contexts:
        prompt = generate_prompt(context, num_instances)
        prompts.extend(prompt)
    return prompts


# Function to generate prompts for each context
def generate_prompt(context, num_instances):
    template = get_template(context)
    prompts = []
    for i in range(num_instances):
        prompt = f"{template}\n\n[English-Spanish Code-Switched Utterances {i+1}]:"
        prompts.append(prompt)
    return prompts


# Function to get template for a given context
def get_template(context):
    if context == "Business Meeting":
        return """
        [Business Meeting]:
        - Use a mix of formal and informal language to reflect different communication settings.
        - Add code-switching examples in other languages depending on the target audience and purpose of the dataset.
        [English-Spanish Code-Switched Utterances]:
        1. [English Sentence with Code-Switched Noun Phrase]: [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        2. [English Sentence with Code-Switched Action Verb Phrase]: [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        3. [English Sentence with Code-Switched Object Phrase]: [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        4. [Spanish Sentence with Code-Switched Noun Phrase]: [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        5. [Spanish Sentence with Code-Switched Action Verb Phrase]: [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        6. [Spanish Sentence with Code-Switched Object Phrase]: [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """
    elif context == "Helpdesk Communication":
        return """
        [Helpdesk Communication]:
        - Use a conversational tone and natural language in the sentences to make them sound authentic.
        - Use formal language and avoid slang or colloquial terms.
        [English-Spanish Code-Switched Utterances]:
        1. [English Sentence with Code-Switched Noun Phrase]: [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        2. [English Sentence with Code-Switched Action Verb Phrase]: [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        3. [English Sentence with Code-Switched Object Phrase]: [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        4. [Spanish Sentence with Code-Switched Noun Phrase]: [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        5. [Spanish Sentence with Code-Switched Action Verb Phrase]: [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        6. [Spanish Sentence with Code-Switched Object Phrase]: [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """
    elif context == "Contact Center Communication":
        return """
        [Contact Center Communication]:
        - Use a mix of formal and informal language to reflect the diverse communication styles and preferences of speakers in different contexts.
        - Consider incorporating non-verbal cues or gestures that may accompany code-switching in a particular context.
        [English-Spanish Code-Switched Utterances]:
        1. [English Sentence with Code-Switched Noun Phrase]: [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        2. [English Sentence with Code-Switched Action Verb Phrase]: [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        3. [English Sentence with Code-Switched Object Phrase]: [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        4. [Spanish Sentence with Code-Switched Noun Phrase]: [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        5. [Spanish Sentence with Code-Switched Action Verb Phrase]: [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        6. [Spanish Sentence with Code-Switched Object Phrase]: [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """
    else:
        return ""


# Generate code-switched prompts for different contexts
contexts = [
    "Business Meeting",
    "Helpdesk Communication",
    "Contact Center Communication",
]
num_instances = 10  # Change this to the desired number of instances
prompts = generate_code_switched_prompts(contexts, num_instances)

# Create DataFrame
df = pd.DataFrame({"Prompt": prompts})
print(df)
# Write DataFrame to CSV
df.to_csv("code_switched_prompts.csv", index=False)

print(
    f"{len(prompts)} prompts have been generated and saved to 'code_switched_prompts.csv'."
)
