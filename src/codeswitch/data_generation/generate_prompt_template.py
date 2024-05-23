import pandas as pd

# Load OpenAI API key from environment variable
# openai.api_key = os.getenv("open_ai_key")


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
        prompt_label = f"[{context}]"
        prompts.append(
            (prompt_label, prompt, context)
        )  # Modified to include the prompt label and domain
    return prompts


# Function to get template for a given context
def get_template(context):
    templates = {
        "Business Meeting": """
        [Business Meeting]:
        - Use a mix of formal and informal language to reflect different communication settings.
        - Add code-switching examples in other languages depending on the target audience and purpose of the dataset.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
        "Helpdesk Communication": """
        [Helpdesk Communication]:
        - Use a conversational tone and natural language in the sentences to make them sound authentic.
        - Use formal language and avoid slang or colloquial terms.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
        "Contact Center Communication": """
        [Contact Center Communication]:
        - Use a mix of formal and informal language to reflect the diverse communication styles and preferences of speakers in different contexts.
        - Consider incorporating non-verbal cues or gestures that may accompany code-switching in a particular context.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
        "Customer Service/Support": """
        [Customer Service/Support]:
        - Use a friendly and helpful tone to make the customer feel valued.
        - Provide clear and concise information.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
        "Logistics or Delivery Coordination": """
        [Logistics or Delivery Coordination]:
        - Use clear and precise language to ensure accurate communication.
        - Include specific terms related to logistics and delivery processes.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
        "Technical Support or IT Assistance": """
        [Technical Support or IT Assistance]:
        - Use technical terms appropriately and provide clear troubleshooting steps.
        - Maintain a patient and supportive tone to assist users with varying technical skills.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
        "Transportation or Ride-Sharing Services": """
        [Transportation or Ride-Sharing Services]:
        - Use polite and professional language to ensure a positive interaction.
        - Include terms related to transportation logistics and passenger communication.
        [Insert an English sentence with a noun phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an action verb phrase that is likely to be code-switched to Spanish.]
        [Insert an English sentence with an object phrase that is likely to be code-switched to Spanish.]
        [Insert a Spanish sentence with a noun phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an action verb phrase that is likely to be code-switched to English.]
        [Insert a Spanish sentence with an object phrase that is likely to be code-switched to English.]
        """,
    }

    return templates.get(context, "")


# Generate code-switched prompts for different contexts
CONTEXTS = [
    "Business Meeting",
    "Helpdesk Communication",
    "Contact Center Communication",
    "Customer Service/Support",
    "Logistics or Delivery Coordination",
    "Technical Support or IT Assistance",
    "Transportation or Ride-Sharing Services",
]


def main():
    num_instances = 10  # Change this to the desired number of instances
    prompts = generate_code_switched_prompts(CONTEXTS, num_instances)

    # Create DataFrame
    df = pd.DataFrame({"Prompt": [prompt[1] for prompt in prompts]})

    # Write DataFrame to CSV
    df.to_csv("generated_prompts.csv", index=False)

    print(df)
    print(
        f"{len(prompts)} prompts have been generated and saved to generated_prompts.csv'."
    )


if __name__ == "__main__":
    main()
