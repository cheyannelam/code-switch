import openai
import csv
import os
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()

OPENAI_API_KEY = os.environ.get("open_ai_key")
PROMPT = (
    "Generate 1000 random code-switched utterances between English and Spanish in CSV format. Each line should contain the following fields separated by a comma:\n"
    "code-switched text, english translation, spanish translation, audio filename\n"
    "Please format each line exactly like this (without double quotes):\n"
    "Example:\n"
    "I can't wait to see you mañana!, I can't wait to see you mañana!, No puedo esperar a verte mañana!, see_you_manana.mp3\n"
)
NUM_OF_CALLS = 1

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

openai.api_key = OPENAI_API_KEY

def generate_text():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def translate_text(text, target_lang):
    # Placeholder function for translation
    return text  # Replace this with actual translation

def generate_audio(text, lang, audio_filename):
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = f"audio/{audio_filename}"  # Save audio in 'audio' folder
    tts.save(filename)
    return filename

def save_to_csv(data, file_name):
    with open(file_name, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code-switched", "english_translation", "spanish_translation", "audio_filename"])  # Header
        writer.writerows(data)

if __name__ == "__main__":
    data = []

    try:
        for i in range(NUM_OF_CALLS):
            response = generate_text()
            lines = response.split("\n")
            for line in lines:
                if line:
                    try:
                        code_switched, english_translation, spanish_translation, audio_filename = line.split(",")
                        audio_file = generate_audio(code_switched.strip(), lang="en", audio_filename=audio_filename.strip())
                        data.append((
                            code_switched.strip(),
                            english_translation.strip(),
                            spanish_translation.strip(),
                            audio_filename.strip()
                        ))
                    except ValueError:
                        print(f"Skipping line due to invalid format: '{line}'")
            print(f"Completed API call {i + 1} of {NUM_OF_CALLS}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving partial data to the CSV file...")

    finally:
        save_to_csv(data, "output.csv")
        print("Data saved to 'output.csv'.")
