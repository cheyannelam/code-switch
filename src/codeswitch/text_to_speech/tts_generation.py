import os
import random

import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

from codeswitch.dataloader import read_text_data


def generate_speech_sentence(
    text, preset=None, output_filename="bark_tts.wav", output_foldername=""
):
    path = os.path.join(output_foldername, output_filename)
    with torch.no_grad():
        audio_array = generate_audio(f"{text}", history_prompt=preset)
    write_wav(path, SAMPLE_RATE, audio_array)


def generate_speech_sentences(sentences, output_foldername="", es_ratio=0.5):
    """
    es_ratio: [0, 1], the percentage of spanish preset use, set to -1 to generate without presets
    """
    if output_foldername != "" and not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    preset_en = [
        "v2/en_speaker_2",
        "v2/en_speaker_3",
        "v2/en_speaker_6",
        "v2/en_speaker_7",
        "v2/en_speaker_8",
        "v2/en_speaker_9",
    ]
    preset_es = [
        "v2/es_speaker_1",
        "v2/es_speaker_4",
        "v2/es_speaker_6",
        "v2/es_speaker_7",
        "v2/es_speaker_8",
        "v2/es_speaker_9",
    ]
    use_preset = isinstance(es_ratio, (int, float)) and 0 >= es_ratio <= 1
    preset = None
    for i, sentence in enumerate(sentences):
        output_filename = f"{i}.wav"
        if use_preset:
            if random.uniform(0, 1) < es_ratio:
                preset = random.choice(preset_es)
            else:
                preset = random.choice(preset_en)
        print(i, preset)
        generate_speech_sentence(sentence, preset, output_filename, output_foldername)


def main():
    # device = check_device()
    data = read_text_data(data_path="/home/public/data/utterance_20240605_test.txt")

    # download and load all models
    preload_models()

    sentences = data
    print("sentences length:", len(sentences))
    generate_speech_sentences(
        sentences,
        output_foldername="/home/public/data/utterance_20240605_test_audio",
        es_ratio=-1,
    )


if __name__ == "__main__":
    main()
