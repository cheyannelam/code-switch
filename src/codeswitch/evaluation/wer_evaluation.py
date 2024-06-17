import re

import jiwer
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

from codeswitch import dataloader


def clean_text(x):
    x = x.lower()
    x = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ'\s]", "", x)
    return x


def wer_sentence(gold_sentence, generated_sentence, case_sensitive=False):
    if not case_sensitive:
        gold_sentence = clean_text(gold_sentence)
        generated_sentence = clean_text(generated_sentence)

    return jiwer.wer(gold_sentence, generated_sentence)


def wer_sentences(gold_sentences, generated_sentences, case_sensitive=False):
    if not case_sensitive:
        gold_sentences = [clean_text(gold_sentence) for gold_sentence in gold_sentences]
        generated_sentences = [
            clean_text(generated_sentence) for generated_sentence in generated_sentences
        ]

    return jiwer.wer(gold_sentences, generated_sentences)


def base_language(sentences):
    languages = [Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    en_confidence = []
    es_confidence = []

    for sentence in sentences:

        confidence_values = detector.compute_language_confidence_values(sentence)
        en_confidence.append(confidence_values[0].value)
        es_confidence.append(confidence_values[1].value)

    return en_confidence, es_confidence


def main():
    data = dataloader.read_json("/home/public/data/synthetic/manifest.json")
    gold_sentences = [line["text"] for line in data]
    audio_paths = [line["audio_filepath"] for line in data]

    data = dataloader.read_json("/home/public/data/synthetic/utterance_20240605_test_whisper_transcripts.json")
    generated_sentences = [line["text"] for line in data]

    wer_lst = []
    for gold_sentence, generated_sentence in zip(gold_sentences, generated_sentences):
        wer_lst.append(
            wer_sentence(gold_sentence, generated_sentence, case_sensitive=False)
        )

    print(
        "wer:", wer_sentences(gold_sentences, generated_sentences, case_sensitive=False)
    )
    # en_confidence, es_confidence = base_language(gold_sentences)

    csv_dict = {
        "audio_path": audio_paths,
        "gold_sentences": gold_sentences,
        "generated_sentences": generated_sentences,
        "wer": wer_lst,
        # "en_confidence": en_confidence,
        # "es_confidence": es_confidence
    }

    df_csv = pd.DataFrame(csv_dict)
    # df_csv.to_csv('maes_bw16_ma2_mg2.3_nolm_utterance_20240605_test_stat.csv', index=False)
    df_csv.to_csv("/home/public/data/synthetic/utterance_20240605_test_whisper_transcripts_stat.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
