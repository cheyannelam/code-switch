import json
import re

import click
import jiwer
import pandas as pd

from codeswitch import dataloader


def clean_text(x):
    x = x.lower()
    x = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ'\s]", "", x)
    x = x.strip()
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


@click.command()
@click.option("--data-path", type=str, default="data/synthetic/manifest.json")
@click.option(
    "--groundtruth-path",
    type=str,
    default="data/synthetic/utterance_20240605_test_audio/manifest.json",
)
@click.option("--output-stats-path", type=str, default="data/synthetic/stats.tsv")
@click.option(
    "--output-manifest-path", type=str, default="data/synthetic/manifest_test.json"
)
def main(
    data_path, groundtruth_path, output_stats_path, output_manifest_path
):  # pylint: disable=too-many-locals
    data = dataloader.read_json(groundtruth_path)
    gold_sentences = [line["text"] for line in data]
    audio_paths = [line["audio_filepath"] for line in data]

    data = dataloader.read_json(
        data_path,
    )
    generated_sentences = [line["text"] for line in data]

    wer_lst = []
    for gold_sentence, generated_sentence in zip(gold_sentences, generated_sentences):
        wer_lst.append(
            wer_sentence(gold_sentence, generated_sentence, case_sensitive=False)
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

    pd.DataFrame(csv_dict).to_csv(
        output_stats_path,
        index=False,
        sep="\t",
    )

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for audio_path, gold_sentence, wer in zip(audio_paths, gold_sentences, wer_lst):
            if wer > 0.8:
                continue
            output = {
                "audio_filepath": audio_path,
                "text": gold_sentence,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
