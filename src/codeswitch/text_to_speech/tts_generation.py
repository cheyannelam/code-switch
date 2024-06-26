import json
import os
import random
from pathlib import Path

import bark
import click
import ray
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

from codeswitch.dataloader import read_text_data


def settings(row, es_ratio, output_foldername):
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
    use_preset = isinstance(es_ratio, (int, float)) and 0 <= es_ratio <= 1
    preset = None

    if use_preset:
        if random.uniform(0, 1) < es_ratio:
            preset = random.choice(preset_es)
        else:
            preset = random.choice(preset_en)
    row["preset"] = preset
    row["output_fpath"] = output_foldername / "audio" / f"{row['id']}.wav"
    row["output_fpath"].parent.mkdir(exist_ok=True, parents=True)

    row["manifest_info"] = {
        "audio_filepath": str(row["output_fpath"]),
        "text": row["text"],
        "preset": preset,
    }
    return row


# pylint: disable=too-few-public-methods
class TTSGenerator:
    def __init__(self):
        preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)
        if torch.cuda.is_available():
            bark.generation.models["text"]["model"] = torch.compile(
                bark.generation.models["text"]["model"].half()
            )
            bark.generation.models["coarse"] = torch.compile(
                bark.generation.models["coarse"].half()
            )
            bark.generation.models["fine"] = torch.compile(
                bark.generation.models["fine"].half()
            )

    @torch.inference_mode()
    def __call__(self, row):
        if row["output_fpath"].exists():
            return row
        row["audio"] = generate_audio(
            row["text"], history_prompt=row["preset"], silent=True
        )
        return row


def save_audio(row):
    if row["output_fpath"].exists():
        return row
    write_wav(row["output_fpath"], SAMPLE_RATE, row["audio"])
    return row


@click.command()
@click.option(
    "--data-path",
    default="./utterance_20240619_test.txt",
    help="Path to the text file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-path",
    default="./utterance_20240619_test_audio",
    help="Output path for audio files",
    type=click.Path(path_type=Path),
)
@click.option(
    "--num-workers", default=2, help="Number of parallel workers to use", type=int
)
@click.option("--use-gpu", default=False, help="Use GPU for TTS", type=bool)
@click.option(
    "--limit-amount", default=None, help="Limit the amount of data to process", type=int
)
def main(data_path, output_path, num_workers, use_gpu, limit_amount):
    output_path.mkdir(exist_ok=True, parents=True)
    data = [
        {"id": i, "text": text}
        for i, text in enumerate(read_text_data(data_path=data_path))
    ]
    if limit_amount:
        data = data[:limit_amount]
    print(f"Generating {len(data)} audio files...")
    ds = ray.data.from_items(data)
    ds = ds.map(settings, fn_kwargs={"es_ratio": -1, "output_foldername": output_path})
    ds = ds.map(
        TTSGenerator,
        concurrency=num_workers,
        num_cpus=(os.cpu_count() - 1) / num_workers,  # leave 1 core for scheduler
        num_gpus=1 / num_workers if use_gpu else 0,
    )
    ds = ds.map(save_audio)
    output = ds.materialize()

    manifest_lst = [row["manifest_info"] for row in output.iter_rows()]
    with (output_path / "tts_manifest.json").open("w", encoding="utf-8") as file:
        for line in manifest_lst:
            j = json.dumps(line, ensure_ascii=False)
            file.write(f"{j}\n")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
