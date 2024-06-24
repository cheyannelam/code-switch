# pylint: disable=R0801
import json
import re
from pathlib import Path

import click


def clean_text(x):
    x = x.lower()
    x = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ'\s]", "", x)
    return x


@click.command()
@click.option(
    "--text_fpath",
    type=Path,
    default="/home/public/data/synthetic/utterance_20240605_test.txt",
)
@click.option(
    "--audio_fpath",
    type=Path,
    default="/home/public/data/synthetic/utterance_20240605_test_audio",
)
def main(text_fpath, audio_fpath):
    transcripts = Path(text_fpath).read_text(encoding="utf-8").splitlines()

    manifest = []
    for i, text in enumerate(transcripts):
        manifest.append(
            {"audio_filepath": f"{audio_fpath}/{i}.wav", "text": clean_text(text)}
        )

    save_fpath = text_fpath.parent / "manifest.json"
    save_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fpath, "w", encoding="utf-8") as f:
        for row in manifest:
            json_repr = json.dumps(row, ensure_ascii=False)
            f.write(json_repr + "\n")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
