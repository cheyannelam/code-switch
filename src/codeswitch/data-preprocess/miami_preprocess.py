# pylint: disable=R0801
import json
from pathlib import Path

import click
import pylangacq
from pydub import AudioSegment
from tqdm import tqdm


def extract_transcript(utterance):
    sent = " ".join(
        [token.word for token in utterance.tokens if not token.word.isupper()]
    )
    sent = sent.replace("/", "")
    sent = sent.replace(" ?", "?")
    sent = sent.replace(" .", ".")
    sent = sent.replace(" !", "!")
    sent = sent.replace("_", " ")
    sent = sent.strip()
    return sent


@click.command()
@click.option("--data_root", type=Path, default="/home/public/data/Miami")
def main(data_root):
    transcipts_fpaths = (data_root / "transcript").rglob("*.cha")
    audio_root = data_root / "audio"

    for fpath in tqdm(transcipts_fpaths):
        data = pylangacq.read_chat(str(fpath))
        audio_relfpath = Path(
            *Path(data.file_paths()[0]).with_suffix(".mp3").parts[-2:]
        )
        audio_fpath = audio_root / audio_relfpath
        audio = AudioSegment.from_file(audio_fpath)

        manifest = []
        for i, utt in enumerate(data.utterances()):
            if utt.time_marks is None:
                continue
            save_fpath = (
                audio_root.parent
                / "audio_slices"
                / audio_relfpath.with_suffix("")
                / f"{i}.wav"
            )
            save_fpath.parent.mkdir(parents=True, exist_ok=True)
            audio[slice(*utt.time_marks)].set_channels(1).export(
                save_fpath, format="wav"
            )
            manifest.append(
                {"audio_filepath": str(save_fpath), "text": extract_transcript(utt)}
            )

        save_fpath = (
            audio_root.parent / "manifests" / audio_relfpath.with_suffix(".json")
        )
        save_fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(save_fpath, "w", encoding="utf-8") as f:
            for row in manifest:
                json_repr = json.dumps(row, ensure_ascii=False)
                f.write(json_repr + "\n")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
