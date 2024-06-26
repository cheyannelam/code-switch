import json
import os

import click
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from codeswitch import dataloader


def model_whisper_large_3(torch_dtype=None):
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def built_pipe(model, processor, device=None):
    if device is None:
        device = dataloader.check_device()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device,
    )
    return pipe


def generate_transcript(model, processor, audio_filepath, prompt=None):
    input_speech, sample_rate = torchaudio.load(audio_filepath)
    input_speech = input_speech[0]
    input_speech = torchaudio.functional.resample(input_speech, sample_rate, 16000)
    input_features = processor(
        input_speech, sampling_rate=16000, return_tensors="pt"
    ).input_features
    if prompt is None:
        output_with_prompt = model.generate(input_features)
    else:
        prompt_ids = torch.tensor(processor.get_prompt_ids(prompt))
        output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids)
    transcription = processor.decode(output_with_prompt[0], skip_special_tokens=True)
    # print(transcription)
    return transcription


def generate_transcripts(model, processor, pairs, prompt=None, output_foldername=""):
    """
    pairs: list of (audio_filepath, text) pair

    """
    if output_foldername != "" and not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    output = []
    for i, (audio_filepath, _) in tqdm(enumerate(pairs)):
        transcription = generate_transcript(model, processor, audio_filepath, prompt)
        line = {"audio_filepath": audio_filepath, "text": transcription}

        output_filename = f"{i}.json"
        write_filepath = os.path.join(output_foldername, output_filename)

        with open(write_filepath, "w", encoding="utf-8") as file:
            j = json.dumps(line, ensure_ascii=False)
            file.write(j)

        output.append((write_filepath, j))

    write_filepath = os.path.join(output_foldername, "transcripts.json")
    with open(write_filepath, "w", encoding="utf-8") as file:
        for _, j in output:
            file.write(f"{j}\n")

    return output


@click.command()
@click.option(
    "--data-path",
    default="/home/public/data/synthetic_temp/utterance_20240605_test_audio/manifest.json",
    help="Path to the manifest file",
)
@click.option(
    "--output-foldername",
    default="/home/public/data/synthetic_temp/utterance_20240605_test_whisper_transcriptions",
    help="Output foldername",
)
def main(data_path, output_foldername):
    data = dataloader.read_json(data_path)
    pairs = [(line["audio_filepath"], line["text"]) for line in data]

    model, processor = model_whisper_large_3()
    generate_transcripts(
        model,
        processor,
        pairs,
        output_foldername=output_foldername,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
