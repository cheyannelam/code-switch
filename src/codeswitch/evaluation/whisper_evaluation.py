import os

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from wer_evaluation import wer_sentence

from codeswitch.dataloader import check_device  # noqa
from codeswitch.dataloader import read_text_data  # noqa

# from codeswitch.dataloader import (  # pylint: disable=unused-import
#     read_splitted_miami,
#     read_synthetic_data,
# )


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
        device = check_device()

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


def generate_transcripts(model, processor, data, prompt=None):
    output = []
    for path, _ in tqdm(data):
        input_speech, sample_rate = torchaudio.load(path)
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
        transcription = processor.decode(
            output_with_prompt[0], skip_special_tokens=True
        )
        # print(transcription)
        output.append(transcription)
    return output


def main():
    # setup device
    # device = check_device()

    # load data
    # data = read_splitted_miami()
    # data = read_synthetic_data()
    # data = data[:1]  # adjust for testing
    # gold_sentences = [text for _, text in data]

    gold_sentences = read_text_data()

    audio_folderpath = os.path.join(
        os.path.dirname(__file__), "../text-to-speech/utterance"
    )
    audio_folderpath = os.path.normpath(audio_folderpath)

    data = [
        (os.path.join(audio_folderpath, f"{i}.wav"), sent)
        for i, sent in enumerate(gold_sentences[:5])
    ]

    # print(data)

    # load model
    model_dict = {"whisper-large-3": model_whisper_large_3()}

    for model_name, (model, processor) in model_dict.items():
        print(model_name)
        # pipe = built_pipe(model, processor, device=device)
        # generated_sentences = generate_transcripts(pipe, data)
        generated_sentences = generate_transcripts(model, processor, data)

        for gold_sentence, generated_sentence in zip(
            gold_sentences, generated_sentences
        ):
            print("gold:     ", gold_sentence)
            print("generated:", generated_sentence)
            print(
                "wer:",
                wer_sentence(gold_sentence, generated_sentence, case_sensitive=False),
            )
            print("-----")

        # wer = jiwer.wer(gold_sentences, generated_sentences)
        # print("wer:", wer)
        # mer = jiwer.mer(gold_sentences, generated_sentences)
        # print("mer:", mer)
        # wil = jiwer.wil(gold_sentences, generated_sentences)
        # print("wil:", wil)


if __name__ == "__main__":
    main()
