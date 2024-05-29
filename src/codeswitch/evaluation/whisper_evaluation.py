import jiwer
import torch
from lm_baseline_evaluation import check_device, read_synthetic_data
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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


def generate_transcripts(pipe, data):
    paths = [path for path, _ in data]
    results = pipe(paths)
    sentences = [result["text"] for result in results]
    return sentences


def main():
    # setup device
    device = check_device()

    # load data
    # data = read_splitted_miami()
    data = read_synthetic_data()
    data = data[:10]  # adjust for testing
    gold_sentences = [text for _, text in data]

    # load model
    model_dict = {"whisper-large-3": model_whisper_large_3()}

    for model_name, (model, processor) in model_dict.items():
        print(model_name)
        pipe = built_pipe(model, processor, device=device)
        generated_sentences = generate_transcripts(pipe, data)

        wer = jiwer.wer(gold_sentences, generated_sentences)
        print("wer:", wer)
        mer = jiwer.mer(gold_sentences, generated_sentences)
        print("mer:", mer)
        wil = jiwer.wil(gold_sentences, generated_sentences)
        print("wil:", wil)


if __name__ == "__main__":
    main()
