from codeswitch.evaluation.data_filter import extract_codeswitch_sentences
from whisper_evaluation import generate_transcripts, model_whisper_large_3

from codeswitch.dataloader import (  # noqa; check_device,; read_synthetic_data,
    read_splitted_miami,
    read_text_data,
)


def synthetic_0():
    data = read_splitted_miami()
    gold_sentences = [text for _, text in data]
    gold_sentences, indexs = extract_codeswitch_sentences(gold_sentences, cs_coef=0.8)
    data = [(data[index][0], data[index][1]) for index in indexs]

    print("length of data:", len(data))
    model, processor = model_whisper_large_3()
    generated_sentences = generate_transcripts(model, processor, data)

    with open("whisper_miami_output.txt", "w", encoding="utf-8") as file:
        for gold_sentence, generated_sentence in zip(
            gold_sentences, generated_sentences
        ):
            file.write(f"{gold_sentence}\t{generated_sentence}\n")


def main():
    gold_sentences = read_text_data(
        "/home/public/data/synthetic/utterance_20240605_test.txt"
    )
    data = [
        (f"/home/public/data/synthetic/utterance_20240605_test_audio/{i}.wav", sent)
        for i, sent in enumerate(gold_sentences)
    ]
    print("length of data:", len(data))

    model, processor = model_whisper_large_3()

    number_of_batch = 10
    batch_size = len(data) // number_of_batch

    for i in range(number_of_batch):
        generated_sentences = generate_transcripts(
            model, processor, data[i * batch_size : (i + 1) * batch_size]
        )

        with open(
            f"whisper_utterance_20240605_test_{i}.txt", "w", encoding="utf-8"
        ) as file:
            for gold_sentence, generated_sentence in zip(
                gold_sentences[i * batch_size : (i + 1) * batch_size],
                generated_sentences,
            ):
                file.write(f"{gold_sentence}\t{generated_sentence}\n")

    generated_sentences = generate_transcripts(
        model, processor, data[number_of_batch * batch_size :]
    )
    with open(
        f"whisper_utterance_20240605_test_{number_of_batch}.txt", "w", encoding="utf-8"
    ) as file:
        for gold_sentence, generated_sentence in zip(
            gold_sentences[number_of_batch * batch_size :], generated_sentences
        ):
            file.write(f"{gold_sentence}\t{generated_sentence}\n")


if __name__ == "__main__":
    main()
