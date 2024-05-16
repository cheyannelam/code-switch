import time

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def predict_next_token(logits, top_k, tokenizer):
    # Get top-k predicted token IDs and their probabilities
    top_k_ids = torch.topk(logits, k=top_k, dim=-1)
    top_k_token_probs = torch.softmax(top_k_ids.values, dim=-1)

    # Decode token IDs to tokens
    tokens = [tokenizer.decode(idx.item()) for idx in top_k_ids.indices[0]]

    # Pair tokens with their probabilities
    token_probabilities = list(zip(tokens, top_k_token_probs.tolist()[0]))
    return token_probabilities


def calculate_cross_entropy(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        # sent = tokenizer.decode(input_ids[0])
        # print("\ninput sent:", sent)

        outputs = model(input_ids=input_ids, labels=input_ids)

    logits = outputs.logits
    next_token_id = torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    target_probabilities = probabilities[
        0, range(len(next_token_id[0])), next_token_id[0]
    ]

    # print("logits", logits.size())
    # print("next_token_id", next_token_id.size())
    # print(tokenizer.decode(next_token_id[0]))
    # print("probabilities", probabilities.size())
    # print("target_probabilities", target_probabilities.size())

    return -torch.log(target_probabilities)


def calculate_perplexity(model, tokenizer, sentences, average="micro"):
    assert(average in ["micro", "macro"])
    
    entropies = []
    for sent in sentences:
        cross_entropy = calculate_cross_entropy(model, tokenizer, sent)
        entropies.append(cross_entropy)
    
    if average == "micro":
        entropies_1d = torch.cat(entropies)
        # print("cat", entropies_1d.size())

    elif average == "macro":
        entropies_1d = torch.Tensor([torch.mean(entropy) for entropy in entropies])

    perplexity = torch.exp(torch.mean(entropies_1d))
    return perplexity


def measure_sentence_time(model, tokenizer, sent):
    inputs = tokenizer(sent, return_tensors="pt")
    input_ids = inputs["input_ids"].squeeze(0)

    input_ids_stream = [input_ids[: i + 1].unsqueeze(0) for i in range(len(input_ids))]
    # print("input_ids_stream[0]", input_ids_stream[0])

    start_time = time.time()
    outputs = model(input_ids=input_ids_stream[0])
    past_key_values = outputs.past_key_values

    for ids in input_ids_stream[1:]:
        outputs = model(input_ids=ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

    end_time = time.time()

    sentence_time = end_time - start_time
    ids_length = len(input_ids)

    return sentence_time, ids_length


def calculate_throughput(model, tokenizer, sentences, average="micro"):
    assert(average in ["micro", "macro"])
    
    timespan_lst = []
    length_lst = []

    for sent in sentences:
        timespan, length = measure_sentence_time(model, tokenizer, sent)
        timespan_lst.append(timespan)
        length_lst.append(length)

    if average == "micro":
        throughput = length_lst.sum() / timespan_lst.sum()
    elif average == "macro":
        throughput = len(time_lst) / timespan_lst.sum()

    return throughput


def main():
    # setup device
    # torch.random.manual_seed(0)
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    # else:
    #     device = "cpu"
    # print(device)

    # validation dataset
    data_path = "../../../data/synthetic_code_switch_data"
    val_dataset = pd.read_csv(data_path + "/output.csv")
    val_dataset["audio_filename"] = val_dataset["audio_filename"].apply(
        lambda x: data_path + "/audio/" + x
    )

    sentences = val_dataset["code-switched"][:2]

    # load lm

    # Blenderbot_small
    # original_repo_id = "facebook/blenderbot_small-90M"
    # repo_id = "remzicam/xs_blenderbot_onnx"
    # model_file_names = [
    #     "blenderbot_small-90M-encoder-quantized.onnx",
    #     "blenderbot_small-90M-decoder-quantized.onnx",
    #     "blenderbot_small-90M-init-decoder-quantized.onnx",
    # ]
    # model=OnnxBlender(original_repo_id, repo_id, model_file_names)
    # tokenizer = BlenderbotSmallTokenizer.from_pretrained(original_repo_id)

    # perplexity = calculate_perplexity(model, tokenizer, sentences)
    # print("Blenderbot_small", perplexity)

    # T5 model
    print("T5")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    perplexity = calculate_perplexity(model, tokenizer, sentences, average="micro")
    throughput = calculate_throughput(model, tokenizer, sentences)
    print("perplexity", perplexity)  # perplexity 1.1774731633714555
    print("throughput", throughput)  # throughput 0.6550413397877288

    # Phi-3-mini
    # model = AutoModelForCausalLM.from_pretrained(
    #     "microsoft/Phi-3-mini-128k-instruct-onnx",
    #     torch_dtype="auto",
    #     trust_remote_code=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # perplexity = calculate_perplexity(model, tokenizer, sentences)
    # print("Phi-3-mini", perplexity)


if __name__ == "__main__":
    main()
