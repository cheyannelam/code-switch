import time

import pandas as pd
import torch
from blender_model import OnnxBlender  # noqa
from transformers import (  # noqa
    AutoModelForCausalLM,
    AutoTokenizer,
    BlenderbotForCausalLM,
    BlenderbotSmallTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# from transformers import Phi3ForCausalLM


def predict_next_token(tokenizer, logits, top_k=5):
    # Get top-k predicted token IDs and their probabilities
    logits_softmax = torch.softmax(logits, dim=-1)
    top_k_ids = torch.topk(logits_softmax, k=top_k, dim=-1)
    top_k_token_probs = top_k_ids.values

    # Decode token IDs to tokens
    tokens = [tokenizer.decode(idx.item()) for idx in top_k_ids.indices[0]]

    # Pair tokens with their probabilities
    token_probabilities = list(zip(tokens, top_k_token_probs.tolist()[0]))
    return token_probabilities


def sentence_predict_next_token(model, tokenizer, sent):
    inputs = tokenizer(sent, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        sent = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print("\ninput sent:", sent)

        outputs = model(input_ids=input_ids, labels=input_ids)
    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    target_probabilities = probabilities[
        0, range(len(input_ids[0]) - 1), input_ids[0][1:]
    ]

    for i in range(len(input_ids[0]) - 1):
        input_sent = tokenizer.decode(input_ids[0][: i + 1], skip_special_tokens=False)
        gold = (
            tokenizer.decode(input_ids[0][i + 1], skip_special_tokens=False),
            float(target_probabilities[i]),
        )
        next_token = predict_next_token(tokenizer, logits[:, i, :])
        print(f'input: "{input_sent}"')
        print(f"gold token: {gold}")
        print(f"predicted token: {next_token}")


def calculate_cross_entropy(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        # sent = tokenizer.decode(input_ids[0])
        # print("\ninput sent:", sent)

        outputs = model(input_ids=input_ids, labels=input_ids)

    logits = outputs.logits
    # next_token_id = torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    target_probabilities = probabilities[
        0, range(len(input_ids[0]) - 1), input_ids[0][1:]
    ]

    # print("logits", logits.size())
    # print("next_token_id", next_token_id.size())
    # print(tokenizer.decode(next_token_id[0]))
    # print("probabilities", probabilities.size())
    # print("target_probabilities", target_probabilities)

    return -torch.log(target_probabilities)


def calculate_perplexity(model, tokenizer, sentences, average="micro"):
    assert average in ["micro", "macro"]

    entropies = []
    for sent in sentences:
        cross_entropy = calculate_cross_entropy(model, tokenizer, sent)
        entropies.append(cross_entropy)

    entropies_1d = None
    if average == "micro":
        entropies_1d = torch.cat(entropies)
        # print("cat", entropies_1d.size()
    elif average == "macro":
        entropies_1d = torch.Tensor([torch.mean(entropy) for entropy in entropies])

    perplexity = torch.exp(torch.mean(entropies_1d))
    return float(perplexity)


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
    assert average in ["micro", "macro"]

    timespan_lst = []
    length_lst = []

    for sent in sentences:
        timespan, length = measure_sentence_time(model, tokenizer, sent)
        timespan_lst.append(timespan)
        length_lst.append(length)

    throughput = None
    if average == "micro":
        throughput = sum(length_lst) / sum(timespan_lst)
    elif average == "macro":
        throughput = len(timespan_lst) / sum(timespan_lst)

    return throughput


def model_blendersmall():
    # Blenderbot_small
    original_repo_id = "facebook/blenderbot_small-90M"
    repo_id = "remzicam/xs_blenderbot_onnx"
    model_file_names = [
        "blenderbot_small-90M-encoder-quantized.onnx",
        "blenderbot_small-90M-decoder-quantized.onnx",
        "blenderbot_small-90M-init-decoder-quantized.onnx",
    ]
    model = OnnxBlender(original_repo_id, repo_id, model_file_names)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(original_repo_id)

    return model, tokenizer


def model_t5():
    # T5 model
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    return model, tokenizer


def model_phi3_mini():
    # Phi-3-mini
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct-onnx",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    return model, tokenizer


def model_gpt2():
    # gpt2
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    return model, tokenizer


def model_blenderbotForCausalLM():  # noqa  # pylint: disable=C0103
    model = BlenderbotForCausalLM.from_pretrained(
        "facebook/blenderbot-400M-distill", add_cross_attention=False
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    return model, tokenizer


def model_phi3ForCausalLM():  # noqa  # pylint: disable=C0103
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-3-mini-4k-instruct", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
    return model, tokenizer


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

    sentences = val_dataset["code-switched"][:]

    # load lm
    model_dict = {
        "gpt2": model_gpt2(),
        # "blenderbotForCausalLM": model_blenderbotForCausalLM(),
        # "phi3ForCausalLM": model_phi3ForCausalLM()
    }

    # for model_name, (model, tokenizer) in model_dict.items():

    #     print("\nmodel_name", model_name)
    #     sentence_predict_next_token(model, tokenizer, sentences[2])

    for model_name, (model, tokenizer) in model_dict.items():

        print("\nmodel_name", model_name)
        perplexity = calculate_perplexity(model, tokenizer, sentences, average="micro")
        print("perplexity", perplexity)
        throughput = calculate_throughput(model, tokenizer, sentences)
        print("throughput", throughput)


if __name__ == "__main__":
    main()
