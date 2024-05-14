import torch
import math
import time

def predict_next_token(logits, top_k, tokenizer):
    # Get top-k predicted token IDs and their probabilities
    top_k_ids = torch.topk(logits, k=top_k, dim=-1)
    top_k_token_probs = torch.softmax(top_k_ids.values, dim=-1)
    
    # Decode token IDs to tokens
    tokens = [tokenizer.decode(idx.item()) for idx in top_k_ids.indices[0]]
    
    # Pair tokens with their probabilities
    token_probabilities = list(zip(tokens, top_k_token_probs.tolist()[0]))
    return token_probabilities

def calculate_perplexity_ex(model, tokenizer, input_text, top_k=5):
    inputs = tokenizer(input_text, return_tensors="pt")

    input_ids = inputs["input_ids"].squeeze(0)
    # print(input_ids)

    total_loss = 0
    total_tokens = 0

    # Loop through all position
    for i in range(len(input_ids)):
        # Generate next token predictions
        with torch.no_grad():
            ids = input_ids[:i+1].unsqueeze(0)
            sent = tokenizer.decode(ids[0])
            # print(ids)
            
            outputs = model(input_ids = ids, labels = ids)
            # print(outputs)
        
        # predictions = outputs.logits[:, -1, :]
        loss = outputs.loss
        total_loss += loss.item()
        total_tokens += ids.size(1)
        
        # print(predictions)
        # next_token = predict_next_token(predictions, top_k, tokenizer)
        # print(next_token)
    
    return total_loss, total_tokens

def calculate_perplexity(model, tokenizer, sentences, top_k=5):
    total_loss = 0
    total_tokens = 0
    for sent in sentences:
        loss, tokens = calculate_perplexity_ex(model, tokenizer, sent, top_k)
        total_loss += loss
        total_tokens += tokens
        break

    return math.exp(total_loss / total_tokens)

def generate_sentenceids(model, tokenizer, sentences):
    sentenceids = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        sentenceids.extend([input_ids[:i+1].unsqueeze(0) for i in range(len(input_ids))])
    return sentenceids

def calculate_throughput(model, tokenizer, sentences, verbose=False):
    sentenceids = generate_sentenceids(model, tokenizer, sentences)
    start_time = time.time()
    for input_ids in sentenceids:
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=len(input_ids[0])+1)
        
        # print(tokenizer.decode(input_ids[0]))
        # print(tokenizer.decode(outputs[0],
        #                 skip_special_tokens = False))
        # print('\n')
    
    end_time = time.time()

    # calculate throughput using time difference
    return len(sentences)/(end_time - start_time)


if __name__ == "__main__":
    import torch
    import pandas as pd
    from tqdm import tqdm

    from blender_model import OnnxBlender
    from transformers import BlenderbotSmallTokenizer
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ## setup device
    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    #print(device)

    ## validation dataset
    data_path = "../data/synthetic_code_switch_data"
    val_dataset = pd.read_csv(data_path+"/output.csv")
    val_dataset['audio_filename'] = val_dataset['audio_filename'].apply(lambda x: data_path+"/audio/"+x)

    sentences = val_dataset['code-switched']

    ## load lm
    
    # Blenderbot_small
    original_repo_id = "facebook/blenderbot_small-90M"
    repo_id = "remzicam/xs_blenderbot_onnx"
    model_file_names = [
        "blenderbot_small-90M-encoder-quantized.onnx",
        "blenderbot_small-90M-decoder-quantized.onnx",
        "blenderbot_small-90M-init-decoder-quantized.onnx",
    ]
    # model=OnnxBlender(original_repo_id, repo_id, model_file_names)
    # tokenizer = BlenderbotSmallTokenizer.from_pretrained(original_repo_id)

    # perplexity = calculate_perplexity(model, tokenizer, sentences)
    # print("Blenderbot_small", perplexity)
    


    # T5 model
    print("T5")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    perplexity = calculate_perplexity(model, tokenizer, sentences)
    throughput = calculate_throughput(model, tokenizer, sentences)
    print("perplexity", perplexity) # perplexity 1.1774731633714555
    print("throughput", throughput) # throughput 0.6550413397877288
    
    # Phi-3-mini
    # model = AutoModelForCausalLM.from_pretrained(
    #     "microsoft/Phi-3-mini-128k-instruct-onnx",
    #     torch_dtype="auto", 
    #     trust_remote_code=True, 
    # )
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # perplexity = calculate_perplexity(model, tokenizer, sentences)
    # print("Phi-3-mini", perplexity)


