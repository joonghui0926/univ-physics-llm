import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def ask_question(model, tokenizer, question):
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Environment: ipython\n"
        "solve it"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{question}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask.to(model.device)
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def main():
    model, tokenizer = load_model()
    while True:
        question = input("Enter your physics question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        response = ask_question(model, tokenizer, question)
        print("\n=== Answer ===\n")
        print(response)

if __name__ == "__main__":
    main()
