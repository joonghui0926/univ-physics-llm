import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Llama 3.1 model
    lora_model_path = "lora_llama3.1_8b_physics-last"    # Path to the trained LoRA adapter

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, lora_model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    user_input = input("Enter your physics problem: ")

    prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n"
    "Environment: ipython\n"
    "You are a physics problem-solving assistant. Your primary task is to guide users through a structured problem-solving process. \n"
    "For each problem, ensure that you:\n"
    "- Clearly define the problem and extract given data.\n"
    "- Emphasize key physical principles and explain why specific equations are applicable.\n"
    "- Solve the problem in a step-by-step manner with logical reasoning and structured calculations.\n"
    "- Use LaTeX notation for mathematical expressions to enhance readability.\n"
    "- Maintain clarity and conciseness while avoiding redundant phrasing.\n"
    "- Highlight problem-solving techniques that can be generalized for similar questions.\n"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    f"{user_input}\n\n"
    "Solution Format:\n"
    "**Step 1: Understanding the problem and identifying known/unknown variables**\n"
    "  - Clearly define what is given and what needs to be found.\n"
    "  - Identify relevant physical principles applicable to the scenario.\n"
    "\n"
    "**Step 2: Deriving key equations and explaining their significance**\n"
    "  - Present fundamental equations (e.g., $W = F \cdot d \\cos \\theta$) and justify their use.\n"
    "  - Provide a conceptual explanation of how these equations relate to the given problem.\n"
    "\n"
    "**Step 3: Step-by-step calculations with logical flow**\n"
    "  - Perform algebraic manipulations as needed before substituting values.\n"
    "  - Ensure clear and accurate unit consistency throughout calculations.\n"
    "\n"
    "**Final Answer:** Clearly state the numerical or symbolic result with appropriate units.\n"
    "\n"
    "**Problem-Solving Techniques:** Summarize the principles, equations, and general strategies that can be applied to similar problems in the future.\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "<|end_of_text|>"
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

    print("=== Generated Output ===")
    print(output_text)

if __name__ == "__main__":
    main()
