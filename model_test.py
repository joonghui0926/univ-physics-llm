import torch
import faiss
import pickle
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# FAISS Index 및 모델 경로 설정
FAISS_INDEX_PATH = "physics_faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_MODEL_PATH = "lora_llama3.1_8b_physics-last"

print(f"LORA_MODEL_PATH: {os.path.abspath(LORA_MODEL_PATH)}")
print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE')}")
print(f"Model Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

# FAISS 인덱스 로드 함수
def load_faiss_index():
    """
    FAISS 인덱스를 로드하는 함수.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    doc_texts = None
    pkl_path = os.path.join(FAISS_INDEX_PATH, "index.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            doc_texts = pickle.load(f)  # 메타데이터 로드

    return vectorstore, doc_texts

# FAISS 검색 시스템 로드
vectorstore, doc_texts = load_faiss_index()
print("✅ FAISS Index 로드 완료!")

# 문장 임베딩 모델 로드
embedder = SentenceTransformer(EMBEDDING_MODEL)

def retrieve_reference_material(query, top_k=3):
    """
    사용자의 물리 문제와 관련된 문단을 FAISS로 검색.
    """
    docs = vectorstore.similarity_search(query, k=top_k)
    retrieved_texts = [doc.page_content for doc in docs]
    return "\n\n".join(retrieved_texts)

def generate_solution(model, tokenizer, user_input):
    """
    사용자가 입력한 물리 문제를 해결하기 위한 모델 응답 생성.
    """
    # 검색된 문단 가져오기
    reference_material = retrieve_reference_material(user_input)

    # 프롬프트 생성
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Environment: ipython\n"
        "You are a physics problem-solving assistant. Your primary task is to guide users through a structured problem-solving process.\n"
        "For each problem, ensure that you:\n"
        "- Clearly define the problem and extract given data.\n"
        "- Emphasize key physical principles and explain why specific equations are applicable.\n"
        "- Solve the problem in a step-by-step manner with logical reasoning and structured calculations.\n"
        "- Use LaTeX notation for mathematical expressions to enhance readability.\n"
        "- Maintain clarity and conciseness while avoiding redundant phrasing.\n"
        "- After provide a dinal answer, do not forget to Highlight problem-solving techniques that can be generalized for similar questions.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Reference Material:\n{reference_material}\n\n"
        f"Question:\n{user_input}\n\n"
        "Solution Format:\n"
        "**Step 1: Understanding the problem and identifying known/unknown variables**\n"
        "  - Clearly define what is given and what needs to be found.\n"
        "  - Identify relevant physical principles applicable to the scenario.\n"
        "\n"
        "**Step 2: Deriving key equations and explaining their significance**\n"
        "  - Present fundamental equations (e.g., $W = F \\cdot d \\cos \\theta$) and justify their use.\n"
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

    # 토큰 변환
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(model.device)

    # 모델 추론 실행
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    # 출력 후처리
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

def extract_assistant_response(full_text):
    """
    'assistant' 이후 부분만 추출하여 반환.
    """
    if "assistant" in full_text:
        return full_text.split("assistant", 1)[-1].strip()
    return full_text  # 혹시 "assistant"가 없으면 원본 그대로 반환

def generate_latex_solution(model, tokenizer, plain_text_solution):
    """
    기존 텍스트 해설을 LaTeX 문서 형태로 변환.
    LoRA 모델을 그대로 사용하여 변환 수행.
    """
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are an AI assistant that converts structured text into LaTeX format for Overleaf.\n"
        "Convert the given physics solution into a well-formatted LaTeX document.\n"
        "Ensure that mathematical expressions are enclosed within \\( \\) for inline math "
        "or \\[ \\] for display math mode.\n"
        "Use \\textbf{} for headings and keep the document structured.\n"
        "To present bold characters, you should use \subsection*{ } instead of ##\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Convert the following solution to LaTeX format:\n\n{plain_text_solution}\n\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    latex_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return latex_output

def main():
    """
    LoRA 기반 Llama 3.1 모델을 사용하여 물리 문제를 해결한 후,
    LaTeX 변환 시에는 "assistant" 이후 응답 부분만 변환하도록 수정.
    """
    # LoRA 모델 로드 (문제 풀이용)
    base_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_MODEL,
        device_map="balanced_low_0",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    user_input = input("Enter your physics problem: ")
    
    # 문제 풀이 수행
    full_solution = generate_solution(model, tokenizer, user_input)
    
    # "assistant" 이후 부분만 추출
    extracted_solution = extract_assistant_response(full_solution)

    print("\n=== Generated Solution ===")
    print(extracted_solution)

    # LaTeX 변환 수행
    latex_solution = generate_latex_solution(model, tokenizer, extracted_solution)

    print("\n=== LaTeX Formatted Solution ===")
    print(latex_solution)

if __name__ == "__main__":
    main()
