import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

def preprocess(example):
    """
    입력 예제:
      {
        "input": "문제 내용",
        "reference": "검색된 문단",
        "output": {
              "solution": "해설 및 최종 답변",
              "skills": [ ... ]
         }
      }
    수정된 프롬프트 형식 (RAG 반영):
      Reference Material:
      <검색된 문단>
      Question: <input>
      Answer:
      <solution>
      Skills: <콤마로 구분된 스킬 목록>
    """

    reference_material = example.get("reference", "")  # 참조 문단이 있으면 포함
    prompt = f"Reference Material:\n{reference_material}\n\nQuestion: {example['input']}\n\nAnswer:\n"

    solution = example["output"]["solution"]
    skills = ", ".join(example["output"]["skills"])
    full_text = prompt + solution + "\n\nSkills: " + skills
    
    return {"text": full_text}

def main():
    # 모델 및 데이터셋 경로 설정
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/SSD/guest/chojoonghui/physics_test_dataset_rag.jsonl"  # RAG 적용된 JSONL 파일

    # 데이터셋 로드 (JSONL 형식)
    raw_dataset = load_dataset("json", data_files=dataset_path, keep_in_memory=True)["train"]
    
    # 데이터셋을 훈련/테스트 세트로 분리 (90% 훈련, 10% 테스트)
    train_dataset, test_dataset = train_test_split(raw_dataset, test_size=0.1, random_state=42)

    # Dataset 객체 변환
    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    # RAG 기반 데이터 전처리 (검색된 문단 추가)
    processed_train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    processed_test_dataset = test_dataset.map(preprocess, remove_columns=test_dataset.column_names)

    # 토크나이저 로드 및 패딩 토큰 설정
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 텍스트 토큰화 (최대 4096 토큰)
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=4096)
    
    tokenized_train_dataset = processed_train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test_dataset = processed_test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 모델 로드 (4-bit 양자화 적용)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 설정
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="lora_llama3.1_8b_physics-last",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
    )

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer 초기화 및 학습 시작
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()

    # 학습된 모델 및 토크나이저 저장
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"✅ 모델이 {training_args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
