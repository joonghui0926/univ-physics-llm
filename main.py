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
        "output": {
              "solution": "해설 및 최종 답변",
              "skills": [ ... ]
         }
      }
    프롬프트 형식:
      Question: <input>
      Answer:
      <solution>
      Skills: <콤마로 구분된 스킬 목록>
    """
    prompt = f"Question: {example['input']}\n\nAnswer:\n"
    solution = example['output']['solution']
    skills = ", ".join(example['output']['skills'])
    full_text = prompt + solution + "\n\nSkills: " + skills
    return {"text": full_text}

def main():
    # 모델 및 데이터셋 경로 설정
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Llama 3.1 8B Instruct 모델
    dataset_path = "/SSD/guest/chojoonghui/physics_test_dataset.jsonl"  # JSONL 데이터셋 파일 경로

    # 데이터셋 로드 (JSONL 형식; 메모리 내 로딩)
    raw_dataset = load_dataset("json", data_files=dataset_path, keep_in_memory=True)["train"]
    
    # 데이터셋을 훈련용과 테스트용으로 분리 (90% 훈련, 10% 테스트)
    train_dataset, test_dataset = train_test_split(raw_dataset, test_size=0.1, random_state=42)

    # Dataset 객체로 변환
    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    # 훈련 데이터 전처리
    processed_train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    processed_test_dataset = test_dataset.map(preprocess, remove_columns=test_dataset.column_names)

    # 토크나이저 로드 및 패딩 토큰 설정
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 텍스트 토큰화 (최대 4096 토큰, 필요에 따라 조정)
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=4096)
    
    tokenized_train_dataset = processed_train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test_dataset = processed_test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4-bit 양자화 설정 (bitsandbytes)
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

    # 4-bit(QLoRA) 파인튜닝을 위한 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정 (4-bit 양자화 상태에서 파인튜닝)
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

    # 데이터 콜레이터 (MLM 비활성, CAUSAL LM 학습)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer 초기화 및 학습 시작
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,  # 테스트 데이터셋 추가
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()

    # 학습된 모델과 토크나이저 저장
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"모델이 {training_args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
