# Univ-Physics-LLM

**Univ-Physics-LLM**은 대학 물리학 문제 해결을 위한 **Llama 3.1 8B** 기반의 AI 모델입니다.  
이 모델은 **QLoRA + 4-bit Quantization** 기법을 적용하여 최적화되었으며, 일반 물리학 문제를 단계적으로 해결하는 기능을 제공합니다.

---

## 1. 프로젝트 개요

- **목표:** AI를 활용하여 대학 물리학 문제를 보다 체계적으로 해결
- **사용 모델:** `Meta-Llama-3.1-8B-Instruct`
- **주요 기능:**
  - 물리 문제를 이해하고 핵심 원리를 도출
  - 단계별 풀이 제공 및 LaTeX 수식 지원
  - JSONL 형식의 학습 데이터를 기반으로 모델 학습

---

## 2. 설치 방법

### 필수 패키지 설치

아래 명령어를 실행하여 필요한 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

### 모델 다운로드 (Hugging Face 활용)
```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
```

### GitHub 저장소 클론
```bash
git clone https://github.com/joonghui0926/univ-physics-llm.git
cd univ-physics-llm
```

---

## 3. 실행 방법

### 기본 모델 테스트
```bash
python basic_model_test.py
```
기본 Llama 3.1 모델을 사용하여 간단한 물리 문제를 테스트합니다.

### 모델 학습 시작 (QLoRA 적용)
```bash
python main.py
```
physics_test_dataset.jsonl을 기반으로 QLoRA를 활용한 파인튜닝을 수행합니다.

### 학습된 모델 테스트
```bash
python model_test.py
```
훈련된 모델이 일반 물리학 문제를 얼마나 잘 해결하는지 평가합니다.

---

## 4. 프로젝트 구조
```plaintext
📦 univ-physics-llm
├── 📂 datasets_cache          # 데이터셋 캐시
├── 📂 lora_finetuned         # LoRA 훈련된 가중치
├── 📂 results                # 학습 결과 저장 폴더
├── 📜 basic_model_test.py    # 기본 모델 테스트 코드
├── 📜 model_test.py          # 훈련된 모델 테스트 코드
├── 📜 main.py                # 학습 스크립트
├── 📜 requirements.txt       # 필수 라이브러리 목록
├── 📜 llama3.yaml            # 모델 및 학습 설정 파일
├── 📜 physics_test_dataset.jsonl  # 학습용 물리 문제 데이터셋
└── 📜 README.md              # 프로젝트 설명 파일
```

---

## 5. 데이터셋 정보 (physics_test_dataset.jsonl)
예제 Jsonl 형식:
```json
{
  "input": "A ball is thrown vertically upward with a velocity of 10 m/s. How high does it go?",
  "output": {
    "solution": "Using kinematics equation h = v^2 / (2g)...",
    "skills": ["Kinematics", "Energy Conservation"]
  }
}
```

구성 요소:
input: 문제 내용
solution: AI가 제공할 풀이
skills: 문제 해결에 필요한 개념

---

## 6. 향후 계획

데이터 증강
