# 프롬프트
PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

풀이 지침:
- 지문과 질문을 분석한 뒤 선택지를 비교한다.
- 풀이 과정은 출력하지 않는다.

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
다른 말은 절대 하지 마세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

풀이 지침:
- 지문과 질문을 분석한 뒤 선택지를 비교한다.
- 풀이 과정은 출력하지 않는다.

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
다른 말은 절대 하지 마세요.
정답:"""

import torch
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from unsloth import FastLanguageModel
import random

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)
# TODO 1. 모델 로드 
checkpoint_path = "outputs_qwen/checkpoint-457" 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path,
    max_seq_length = 3072,
    load_in_4bit = True,
    device_map = "auto",
)
# FastLanguageModel.for_inference(model)

# 2. 데이터 로드 및 전처리
test_df = pd.read_csv('val_stratified_v1.csv')
records = []
for _, row in test_df.iterrows():
    problems = literal_eval(row['problems'])
    records.append({
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': str(problems.get('answer')), # 정답(라벨) 저장
        "question_plus": problems.get('question_plus', None),
    })
test_df = pd.DataFrame(records)

# 3. 인퍼런스 및 결과 저장용 리스트
y_true = [] # 실제 정답
y_pred = [] # 모델 예측값

pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
choice_tokens = ["1", "2", "3", "4", "5"]
choice_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in choice_tokens]

with torch.inference_mode():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # 프롬프트 생성 (질문+보기 여부에 따라)
        choices_string = "\n".join([f"{idx + 1} - {c}" for idx, c in enumerate(row["choices"])])
        if row["question_plus"]:
            user_msg = PROMPT_QUESTION_PLUS.format(paragraph=row["paragraph"], question=row["question"], question_plus=row["question_plus"], choices=choices_string)
        else:
            user_msg = PROMPT_NO_QUESTION_PLUS.format(paragraph=row["paragraph"], question=row["question"], choices=choices_string)

        messages = [
            {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
            {"role": "user", "content": user_msg},
        ]

        # 모델 예측
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model(input_ids)
        last_token_logits = outputs.logits[0, -1, :]
        
        len_choices = len(row["choices"])
        choice_logits = last_token_logits[choice_ids[:len_choices]]
        probs = torch.softmax(choice_logits, dim=0).cpu().numpy()
        
        predict_value = pred_choices_map[np.argmax(probs)]
        
        # 결과 수집
        y_true.append(row["answer"])
        y_pred.append(predict_value)

print(y_true[:10])
print(y_pred[:10])

# 4. 성능 평가 (F1 Score 및 Accuracy)
accuracy = accuracy_score(y_true, y_pred)
# 객관식은 다중 클래스(Multi-class)이므로 average='macro' 또는 'weighted' 사용
f1_macro = f1_score(y_true, y_pred, average='macro')

print("\n" + "="*30)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print("="*30)

# 상세 리포트 (클래스별 성능 확인)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))