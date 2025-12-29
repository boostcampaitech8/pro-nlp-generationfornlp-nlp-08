import os
import ast
import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from src.agent.graph import build_graph

def parse_choices(choices_raw):
    try:
        if isinstance(choices_raw, list):
            return choices_raw
        return ast.literal_eval(choices_raw)
    except:
        return []

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(f"🚀 [Inference] Model: {cfg.model.name}")
    
    # Config에서 경로 가져오기
    data_path = cfg.path.input
    output_path = cfg.path.output
    
    print(f"📂 Input: {data_path}")
    print(f"💾 Output: {output_path}")

    # 1. 그래프 빌드
    app = build_graph(cfg)

    # 2. 데이터 로드
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} rows.")
    else:
        print(f"⚠️ {data_path} not found. Running with Dummy Data.")
        df = pd.DataFrame([
            {
                "id": "dummy_001",
                "paragraph": "테스트 지문...",
                "question": "테스트 질문...",
                "choices": "['1. A', '2. B']"
            }
        ])

    results = []

    # 3. 추론 루프
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            inputs = {
                "paragraph": row.get('paragraph', ''),
                "problem": {
                    "question": row.get('question', ''),
                    "choices": parse_choices(row.get('choices', '[]'))
                }
            }
            
            output = app.invoke(inputs)
            
            final_ans = output.get("final_answer", -1)
            results.append({
                "id": row['id'],
                "answer": final_ans
            })
            
        except Exception as e:
            print(f"❌ Error at ID {row.get('id')}: {e}")
            results.append({"id": row['id'], "answer": -1})

    # 4. 결과 저장
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    print(f"🎉 Saved to {output_path}")

if __name__ == "__main__":
    main()