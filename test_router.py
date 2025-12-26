import yaml
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.agent.nodes.router import RouterNode

load_dotenv()


def test_router():
    # =========================
    # 1. prompt/router.yaml 로드
    # =========================
    with open("config/prompt/router.yaml", "r", encoding="utf-8") as f:
        router_prompt_cfg = yaml.safe_load(f)

    # =========================
    # 2. model/router.yaml 로드
    # =========================
    with open("config/model/qwen2.5_7b_it.yaml", "r", encoding="utf-8") as f:
        router_model_cfg = yaml.safe_load(f)

    router_model_cfg = OmegaConf.create(router_model_cfg)

    # =========================
    # 3. cfg 최소 구성 (RouterNode가 기대하는 형태)
    # =========================
    cfg = OmegaConf.create({
        "prompt": {
            "router": {
                "system": router_prompt_cfg["system"]
            }
        },
        "model": {
            "router": router_model_cfg
        }
    })

    # =========================
    # 4. RouterNode 생성 (여기서 모델 로드됨)
    # =========================
    router = RouterNode(cfg)

    # =========================
    # 5. 데이터 로드
    # =========================
    df = pd.read_csv("/data/ephemeral/home/data/train.csv")
    samples = df.sample(n=3, random_state=42)

    # =========================
    # 6. 테스트 실행
    # =========================
    for i, (_, row) in enumerate(samples.iterrows(), start=1):
        state = {
            "paragraph": row["paragraph"],
            "problem": {
                "question": row["problems"],
            }
        }

        print(f"\n===== 테스트 입력 {i} =====")
        print(f"Paragraph: {state['paragraph'][:200]}...")
        print(f"Problem: {state['problem']['question']}")

        try:
            result = router(state)
            print("✅ Router 결과")
            print(result)

        except Exception as e:
            print("❌ Router 실패")
            print(str(e))


if __name__ == "__main__":
    test_router()
