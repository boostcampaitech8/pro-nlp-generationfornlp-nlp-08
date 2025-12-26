import yaml
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import StoppingCriteriaList

from src.agent.nodes.router import router_node
from src.model.factory import ModelFactory
from src.model.pipeline_builder import PipelineBuilder
from src.model.stopping import StopOnSubstrings

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

    # Dict → OmegaConf (현재 코드 흐름과 맞추기 위함)
    router_model_cfg = OmegaConf.create(router_model_cfg)

    # =========================
    # 3. cfg 최소 구성 (router 테스트용)
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
    # 4. Router LLM 빌드
    # =========================
    factory = ModelFactory(cfg.model)
    router_model, router_tokenizer = factory.get_model("router")

    router_stopping = StoppingCriteriaList([
        StopOnSubstrings(
            cfg.model.router.stopping.stop_strings,
            router_tokenizer
        )
    ])

    router_llm = PipelineBuilder(
        model=router_model,
        tokenizer=router_tokenizer,
        generation_cfg=cfg.model.router.generation,
        stopping_criteria=router_stopping,
    ).build()

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
            result = router_node(
                state=state,
                cfg=cfg,
                llm=router_llm
            )
            print("✅ Router 결과")
            print(result)

        except Exception as e:
            print("❌ Router 실패")
            print(str(e))


if __name__ == "__main__":
    test_router()
