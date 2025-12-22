import hydra
from omegaconf import DictConfig, OmegaConf
from src.agent.graph import build_graph

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(f"🔥 System Initializing... Model: {cfg.model.name}")
    print(f"⚙️  Prompt Loaded: {list(cfg.prompt.keys())}") # router, solver, critic 로드 확인

    # Hydra 설정을 그래프에 주입하여 빌드
    app = build_graph(cfg)
    
    # 테스트용 입력 데이터
    inputs = {
        "paragraph": "신라 하기는 진골 귀족의 왕위 쟁탈전이 심화되던 시기이다...",
        "problem": {
            "question": "윗글의 시기에 일어난 사건으로 적절한 것은?", 
            "choices": ["1. 적고적의 난", "2. 웅진 천도", "3. 묘청의 난"]
        }
    }
    
    # 실행
    result = app.invoke(inputs)
    
    print("-" * 50)
    print(f"✅ Final Result: {result.get('final_answer')}")

if __name__ == "__main__":
    main()