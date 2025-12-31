import hydra
from omegaconf import DictConfig, OmegaConf
from src.agent.graph import build_graph
from dotenv import load_dotenv
from src.data_loader.data_loader import load_dataset

load_dotenv()  # langsmith 설정을 위해

# TODO: 랜덤 시드 고정 기능 추가

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    app = build_graph(cfg)
    dataset = load_dataset(cfg.path.validation)
    for i in cfg.debug.test_indices:
        print(f"--- Test Index: {i} ---")
        data = dataset[i]
        app.invoke(data)


if __name__ == "__main__":
    main()
