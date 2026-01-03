import hydra
from typing import cast
from omegaconf import DictConfig
from src.agent.graph import build_graph
from dotenv import load_dotenv
from src.data_loader.data_loader import load_dataset
from src.agent.state import AgentState

# TODO: 랜덤 시드 고정 기능 추가


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    app = build_graph(cfg)
    dataset = load_dataset(cfg.path.data.validate)
    for i in cfg.debug.test_indices:
        print(f"--- Test Index: {i} ---")
        data = dataset[i]
        app.invoke(cast(AgentState, {"problem": data}))


if __name__ == "__main__":
    main()
