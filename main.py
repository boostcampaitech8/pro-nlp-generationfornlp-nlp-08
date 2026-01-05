import hydra
from typing import cast
from omegaconf import DictConfig
from src.agent.graph import build_graph
from dotenv import load_dotenv
from src.data_loader.data_loader import load_dataset
from src.agent.state import AgentState
import os
import random
import numpy as np
import torch


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    app = build_graph(cfg)
    dataset = load_dataset(cfg.path.data.validate)
    for i in cfg.debug.test_indices:
        print(f"--- Test Index: {i} ---")
        data = dataset[i]
        app.invoke(cast(AgentState, {"problem": data}))

def set_seed(seed: int) -> None:
    '''
    시드를 고정하여 실험의 재현성을 확보
    Args:
        param seed: 
        type seed: int
    '''
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경일 경우
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()
