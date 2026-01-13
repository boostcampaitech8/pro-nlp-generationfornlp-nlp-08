import hydra
from omegaconf import OmegaConf
from src.agent.graph import build_graph

# Config 파일 로드 (기본 설정)
base_conf = OmegaConf.load("config/config.yaml")

app = build_graph(base_conf)
