import hydra
from omegaconf import OmegaConf
from src.agent.graph import build_graph

# 1. Config 파일 로드 (기본 설정)
base_conf = OmegaConf.load("config/config.yaml")

# 2. 그래프 생성
app = build_graph(base_conf)