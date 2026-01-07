import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(".."))
import pickle
import src.agent.state
import hydra
from typing import cast
from omegaconf import DictConfig
from dotenv import load_dotenv
from src.data_loader.data_loader import load_dataset
from src.agent.state import AgentState
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import warnings
import logging
from transformers import logging as transformers_logging
from transformers.utils import logging as hf_logging
import pickle


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)
hf_logging.disable_progress_bar()
logging.getLogger("ddgs").setLevel(logging.ERROR)  # DuckDuckGo 검색 로그
logging.getLogger("primp").setLevel(logging.ERROR)  # 검색 라이브러리 로그
logging.getLogger("httpx").setLevel(logging.ERROR)  # HTTP 요청 로그
logging.getLogger("httpcore").setLevel(logging.ERROR)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    app_with_critic = build_graph_with_critic(cfg)
    app_without_critic = build_graph_without_critic(cfg)
    problem_result = pickle.load(open("../outputs/validation_solver_results.pkl", "rb"))

    output_path = "outputs/if_critic_results.csv"

    if os.path.exists(output_path):
        df_results = pd.read_csv(output_path)
        for id in df_results["id"].tolist():
            problem_result = [p for p in problem_result if str(p["problem"].id) != str(id)]
    else:
        df_results = pd.DataFrame()


    for data in tqdm(problem_result):
        data_id = str(data["problem"].id)

        try:
            result_state_with_critic = app_with_critic.invoke(data)
            result_state_without_critic = app_without_critic.invoke(data)
            df_results = pd.concat([
                df_results,
                pd.DataFrame([{
                    "id": data_id,
                    "answer_with_critic": result_state_with_critic["final_answer"]["final_answer"],
                    "answer_without_critic": result_state_without_critic["final_answer"]["final_answer"],
                }])
            ], ignore_index=True)
            df_results.to_csv(output_path, index=False)
        except Exception as e:
            print(f"[Inference Error] id={data_id} | {type(e).__name__}: {e}")
            continue


def set_seed(seed: int) -> None:
    """
    시드를 고정하여 실험의 재현성을 확보
    Args:
        param seed:
        type seed: int
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경일 경우
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




from langgraph.graph import StateGraph, END
from src.agent.state import AgentState, CriticResult
from typing import Dict, List
from src.agent.nodes import (
    critic,
    ensemble,
)


class AllPassCiriticNode:
    def __init__(self, config):
        pass

    def __call__(self, state: AgentState) -> Dict[str, List[CriticResult]]:
        critic_results = []
        for solver_result in state["solver_results"]:
            critic_results.append(
                CriticResult(critic_result = "Pass")
            )
        return {"critic_results": critic_results}


def build_graph_with_critic(cfg):
    workflow = StateGraph(AgentState)

    workflow.add_node("critic", critic.CriticNode(cfg))
    workflow.add_node("ensemble", ensemble.EnsembleNode(cfg))

    workflow.set_entry_point("critic")
    workflow.add_edge("critic", "ensemble")
    workflow.add_edge("ensemble", END)

    return workflow.compile()


def build_graph_without_critic(cfg):
    workflow = StateGraph(AgentState)

    workflow.add_node("critic", AllPassCiriticNode(cfg))
    workflow.add_node("ensemble", ensemble.EnsembleNode(cfg))

    workflow.set_entry_point("critic")
    workflow.add_edge("critic", "ensemble")
    workflow.add_edge("ensemble", END)

    return workflow.compile()










if __name__ == "__main__":
    main()
