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
    app = build_graph(cfg)
    dataset = load_dataset(cfg.path.data.validation)
    output_path = "outputs/validation_solver_results.pkl"
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            solver_results = pickle.load(f)
        for predicted_problem in solver_results:
            data_id = str(predicted_problem["problem"].id)
            dataset = [data for data in dataset if str(data.id) != data_id]
    else:
        solver_results = []
    
    app = build_graph(cfg)

    for data in tqdm(dataset):
        data_id = str(data.id)

        try:
            result_state = app.invoke(cast(AgentState, {"problem": data}))
            solver_results.append(result_state)
            with open(output_path, "wb") as f:
                pickle.dump(solver_results, f)
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
from src.agent.state import AgentState
from src.agent.nodes import (
    router,
    # self_querying_retriever as retriever,
    retrieval,
    # prompt,
    solver,
    # solver_no_tta as solver,
    # critic,
    # ensemble,
)


def build_graph(cfg):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router.RouterNode(cfg))
    workflow.add_node("retriever", retrieval.RetrievalNode(cfg))
    # workflow.add_node("prompt_builder", prompt.PromptNode(cfg))
    workflow.add_node("solver", solver.SolverNode(cfg))
    # workflow.add_node("critic", critic.CriticNode(cfg))
    # workflow.add_node("ensemble", ensemble.EnsembleNode(cfg))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda x: (
            "retriever"
            if x["track_info"]["is_rag_required"]
            else "solver"
        ),
        {
            "retriever": "retriever",
            "solver": "solver",
        },
    )

    workflow.add_edge("retriever", "solver")
    workflow.add_edge("solver", END)
    # workflow.add_edge("solver", "critic")
    # workflow.add_edge("critic", "ensemble")
    # workflow.add_edge("ensemble", END)

    return workflow.compile()













if __name__ == "__main__":
    main()
