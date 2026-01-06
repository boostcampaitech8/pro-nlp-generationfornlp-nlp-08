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
from tqdm import tqdm
import pandas as pd
import warnings
import logging
from transformers import logging as transformers_logging
from transformers.utils import logging as hf_logging


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
    dataset = load_dataset(cfg.path.data.test)
    output_path = cfg.path.data.output

    output_df = prepare_output_file(dataset, output_path)

    # answer==0 인 데이터만 남기고 미리 필터링
    pending_ids = set(output_df.index[output_df["answer"] == 0])
    dataset = [data for data in dataset if str(data.id) in pending_ids]

    if not dataset:
        print("[Inference] answer==0 인 데이터가 없습니다. 종료합니다.")
        return

    for data in tqdm(dataset):
        data_id = str(data.id)

        try:
            result_state = app.invoke(cast(AgentState, {"problem": data}))
            predicted_answer = extract_answer(result_state)
            output_df.loc[data_id] = {
                "id": data_id,
                "answer": predicted_answer,
            }
            output_df.to_csv(output_path, index=False)
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


def prepare_output_file(dataset, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        output_df = pd.read_csv(output_path)
    else:
        output_df = pd.DataFrame(
            {
                "id": [str(problem.id) for problem in dataset],
                "answer": [0 for _ in dataset],
            }
        )

    output_df["id"] = output_df["id"].astype(str)
    existing_ids = set(output_df["id"])
    missing_ids = [
        str(problem.id)
        for problem in dataset
        if str(problem.id) not in existing_ids
    ]
    if missing_ids:
        missing_df = pd.DataFrame(
            {"id": missing_ids, "answer": [0] * len(missing_ids)}
        )
        output_df = pd.concat([output_df, missing_df], ignore_index=True)

    output_df["answer"] = (
        pd.to_numeric(output_df["answer"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    output_df = output_df.drop_duplicates(subset="id", keep="first").set_index(
        "id", drop=False
    )
    output_df.to_csv(output_path, index=False)
    return output_df


def extract_answer(state):
    try:
        final_answer = state.get("final_answer", {})
        answer = final_answer.get("final_answer", 0)
        return int(answer) if answer not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
