import torch
import random
import numpy as np
import os
import sys
import pandas as pd
import ast
import time
import wandb
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.agent.nodes.solver import SolverNode


class SolverEvaluation:
    """
    Solver 노드의 성능을 평가하는 클래스

    Args:
        cfg: 설정 객체

    Returns:
        List[Dict]: 평가 결과 리스트
    """

    def __init__(self, config):
        self.config = config
        self.model = SolverNode(config)
        self.data = []
        self.project_root = project_root
        self.__load_data()
        self.true_answers = []
        self.pred_answers = []

    def __call__(self, dataset):
        """
        데이터셋에 대해 Solver 노드를 평가

        Args:
            dataset: 평가할 데이터셋 리스트

        Returns:
            List[Dict]: 각 데이터에 대한 Solver 결과 리스트
        """
        result = []
        for data in tqdm(
            dataset, desc="Evaluating Solver", total=len(dataset)
        ):
            res = self.model(data)
            result.append(res)
            self.pred_answers.append(res["solver_results"][0]["answer"])
            self.true_answers.append(data["answer"])
        return result

    def __load_data(self):
        """
        설정 파일에서 지정한 경로의 데이터를 로드하고 RAG가 필요하지 않은 문제만 필터링

        RAG 컬럼이 'F'(False)인 경우만 로드하여 Track A 문제만 평가 대상으로 설정
        """
        data_path = os.path.join(
            self.project_root, self.config.path.data.validate
        )
        df = pd.read_csv(data_path)

        for _, row in df.iterrows():
            problems_dict = ast.literal_eval(row["problems"])
            if row["rag"] != "F":
                continue
            self.data.append(
                {
                    "id": row["id"],
                    "paragraph": row["paragraph"],
                    "question": problems_dict["question"],
                    "question_plus": row.get("question_plus", None),
                    "choices": problems_dict["choices"],
                    "answer": row["answer"],
                }
            )
        print(len(self.data), "length of data")


def set_random_seed(random_seed=42):
    """
    재현 가능한 결과를 위한 랜덤 시드 설정

    Args:
        random_seed: 랜덤 시드 값
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def add_project_root():
    """
    프로젝트 루트 경로를 sys.path에 추가
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    if project_root not in sys.path:
        sys.path.append(project_root)


if __name__ == "__main__":
    set_random_seed()
    cfg = load_config()
    experiment_name = "test"
    wandb_config = {
        "prompt_solver": cfg["prompt"]["solver"],
        "main_solver": cfg["model"]["main_solver"],
    }
    wandb.init(
        project="solver-evaluation",
        name=f"{experiment_name}",
        config=wandb_config,
    )
    solver = SolverEvaluation(cfg)

    start_time = time.time()
    result = solver(solver.data)
    elapsed_time = time.time() - start_time

    log_data = []
    log_wrong_data = []

    for i, item in enumerate(result):
        for res in item["solver_results"]:
            if res["answer"] != solver.true_answers[i]:
                log_wrong_data.append(
                    [
                        solver.data[i]["id"],
                        res["reasoning"],
                        res["answer"],
                        solver.true_answers[i],
                    ]
                )

            log_data.append(
                [
                    solver.data[i]["id"],
                    res["reasoning"],
                    res["answer"],
                    solver.true_answers[i],
                ]
            )
    print(
        f"\n 추론 시간: {elapsed_time:.2f}초 (문제당 {elapsed_time/len(result):.2f}초)"
    )

    f1 = f1_score(solver.true_answers, solver.pred_answers, average="macro")
    accuracy = accuracy_score(solver.true_answers, solver.pred_answers)
    print(f" F1 Score: {f1:.4f}")
    print(f" Accuracy: {accuracy:.4f}")
    print(
        f" Correct Data: {len(log_data) - len(log_wrong_data)}/{len(log_data)}"
    )

    wandb.log(
        {
            "f1_score": f1,
            "accuracy": accuracy,
            "elapsed_time": elapsed_time,
            "evaluation_results": wandb.Table(
                data=log_data,
                columns=["id", "reasoning", "pred_answer", "true_answer"],
            ),
            "wrong_results": wandb.Table(
                data=log_wrong_data,
                columns=["id", "reasoning", "pred_answer", "true_answer"],
            ),
        }
    )

    wandb.finish()
