import pandas as pd
import ast
from typing import List
from src.data_loader.schema import Problem


def load_dataset(file_path: str) -> List[Problem]:
    """
    CSV 파일에서 문제 데이터를 로드하여 Problem 객체 리스트로 반환.

    Args:
        file_path (str): CSV 파일 경로

    Returns:
        List[Problem]: 로드된 문제 데이터 리스트
    """
    df = pd.read_csv(file_path)
    problems = []
    for _, row in df.iterrows():
        problem_raw = ast.literal_eval(row["problems"])
        question = problem_raw["question"]
        choices = problem_raw["choices"]
        answer_raw = problem_raw.get("answer")
        if pd.isna(answer_raw) or str(answer_raw).strip() == "":
            answer = None
        else:
            answer = int(answer_raw)

        question_plus_raw = problem_raw.get("question_plus")
        if pd.isna(question_plus_raw) or str(question_plus_raw).strip() == "":
            question_plus = None
        else:
            question_plus = question_plus_raw

        # 추가 정보가 없는 경우 처리
        problem = Problem(
            id=row["id"],
            paragraph=row["paragraph"],
            question=question,
            choices=choices,
            answer=answer,
            question_plus=question_plus,
        )
        problems.append(problem)
    return problems
