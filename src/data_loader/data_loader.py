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
        choices = row['choices']
        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except:
                choices = []

        problem = Problem(
            id=row['id'],
            paragraph=row['paragraph'],
            question=row['question'],
            choices=choices,
            answer=row.get('answer', None)
        )
        problems.append(problem)

    return problems
