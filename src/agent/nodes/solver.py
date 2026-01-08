# src/agent/nodes/solver.py
import json
import re
import random
from typing import List, Tuple, Dict, Any
from .base import BaseLLMNode
from src.agent.state import AgentState, SolverResult
from src.utils.text import extract_json_from_text
from langsmith import traceable


class SolverNode(BaseLLMNode):
    """
    Module 3. Solver Engine with TTA

    - 5번의 TTA를 사용하여 편향을 제거합니다.
    - 선지의 순서를 무작위로 섞은 5가지 버전을 생성하고, 각각에 대해 답안을 생성합니다.
    - Index Remapping을 통해 섞인 선지에서 고른 답을 원본 번호로 변환합니다.
    """

    NUM_TTA_VERSIONS = 3  # TTA 버전 수

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.system_prompt = {
            "track_a": config.prompt.solver.track_a.system,
            "track_b": config.prompt.solver.track_b.system,
        }
        self.user_prompt_template = {
            "track_a": config.prompt.solver.track_a.user,
            "track_b": config.prompt.solver.track_b.user,
        }
        self.enable_thinking = config.prompt.solver.strategy.get(
            "enable_thinking", False
        )

    @traceable(name="SolverNode")
    def __call__(self, state: AgentState) -> Dict[str, List[SolverResult]]:

        track_info = state.get("track_info", {})
        is_rag_required = track_info.get("is_rag_required", False)
        track = "track_b" if is_rag_required else "track_a"

        retrieved_context = state.get("retrieved_context", [])
        context = "\n".join(retrieved_context) if retrieved_context else ""

        tta_versions = self._generate_tta_versions(state["problem"].choices)
        solver_results = []

        for shuffled_choices, original_indices in tta_versions:

            choices_str = self._format_choices(shuffled_choices)

            input_kwargs = {
                "paragraph": state["problem"].paragraph,
                "question": state["problem"].question,
                "choices": choices_str,
            }
            if track == "track_b":
                input_kwargs["context"] = context

            raw_output = self.generate(
                user_prompt=self.user_prompt_template[track],
                system_prompt=self.system_prompt[track],
                enable_thinking=self.enable_thinking,
                **input_kwargs, #type: ignore
            )

            parsed_result = self._parse_and_remap(raw_output, original_indices)

            solver_results.append(parsed_result)

        return {"solver_results": solver_results}

    def _generate_tta_versions(
        self, choices: List[str]
    ) -> List[Tuple[List[str], List[int]]]:
        """
        선지의 순서를 무작위로 섞은 5가지 버전을 생성합니다.

        Returns:
            List[Tuple[shuffled_choices, original_indices]]
            - shuffled_choices: 섞인 선지 리스트
            - original_indices: 섞인 순서의 원본 인덱스 (1-indexed)
              예: [3, 1, 4, 2, 5] → 첫 번째 위치에 원본 3번이 있음
        """
        n = len(choices)
        original_indices = list(range(1, n + 1))  # 1-indexed: [1, 2, 3, 4, 5]

        versions = []
        used_permutations = set()

        for _ in range(self.NUM_TTA_VERSIONS):
            # 중복되지 않는 순열 생성
            while True:
                shuffled_indices = original_indices.copy()
                random.shuffle(shuffled_indices)
                perm_tuple = tuple(shuffled_indices)

                if perm_tuple not in used_permutations:
                    used_permutations.add(perm_tuple)
                    break

            # 섞인 순서대로 선지 재배열
            shuffled_choices = [choices[i - 1] for i in shuffled_indices]

            versions.append((shuffled_choices, shuffled_indices))

        return versions

    def _format_choices(self, choices: List[str]) -> str:
        """선지 리스트를 포맷팅된 문자열로 변환합니다."""
        return "\n".join(
            [f"{i + 1}. {choice}" for i, choice in enumerate(choices)]
        )


    def _parse_and_remap(
        self, raw_output: str, original_indices: List[int]
    ) -> Dict[str, Any]:
        """결과 파싱 및 인덱스 복원 전담"""
        try:
            data = extract_json_from_text(raw_output)
            shuffled_idx = int(data.get("answer", 0))

            if 1 <= shuffled_idx <= len(original_indices):
                original_answer = original_indices[shuffled_idx - 1]
            else:
                original_answer = 0

            return {
                "reasoning": data.get("think", data.get("reasoning", "")).strip(),
                "answer": original_answer,
            }
        except Exception:
            return {"reasoning": "Parsing failed", "answer": 0}
