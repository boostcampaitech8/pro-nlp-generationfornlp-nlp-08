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

<<<<<<< HEAD
    NUM_TTA_VERSIONS = 3  # TTA 버전 수
=======
    NUM_TTA_VERSIONS = 1  # TTA 버전 수
>>>>>>> fc95ce6 ([Exp] CoT 데이터 셋으로 모델 SFT한 테스트 결과(#52) (#63))

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

<<<<<<< HEAD
    @traceable(name="SolverNode")
    def __call__(self, state: AgentState) -> Dict[str, List[SolverResult]]:
=======
    def __call__(self, state: dict) -> dict:
        # 1. State에서 필요한 정보 추출
        paragraph = state.get("paragraph", "")
        question = state.get("question", "")
        choices = state.get("choices", [])
>>>>>>> fc95ce6 ([Exp] CoT 데이터 셋으로 모델 SFT한 테스트 결과(#52) (#63))

        track_info = state.get("track_info", {})
        is_rag_required = track_info.get("is_rag_required", False)

<<<<<<< HEAD
        context_str = ""
        track = ""
        if is_rag_required:
            for i, document in enumerate(state["retrieval_results"]):
                context_str += f"[context_{i+1}: {document['title']}]\n{document['body']}\n\n"
            track = "track_b"
        else:
            track = "track_a"

        tta_versions = self._generate_tta_versions(state["problem"].choices)
        solver_results = []

        for shuffled_choices, original_indices in tta_versions:
=======
        # 2. TTA: 5가지 선지 순서 변형 생성
        tta_versions = self._generate_tta_versions(choices)
        solver_results = []

        # 3. 각 버전에 대해 추론 수행
        for version_idx, (shuffled_choices, original_indices) in enumerate(tta_versions):
>>>>>>> fc95ce6 ([Exp] CoT 데이터 셋으로 모델 SFT한 테스트 결과(#52) (#63))

            choices_str = self._format_choices(shuffled_choices)

            input_kwargs = {
                "paragraph": state["problem"].paragraph,
                "question": state["problem"].question,
                "choices": choices_str,
            }
            if track == "track_b":
                input_kwargs["context"] = context_str

            raw_output = self.generate(
                user_prompt=self.user_prompt_template[track],
                system_prompt=self.system_prompt[track],
                enable_thinking=self.enable_thinking,
                **input_kwargs, #type: ignore
            )

            parsed_result = self._parse_and_remap(raw_output, original_indices)

<<<<<<< HEAD
            parsed_result["raw_choice"] = choices_str

            solver_results.append(parsed_result)
=======
            solver_results.append({
                "reasoning": reasoning,
                "answer": original_answer,
            })
>>>>>>> fc95ce6 ([Exp] CoT 데이터 셋으로 모델 SFT한 테스트 결과(#52) (#63))

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


        예상 형식: {"reasoning": "...", "answer": 3}
        """
        # 1. <think> 태그 제거
        clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
        parsed_answer = None
        parsed_reasoning = ""

        # 2. JSON 파싱 시도 (가장 바깥쪽 중괄호 탐색)
        start_idx = clean_output.find('{')
        end_idx = clean_output.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = clean_output[start_idx : end_idx + 1]
            result = json.loads(json_str)
            
            val = result.get("answer")
            if val is not None:
                parsed_answer = int(val)
            parsed_reasoning = result.get("reasoning", "")
            
        if parsed_answer is None:
            print(f"parsing error:\n{raw_output[:500]}")
            parsed_answer = -1 
        return parsed_reasoning, parsed_answer

    def _remap_index(self, shuffled_answer: int, original_indices: List[int]) -> int:
        """
        섞인 선지에서 선택한 답을 원본 문제의 번호로 변환합니다.

        Args:
            shuffled_answer: 섞인 선지에서 선택한 번호 (1-indexed)
            original_indices: 섞인 순서의 원본 인덱스 리스트
                예: [3, 1, 4, 2, 5] → 셔플된 1번 = 원본 3번

        Returns:
            원본 문제에서의 정답 번호
        """
        shuffled_answer = int(shuffled_answer)
        return original_indices[shuffled_answer - 1]
