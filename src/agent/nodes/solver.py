# src/agent/nodes/solver.py
import json
import re
import random
from typing import List, Tuple
from .base import BaseLLMNode


class SolverNode(BaseLLMNode):
    """
    Module 3. Solver Engine with TTA

    - 5번의 TTA를 사용하여 편향을 제거합니다.
    - 선지의 순서를 무작위로 섞은 5가지 버전을 생성하고, 각각에 대해 답안을 생성합니다.
    - Index Remapping을 통해 섞인 선지에서 고른 답을 원본 번호로 변환합니다.
    """

    NUM_TTA_VERSIONS = 1  # TTA 버전 수

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.templates = {
            "track_a": config.prompt.solver.track_a,
            "track_b": config.prompt.solver.track_b,
        }

    def __call__(self, state: dict) -> dict:
        # 1. State에서 필요한 정보 추출
        paragraph = state.get("paragraph", "")
        question = state.get("question", "")
        choices = state.get("choices", [])

        # Track 정보와 RAG 컨텍스트 가져오기
        track_info = state.get("track_info", {})
        is_rag_required = track_info.get("is_rag_required", False)
        track = "track_b" if is_rag_required else "track_a"
        
        retrieved_context = state.get("retrieved_context", [])
        context = "\n".join(retrieved_context) if retrieved_context else ""

        # 2. TTA: 5가지 선지 순서 변형 생성
        tta_versions = self._generate_tta_versions(choices)
        solver_results = []

        # 3. 각 버전에 대해 추론 수행
        for version_idx, (shuffled_choices, original_indices) in enumerate(tta_versions):

            choices_str = self._format_choices(shuffled_choices) # 선지 문자열 생성
            template = self.templates.get(track, self.templates["track_a"]) # 적절한 템플릿 선택 및 프롬프트 생성

            # 입력 변수 동적 구성
            input_kwargs = {
                "category": track_info.get("category", "others"),
                "paragraph": paragraph,
                "question": question,
                "choices": choices_str
            }
            if track == "track_b":
                input_kwargs["context"] = context

            # LLM 호출
            raw_output = self.generate(template, **input_kwargs)

            # 4. 답안 파싱 및 Index Remapping
            reasoning, shuffled_answer = self._parse_answer(raw_output)
            original_answer = self._remap_index(shuffled_answer, original_indices)

            solver_results.append({
                "reasoning": reasoning,
                "answer": original_answer,
            })

        return {"solver_results": solver_results}

    def _generate_tta_versions(self, choices: List[str]) -> List[Tuple[List[str], List[int]]]:
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
        return "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])

    def _parse_answer(self, raw_output: str) -> int:
        """
        LLM 출력에서 JSON을 파싱하여 답안 번호를 추출합니다.

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