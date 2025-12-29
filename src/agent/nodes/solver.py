# src/agent/nodes/solver.py
import re
import random
from typing import List, Tuple
from .base import BaseLLMNode


class SolverNode(BaseLLMNode):
    """
    Module 3. Dual Solver Engine with TTA

    "다양한 관점에서 문제를 동시에 풀어보는 집단 지성"

    - 1개의 모델과 5번의 TTA(Test Time Augmentation)를 사용하여 편향을 제거합니다.
    - 선지의 순서를 무작위로 섞은 5가지 버전을 생성하고, 각각에 대해 답안을 생성합니다.
    - Index Remapping을 통해 섞인 선지에서 고른 답을 원본 번호로 변환합니다.
    """

    NUM_TTA_VERSIONS = 5  # TTA 버전 수

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.templates = {
            "track_a": config.prompt.solver.track_a,
            "track_b": config.prompt.solver.track_b,
        }

    def __call__(self, state: dict) -> dict:
        # 1. State에서 필요한 정보 추출
        paragraph = state.get("paragraph", "")
        problem = state.get("problem", {})
        question = problem.get("question", "")
        choices = problem.get("choices", [])

        # Track 정보와 RAG 컨텍스트 가져오기
        track_info = state.get("track_info", {})
        is_rag_required = track_info.get("is_rag_required","false")
        category = track_info.get("category","others")
        track = "track_a" if is_rag_required == "false" else "track_b"
        retrieved_context = state.get("retrieved_context", [])
        context = "\n".join(retrieved_context) if retrieved_context else ""

        print(f"🤖 [Solver] TTA 기반 추론 시작...")
        print(f"   - 트랙: {track}")
        print(f"   - 선지 수: {len(choices)}개")
        print(f"   - TTA 버전: {self.NUM_TTA_VERSIONS}개 생성 예정")

        # 2. TTA: 5가지 선지 순서 변형 생성
        tta_versions = self._generate_tta_versions(choices)

        # 3. 각 버전에 대해 추론 수행
        solver_results = []

        for version_idx, (shuffled_choices, original_indices) in enumerate(tta_versions):
            print(f"   ▶️ [TTA V{version_idx + 1}] 추론 중... (순서: {original_indices})")

            # 선지 문자열 생성
            choices_str = self._format_choices(shuffled_choices)

            # 적절한 템플릿 선택 및 프롬프트 생성
            template = self.templates.get(track, self.templates["track_a"])

            # LLM 호출
            if track == "track_b" and context:
                raw_output = self.generate(
                    template,
                    paragraph=paragraph,
                    question=question,
                    choices=choices_str,
                    context=context
                )
            else:
                raw_output = self.generate(
                    template,
                    paragraph=paragraph,
                    question=question,
                    choices=choices_str
                )

            # 4. 답안 파싱 및 Index Remapping
            shuffled_answer = self._parse_answer(raw_output)
            original_answer = self._remap_index(shuffled_answer, original_indices)

            print(f"      ↳ 셔플된 답: {shuffled_answer} → 원본 답: {original_answer}")

            solver_results.append({
                "model": "main_solver",
                "version": version_idx + 1,
                "shuffled_order": original_indices,
                "shuffled_answer": shuffled_answer,
                "answer": original_answer,
                "raw_output": raw_output
            })

        print(f"✅ [Solver] {len(solver_results)}개 답안 생성 완료!")

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
        LLM 출력에서 답안 번호를 추출합니다.

        다양한 형식 지원:
        - "정답: 3", "정답은 3번입니다", "3번", "③" 등
        """
        # <think> 태그 제거
        clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)

        # 원문자 숫자 매핑 (①②③④⑤)
        circled_numbers = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}
        for symbol, num in circled_numbers.items():
            if symbol in clean_output:
                return num

        # "정답: N", "정답은 N번", "N번이" 등의 패턴
        patterns = [
            r'정답[은:\s]*(\d)',
            r'(\d)\s*번[이을를]?\s*(정답|입니다|이다|선택)',
            r'답[은:\s]*(\d)',
            r'\[(\d)\]',
            r'(\d)\s*$',  # 마지막 숫자
        ]

        for pattern in patterns:
            match = re.search(pattern, clean_output)
            if match:
                return int(match.group(1))

        # 패턴 매칭 실패 시 첫 번째 숫자 추출
        numbers = re.findall(r'\d', clean_output)
        if numbers:
            return int(numbers[0])

        # 기본값 (파싱 실패)
        print(f"      ⚠️ 답안 파싱 실패: {clean_output[:100]}...")
        return 1

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
        if 1 <= shuffled_answer <= len(original_indices):
            return original_indices[shuffled_answer - 1]

        # 범위 초과 시 그대로 반환
        print(f"      ⚠️ 답안 범위 초과: {shuffled_answer}")
        return shuffled_answer