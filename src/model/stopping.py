# src/model/stopping.py
from typing import List
from transformers import StoppingCriteria


class StopOnSubstrings(StoppingCriteria):
    """
    지정된 문자열(stop_strings) 중 하나라도
    생성된 토큰의 suffix에 등장하면 generation을 중단한다.

    - Router처럼 role token(System/Human/Assistant)을 차단할 때 사용
    - JSON-only 출력 강제할 때도 활용 가능
    """

    def __init__(self, stop_strings: List[str], tokenizer):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

        # stop string들을 token id sequence로 변환
        self.stop_token_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in stop_strings
            if s
        ]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """
        input_ids: (batch, seq_len)
        """
        if input_ids is None or input_ids.shape[0] == 0:
            return False

        generated_ids = input_ids[0].tolist()

        for stop_ids in self.stop_token_ids:
            n = len(stop_ids)
            if n == 0:
                continue

            if generated_ids[-n:] == stop_ids:
                return True

        return False
