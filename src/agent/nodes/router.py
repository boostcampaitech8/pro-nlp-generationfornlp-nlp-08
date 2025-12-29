from typing import Dict
import json, re
from src.agent.nodes.base import BaseLLMNode
from langsmith import traceable

def extract_json_from_text(text: str) -> Dict:
    """
    문자열 어디에 있든 첫 번째 JSON 객체만 추출하여 파싱
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON object not found in output: {text}")

    json_str = match.group(0)
    return json.loads(json_str)

class RouterNode(BaseLLMNode):
    '''
    문제 유형(category)과 RAG 필요 여부(is_rag_required)를 판별하는 Router 노드
    cfg.prompt.router.system을 사용하여 LLM에 문제 유형 분류 요청
    1) RAG 필요: {"track_info": {"category": str, "is_rag_required": True}}
    2) RAG 불필요: {"track_info": {"category": str, "is_rag_required": False}}
    '''

    def __init__(self, cfg):
        # router 전용 모델을 사용하도록 명시
        super().__init__(cfg, model_name="router")

        self.prompt_template = cfg.prompt.router.system

    @traceable(name="RouterNode")
    def __call__(self, state: Dict) -> Dict:
        print("🚦 [Router] 문제 유형 분석 중...")

        paragraph = state["paragraph"]
        question = state["problem"]["question"]

        # BaseLLMNode.generate() 사용
        raw_output = self.generate(
            self.prompt_template,
            paragraph=paragraph,
            question=question,
        ).strip()

        print("🧾 [Router Raw Output]")
        print(raw_output)

        # JSON 파싱
        parsed = extract_json_from_text(raw_output)

        track_info = {
            "category": parsed["category"],
            "is_rag_required": parsed["is_rag_required"],
        }

        return {"track_info": track_info}