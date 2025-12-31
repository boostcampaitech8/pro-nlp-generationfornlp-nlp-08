from typing import Dict
from langsmith import traceable
from src.agent.nodes.base import BaseLLMNode
from src.agent.state import AgentState, RouterResult
from src.utils.text import extract_json_from_text


class RouterNode(BaseLLMNode):
    """
    문제 유형(category)과 RAG 필요 여부(is_rag_required)를 판별하는 Router 노드
    cfg.prompt.router.system을 사용하여 LLM에 문제 유형 분류 요청
    1) RAG 필요: {"track_info": {"category": str, "is_rag_required": True}}
    2) RAG 불필요: {"track_info": {"category": str, "is_rag_required": False}}
    """

    def __init__(self, cfg):
        super().__init__(cfg, model_name="router")
        self.prompt_template = cfg.prompt.router.system

    @traceable(name="RouterNode")
    def __call__(self, state: AgentState) -> Dict:
        raw_output = self.generate(
            self.prompt_template,
            paragraph=state["problem"].paragraph,
            question=state["problem"].question,
        ).strip()
        parsed = extract_json_from_text(raw_output)
        track_info = RouterResult(
            category=parsed["category"],
            is_rag=parsed["is_rag_required"],
        )
        return {"track_info": track_info}