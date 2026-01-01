from typing import Dict
from langsmith import traceable
from src.agent.nodes.base import BaseLLMNode
from src.agent.state import AgentState, RouterResult
from src.utils.text import extract_json_from_text


class RouterNode(BaseLLMNode):
    """
    문제 유형(category)과 RAG 필요 여부(is_rag_required)를 판별하는 Router 노드
    cfg.prompt.router.system을 사용하여 LLM에 문제 유형 분류 요청

    Args:
        cfg: 설정 객체

    Returns:
        Dict[str, RouterResult]: 문제 유형과 RAG 필요 여부를 담은 딕셔너리
    """

    def __init__(self, cfg):
        super().__init__(cfg, model_name="router")
        self.prompt_template = cfg.prompt.router.template

    @traceable(name="RouterNode")
    def __call__(self, state: AgentState) -> Dict[str, RouterResult]:
        raw_output = self.generate(
            self.prompt_template,
            paragraph=state["problem"].paragraph,
            question=state["problem"].question,
        ).strip()
        parsed = extract_json_from_text(raw_output)
        track_info_result = RouterResult(
            category=parsed["category"],
            is_rag=parsed["is_rag_required"],
        )
        return {"track_info": track_info_result}
