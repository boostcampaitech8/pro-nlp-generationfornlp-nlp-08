from typing import Dict
from langsmith import traceable
from src.agent.state import AgentState, RouterResult


class RouterNode:
    """
    RAG 필요 여부(is_rag_required)를 판별하는 Router 노드

    Args:
        cfg: 설정 객체

    Returns:
        Dict[str, RouterResult]: 문제 유형과 RAG 필요 여부를 담은 딕셔너리
    """

    def __init__(self, cfg):
        pass

    @traceable(name="RouterNode")
    def __call__(self, state: AgentState) -> Dict[str, RouterResult]:
        track_info_result = RouterResult(
            is_rag_required=(
                True if len(state["problem"].choices) == 4 else False
            ),  # 선지가 4개인 문제는 RAG 필요
        )
        return {"track_info": track_info_result}
