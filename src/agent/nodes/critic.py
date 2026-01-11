import re
import json
from .base import BaseLLMNode
from typing import Dict, List
from langsmith import traceable

from src.agent.state import AgentState, CriticResult
from src.utils.text import extract_json_from_text


class CriticNode(BaseLLMNode):
    """
    Solver 노드에서 생성한 답안을 평가하는 Critic 노드

    Args:
        config: 설정 객체

    Returns:
        Dict[str, List[CriticResult]]: 평가 결과 리스트를 담은 딕셔너리
    """

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.cfg = config
        self.enable_thinking = config.prompt.critic.strategy.get(
            "enable_thinking", False
        )

    @traceable(name="CriticNode")
    def __call__(self, state: AgentState) -> Dict[str, List[CriticResult]]:

        track_info = state.get("track_info", {})
        is_rag_required = track_info.get("is_rag_required", False)

        context_str = ""
        track = ""
        if is_rag_required:
            for i, document in enumerate(state["retrieval_results"]):
                context_str += f"[context_{i+1}: {document['title']}]\n{document['body']}\n\n"
            track = "track_b"
        else:
            track = "track_a"

        solver_results = state["solver_results"]

        critic_results = []

        # 각 결과에 대한 평가 진행
        for i, solver_result in enumerate(solver_results):

            payload = {
                "user_prompt": self.cfg.prompt.critic[track].user,
                "system_prompt": self.cfg.prompt.critic[track].system,
                "paragraph": state["problem"].paragraph,
                "question": state["problem"].question,
                "choices": solver_result["raw_choice"],
                "predicted_answer": solver_result["raw_answer"],
                "reasoning": solver_result["reasoning"],
            }
            if track == "track_b":
                payload["context"] = context_str

            raw_output = self.generate(**payload)
            critic_result = extract_json_from_text(raw_output)
            critic_results.append(CriticResult(**critic_result))

        return {"critic_results": critic_results}
