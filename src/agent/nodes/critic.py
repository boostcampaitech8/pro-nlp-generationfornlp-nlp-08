import re
import json
from .base import BaseLLMNode
from typing import Dict, List
from langsmith import traceable

from src.agent.state import AgentState, CriticResult
from src.utils.text import extract_json_from_text


class CriticNode(BaseLLMNode):
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

        critic_results: List[CriticResult] = []
        
        def evaluate(solver_results: List[dict], source: str):
            # 각 결과에 대한 평가 진행
            for solver_result in solver_results:

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

                # 템플릿에 따른 모델 추론 진행
                raw_output = self.generate(**payload)

                # raw 결과 json으로 전처리
                parsed = extract_json_from_text(raw_output)
                parsed["source"] = source

                # 결과 형식에 추가
                critic_results.append(CriticResult(**parsed))
            # 5. 결과 반환 (List[Dict[str, str]])
        
        evaluate(state["main_solver_results"], source="main_solver")
        evaluate(state["sub_solver_results"], source="sub_solver")
        
        self.unload_model()
        return {"critic_results": critic_results}
