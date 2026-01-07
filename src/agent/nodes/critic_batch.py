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
        self.user_prompt_template = config.prompt.critic.user
        self.system_prompt = config.prompt.critic.system
        self.enable_thinking = config.prompt.critic.strategy.get(
            "enable_thinking", False
        )

    @traceable(name="CriticNode")
    def __call__(self, state: AgentState) -> Dict[str, List[CriticResult]]:
        # State에서 필요한 모든 정보 추출
        problem = state.get("problem", {})
        solver_results = state["solver_results"]

        critic_results = []

        # 각 결과에 대한 평가 진행
        for i, solver_result in enumerate(solver_results):

            # 템플릿에 따른 모델 추론 진행
            raw_output = self.generate(
                self.user_prompt_template,
                system_prompt=self.system_prompt,
                paragraph=state["problem"].paragraph,
                question=state["problem"].question,
                choices= solver_result["raw_choice"],
                predicted_answer=solver_result["raw_answer"],
                reasoning=solver_result["reasoning"],
            )

            # raw 결과 json으로 전처리
            critic_result = extract_json_from_text(raw_output)

            # 결과 형식에 추가
            critic_results.append(CriticResult(**critic_result))
        # 5. 결과 반환 (List[Dict[str, str]])
        return {"critic_results": critic_results}
