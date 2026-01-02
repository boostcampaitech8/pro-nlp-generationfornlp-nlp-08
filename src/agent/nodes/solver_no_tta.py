from .base import BaseLLMNode
from src.agent.state import AgentState, SolverResult
from typing import Dict, List
from src.utils.text import extract_json_from_text
from langsmith import traceable

class SolverNode(BaseLLMNode):
    """
    TTA를 사용하지 않는 기본 Solver 노드
    """

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")

    @traceable(name="SolverNode")
    def __call__(self, state: AgentState) -> Dict[str, List[SolverResult]]:
        raw_output = self.generate(
            user_prompt=state["solver_prompt"]["user_prompt"],
            system_prompt=state["solver_prompt"]["system_prompt"],
            enable_thinking=True,
        )
        
        output = extract_json_from_text(raw_output)
        answer = output.get("answer", "")
        reasoning = output.get("reasoning", "").strip()

        if "<think>" not in raw_output and "</think>" in raw_output:
            raw_output = "<think>\n" + raw_output
        return {"solver_results": [SolverResult(answer=answer, reasoning=reasoning)]}