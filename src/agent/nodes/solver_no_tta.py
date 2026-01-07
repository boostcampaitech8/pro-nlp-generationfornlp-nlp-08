from .base import BaseLLMNode
from src.agent.state import AgentState, EnsembleResult
from typing import Dict, List, Any, Optional
from src.utils.text import extract_json_from_text
from langsmith import traceable


class SolverNode(BaseLLMNode):
    """
    TTA를 사용하지 않는 기본 Solver 노드
    """

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.enable_thinking = config.prompt.solver.strategy.get(
            "enable_thinking", False
        )

    @traceable(name="SolverNode")
    def __call__(self, state: AgentState) -> Dict[str, EnsembleResult]:
        raw_output = self.generate(
            user_prompt=state["solver_prompt"]["user_prompt"],
            system_prompt=state["solver_prompt"]["system_prompt"],
            enable_thinking=self.enable_thinking,
        )

        output: Dict[str, Any] = extract_json_from_text(raw_output)
        raw_answer = output.get("answer", None)

        answer: Optional[int]
        if raw_answer is None or raw_answer == "":
            answer = None
        else:
            try:
                answer = int(raw_answer)
            except (TypeError, ValueError):
                answer = None
        # reasoning = output.get("reasoning", "").strip()

        if "<think>" not in raw_output and "</think>" in raw_output:
            raw_output = "<think>\n" + raw_output
        return {"final_answer": EnsembleResult(final_answer=answer)}
