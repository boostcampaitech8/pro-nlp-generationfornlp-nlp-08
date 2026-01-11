from collections import Counter
from src.agent.state import AgentState, EnsembleResult
from langsmith import traceable


class EnsembleNode:
    """
    검증된 답안들을 앙상블하여 최종 정답을 도출하는 Ensemble 노드

    Args:
        cfg: 설정 객체

    Returns:
        EnsembleResult: 최종 정답을 담은 결과
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @traceable(name="EnsembleNode")
    def __call__(self, state: AgentState) -> EnsembleResult:
        solver_results = state.get("solver_results", [])
        critic_results = state.get("critic_results", [])

        pass_answers = []
        for i in range(len(critic_results)):
            if critic_results[i].get("critic_result") == "Pass":
                pass_answers.append(solver_results[i].get("answer"))

        if pass_answers:
            voting_targets = pass_answers
        else:
            voting_targets = [res.get("answer") for res in solver_results]

        if not voting_targets:
            return {"final_answer": None}

        vote_counts = Counter(voting_targets)
        final_choice, count = vote_counts.most_common(1)[0]

        return {"final_answer": int(final_choice)}
