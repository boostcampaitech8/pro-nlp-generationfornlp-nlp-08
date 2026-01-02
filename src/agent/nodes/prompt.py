from typing import Dict
from src.agent.state import AgentState, PromptResult
from langsmith import traceable

class PromptNode:
    """
    문제 해결을 위한 최종 프롬프트 메시지를 생성하는 노드
    cfg.prompt.solver의 템플릿을 사용하여 프롬프트를 구성

    Args:
        cfg: 설정 객체
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @traceable(name="PromptNode")
    def __call__(self, state: AgentState) -> Dict[str, PromptResult]:

        # Config에서 템플릿 가져오기 (cfg.prompt.solver 활용)
        if state["track_info"].get("is_rag_required"):
            # Track B: RAG 포함
            system_prompt = self.cfg.prompt.solver.track_b.system
            user_prompt_template = self.cfg.prompt.solver.track_b.user
            context_str = ""
            for i, document in enumerate(state["retrieval_results"]):
                context_str += f"[context_{i+1}: {document['title']}]\n{document['body']}\n\n"

            user_prompt = user_prompt_template.format(
                context=context_str,
                pragraph = state["problem"].paragraph,
                question=state["problem"].question,
                choices=", ".join(
                    [f"{i+1}. {c}" for i, c in enumerate(state["problem"].choices)]
                ),
            )
        else:
            system_prompt = self.cfg.prompt.solver.track_a.system
            user_prompt_template = self.cfg.prompt.solver.track_a.user
            user_prompt = user_prompt_template.format(
                paragraph=state["problem"].paragraph,
                question=state["problem"].question,
                choices=", ".join(
                    [f"{i+1}. {c}" for i, c in enumerate(state["problem"].choices)]
                ),
            )
        return {"solver_prompt": PromptResult(system_prompt=system_prompt, user_prompt=user_prompt)}
