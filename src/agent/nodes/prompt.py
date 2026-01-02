from typing import Dict
from src.agent.state import AgentState, PromptResult
from langchain_core.messages import SystemMessage, HumanMessage


class PromptNode:
    """
    문제 해결을 위한 최종 프롬프트 메시지를 생성하는 노드
    cfg.prompt.solver의 템플릿을 사용하여 프롬프트를 구성

    Args:
        cfg: 설정 객체

    Returns:
        Dict[str, List[BaseMessage]]: 최종 프롬프트 메시지 리스트를 담은 딕셔너리
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, state: AgentState):

        # 데이터 추출
        paragraph = state["paragraph"]
        question = state["problem"]["question"]
        choices = "\n".join(state["problem"]["choices"])

        formatted_text = ""

        # Config에서 템플릿 가져오기 (cfg.prompt.solver 활용)
        if state["track_info"].get("is_rag_required"):
            # Track B: RAG 포함
            template = cfg.prompt.solver.track_b
            context_str = "\n".join(state.get("retrieved_context", []))

            formatted_text = template.format(
                context=context_str,
                paragraph=paragraph,
                question=question,
                choices=choices,
            )
        else:
            # Track A: 지문만 사용
            template = cfg.prompt.solver.track_a
            formatted_text = template.format(
                paragraph=paragraph, question=question, choices=choices
            )

        # Chat Model용 메시지 객체 생성
        messages = [
            SystemMessage(content="당신은 입시 문제 풀이 AI입니다."),
            HumanMessage(content=formatted_text),
        ]

        return {"final_prompt_messages": messages}
