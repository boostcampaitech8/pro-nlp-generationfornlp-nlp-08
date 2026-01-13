from langsmith import traceable
from typing import List, Dict
from .base import BaseLLMNode
from src.utils.tools import duckduckgo_search
from src.utils.text import extract_json_from_text
from src.agent.state import AgentState, RetrievalResult


class RetrievalNode(BaseLLMNode):
    """
    문제 해결에 필요한 검색 키워드를 추출하고, 위키피디아에서 관련 문서를 검색하는 Retrieval 노드
    cfg.prompt.retrieval.template을 사용하여 LLM에 검색 키워드 추출 요청

    Args:
        cfg: 설정 객체

    Returns:
        Dict[str, List[RetrievalResult]]: 검색된 문서들을 담은 딕셔너리
    """

    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.system_prompt = config.prompt.retrieval.system
        self.user_prompt = config.prompt.retrieval.user
        self.enable_thinking = config.prompt.retrieval.strategy.get(
            "enable_thinking", False
        )

    @traceable(name="RetrievalNode")
    def __call__(self, state: AgentState) -> Dict[str, List[RetrievalResult]]:
        choices_str = ", ".join(
            [f"{i+1}. {c}" for i, c in enumerate(state["problem"].choices)]
        )

        raw_output = self.generate(
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            enable_thinking=self.enable_thinking,
            paragraph=state["problem"].paragraph,
            question=state["problem"].question,
            choices=choices_str,
        )

        keywords = extract_json_from_text(raw_output)["search_keywords"]

        search_results = []
        for keyword in keywords:
            keyword = keyword[:100]
            search_results.extend(
                [
                    RetrievalResult(**result)
                    for result in duckduckgo_search(keyword)
                ]
            )

        return {"retrieval_results": search_results}
