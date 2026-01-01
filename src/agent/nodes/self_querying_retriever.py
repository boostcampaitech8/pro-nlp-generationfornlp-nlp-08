from langsmith import traceable
from typing import List, Dict
from .base import BaseLLMNode
from src.utils.tools import search_wikipedia
from src.utils.text import extract_json_from_text
from src.agent.state import AgentState, RetrievalResult


class RetrievalNode(BaseLLMNode):
    """
    문제 해결에 필요한 검색 키워드를 추출하고, 위키피디아에서 관련 문서를 검색하는 Retrieval 노드
    cfg.prompt.retrieval.template을 사용하여 LLM에 검색 키워드 추출 요청

    Args:
        config: 설정 객체

    Returns:
        Dict[str, List[RetrievalResult]]: 검색된 문서들을 담은 딕셔너리
    """
    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.template = config.prompt.retrieval.template

    @traceable(name="RetrievalNode")
    def __call__(self, state: AgentState) -> Dict[str, List[RetrievalResult]]:
        choices_str = ", ".join(
            [f"{i+1}. {c}" for i, c in enumerate(state["problem"].choices)]
        )

        raw_output = self.generate(
            self.template,
            paragraph=state["problem"].paragraph,
            question=state["problem"].question,
            choices=choices_str,
        )

        keywords = extract_json_from_text(raw_output)["search_keywords"]

        search_results = []
        for keyword in keywords:
            keyword = keyword[:100]
            search_results.append(RetrievalResult(context=search_wikipedia(keyword)))

        return {"retrieval_results": search_results}