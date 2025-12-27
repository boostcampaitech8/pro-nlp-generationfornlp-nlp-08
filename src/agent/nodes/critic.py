# def critic_node(state, cfg):
#     print("🧐 [Critic] 검증 중...")
#     # cfg.prompt.critic.system 활용 가능
#     return {}

import re
from .base import BaseLLMNode
from src.utils.tools import search_wikipedia

class critic_node(BaseLLMNode):
    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.template = config.prompt.retrieval.template

    # def __call__(self, state: dict) -> dict:
    #     # 1. State에서 필요한 모든 정보 추출
    #     paragraph = state.get("paragraph", "")
    #     problem = state.get("problem", {})
    #     question = problem.get("question", "")
    #     choices = problem.get("choices", [])
        
    #     # 선지 리스트를 보기 좋게 문자열로 변환 (예: "1. 선택지A, 2. 선택지B...")
    #     choices_str = ", ".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    #     print(f"▶️ [Retrieval] 검색 키워드 추출 중...")
    #     print(f"   - 질문: {question[:30]}...")
    #     print(f"   - 지문 길이: {len(paragraph)}자")

    #     # 2. LLM에게 지문+질문+선지 모두 제공 (kwargs로 전달)
    #     # 템플릿의 {paragraph}, {question}, {choices} 자리에 들어갑니다.
    #     raw_output = self.generate(
    #         self.template, 
    #         paragraph=paragraph, 
    #         question=question, 
    #         choices=choices_str
    #     )
        
    #     # 3. <think> 태그 제거 및 정제
    #     clean_keyword = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
    #     keyword = clean_keyword.strip()
        
    #     # 안전장치 (너무 길면 자름)
    #     if len(keyword) > 300: keyword = keyword[:300]

    #     print(f"   ↳ 🔑 추출된 키워드: [{keyword}]")
        
    #     # 4. 위키피디아 검색 수행
    #     search_result = search_wikipedia(keyword)
        
    #     # 5. 결과 반환 (List[str] 형태)
    #     return {"retrieved_context": [search_result]}