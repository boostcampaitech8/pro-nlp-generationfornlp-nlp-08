from .base import BaseLLMNode
from src.utils.tools import search_wikipedia

class RetrievalNode(BaseLLMNode):
    def __init__(self, config):
        # 1. 부모 초기화 (모델 로드)
        super().__init__(config, model_name="main_solver")
        
        # 2. 프롬프트 템플릿 가져오기
        self.template = config.prompt.retrieval.template

    def __call__(self, state: dict) -> dict:
        print(f"▶️ [Retrieval] 검색 노드 시작 (질문: {state['question']})")
        
        # 1. LLM이 질문에서 '키워드' 추출
        keyword = self.generate(self.template, question=state["question"])
        keyword = keyword.strip() # 공백 제거
        print(f"   ↳ 🔑 추출된 키워드: [{keyword}]")
        
        # 2. 위키피디아 검색 수행
        search_result = search_wikipedia(keyword)
        print(f"   ↳ 📄 검색 결과: {search_result[:50]}...") # 앞부분만 로그 출력
        
        # 3. 결과 반환 (State의 'context'에 저장)
        return {"context": search_result}