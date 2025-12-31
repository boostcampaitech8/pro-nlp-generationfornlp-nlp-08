import re
import json
from .base import BaseLLMNode

def raw_critic_to_json(raw_text: str) -> dict:
    """
    LLM의 가공되지 않은 답변에서 JSON 객체를 찾아 파싱하는 함수.
    """
    try:
        # 정규표현식으로 가장 바깥쪽의 { ... } 내용을 추출
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # JSON 파싱 및 검증
            parsed_data = json.loads(json_str)
            if "critic_result" in parsed_data and "critic_reason" in parsed_data:
                return parsed_data
            else:
                return {"critic_result": "Fail", "critic_reason": "필수 키(result/reason) 누락"}
        else:
            return {"critic_result": "Fail", "critic_reason": "JSON 형식을 찾을 수 없음"}
            
    except json.JSONDecodeError:
        return {"critic_result": "Fail", "critic_reason": "JSON 파싱 에러 (형식 불일치)"}

class CriticNode(BaseLLMNode):
    def __init__(self, config):
        super().__init__(config, model_name="main_solver")
        self.template = config.prompt.critic.template

    def __call__(self, state: dict) -> dict:
        # State에서 필요한 모든 정보 추출
        problem = state.get("problem", {})
        solver_results = state.get("solver_results", [])
        
        critic_results = []
        
        # 각 결과에 대한 평가 진행
        for i, solver_result in enumerate(solver_results):
            predicted_answer = solver_result.get("answer", "")
            reason = solver_result.get("reasoning", "")

            choices_str = ", ".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

            # 템플릿에 따른 모델 추론 진행
            raw_output = self.generate(
                self.template, 
                paragraph=paragraph, 
                question=question, 
                choices=choices_str,
                predicted_answer=predicted_answer,
                reasoning=reason,
            )

            # raw 결과 json으로 전처리
            critic_result = raw_critic_to_json(raw_output)
            
            # 결과 형식에 추가
            critic_results.append(critic_result)
        
        # 5. 결과 반환 (List[Dict[str, str]])
        return {"critic_results": critic_results}