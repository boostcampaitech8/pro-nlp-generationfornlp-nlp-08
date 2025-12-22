from typing import Dict
import json
from langchain_core.messages import SystemMessage, HumanMessage

def router_node(state: Dict, cfg) -> Dict:
    '''
    Router 노드: 문제 유형을 분석하여 RAG 필요 여부 결정
    cfg.prompt.router.system을 사용하여 LLM에 문제 유형 분류 요청
    1) RAG 필요: {"track_info": {"category": str, "is_rag_required": True}}
    2) RAG 불필요: {"track_info": {"category": str, "is_rag_required": False}}
    '''
    print("🚦 [Router] 문제 유형 분석 중...")
    
    
    paragraph = state["paragraph"]
    problem = state["problem"]["question"]
    
    # LLM 프롬프트 구성
    prompt = cfg.prompt.router.system
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Paragraph: {paragraph}\nProblem: {problem}")
    ]
    
    # LLM 호출
    llm = cfg.model.model
    response = llm.invoke(messages)

    raw_output = response.content.strip()
    print(f"🚦 [Router] LLM 응답: {raw_output}")
    
    # LLM 출력(JSON) 파싱
    try:
        parsed_output = json.loads(raw_output)
        
        category = parsed_output["category"]
        is_rag_required = parsed_output["is_rag_required"]
    
    except Exception as e:
        print(f"❗ [Router] JSON 파싱 오류: {e}")
        raise ValueError("LLM 응답을 JSON으로 파싱하는 데 실패했습니다.")
    
    # 상태 업데이트
    state["track_info"] = {
        "category": category,
        "is_rag_required": is_rag_required
    }
    print(f"🚦 [Router] 문제 유형: {category}, RAG 필요 여부: {is_rag_required}")
    return state
