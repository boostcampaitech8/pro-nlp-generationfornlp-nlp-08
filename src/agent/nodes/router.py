from typing import Dict
import json, re
from langchain_core.messages import SystemMessage, HumanMessage

import json
import re
from typing import Dict

def extract_json_from_text(text: str) -> Dict:
    """
    문자열 어디에 있든 첫 번째 JSON 객체만 추출하여 파싱
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON object not found in output: {text}")

    json_str = match.group(0)
    return json.loads(json_str)


def router_node(state: Dict, cfg) -> Dict:
    '''
    Router 노드: 문제 유형(category)을 분석하여 RAG 필요 여부(is_rag_required) 결정
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
        HumanMessage(content=(
            "다음은 분류 대상 문제입니다.\n\n"
            "[지문]\n"
            f"{paragraph}\n\n"
            "[문제]\n"
            f"{problem}"
        ))
    ]
    
    # LLM 호출
    llm = cfg.router_llm
    response = llm.invoke(messages)
    print(response)
    raw_output = (
        response.content if hasattr(response, "content") else str(response)
    ).strip()
    
     # JSON 파싱
    parsed_output = extract_json_from_text(raw_output)

    track_info = {
        "category": parsed_output["category"],
        "is_rag_required": parsed_output["is_rag_required"]
    }
    return {"track_info": track_info}
