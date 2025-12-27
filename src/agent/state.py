from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    # --- 입력 데이터 ---
    paragraph: str
    problem: Dict[str, Any]  # question, choices

    # --- 처리 데이터 ---
    track_info: Dict[str, Any]       # category, is_rag_required
    retrieved_context: List[str]     # RAG 검색 결과
    
    # Solver에게 던져질 완성된 프롬프트
    final_prompt_messages: List[Any] 
    solver_results: List[Dict[Any, str]] # 10개 답안{"reasoning" "answer"}
    
    # --- 최종 결과 ---
    final_answer: Optional[int]