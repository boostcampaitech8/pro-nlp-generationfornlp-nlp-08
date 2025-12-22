def router_node(state, cfg):
    print("🚦 [Router] 문제 유형 분석 중...")
    
    # 실제로는 cfg.prompt.router.system 을 LLM에 넣어 분류해야 함
    instruction = cfg.prompt.router.system
    
    # Dummy Logic: 무조건 RAG가 필요한 역사 문제로 가정
    return {"track_info": {"category": "HISTORY", "is_rag_required": True}}