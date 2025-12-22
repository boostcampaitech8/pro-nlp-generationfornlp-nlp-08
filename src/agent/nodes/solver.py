def solver_node(state, cfg):
    # Solver는 이제 복잡한 생각 없이 '주어진 메시지'대로 답만 생성하면 됨
    messages = state["final_prompt_messages"]
    print(f"🤖 [Solver] 준비된 프롬프트({len(messages)} msgs)로 추론 시작...")
    
    # TODO: model.invoke(messages) 구현 필요
    
    # Dummy: 1번이 정답이라고 10번 외침
    dummy_results = [{"model": "Qwen", "answer": 1} for _ in range(10)]
    return {"solver_results": dummy_results}