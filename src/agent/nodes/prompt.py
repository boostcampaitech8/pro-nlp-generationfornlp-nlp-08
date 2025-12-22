from langchain_core.messages import SystemMessage, HumanMessage

def prompt_node(state, cfg):
    print("📝 [Prompt Builder] Config 기반 최적 프롬프트 조립 중...")
    
    # 데이터 추출
    paragraph = state["paragraph"]
    question = state["problem"]["question"]
    choices = "\n".join(state["problem"]["choices"])
    
    formatted_text = ""
    
    # Config에서 템플릿 가져오기 (cfg.prompt.solver 활용)
    if state["track_info"].get("is_rag_required"):
        # Track B: RAG 포함
        template = cfg.prompt.solver.track_b
        context_str = "\n".join(state.get("retrieved_context", []))
        
        formatted_text = template.format(
            context=context_str,
            paragraph=paragraph,
            question=question,
            choices=choices
        )
    else:
        # Track A: 지문만 사용
        template = cfg.prompt.solver.track_a
        formatted_text = template.format(
            paragraph=paragraph,
            question=question,
            choices=choices
        )

    # Chat Model용 메시지 객체 생성
    messages = [
        SystemMessage(content="당신은 입시 문제 풀이 AI입니다."),
        HumanMessage(content=formatted_text)
    ]
    
    return {"final_prompt_messages": messages}