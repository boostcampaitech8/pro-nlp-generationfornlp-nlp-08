from collections import Counter

def ensemble_node(state, cfg):
    solver_results = state.get("solver_results", [])
    critic_results = state.get("critic_results", [])
    
    # Pass 판정 받은 답변 필터링
    pass_answers = []
    for i in range(len(critic_results)):
        if critic_results[i].get("critic_result") == "Pass":
            pass_answers.append(solver_results[i].get("answer"))
            
    if pass_answers:
        # Pass가 하나라도 있으면 Pass 답변들로 hard voting
        voting_targets = pass_answers
        strategy = "Pass Filtering"
    else:
        # 모든 critic이 Fail이면 전체 답변으로 hard voting
        voting_targets = [res.get("answer") for res in solver_results]
        strategy = "Fallback (All Fail)"

    if not voting_targets:
        return {"final_answer": None}
    
    # hard voting 수행
    vote_counts = Counter(voting_targets)
    final_choice, count = vote_counts.most_common(1)[0]
    
    print(f"전략: {strategy} | 결과: {final_choice} (득표: {count}/{len(voting_targets)})")

    # 결과 반환
    try:
        return {"final_answer": int(final_choice)}
    except (ValueError, TypeError):
        return {"final_answer": None}