from functools import partial
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    router_node, retrieval_node, prompt_node, 
    solver_node, critic_node, ensemble_node
)

def build_graph(cfg):
    workflow = StateGraph(AgentState)

    # [핵심] functools.partial을 사용해 Config를 각 노드에 주입
    # 이제 각 노드는 호출될 때 (state, cfg)를 인자로 받습니다.
    workflow.add_node("router", partial(router_node, cfg=cfg))
    workflow.add_node("retriever", partial(retrieval_node, cfg=cfg))
    workflow.add_node("prompt_builder", partial(prompt_node, cfg=cfg))
    workflow.add_node("solver", partial(solver_node, cfg=cfg))
    workflow.add_node("critic", partial(critic_node, cfg=cfg))
    workflow.add_node("ensemble", partial(ensemble_node, cfg=cfg))

    # 엔트리 포인트
    workflow.set_entry_point("router")

    # 조건부 분기: Router -> Retriever 또는 바로 Prompt Builder
    workflow.add_conditional_edges(
        "router",
        lambda x: "retriever" if x["track_info"]["is_rag_required"] else "prompt_builder",
        {"retriever": "retriever", "prompt_builder": "prompt_builder"}
    )

    # Retriever -> Prompt Builder (Solver로 바로 가지 않음!)
    workflow.add_edge("retriever", "prompt_builder")
    
    # Prompt Builder -> Solver (여기서 프롬프트가 완성된 상태로 넘어감)
    workflow.add_edge("prompt_builder", "solver")
    
    # 이후 파이프라인
    workflow.add_edge("solver", "critic")
    workflow.add_edge("critic", "ensemble")
    workflow.add_edge("ensemble", END)

    return workflow.compile()