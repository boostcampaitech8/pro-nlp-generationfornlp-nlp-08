from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    router,
    retrieval,
    main_solver, 
    sub_solver,
    critic,
    ensemble,
)


def build_graph(cfg):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router.RouterNode(cfg))
    workflow.add_node("retriever", retrieval.RetrievalNode(cfg))
    workflow.add_node("sub_solver", sub_solver.SolverNode(cfg))
    workflow.add_node("main_solver", main_solver.SolverNode(cfg))
    workflow.add_node("critic", critic.CriticNode(cfg))
    workflow.add_node("ensemble", ensemble.EnsembleNode(cfg))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: (
            "rag_required"
            if state["track_info"]["is_rag_required"]
            else "non_rag_required"
        ),
        {
            "rag_required": "retriever",
            "non_rag_required": "sub_solver",
        },
    )

    workflow.add_edge("retriever", "sub_solver")
    workflow.add_edge("sub_solver", "main_solver")
    workflow.add_edge("main_solver", "critic")
    workflow.add_edge("critic", "ensemble")
    workflow.add_edge("ensemble", END)

    return workflow.compile()
