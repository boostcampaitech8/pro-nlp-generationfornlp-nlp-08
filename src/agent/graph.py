from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    router,
    self_querying_retriever,
    prompt,
    solver,
    critic,
    ensemble,
)


def build_graph(cfg):
    workflow = StateGraph(AgentState)

    # =========================
    # Node 등록 (LangGraph 규칙)
    # =========================
    workflow.add_node("router", router.RouterNode(cfg))
    workflow.add_node("retriever", self_querying_retriever.SelfQueryingRetrieverNode(cfg))
    # workflow.add_node("prompt_builder", prompt.PromptBuilderNode(cfg))
    workflow.add_node("solver", solver.SolverNode(cfg))
    workflow.add_node("critic", critic.CriticNode(cfg))
    workflow.add_node("ensemble", ensemble.EnsembleNode(cfg))

    # =========================
    # Graph structure
    # =========================
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda x: (
            "retriever"
            if x["track_info"]["is_rag_required"]
            else "prompt_builder"
        ),
        {
            "retriever": "retriever",
            "prompt_builder": "prompt_builder",
        },
    )

    workflow.add_edge("retriever", "prompt_builder")
    workflow.add_edge("prompt_builder", "solver")
    workflow.add_edge("solver", "critic")
    workflow.add_edge("critic", "ensemble")
    workflow.add_edge("ensemble", END)

    return workflow.compile()
