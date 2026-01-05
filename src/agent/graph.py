from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    router,
    # self_querying_retriever as retriever,
    retrieval,
    prompt,
    # solver,
    solver_no_tta as solver,
    # critic,
    # ensemble,
)


def build_graph(cfg):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router.RouterNode(cfg))
    workflow.add_node("retriever", retrieval.RetrievalNode(cfg))
    workflow.add_node("prompt_builder", prompt.PromptNode(cfg))
    workflow.add_node("solver", solver.SolverNode(cfg))
    # workflow.add_node("critic", critic.CriticNode(cfg))
    # workflow.add_node("ensemble", ensemble.EnsembleNode(cfg))

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
    workflow.add_edge("solver", END)
    # workflow.add_edge("solver", "critic")
    # workflow.add_edge("critic", "ensemble")
    # workflow.add_edge("ensemble", END)

    return workflow.compile()
