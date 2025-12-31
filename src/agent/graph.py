from functools import partial
from langgraph.graph import StateGraph, END
from transformers import StoppingCriteriaList

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
    # LangGraph node wrappers
    # =========================
    def router_node_lg(state):
        return router(state, cfg=cfg)

    def retrieval_node_lg(state):
        return self_querying_retriever(state, cfg=cfg)
    def prompt_node_lg(state):
        return prompt(state, cfg=cfg)

    def solver_node_lg(state):
        return solver(state, cfg=cfg)

    def critic_node_lg(state):
        return critic(state, cfg=cfg)
    def ensemble_node_lg(state):
        return ensemble_node(state, cfg=cfg)

    # =========================
    # Node 등록 (LangGraph 규칙)
    # =========================
    workflow.add_node("router", router_node_lg)
    workflow.add_node("retriever", retrieval_node_lg)
    workflow.add_node("prompt_builder", prompt_node_lg)
    workflow.add_node("solver", solver_node_lg)
    workflow.add_node("critic", critic_node_lg)
    workflow.add_node("ensemble", ensemble_node_lg)

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
