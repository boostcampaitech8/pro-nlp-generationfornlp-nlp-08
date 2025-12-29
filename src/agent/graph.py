from functools import partial
from langgraph.graph import StateGraph, END
from transformers import StoppingCriteriaList

from .state import AgentState
from .nodes import (
    router_node, retrieval_node, prompt_node,
    solver_node, critic_node, ensemble_node
)

from src.model.factory import ModelFactory
from src.model.pipeline_builder import PipelineBuilder
from src.model.stopping import StopOnSubstrings


def build_graph(cfg):
    workflow = StateGraph(AgentState)

    factory = ModelFactory(cfg.model)

    # =========================
    # Router LLM
    # =========================
    router_model, router_tokenizer = factory.get_model("router")

    router_stopping = StoppingCriteriaList([
        StopOnSubstrings(
            cfg.model.router.stopping.stop_strings,
            router_tokenizer
        )
    ])

    router_llm = PipelineBuilder(
        model=router_model,
        tokenizer=router_tokenizer,
        generation_cfg=cfg.model.router.generation,
        stopping_criteria=router_stopping,
    ).build()

    # =========================
    # Solver LLM (main_solver!)
    # =========================
    solver_model, solver_tokenizer = factory.get_model("main_solver")

    solver_llm = PipelineBuilder(
        model=solver_model,
        tokenizer=solver_tokenizer,
        generation_cfg=cfg.model.main_solver.generation,
    ).build()

    # =========================
    # LangGraph node wrappers
    # =========================
    def router_node_lg(state):
        return router_node(state, cfg=cfg, llm=router_llm)

    def retrieval_node_lg(state):
        return retrieval_node(state, cfg=cfg)

    def prompt_node_lg(state):
        return prompt_node(state, cfg=cfg)

    def solver_node_lg(state):
        return solver_node(state, cfg=cfg, llm=solver_llm)

    def critic_node_lg(state):
        return critic_node(state, cfg=cfg)

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
        lambda x: "retriever"
        if x["track_info"]["is_rag_required"]
        else "prompt_builder",
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
