from typing import TypedDict, List, Optional
from src.data_loader.schema import Problem

class RouterResult(TypedDict):
    is_rag_required: bool

class RetrievalResult(TypedDict):
    title: str
    body: str

class PromptResult(TypedDict):
    system_prompt: str
    user_prompt: str

class SolverResult(TypedDict):
    answer: int
    reasoning: str

class CriticResult(TypedDict):
    critic_result: str

class EnsembleResult(TypedDict):
    final_answer: Optional[int]

class AgentState(TypedDict):
    problem: Problem
    track_info: RouterResult
    retrieval_results: List[RetrievalResult]
    solver_prompt: PromptResult
    solver_results: List[SolverResult]
    critic_results: List[CriticResult]
    final_answer: EnsembleResult
