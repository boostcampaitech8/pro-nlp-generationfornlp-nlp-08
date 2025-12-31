from typing import TypedDict, List, Optional
from src.data_loader.schema import Problem

class RouterResult(TypedDict):
    category: str
    is_rag: bool

class RetrievalResult(TypedDict):
    context: str

class PromptResult(TypedDict):
    prompt: str

class SolverResult(TypedDict):
    answer: str
    reasoning: str

class CriticResult(TypedDict):
    evaluation: str
    feedback: str

class EnsembleResult(TypedDict):
    final_answer: Optional[int]

class AgentState(TypedDict):
    problem: Problem
    track_info: RouterResult
    retrival_results: List[RetrievalResult]
    solver_results: List[SolverResult]
    critic_results: List[CriticResult]
    final_answer: EnsembleResult
