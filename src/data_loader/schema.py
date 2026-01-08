from typing import TypedDict, List, Optional

class MCQSample(TypedDict):
    id: str
    paragraph: str
    question: str
    choices: List[str]
    answer: Optional[str]
    question_plus: Optional[str]
