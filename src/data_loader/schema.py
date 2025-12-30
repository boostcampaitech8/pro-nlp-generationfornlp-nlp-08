from typing import List, Optional
from pydantic import BaseModel, Field

class Problem(BaseModel):
    """
    하나의 수능 문제 데이터를 정의하는 스키마
    CSV의 한 행에 해당하는 데이터 구조
    """
    id: str = Field(..., description="문제의 고유 식별자")
    paragraph: str = Field(..., description="문제 지문")
    question: str = Field(..., description="문제 질문")
    choices: List[str] = Field(..., description="선택지 목록")
    answer: Optional[str] = Field(None, description="정답 (있을 경우)")