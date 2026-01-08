# src/utils/tools.py
import wikipedia
from langsmith import traceable
from ddgs import DDGS
from typing import List, Dict

@traceable(name="search_wikipedia")
def search_wikipedia(query: str) -> str:
    """
    위키피디아에서 검색하여 가장 관련성 높은 문서의 본문 일부를 가져옵니다.

    Args:
        query (str): 검색할 키워드

    Returns:
        str: 검색된 문서
    """
    try:
        wikipedia.set_lang("ko")
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "검색 결과가 없음."
        page = wikipedia.page(search_results[0])
        return page.content[:500]  # 문서 본문 일부 반환 (최대 500자)
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"

@traceable(name="duckduckgo_search")
def duckduckgo_search(query: str) -> List[Dict[str, str]]:
    """
    Args:
        query (str): 검색할 키워드

    Returns:
        List[Dict[str, str]]: 검색 결과 리스트 {title: str, body: str}
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "title": result.get("title", ""),
            "body": result.get("body", "")
        })
    return formatted_results
    
