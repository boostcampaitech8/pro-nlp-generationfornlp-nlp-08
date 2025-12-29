# src/utils/tools.py
import wikipedia

def search_wikipedia(query: str) -> str:
    """위키피디아에서 요약 정보를 가져오는 함수"""
    try:
        wikipedia.set_lang("ko") # 한국어 설정
        # 검색 결과 중 가장 첫 번째 문서의 요약을 가져옴
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"검색 결과가 너무 많습니다. 더 구체적으로 검색해주세요. (관련: {e.options[:3]})"
    except wikipedia.exceptions.PageError:
        return "해당하는 문서를 찾을 수 없습니다."