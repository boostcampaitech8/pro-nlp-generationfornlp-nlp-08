import re
import json
from typing import Dict
from langsmith import traceable

@traceable(name="extract_json_from_text")
def extract_json_from_text(text: str) -> Dict:
    """
    Args:
        text (str): JSON 객체를 포함하는 문자열

    Returns:
        Dict: 추출된 JSON 객체

    Raises:
        ValueError: 문자열에서 JSON 객체를 찾지 못한 경우
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON object not found in output: {text}")

    json_str = match.group(0)
    return json.loads(json_str)
