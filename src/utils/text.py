import json
from typing import Dict, Any
from langsmith import traceable

@traceable(name="extract_last_json")
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    텍스트 내에서 '가장 마지막에 위치한' 유효한 JSON 객체를 찾아 반환합니다.
    """
    stack = 0
    end_index = -1
    
    for i in range(len(text) - 1, -1, -1):
        char = text[i]
        
        if char == '}':
            if stack == 0:
                end_index = i
            stack += 1
        
        elif char == '{':
            if stack > 0:
                stack -= 1
                if stack == 0:
                    json_str = text[i : end_index + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
    
    raise ValueError(f"JSON object not found in output: {text}")