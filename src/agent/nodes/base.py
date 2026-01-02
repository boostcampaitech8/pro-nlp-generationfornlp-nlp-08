import torch
from omegaconf import DictConfig
from src.model.factory import ModelFactory
from transformers import TextStreamer
from langsmith import traceable

class BaseLLMNode:
    """
    [역할]
    모든 Agent Node(Solver, Router, Critic 등)가 공통으로 상속받는 부모 클래스입니다.
    
    복잡한 모델 로딩, 토크나이징, GPU 이동, 디코딩 과정을 이 클래스 내부로 숨기고(캡슐화),
    자식 클래스에서는 간단히 'generate()' 함수만 호출하여 사용할 수 있도록 돕습니다.

    Args:
        cfg (DictConfig): Hydra를 통해 로드된 전체 설정 객체 (cfg)
        model_name (str): 사용할 모델의 설정 키 이름 (기본값: "main_solver")
                            - config.yaml의 model 섹션에 정의된 이름이어야 합니다.
                            - 예: "main_solver" (Qwen), "sub_solver" (Gemma)
    """

    def __init__(self, cfg: DictConfig, model_name: str = "main_solver"):
        if model_name in cfg.model:
            self.factory = ModelFactory(cfg.model)
            self.model, self.tokenizer = self.factory.get_model(model_name)
            self.gen_params = cfg.model[model_name].generation
        else: 
            raise ValueError(f"❌ [No such model] '{model_name}'에 해당하는 모델 설정이 cfg.model에 없습니다.")

    @traceable(name="BaseLLMNode.generate")
    def generate(self, user_prompt: str, system_prompt: str = '', enable_thinking: bool = False, **kwargs) -> str:
        """
        LLM에게 프롬프트를 입력하고, 생성된 텍스트(답변)를 반환하는 함수

        Args:
            user_prompt (str): 사용자 메시지 (실제 질문이나 요청 내용)
            system_prompt (str): 시스템 메시지 (모델에게 역할을 지시하는 용도)
            enable_thinking (bool): '생각하는' 프롬프트 기법 활성화 여부
            **kwargs: user_prompt 내에 포맷팅할 변수들
        Returns:
            str: 모델이 생성한 순수 텍스트 답변 (특수 토큰 제외)
        """
        
        if kwargs:
            try:
                formatted_content = user_prompt.format(**kwargs)
            except KeyError as e:
                raise KeyError(f"❌ [Formatting Error] user_prompt 포맷팅 중 누락된 키: {e}")
        else:
            formatted_content = user_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": formatted_content})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.gen_params,
                streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        
        decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return decoded_output