import torch
from omegaconf import DictConfig
from src.model.factory import ModelFactory
from transformers import TextStreamer

class BaseLLMNode:
    """
    [역할]
    모든 Agent Node(Solver, Router, Critic 등)가 공통으로 상속받는 부모 클래스입니다.
    
    복잡한 모델 로딩, 토크나이징, GPU 이동, 디코딩 과정을 이 클래스 내부로 숨기고(캡슐화),
    자식 클래스에서는 간단히 'generate()' 함수만 호출하여 사용할 수 있도록 돕습니다.
    """

    def __init__(self, config: DictConfig, model_name: str = "main_solver"):
        """
        클래스 초기화 메서드입니다. 그래프 빌드 시 모델을 메모리에 로드합니다.

        Args:
            config (DictConfig): Hydra를 통해 로드된 전체 설정 객체 (cfg)
            model_name (str): 사용할 모델의 설정 키 이름 (기본값: "main_solver")
                              - config.yaml의 model 섹션에 정의된 이름이어야 합니다.
                              - 예: "main_solver" (Qwen), "sub_solver" (Gemma)
        """
        print(f"🔧 [{self.__class__.__name__}] 초기화 중... (사용 모델: {model_name})")
        
        # 1. ModelFactory를 사용하여 모델과 토크나이저 로드
        # (src/model/factory.py 의존)
        self.factory = ModelFactory(config.model)
        
        # 실제 무거운 모델이 여기서 메모리에 올라갑니다.
        self.model, self.tokenizer = self.factory.get_model(model_name)
        
        # 2. 생성(Generation) 파라미터 캐싱
        # yaml 파일의 [generation] 항목(temperature, top_p 등)을 미리 저장해둡니다.
        # 추론할 때마다 매번 config를 뒤지지 않기 위함입니다.
        if model_name in config.model:
            self.gen_params = config.model[model_name].generation
        else:
            print(f"⚠️ 경고: '{model_name}'에 대한 generation 설정이 없습니다. 기본값을 사용합니다.")
            self.gen_params = {}

    def generate(self, prompt_template: str, **kwargs) -> str:
        """
        LLM에게 프롬프트를 입력하고, 생성된 텍스트(답변)를 반환하는 핵심 함수입니다.

        Args:
            prompt_template (str): 변수({key})가 포함된 프롬프트 문자열 (YAML에서 로드)
            **kwargs: 프롬프트 템플릿의 변수 구멍을 채울 값들 (예: question="...")

        Returns:
            str: 모델이 생성한 순수 텍스트 답변 (특수 토큰 제외)

        Example:
            >>> template = "문제: {question} 정답:"
            >>> answer = self.generate(template, question="1+1은?")
        """
        
        # 1. 프롬프트 포맷팅 (변수 채우기)
        try:
            formatted_content = prompt_template.format(**kwargs)
        except KeyError as e:
            error_msg = f"❌ [Prompt Error] 템플릿 변수 '{e}'가 누락되었습니다."
            print(error_msg)
            return error_msg

        # 2. Chat Template 적용 (Instruct 모델 필수 과정)
        # 단순히 텍스트만 넣는 게 아니라, 모델이 학습한 대화 형식(<|user|> 등)으로 변환합니다.
        messages = [{"role": "user", "content": formatted_content}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False
        )
        
        # 3. 토크나이징 및 GPU 이동
        # return_tensors="pt" -> PyTorch 텐서로 변환
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 4. 모델 추론 (Inference)
        # torch.no_grad()를 사용하여 불필요한 그라디언트 계산을 막아 메모리를 아낍니다.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.gen_params,  # YAML에 설정된 top_p, temp 등이 여기서 적용됨
                streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )
            
        # 5. 디코딩 (Decoding)
        # 입력으로 넣은 프롬프트 길이를 계산하여, 새로 생성된 뒷부분만 잘라냅니다.
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        
        # 숫자를 다시 사람이 읽을 수 있는 문자열로 변환합니다.
        decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return decoded_output