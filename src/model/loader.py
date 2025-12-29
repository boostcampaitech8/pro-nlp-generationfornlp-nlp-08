# src/model/loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from omegaconf import DictConfig

class HuggingFaceLoader:
    def __init__(self, config: DictConfig):
        """
        config: model/qwen_32b.yaml 등의 내용이 담긴 DictConfig
        """
        self.config = config

    def load(self):
        print(f"🔄 [Loader] 모델 로딩 시작: {self.config.name} ({self.config.path})")

        # 1. 양자화(Quantization) 설정 처리
        bnb_config = None
        if "quantization" in self.config and self.config.quantization:
            print("   ↳ ⚡ 양자화 설정 적용 중...")
            q_params = dict(self.config.quantization)
            
            # 문자열 "bfloat16"을 실제 torch.bfloat16 타입으로 변환
            if q_params.get("bnb_4bit_compute_dtype") == "bfloat16":
                q_params["bnb_4bit_compute_dtype"] = torch.bfloat16
            
            bnb_config = BitsAndBytesConfig(**q_params)

        # 2. 모델 로드 인자 준비
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "quantization_config": bnb_config
        }

        # torch_dtype 설정이 있으면 적용 (예: bfloat16)
        # 양자화가 없을 때 주로 사용됨
        if not bnb_config and hasattr(self.config, "torch_dtype"):
             model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype, torch.float16)

        # 3. 모델 & 토크나이저 로드
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.path, 
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.path, 
                trust_remote_code=True
            )
            print(f"✅ [Loader] 로딩 완료!")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ [Loader] 로딩 실패: {e}")
            raise e
