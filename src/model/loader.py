# src/model/loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from omegaconf import DictConfig
from src.utils.memory import free_gpu_memory

class HuggingFaceLoader:
    def __init__(self, config: DictConfig):
        """
        Args:
            config (DictConfig): Hydra 설정 객체
        """
        self.config = config

    def load(self):
        """
        HuggingFace 모델과 토크나이저를 로드하는 함수
        Returns:
            model, tokenizer: 로드된 모델과 토크나이저 객체
        """
        free_gpu_memory()

        # 양자화 설정
        bnb_config = None
        if "quantization" in self.config and self.config.quantization:
            q_params = dict(self.config.quantization)
            
            if q_params.get("bnb_4bit_compute_dtype") == "bfloat16":
                q_params["bnb_4bit_compute_dtype"] = torch.bfloat16
            
            bnb_config = BitsAndBytesConfig(**q_params)

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "quantization_config": bnb_config
        }

        if not bnb_config and hasattr(self.config, "torch_dtype"):
             model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype, torch.float16)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.path, 
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.path, 
                trust_remote_code=True
            )
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ [Loader] 로딩 실패: {e}")
            raise e
