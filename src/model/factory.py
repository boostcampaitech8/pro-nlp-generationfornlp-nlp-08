import gc
import torch
from omegaconf import DictConfig
from src.model.loader import HuggingFaceLoader

class ModelFactory:
    def __init__(self, config: DictConfig):
        """
        config: cfg.model 전체를 받습니다.
        (예: {'main_solver': {...}, 'sub_solver': {...}})
        """
        self.config = config
        self._models = {}

    def get_model(self, model_key: str):
        """
        model_key: config.yaml에서 지정한 이름 (예: "main_solver", "sub_solver")
        """
        if model_key not in self.config:
            available_keys = list(self.config.keys())
            raise ValueError(f"❌ [Factory] '{model_key}' 설정을 찾을 수 없습니다. (가능한 목록: {available_keys})")

        if model_key in self._models:
            return self._models[model_key]
        
        # 해당 키의 설정만 추출
        target_config = self.config[model_key]
        
        # 로더 생성 및 로딩 수행
        # 추후 unsloth 등을 쓴다면 여기서 target_config 내용을 보고 분기 처리 가능
        loader = HuggingFaceLoader(target_config)
        
        model, tokenizer = loader.load()
        
        self._models[model_key] = (model, tokenizer)
        return model, tokenizer
    
    def unload_model(self, model_key: str | None = None):
        """
        특정 모델 또는 전체 모델을 메모리에서 제거
        """
        if model_key:
            if model_key in self._models:
                model, tokenizer = self._models.pop(model_key)
                del model
                del tokenizer
        else:
            for model, tokenizer in self._models.values():
                del model
                del tokenizer
            self._models.clear()

        # 🔥 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
