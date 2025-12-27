from omegaconf import DictConfig
from src.model.loader import HuggingFaceLoader

class ModelFactory:
    def __init__(self, config: DictConfig):
        """
        config: cfg.model 전체를 받습니다.
        (예: {'main_solver': {...}, 'sub_solver': {...}})
        """
        self.config = config

    def get_model(self, model_key: str):
        """
        model_key: config.yaml에서 지정한 이름 (예: "main_solver", "sub_solver")
        """
        if model_key not in self.config:
            available_keys = list(self.config.keys())
            raise ValueError(f"❌ [Factory] '{model_key}' 설정을 찾을 수 없습니다. (가능한 목록: {available_keys})")

        # 해당 키의 설정만 추출
        target_config = self.config[model_key]
        
        # 로더 생성 및 로딩 수행
        # 추후 unsloth 등을 쓴다면 여기서 target_config 내용을 보고 분기 처리 가능
        loader = HuggingFaceLoader(target_config)
        return loader.load()