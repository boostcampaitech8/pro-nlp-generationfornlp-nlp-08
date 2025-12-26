# src/model/factory.py
from src.model.loader import HuggingFaceLoader


class ModelFactory:
    def __init__(self, cfg):
        """
        cfg = cfg.model
        """
        self.cfg = cfg

    def get_model(self, model_key: str):
        if model_key not in self.cfg:
            raise ValueError(
                f"Unknown model key: {model_key}. "
                f"Available keys: {list(self.cfg.keys())}"
            )

        raw_cfg = self.cfg[model_key]
        loader_cfg = self._normalize_cfg(raw_cfg)

        loader = HuggingFaceLoader(loader_cfg)
        return loader.load()

    def _normalize_cfg(self, cfg):
        """
        cfg = cfg.model.<router | main_solver | sub_solver>
        공통 스키마로 정규화
        """
        loader_cfg = {
            "name": cfg.name,
            "path": cfg.loader.path,
            "model_kwargs": dict(cfg.loader.model_kwargs),
            "tokenizer_kwargs": dict(cfg.loader.tokenizer_kwargs),
        }

        # quantization은 있는 경우에만
        if "quantization" in cfg.loader and cfg.loader.quantization:
            loader_cfg["quantization"] = dict(cfg.loader.quantization)

        return loader_cfg
