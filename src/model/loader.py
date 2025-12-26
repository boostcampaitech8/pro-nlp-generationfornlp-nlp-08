# src/model/loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HuggingFaceLoader:
    """
    Factory에서 정규화된 dict 기반 loader cfg를 받는다.
    """

    def __init__(self, config: dict):
        self.config = config

    def load(self):
        name = self.config.get("name")
        path = self.config.get("path")

        print(f"🔄 [Loader] 모델 로딩 시작: {name} ({path})")

        # =========================
        # 1️⃣ Quantization
        # =========================
        bnb_config = None
        quant_cfg = self.config.get("quantization")

        if quant_cfg:
            print("   ↳ ⚡ 양자화 설정 적용 중...")
            q_params = dict(quant_cfg)

            if q_params.get("bnb_4bit_compute_dtype") == "bfloat16":
                q_params["bnb_4bit_compute_dtype"] = torch.bfloat16

            bnb_config = BitsAndBytesConfig(**q_params)

        # =========================
        # 2️⃣ model_kwargs
        # =========================
        model_kwargs = dict(self.config.get("model_kwargs", {}))

        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = getattr(
                torch, model_kwargs["torch_dtype"]
            )

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # =========================
        # 3️⃣ Model load
        # =========================
        model = AutoModelForCausalLM.from_pretrained(
            path,
            **model_kwargs
        )

        # =========================
        # 4️⃣ Tokenizer load
        # =========================
        tokenizer_kwargs = dict(self.config.get("tokenizer_kwargs", {}))

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            **tokenizer_kwargs
        )

        print("✅ [Loader] 로딩 완료")
        return model, tokenizer
