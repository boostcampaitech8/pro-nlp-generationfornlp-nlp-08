# src/model/pipeline_builder.py
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from omegaconf import OmegaConf


class PipelineBuilder:
    """
    - model / tokenizer는 Loader 결과
    - generation_cfg는 cfg 그대로 (router/solver/TTA별로 다름)
    - stopping_criteria는 외부에서 주입
    """

    def __init__(
        self,
        model,
        tokenizer,
        generation_cfg,
        stopping_criteria=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_cfg = generation_cfg
        self.stopping_criteria = stopping_criteria

    def build(self):
        # generation cfg를 dict로 변환 (없는 값 자동 제외)
        gen_kwargs = OmegaConf.to_container(
            self.generation_cfg,
            resolve=True
        )

        pipe_kwargs = {
            "task": "text-generation",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "return_full_text": False,
            **gen_kwargs,
        }

        # stopping criteria는 있을 때만 주입
        if self.stopping_criteria is not None:
            pipe_kwargs["stopping_criteria"] = self.stopping_criteria

        pipe = pipeline(**pipe_kwargs)

        return HuggingFacePipeline(pipeline=pipe)
