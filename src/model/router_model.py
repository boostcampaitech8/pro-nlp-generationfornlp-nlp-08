import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)
from langchain_huggingface import HuggingFacePipeline
from src.base.llm_base import BaseLLM


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_token_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in stop_strings
        ]

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) == 0:
                continue
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False


class RouterLLM(BaseLLM):
    def load(self):
        # self.cfg는 dict
        model_name = self.cfg["model"]["name"]
        generation_cfg = self.cfg["generation"]
        stopping_cfg = self.cfg["stopping"]

        torch_dtype = getattr(torch, self.cfg["dtype"])

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.cfg["device_map"],
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        stopping_criteria = StoppingCriteriaList([
            StopOnSubstrings(stopping_cfg["stop_strings"], tokenizer)
        ])

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=generation_cfg["max_new_tokens"],
            temperature=generation_cfg["temperature"],
            do_sample=generation_cfg["do_sample"],
            stopping_criteria=stopping_criteria,
            return_full_text=False,
        )

        return HuggingFacePipeline(pipeline=pipe)
