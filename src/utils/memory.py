import gc
import torch

def free_gpu_memory(model=None, tokenizer=None):
    """
    GPU 메모리를 해제하는 유틸리티 함수
    
    Args:
        model: (optional) GPU에 로드된 모델 객체
        tokenizer: (optional) GPU에 로드된 토크나이저 객체
    """
    if model:
        del model
    if tokenizer:
        del tokenizer
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()