# src/utils/tools.py
import wikipedia
from langsmith import traceable
from ddgs import DDGS
from typing import List, Dict

import os
import torch
from typing import Dict, Any, List, cast
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from omegaconf import DictConfig
from src.agent.state import AgentState, RetrievalResult
from langsmith import traceable


@traceable(name="search_wikipedia")
def search_wikipedia(query: str) -> str:
    """
    위키피디아에서 검색하여 가장 관련성 높은 문서의 본문 일부를 가져옵니다.

    Args:
        query (str): 검색할 키워드

    Returns:
        str: 검색된 문서
    """
    try:
        wikipedia.set_lang("ko")
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "검색 결과가 없음."
        page = wikipedia.page(search_results[0])
        return page.content[:500]  # 문서 본문 일부 반환 (최대 500자)
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"


@traceable(name="duckduckgo_search")
def duckduckgo_search(query: str) -> List[Dict[str, str]]:
    """
    Args:
        query (str): 검색할 키워드

    Returns:
        List[Dict[str, str]]: 검색 결과 리스트 {title: str, body: str}
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
    formatted_results = []
    for result in results:
        formatted_results.append(
            {"title": result.get("title", ""), "body": result.get("body", "")}
        )
    return formatted_results


@traceable(name="rag_search")
def rag_search(cfg: DictConfig, query: str) -> List[Dict[str, str]]:
    """
    Args:
        query (str): 검색할 키워드

    Returns:
        List[Dict[str, str]]: 검색 결과 리스트 {title: str, body: str}
    """

    model = BGEM3FlagModel(
        cfg.model.bge_m3.path,
        use_fp16=cfg.model.bge_m3.model_kwargs.get("use_fp16", True),
        device="cpu",
    )
    client = QdrantClient("http://localhost:6333")
    collection_name = "wiki_collection"

    output = model.encode(
        [query],
        return_dense=cfg.model.bge_m3.encode_kwargs.return_dense,
        return_sparse=cfg.model.bge_m3.encode_kwargs.return_sparse,
    )

    dense_vec = output["dense_vecs"][0].tolist()
    sparse_vec = output["lexical_weights"][0]

    search_result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="default",
                limit=cfg.model.bge_m3.retriever.top_k,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=[int(k) for k in sparse_vec.keys()],
                    values=list(sparse_vec.values()),
                ),
                using="sparse",
                limit=cfg.model.bge_m3.retriever.top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=cfg.model.bge_m3.retriever.top_k,
    )

    formatted_results = []
    for hit in search_result.points:
        text = hit.payload.get("text", "")
        head, _, body = text.partition("\n\n")
        formatted_results.append(
            {
                "title": head.replace("문서 제목:", "").strip(),
                "body": body.strip(),
            }
        )

    return formatted_results
