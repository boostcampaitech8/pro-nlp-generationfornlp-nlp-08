import os
import torch
from typing import Dict, Any, List, cast
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from omegaconf import DictConfig
from src.agent.state import AgentState, RetrievalResult
from langsmith import traceable


class RetrievalNode:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BGEM3FlagModel(
            cfg.model.bge_m3.path,
            use_fp16=cfg.model.bge_m3.model_kwargs.get("use_fp16", True),
            device=self.device,
        )

        self.client = QdrantClient("http://localhost:6333")
        self.collection_name = "wiki_collection"

    def get_hybrid_embeddings(self, text: str):
        output: Dict[str, Any] = cast(
            Dict[str, Any],
            self.model.encode(
                [text],
                return_dense=self.cfg.model.bge_m3.encode_kwargs.return_dense,
                return_sparse=self.cfg.model.bge_m3.encode_kwargs.return_sparse,
            ),
        )

        dense_vec = cast(Any, output["dense_vecs"][0]).tolist()
        sparse_vec = cast(Dict[str, float], output["lexical_weights"][0])

        return dense_vec, sparse_vec

    @traceable(name="RetrievalNode")
    def __call__(self, state: AgentState) -> Dict[str, List[RetrievalResult]]:

        problem = state["problem"]
        query_text = f"{problem.paragraph} {problem.question} {' '.join(problem.choices)}"

        dense_vec, sparse_vec = self.get_hybrid_embeddings(query_text)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="default",
                    limit=self.cfg.model.bge_m3.retriever.top_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=[int(k) for k in sparse_vec.keys()],
                        values=list(sparse_vec.values()),
                    ),
                    using="sparse",
                    limit=self.cfg.model.bge_m3.retriever.top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.cfg.model.bge_m3.retriever.top_k,
        )

        retrieval_results: List[RetrievalResult] = []
        for hit in search_result.points:
            text = hit.payload.get("text", "")
            head, _, body = text.partition("\n\n")
            retrieval_results.append(
                {
                    "title": head.replace("문서 제목:", "").strip(),
                    "body": body.strip(),
                }
            )

        return {"retrieval_results": retrieval_results}
