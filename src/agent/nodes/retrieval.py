import os
import torch
from typing import Dict, Any, List
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models


class RetrievalNode:

    def __init__(self, cfg: Dict[str, Any]):
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
        output = self.model.encode(
            [text],
            return_dense=self.cfg.model.bge_m3.encode_kwargs.return_dense,
            return_sparse=self.cfg.model.bge_m3.encode_kwargs.return_sparse,
        )
        dense_vec = output["dense_vecs"][0].tolist()
        sparse_vec = output["lexical_weights"][0]

        return dense_vec, sparse_vec

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:

        paragraph = state.get("paragraph", "")
        problem = state.get("problem", {})
        question = problem.get("question", "")
        choices = problem.get("choices", [])

        query_text = f"{paragraph} {question} {' '.join(choices)}"

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

        retrieved_context = [
            hit.payload.get("text", "") for hit in search_result.points
        ]

        return {"retrieved_context": retrieved_context}
