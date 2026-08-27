from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent

@dataclass #Decorator to automatically generate special methods like __init__ and __repr__ for the class.
class Config:
    #Embedding model configuration
    embedding_model = "models/gemini-emmbeding-001"
    embedding_model_dim = 3072

    #chunking configuration
    chunk_size = 512
    chunk_overlap = 64

    #Retrieval configuration
    top_k_dense = 10   #FAISS Can be used for dense vector search, and top_k_dense specifies the number of top results to retrieve from the dense index.
    top_k_sparse = 10  #BM25 is a popular algorithm for sparse vector search, and top_k_sparse specifies the number of top results to retrieve from the sparse index.
    top_k_hybrid = 10  #Hybrid search combines both dense and sparse retrieval methods, and top_k_hybrid specifies the number of top results to retrieve from the hybrid search.
    top_k_final = 5 #Specifies the number of top results to retrieve from the final search.
    rrf_k = 60 #Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked lists of results, and rrf_k specifies the number of top results to consider for fusion.


    #Re-ranking configuration
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2" #Specifies the model to be used for re-ranking the retrieved results.


    #LLM configuration
    llm_model = "gemini-2.0-flash" #Specifies
    max_tokens = 4096 #Specifies the maximum number of tokens that can be processed by the LLM model

    memory_window = 5 #Specifies the number of previous interactions to consider for context in the conversation with the LLM model.

    cache_direc