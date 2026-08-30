import os

from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *
from default_config.DefaultMMMemoryConfig import *  
from default_config.DefaultMMFUMemoryConfig import *  
from default_config.DefaultNGMemoryConfig import *  
from default_config.DefaultAUGUSTUSMemoryConfig import *  
from default_config.DefaultUniversalRAGMemoryConfig import *  

DEFAULT_FUMEMORY = {
    'name': 'FUMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_FUMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_STMEMORY = {
    'name': 'STMMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_STMEMORY_RECALL,
    'store': DEFAULT_STMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_LTMEMORY = {
    'name': 'LTMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_LTMEMORY_RECALL,
    'store': DEFAULT_LTMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_GAMEMORY = {
    'name': 'GAMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_GAMEMORY_RECALL,
    'store': DEFAULT_GAMEMORY_STORE,
    'reflect': DEFAULT_GAMEMORY_REFLECT,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}


DEFAULT_MGMEMORY = {
    'name': 'MGMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_MGMEMORY_RECALL,
    'store': DEFAULT_MGMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_RFMEMORY = {
    'name': 'RFMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_FUMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'optimize': DEFAULT_RFMEMORY_OPTIMIZE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_AMEMORY = {
    'name': 'AMemMemory',
    'is_multimodal': False,
    'top_k': DEFAULT_RETRIEVAL_TOP_K,
    # A-Mem uses SentenceTransformer internally. Reuse the project-local text encoder path.
    'embedding_model': DEFAULT_ENCODER['path'],
    # RobustOpenAIController is used with base_url for OpenAI-compatible APIs.
    'llm_backend': 'openai',
    'llm_model': os.environ.get('CUE_MEM_LLM_MODEL', ''),
    'api_key': DEFAULT_OPENAI_APIKEY,
    'api_base': DEFAULT_OPENAI_APIBASE,
    'evo_threshold': 100,
    'evo_interval': 15,
    'check_connection': False,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_MEMORYOSMEMORY = {
    'name': 'MemoryOSMemory',
    'is_multimodal': False,
    'top_k': DEFAULT_RETRIEVAL_TOP_K,
    'top_k_sessions': DEFAULT_RETRIEVAL_TOP_K,
    'top_k_knowledge': DEFAULT_RETRIEVAL_TOP_K,
    'embedding_model': DEFAULT_ENCODER['path'],
    'embedding_model_kwargs': None,
    'llm_model': os.environ.get('CUE_MEM_LLM_MODEL', ''),
    'api_key': DEFAULT_OPENAI_APIKEY,
    'api_base': DEFAULT_OPENAI_APIBASE,
    'user_id': 'memgallery_user',
    'assistant_id': 'memgallery_assistant',
    'short_term_capacity': 10,
    'fast_mode': True,
    'fast_batch_size': 15,
    'fast_use_llm_summary': True,
    'mid_term_capacity': 2000,
    'long_term_knowledge_capacity': 100,
    'retrieval_queue_capacity': 7,
    # Benchmark mode disables hot-session profile/knowledge updates; otherwise
    # MemoryOS can trigger extra LLM calls while merely ingesting history.
    'mid_term_heat_threshold': 999999.0,
    'mid_term_similarity_threshold': 0.6,
    # Resolved relative to the configured benchmark root by MemoryOSMemory.
    'storage_root': '.memory_cache/MemoryOS',
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_ALL_PARAM = DEFAULT_FUMEMORY
