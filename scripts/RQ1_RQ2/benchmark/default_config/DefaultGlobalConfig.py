import os


DEFAULT_GLOBAL_CONFIG = {
    'usable_gpu': '0,1'
}

# Global Top-K Configuration
# Unified management of all search-related top-k parameters
# Modifying these values ​​will uniformly adjust the number of searches across all memories
DEFAULT_RETRIEVAL_TOP_K = 10  # The default search term is the top-k value (used for text retrieval, multimodal retrieval, etc.).
DEFAULT_GRAPH_MAX_NODES = 10  # Maximum number of nodes traversed in a graph (for NGMemory, AUGUSTUSMemory)
DEFAULT_REFLECTION_TOP_K = 10 # Reflection retrieval of top-k (for GAReflector)

DEFAULT_OPENAI_APIKEY = os.environ.get('CUE_MEM_LLM_API_KEY', '')
DEFAULT_OPENAI_APIBASE = os.environ.get('CUE_MEM_LLM_BASE_URL', '')


DEFAULT_BACKBONE_PATH = os.environ.get('CUE_MEM_BACKBONE_PATH', '')
DEFAULT_GME_QWEN2_VL_7B_PATH = os.environ.get('CUE_MEM_GME_QWEN2_VL_7B_PATH', '')
DEFAULT_GME_QWEN2_VL_2B_PATH = os.environ.get('CUE_MEM_GME_QWEN2_VL_2B_PATH', '')
