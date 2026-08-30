from abc import ABC, abstractmethod
import torch
import hashlib
import pickle
import os
import atexit
import numpy as np
from benchmark.paths import cache_path
from benchmark.security import redact_runtime_text

# ---------------------------------------------------------------------------
# Module-level embedding cache (shared across all encoder instances)
# ---------------------------------------------------------------------------
_encoder_caches = {}        # {model_path: {text_hash: numpy_array}}
_encoder_cache_dirty = {}   # {model_path: bool}
_encoder_save_registered = False

_st_model_cache = {}
_lm_model_cache = {}


def _get_cache_dir():
    cache_dir = str(cache_path())
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cache_file_path(model_path):
    safe = os.path.basename(model_path).replace(' ', '_')
    return os.path.join(_get_cache_dir(), f'emb_cache_{safe}.pkl')


def _load_encoder_cache(model_path):
    if model_path in _encoder_caches:
        return _encoder_caches[model_path]
    path = _cache_file_path(model_path)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            _encoder_caches[model_path] = data
            _encoder_cache_dirty[model_path] = False
            print(f"[EmbCache] Loaded {len(data)} cached embeddings from {path}")
            return data
        except Exception as e:
            print(f"[EmbCache] Failed to load cache: {redact_runtime_text(e)}")
    _encoder_caches[model_path] = {}
    _encoder_cache_dirty[model_path] = False
    return _encoder_caches[model_path]


def _save_all_encoder_caches():
    for model_path, cache in _encoder_caches.items():
        if not _encoder_cache_dirty.get(model_path, False):
            continue
        path = _cache_file_path(model_path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        disk_data = pickle.load(f)
                    disk_data.update(cache)
                    cache = disk_data
                    _encoder_caches[model_path] = cache
                except Exception:
                    pass
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(cache, f)
            os.replace(tmp, path)
            _encoder_cache_dirty[model_path] = False
            print(f"[EmbCache] Saved {len(cache)} cached embeddings to {path}")
        except Exception as e:
            print(f"[EmbCache] Failed to save cache: {redact_runtime_text(e)}")


def _text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _init_cache_for_encoder(model_path):
    global _encoder_save_registered
    cache = _load_encoder_cache(model_path)
    if not _encoder_save_registered:
        atexit.register(_save_all_encoder_caches)
        _encoder_save_registered = True
    return cache


class BaseEncoder(ABC):
    """
    Transfer textual messages into embeddings to represent in latent space by pre-trained models.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LMEncoder(BaseEncoder):
    """
    Embedding vias LM transformers.
    """
    def __init__(self, config):
        super().__init__(config)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'install the optional `transformers` dependency to use LMEncoder'
            ) from exc

        if not self.config.path:
            raise ValueError('LMEncoder requires CUE_MEM_BACKBONE_PATH or an explicit model path')

        if self.config.path in _lm_model_cache:
            self.tokenizer, self.model = _lm_model_cache[self.config.path]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path)
            self.model = AutoModel.from_pretrained(self.config.path).to(self.device)
            self.model.eval()
            _lm_model_cache[self.config.path] = (self.tokenizer, self.model)

        self._cache = _init_cache_for_encoder(self.config.path)
        self._cache_new = 0

    def __call__(self, text, return_type = 'numpy'):
        key = _text_hash(text)
        if key in self._cache:
            cached_np = self._cache[key]
            if return_type == 'numpy':
                return cached_np
            elif return_type == 'tensor':
                return torch.from_numpy(cached_np).to(self.device)
            else:
                return 'Unrecognized Return Type.'

        res = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**res).last_hidden_state[:, -1, :]

        self._cache[key] = embeddings.cpu().numpy()
        _encoder_cache_dirty[self.config.path] = True
        self._cache_new += 1
        if self._cache_new % 100 == 0:
            _save_all_encoder_caches()

        if return_type == 'numpy':
            return embeddings.cpu().numpy()
        elif return_type == 'tensor':
            return embeddings.to(self.device)
        else:
            return 'Unrecognized Return Type.'

class STEncoder(BaseEncoder):
    """
    Embedding vias Sentence Transformer.
    """
    def __init__(self, config):
        super().__init__(config)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'install the optional `sentence-transformers` dependency to use STEncoder'
            ) from exc

        if not self.config.path:
            raise ValueError(
                'STEncoder requires CUE_MEM_TEXT_ENCODER_PATH or an explicit model path'
            )

        if self.config.path in _st_model_cache:
            self.model = _st_model_cache[self.config.path]
        else:
            self.model = SentenceTransformer(self.config.path).to(self.device)
            _st_model_cache[self.config.path] = self.model

        self._cache = _init_cache_for_encoder(self.config.path)
        self._cache_new = 0

    def __call__(self, text, return_type = 'numpy'):
        key = _text_hash(text)
        if key in self._cache:
            cached_np = self._cache[key]
            if return_type == 'numpy':
                return cached_np
            elif return_type == 'tensor':
                return torch.from_numpy(cached_np).to(self.device)
            else:
                return 'Unrecognized Return Type.'

        embeddings = self.model.encode([text], normalize_embeddings=True)

        if isinstance(embeddings, torch.Tensor):
            cache_val = embeddings.cpu().numpy()
        else:
            cache_val = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        self._cache[key] = cache_val
        _encoder_cache_dirty[self.config.path] = True
        self._cache_new += 1
        if self._cache_new % 100 == 0:
            _save_all_encoder_caches()

        if return_type == 'numpy':
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy()
            return embeddings
        elif return_type == 'tensor':
            if isinstance(embeddings, torch.Tensor):
                return embeddings.to(self.device)
            return torch.from_numpy(cache_val).to(self.device)
        else:
            return 'Unrecognized Return Type.'
