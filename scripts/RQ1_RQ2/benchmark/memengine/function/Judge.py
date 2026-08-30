from abc import ABC, abstractmethod
from memengine.function.LLM import *
import hashlib
import pickle
import os
import atexit

from benchmark.memengine.function.prompt import render_prompt
from benchmark.paths import cache_path
from benchmark.security import redact_runtime_text

# ---------------------------------------------------------------------------
# Module-level LLMJudge cache (persists importance scores across runs)
# ---------------------------------------------------------------------------
_judge_cache = {}
_judge_cache_dirty = False
_judge_save_registered = False


def _get_judge_cache_path():
    cache_dir = str(cache_path())
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, 'llm_judge_cache.pkl')


def _load_judge_cache():
    global _judge_cache
    path = _get_judge_cache_path()
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                _judge_cache = pickle.load(f)
            print(f"[JudgeCache] Loaded {len(_judge_cache)} cached importance scores")
        except Exception as e:
            print(f"[JudgeCache] Failed to load cache: {redact_runtime_text(e)}")
            _judge_cache = {}


def _save_judge_cache():
    global _judge_cache_dirty
    if not _judge_cache_dirty:
        return
    path = _get_judge_cache_path()
    try:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    disk_data = pickle.load(f)
                disk_data.update(_judge_cache)
                _judge_cache.update(disk_data)
            except Exception:
                pass
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(_judge_cache, f)
        os.replace(tmp, path)
        _judge_cache_dirty = False
        print(f"[JudgeCache] Saved {len(_judge_cache)} cached importance scores")
    except Exception as e:
        print(f"[JudgeCache] Failed to save cache: {redact_runtime_text(e)}")


class BaseJudge(ABC):
    """
    Assess given observations or intermediate messages on certain aspects.
    """
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LLMJudge(BaseJudge):
    """
    Judge vias large language models.
    """
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)

        global _judge_save_registered
        if not _judge_save_registered:
            _load_judge_cache()
            atexit.register(_save_judge_cache)
            _judge_save_registered = True
        self._cache_new = 0

    def __post_scale__(self, res):
        score = float(eval(res))
        if hasattr(self.config, 'post_scale'):
            return score/self.config.post_scale

    def __post_bool__(self, res):
        if res == 'True':
            return True
        elif res == 'False':
            return False
        else:
            return "LLM Judge Parse Error for Boolean"

    def __call__(self, input_dict, post_process = 'scale'):
        cache_key = hashlib.sha256(
            str(sorted(input_dict.items())).encode('utf-8')
        ).hexdigest()

        if cache_key in _judge_cache:
            return _judge_cache[cache_key]

        prompt = render_prompt(self.config.prompt, input_dict)
        res = self.llm.fast_run(prompt)

        if post_process == 'scale':
            result = self.__post_scale__(res)
        elif post_process == 'bool':
            result = self.__post_bool__(res)
        else:
            raise ValueError("Judge Post Process Type Error!")

        global _judge_cache_dirty
        _judge_cache[cache_key] = result
        _judge_cache_dirty = True
        self._cache_new += 1
        if self._cache_new % 50 == 0:
            _save_judge_cache()

        return result
    
