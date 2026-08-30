"""
兼容入口（仅音频 Caption）：从 ``qa_formatted_data_*.json`` 结构中读取 ``dialog[].audio_path``，
调用本地 Qwen2-Audio，将说明写回对应 ``dialog_list`` 中的 ``*.wav`` 字段。

等价于::
    python qa/gen_media_captions.py --audio-only

（旧版 ``profile['history']`` + v1_profiles 的用法已废弃，请统一使用 qa_formatted_data。）
"""
import sys

from scripts.qa.config import qa_path
from scripts.qa.gen_media_captions import main as media_main


if __name__ == "__main__":
    argv = [
        "--audio-only",
        "--input",
        str(qa_path("qa_formatted_data_000_002.json")),
        "--output",
        str(qa_path("qa_formatted_data_000_002_with_audio_captions.json")),
    ] + sys.argv[1:]
    sys.exit(media_main(argv))
