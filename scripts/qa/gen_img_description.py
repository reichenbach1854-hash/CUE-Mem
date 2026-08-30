"""
兼容入口（仅图像 Caption）：等价于在项目根目录执行

    python qa/gen_media_captions.py --image-only ...

默认输出仍为原先文件名；完整流程请用 ``gen_media_captions.py``（先图后音频）。
"""
import sys

from scripts.qa.config import qa_path
from scripts.qa.gen_media_captions import main as media_main


if __name__ == "__main__":
    argv = [
        "--image-only",
        "--input",
        str(qa_path("qa_formatted_data_000_002.json")),
        "--output",
        str(qa_path("qa_formatted_data_000_002_with_vlm_image_desc.json")),
        "--img-token-stats",
        str(qa_path("qa_formatted_data_000_002_with_vlm_image_desc_tokens.json")),
    ] + sys.argv[1:]
    sys.exit(media_main(argv))
