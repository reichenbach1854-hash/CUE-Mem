# CUE-Mem GitHub Pages demo

This directory is a self-contained static deployment of the CUE-Mem demo.
It does not require `benchmark_demo/server.py`: the browser loads the exported
JSON files from `data/` and the selected images/audio from `media/`.

To publish it with GitHub Pages, copy the contents of this directory to the
repository's `docs/` directory, or publish this directory with a GitHub Actions
Pages workflow. Enable Pages in repository Settings → Pages, then select the
chosen branch and `/docs` source.

The original Python-backed demo remains in `benchmark_demo/`.
