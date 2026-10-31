# Keypoint network latency on an Apple M4 Pro

`uv run python scripts/bench_latency.py --frames 3` · commit `cf65463` · 2026-10-31 · model `pnlcalib-sv@7ea78fa76aaf+d72f4ed71734`

Machine: Apple M4 Pro, macOS 27.0.1; torch 2.15.0, onnxruntime 1.31.1. Load average at start (1, 5, 15 min): 3.34, 3.54, 3.51; other jobs may have shared the machine.

| Backend | Load (s) | Warm-up (ms) | ms / frame, median (min–max) | Frames / s | h per match at 10 fps | Max shift vs PyTorch CPU (px) |
|---|---|---|---|---|---|---|
| PyTorch CPU | 2.2 | 1127 | 1928 (1584–2116) | 0.5 | 28.9 | reference |
| PyTorch MPS | 2.9 | 675 | 563 (463–1322) | 1.8 | 8.4 | 0.0001 |
| ONNX Runtime CPU | 0.2 | 3215 | 1805 (1480–2545) | 0.6 | 27.1 | 0.0002 |
| ONNX Runtime CoreML | 3.1 | 713 | 464 (397–474) | 2.2 | 7.0 | 0.0001 |

Around the networks, on the CPU: JPEG decoding and resizing 16 ms per frame, heatmap decoding 41 ms per frame (medians).

**Method.** Both networks (keypoints and lines, HRNet-W48) run on one 960 × 540 frame at a time (batch 1, float32). The clock covers the copy to the device, both forward passes and the heatmaps back in host memory; MPS is synchronised (`torch.mps.synchronize()`) before the clock starts and stops. The first frame warms each backend up and is excluded; loading (weights and their hashes, session creation, CoreML compilation) is timed apart. Frames: 3 timed frames of dev clip SNGS-056 after one warm-up frame, the same preprocessed inputs for every backend. PyTorch CPU uses 8 threads, ONNX Runtime its defaults. CoreML options {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL'}: CoreML ran 785 of 785 nodes of the keypoint graph in 1 partition(s) and 785 of 785 nodes of the line graph in 1 partition(s). Max shift: the largest distance between the keypoints and line ends decoded from this backend's heatmaps and from PyTorch CPU's, in pixels of the 1920 × 1080 frame (the ONNX export itself is checked within 0.5 px by `scripts/export_onnx.py`).

**Caveats.** 3 timed frames per backend make a smoke run, not a stable estimate: no confidence interval and no thermal steady state. Hours per match extrapolate the median to 54,000 frames (90 minutes at 10 fps), networks only. No T4 / CUDA row yet.
