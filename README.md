# VLM-Zoom-Agent — Phase 1 (Baseline)

A simple video-QA agent built on **Qwen2.5-VL-7B-Instruct** (4-bit quantized) that answers
questions about fine visual details in a video by zooming in *spatio-temporally*:

1. **Overview pass** – look at N evenly-spaced frames at low resolution.
2. If not confident, the model proposes **one frame + one region (bbox)** to inspect.
3. **Coarse-to-fine grounding** – crop and upscale that region.
4. **Zoom pass** – ask the model again on the high-resolution crop and return the final answer.

> This is the baseline **without** backtracking and self-consistency (planned for Phase 2).

## Notebook

| File | Description |
|------|-------------|
| `vlm_zoom_agent_phase1.ipynb` | Full pipeline: model loading, frame sampling, JSON parsing, crop+upscale, agent loop, visualization |

## Requirements

- NVIDIA GPU (tested on Colab **T4 / 16 GB**)
- Python 3.10+

```bash
pip install -r requirements.txt
```

## Usage

1. Open the notebook in Jupyter or Google Colab (Runtime → GPU):
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliFazelM/VLM-Zoom-Agent/blob/main/vlm_zoom_agent_phase1.ipynb)
2. Run the cells in order (the first run downloads the model, ~16 GB).
3. Set your video path and question:

```python
result = joint_zoom_agent(
    "sample_video.mp4",
    "Does she wear lipstick?",
    num_frames=8,
    min_zoom_ratio=0.15,
    fine_min_ratio=0.30,
    verbose=True,
)
visualize_zoom_result(result, frames=result.get("frames"))
```

The notebook downloads a sample video from
[intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) for a quick test.

## Roadmap (Phase 2)

- Self-consistency: ask the verify prompt 2–3 times with `temperature > 0` and check agreement
- Backtracking: keep a list of rejected `(frame_idx, bbox)` regions and tell the model to try elsewhere

## License

MIT — see [LICENSE](LICENSE).

## Author

Ali Fazel Maragheh ([@AliFazelM](https://github.com/AliFazelM))
