---
title: >-
  [Paper Note] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models
description: >-
  [AAAI 2026 Oral][Video Understanding][Long video compression] MambaMia proposes a two-stage hierarchical video token compression framework based on bidirectional Mamba: Gated Patch Aggregation (GPA) for spatial-temporal local compression, and a Temporal Axis Aggregator (TAA) that leverages Mamba's adaptive step size $\Delta_t$ for data-driven keyframe sampling. The method compresses hour-long videos to only 4.7K tokens, achieving 44.6 on LVBench and surpassing Qwen2-VL and mP…
tags:
  - "AAAI 2026 Oral"
  - "Video Understanding"
  - "Long video compression"
  - "state space models"
  - "Mamba"
  - "gated patch aggregation"
  - "adaptive frame sampling"
date: 2026-05-08
content_hash: 7c07d46969f84edf
---

# MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models

**Conference**: AAAI 2026 Oral  
**arXiv**: [2506.13564](https://arxiv.org/abs/2506.13564)  
**Code**: [https://github.com/naver-ai/mambamia](https://github.com/naver-ai/mambamia)  
**Area**: Large Multimodal Models / Video Understanding
**Keywords**: Long video compression, state space models, Mamba, gated patch aggregation, adaptive frame sampling

## TL;DR
MambaMia proposes a two-stage hierarchical video token compression framework based on bidirectional Mamba: Gated Patch Aggregation (GPA) for spatial-temporal local compression, and a Temporal Axis Aggregator (TAA) that leverages Mamba's adaptive step size $\Delta_t$ for data-driven keyframe sampling. The method compresses hour-long videos to only 4.7K tokens, achieving 44.6 on LVBench and surpassing Qwen2-VL and mPLUG-Owl3.

## Background & Motivation

**Background**: Large multimodal models (LMMs) excel at image and short video understanding, but processing hour-long videos poses a severe token explosion problem — hundreds of frames can generate hundreds of thousands of tokens, far exceeding the capacity of standard models and hardware.

**Limitations of Prior Work**: (1) Per-frame spatial pooling and token pruning address only single-frame redundancy, failing to resolve inter-frame temporal accumulation; (2) query-based selection methods are task-specific and sacrifice general-purpose context modeling; (3) brute-force context window scaling demands enormous computational resources, making it impractical for academic or production settings.

**Key Challenge**: Long videos contain two types of redundancy — intra-frame spatial redundancy (many similar patches) and inter-frame temporal redundancy (highly similar content across consecutive frames) — while simultaneously containing fine-grained key events that must be preserved. A general solution is needed that achieves aggressive compression without losing critical information.

**Goal**: How to efficiently compress visual tokens from hour-long videos on standard hardware while maintaining understanding performance?

**Key Insight**: Exploit the linear complexity of state space models (Mamba) for processing ultra-long sequences, and repurpose Mamba's internal adaptive step size $\Delta_t$ as a frame importance signal for adaptive sampling.

**Core Idea**: Apply bidirectional Mamba with gated aggregation for spatial compression, then reuse Mamba's step size for adaptive temporal frame selection, achieving hierarchical long-video compression.

## Method

### Overall Architecture
A two-stage compression pipeline: 384-frame video input → visual encoder extracts 576 patch tokens per frame (≈221K tokens total) → **Stage 1: Spatiotemporal Compression (GPA)** compresses each frame to 24 anchor tokens (≈9.2K total) → **Stage 2: Temporal Axis Aggregator (TAA)** further compresses via delta sampling to ≈4.7K tokens → fed into LLM.

### Key Designs

1. **Gated Patch Aggregation (GPA)**:

    - **Function**: After bidirectional Mamba processing, learnable query anchors aggregate surrounding patch information within the sequence.
    - **Mechanism**: Each row of 24 patches corresponds to one query anchor. GPA applies query-conditioned weighted pooling over neighboring patches: $\boldsymbol{\alpha} = \text{softmax}(\mathbf{W}_\alpha \mathbf{q} + \mathbf{b}_\alpha)$, $\mathbf{a} = \sum_i \alpha_i \mathbf{x}_i$. A gating mechanism then adaptively blends the results: $\mathbf{f} = (1-g)\mathbf{q} + g \cdot \mathbf{a}$, where $g = \sigma(\mathbf{W}_g \mathbf{q} + b_g)$.
    - **Design Motivation**: When $g \approx 0$, the anchor retains its own information (state-space context); when $g \approx 1$, it absorbs local patch information. This is more flexible than the 3D average pooling used in BIMBA; ablations show GPA yields approximately 7% average improvement.

2. **Temporal Axis Aggregator (TAA) + Delta Sampling**:

    - **Function**: Models inter-frame dependencies along the temporal axis and performs data-driven keyframe selection using Mamba's adaptive step size $\Delta_t$.
    - **Mechanism**: A unidirectional Mamba processes the frame-level anchor sequence, with its internal $\Delta_t = \text{softplus}(\mathbf{W}_\Delta \mathbf{f}_t + \mathbf{b}_\Delta)$ learned end-to-end. $\Delta_t$ is interpreted as a frame importance score — frames with larger $\Delta_t$ are considered more informative by the model. A cumulative delta sampling algorithm accumulates $\Delta_t$ and selects a frame when the cumulative value exceeds a threshold $\delta_{\text{thresh}}$, then resets the accumulator. By default, approximately 50% of frames are retained (384→192).
    - **Design Motivation**: The SSM's internal $\Delta_t$ is repurposed directly rather than training a separate selector — this step size inherently reflects the information content of the input (large $\Delta_t$ = larger state update = more new information). Visualizations show $\Delta_t$ peaks align with scene transitions and key events.

3. **Bidirectional Mamba Spatiotemporal Compressor**:

    - **Function**: Processes the entire spatiotemporal token sequence prior to GPA, enabling spatial and temporal information sharing.
    - **Mechanism**: Three layers of bidirectional Mamba2 blocks process sequences of ≈230K tokens. The bidirectional design allows each token to attend to both past and future context.
    - **Design Motivation**: Replacing bidirectional Mamba with unidirectional Mamba causes a drop of approximately 1.7 points, confirming that bidirectional modeling is important for spatiotemporal feature sharing.

### Loss & Training
Three-stage LLaVA-style training: image understanding → module alignment (compression layers only) → video instruction fine-tuning (LLM unfrozen). Training uses 128 frames; inference uses 384 frames. Delta sampling is applied at inference time only. The compression module contains approximately 247M parameters.

## Key Experimental Results

### Main Results — Long Video Benchmarks

| Model | LLM | Max Tokens | LVBench | MLVU | VideoMME | VNBench |
|-------|-----|------------|---------|------|----------|---------|
| Qwen2-VL | Qwen2-7B | - | 42.0 | 64.2 | 55.6 | 33.9 |
| LLaVA-Video | Qwen2-7B | 12.5K | 43.8 | 70.8 | 63.3 | 37.0 |
| mPLUG-Owl3 | Qwen2-7B | - | 43.5 | - | 53.5 | - |
| **MambaMia** | Qwen2-7B | **4.7K** | **44.6** | 68.0 | 58.3 | 41.5 |

### Ablation Study — Compression Module Design

| Configuration | GPA | TAA | LVBench | MLVU | MME | Avg |
|---------------|-----|-----|---------|------|-----|-----|
| BIMBA (3D pool) | ✗ | ✗ | 35.3 | 53.8 | 47.3 | 45.4 |
| +GPA | ✓ | ✗ | 41.1 | 62.4 | 53.2 | 52.2 |
| +GPA+TAA (Full) | ✓ | ✓ | 41.1 | 64.0 | 55.7 | **53.6** |

### Key Findings
- Only 4.7K tokens are needed to achieve performance comparable to LLaVA-Video using 12.5K tokens, representing approximately 2.6× improvement in token efficiency.
- Replacing 3D average pooling with GPA yields approximately 7 points average improvement — learnable gated aggregation substantially outperforms fixed pooling.
- Delta sampling outperforms uniform sampling by 1.2 points on LVBench (44.6 vs. 43.4), a statistically significant difference ($p = 0.047$).
- Even when using Mamba as the LLM backbone, dedicated compression is necessary — a vanilla Mamba LLM performs substantially below the compressed variant.
- Performance saturates at 384 frames; additional frames provide no further gain.
- Strong performance on VNBench (needle-in-a-video-haystack, 41.5) demonstrates that compression does not discard critical fine-grained information.

## Highlights & Insights
- **Repurposing Mamba $\Delta_t$ as a frame importance signal**: The most elegant design choice — the SSM step size inherently encodes input information content, eliminating the need for a separate importance predictor. This idea generalizes to any sequence processing scenario utilizing SSMs.
- **Modular pre-LLM compression**: Unlike VAMBA, which compresses inside the LLM, MambaMia performs compression independently before the LLM, maintaining modularity and lightweight design.
- **Rigorous experimental methodology**: The paper emphasizes from-scratch training, controlled variable comparisons, multi-seed statistical validation, and significance testing — the experimental design is exemplary.

## Limitations & Future Work
- Performance saturation at 384 frames suggests an information bottleneck in the compression layer; exploring more frames with improved compression strategies is worthwhile.
- The query-conditioned pooling in GPA attends only to the query token without content-aware attention across patches (for efficiency), potentially losing inter-patch relationships.
- $\delta_{\text{thresh}}$ is manually set; an adaptive threshold would be preferable.
- Evaluation is limited to 7B-scale models; effectiveness on larger LLMs remains to be verified.
- The train-test mismatch between uniform sampling (training) and delta sampling (inference) may constrain performance.

## Related Work & Insights
- **vs. BIMBA**: Both share a Mamba + periodic query architecture, but BIMBA uses 3D average pooling while MambaMia employs learnable gated aggregation and delta sampling; ablations show MambaMia consistently outperforms BIMBA.
- **vs. LLaVA-Video**: LLaVA-Video uses 12.5K tokens; MambaMia achieves comparable performance with 4.7K tokens, demonstrating a clear efficiency advantage.
- **vs. Video-XL**: Video-XL aggregates inside the LLM with CLIP-based frame selection, whereas MambaMia compresses independently outside the LLM with learned frame selection, yielding a more modular architecture.
- Insight: The inter-frame temporal fusion approach in TTF-VLA and MambaMia's temporal compression are complementary — TTF-style enhancement followed by MambaMia-style compression is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Repurposing $\Delta_t$ for frame sampling is clever; GPA design is also novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven benchmarks, five compression comparisons, multi-seed statistical testing, and cost analysis — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Method descriptions are precise; a 13-section appendix covers all implementation details with strong reproducibility.
- Value: ⭐⭐⭐⭐⭐ Processing hour-long videos with only 4.7K tokens offers substantial practical value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](../../CVPR2025/video_understanding/gg-ssms_graph-generating_state_space_models.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](../../ECCV2024/video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)

</div>

<!-- RELATED:END -->
