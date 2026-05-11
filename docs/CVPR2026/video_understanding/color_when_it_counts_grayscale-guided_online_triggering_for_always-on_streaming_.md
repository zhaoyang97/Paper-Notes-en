---
title: >-
  [Paper Note] Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing
description: >-
  [CVPR 2026][Video Understanding][Streaming Video Understanding] This paper proposes a novel "grayscale always-on, color on demand" paradigm. ColorTrigger detects color redundancy online via lightweight quadratic programm…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Streaming Video Understanding"
  - "Edge Devices"
  - "Grayscale-Guided Triggering"
  - "Energy-Efficient Sensing"
  - "Dynamic Token Routing"
date: 2026-05-08
content_hash: bb17cfa0b23dcb8e
---

# Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing

**Conference**: CVPR 2026
**arXiv**: [2603.22466](https://arxiv.org/abs/2603.22466)
**Code**: [lvgd.github.io/ColorTrigger](https://lvgd.github.io/ColorTrigger/)
**Area**: Video Understanding
**Keywords**: Streaming Video Understanding, Edge Devices, Grayscale-Guided Triggering, Energy-Efficient Sensing, Dynamic Token Routing

## TL;DR

This paper proposes a novel "grayscale always-on, color on demand" paradigm. ColorTrigger detects color redundancy online via lightweight quadratic programming on the grayscale stream, achieving 91.6% of the full-color baseline performance using only 8.1% RGB frames, enabling always-on video sensing on resource-constrained devices.

## Background & Motivation

Always-on sensing is a core requirement for next-generation wearable and edge AI systems, yet continuous high-fidelity RGB video capture imposes prohibitive costs on resource-constrained platforms.

**Energy bottleneck**: Devices such as smart glasses can sustain continuous RGB recording for only approximately 30–60 minutes, far short of the all-day assistant use case. Even when inference is offloaded to the cloud, end-to-end energy consumption remains dominated by continuous camera exposure and wireless transmission.

**Limitations of Prior Work**: Methods such as EgoTrigger trigger the RGB camera via audio cues, but trigger failures result in extended periods of complete visual information loss, making critical context unrecoverable.

**Key finding — color is not always necessary**: Pilot experiments on Qwen2.5-VL-7B show that replacing most frames with grayscale while retaining only a small fraction of RGB frames (preserving temporal structure) causes only marginal degradation in video understanding performance. This reveals pervasive **color redundancy** in natural video: semantic tasks such as action recognition, layout reasoning, and counting largely do not depend on color, requiring chromatic detail only at a small number of critical moments.

Based on this insight, ColorTrigger proposes to maintain temporal continuity via a continuously running grayscale stream and trigger RGB capture only when necessary, fundamentally reducing sensing cost.

## Method

### Overall Architecture

ColorTrigger comprises two core components:

1. **Causal Online Trigger**: Analyzes the affinity matrix of grayscale features within a sliding window to detect redundancy or novelty, and determines whether to trigger RGB capture via lightweight QP solving combined with a credit budget controller.
2. **Dynamic Token Router**: Grayscale frames are processed through a high-compression path (fewer tokens) while RGB frames are processed through a high-capacity path (more tokens); the token sequences are assembled temporally and fed into a frozen MLLM decoder.

The entire pipeline is **training-free**, strictly causal, and integrates seamlessly with frozen MLLMs.

### Key Designs

1. **Sliding-Window Grayscale Affinity Matrix**: At each time step $t$, a causal sliding window $\mathcal{W}_t$ of size $W$ is maintained. The $\ell_2$-normalized CLS token features $\mathbf{f}_i$ are extracted from each frame using a frozen CLIP visual encoder, and the affinity matrix is constructed as:

    $\tilde{\mathbf{A}}_t = \frac{1}{2}(\mathbf{F}_t \mathbf{F}_t^\top + \mathbf{I}_{n_t}) \in [0,1]^{n_t \times n_t}$

   A high $\tilde{A}_{ij}$ indicates that frames $i$ and $j$ are redundant (similar), while a low value indicates novelty or change. The matrix is symmetric, positive semi-definite, and strictly causal (computed only from frames in the current window).

2. **Diversity-Driven Quadratic Programming (QP)**: Frame selection is formulated as a continuous QP problem that assigns weights $\mathbf{w}_t \in [0,1]^{n_t}$ to each frame in the window:

    $\mathbf{w}_t = \arg\min_{\mathbf{w}} \lambda \mathbf{w}^\top \tilde{\mathbf{A}}_t \mathbf{w} \quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = m_t$

   The quadratic term $\mathbf{w}^\top \tilde{\mathbf{A}}_t \mathbf{w}$ penalizes concentrating weight on similar frames, naturally encouraging the budget to be distributed across temporally diverse frames. A higher weight $s_t = (\mathbf{w}_t)_{n_t}$ for the current frame indicates that it contributes novel information not covered by recent history, warranting RGB triggering.

3. **Credit-Budgeted Online Controller**: A scalar credit balance $b_t \in [0, C]$ is maintained, accruing at target rate $r$ per frame and consuming one unit per RGB trigger:

    $b_{t+1} = \text{clip}(b_t - u_t + r,\; 0, C)$

   A trigger decision requires both a geometric criterion ($s_t \geq \theta$) and a budget criterion ($b_t \geq 1$) to be satisfied:

    $u_t = \mathbb{I}[s_t \geq \theta \wedge b_t \geq 1]$

   Long-term RGB usage is bounded as $\sum_{t=1}^T u_t \leq rT + C$, guaranteeing budget compliance.

4. **Dynamic Token Routing**: Grayscale frames are input at low resolution to produce $T_g$ tokens, while RGB frames are input at high resolution to produce $T_c > T_g$ tokens:

    $\mathbf{Z}_t = (1 - u_t)\psi_g(g_t) \oplus u_t \psi_c(c_t)$

   The total computation cost $\sum_t [(1-u_t)T_g + u_t T_c]$ is substantially lower than the full-color cost $T \cdot T_c$ when RGB frames are sparse, concentrating computational resources on the most informative moments.

### Loss & Training

The proposed method is entirely **training-free** and requires no additional supervision or fine-tuning. All triggering and routing decisions are derived solely from the geometric relationships within the grayscale stream and integrate directly with frozen MLLMs.

## Key Experimental Results

### Main Results

Performance on StreamingBench real-time visual understanding tasks:

| Method | #Frames | RGB(%) | All (Acc) | Notes |
|--------|---------|--------|-----------|-------|
| Qwen2.5-VL-7B (full color) | 1fps | 100% | 73.68 | Full-color baseline |
| Human | - | - | 91.46 | Human performance |
| TimeChat-Online-7B | 1fps | 100% | 75.36 | Streaming MLLM SOTA |
| InternVL-3.5-8B | 128 | 100% | - | Strong open-source model |
| **ColorTrigger** | 1fps | **8.1%** | **67.49** | Only 8.1% RGB frames |

ColorTrigger achieves 91.6% of the full-color baseline performance (67.49/73.68) using only 8.1% RGB frames, with balanced performance across sub-tasks including object perception, causal reasoning, and action recognition.

### Ablation Study

| Configuration | RGB(%) | All (Acc) | Notes |
|---------------|--------|-----------|-------|
| All grayscale | 0% | ~60 | Pure grayscale; significant degradation |
| Uniform sampling 5% RGB | 5% | ~63 | Random sparse RGB insertion |
| Uniform sampling 10% RGB | 10% | ~66 | Uniform sampling |
| ColorTrigger 8.1% | 8.1% | 67.49 | Intelligent triggering outperforms uniform sampling |
| ColorTrigger + Token Routing | 8.1% | 67.49 | Token routing further reduces inference cost |
| Full color 100% RGB | 100% | 73.68 | Upper bound |

### Key Findings

- Natural video contains substantial color redundancy: retaining only 5–10% RGB frames recovers most performance.
- Intelligent triggering outperforms uniform sampling at equivalent RGB ratios.
- Dynamic token routing further reduces inference cost without sacrificing performance.
- The paradigm can be directly applied to existing frozen MLLMs without any training.

## Highlights & Insights

- **"Color is not always necessary" is a profound observation**: it fundamentally challenges the implicit assumption that more RGB equates to better performance.
- **The grayscale always-on, color on-demand paradigm** carries significant practical implications for edge AI deployment — it can substantially extend battery life in smart glasses, surveillance cameras, and similar applications.
- **The QP + credit budget design** elegantly balances local triggering flexibility with global budget constraints.
- Fully training-free and plug-and-play, compatible with any frozen MLLM.

## Limitations & Future Work

- Grayscale cameras typically have lower resolution and image quality than RGB cameras; handling hardware heterogeneity in real deployment remains to be validated.
- The current approach uses only the CLIP CLS token for affinity analysis, which may overlook local detail changes.
- Although lightweight, QP solving still incurs computational overhead; real-time feasibility on ultra-low-power chips requires further evaluation.
- Validation is limited to streaming video understanding benchmarks; power consumption and latency measurements on actual edge devices are lacking.
- Tasks that are highly color-dependent (e.g., color recognition) may be more substantially affected.

## Related Work & Insights

- Conceptually analogous to event cameras (activity-driven sampling), but requires no specialized hardware.
- Complementary to EgoTrigger (audio-triggered RGB): ColorTrigger maintains continuous grayscale visual context, avoiding complete visual information blackouts.
- Token pruning and merging methods (e.g., ToMe, ATP-LLaVA) perform compression after capture; ColorTrigger reduces cost **before capture**.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The grayscale always-on, color on-demand paradigm is entirely novel with far-reaching practical implications.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough validation on StreamingBench, but lacks real hardware power consumption evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative arc from pilot study to insight to method to experiments is highly coherent.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable engineering value for always-on sensing in edge AI, with strong inspirational merit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamReady: Learning What to Answer and When in Long Streaming Videos](streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](../../ICCV2025/video_understanding/online_dense_point_tracking_with_streaming_memory.md)

</div>

<!-- RELATED:END -->
