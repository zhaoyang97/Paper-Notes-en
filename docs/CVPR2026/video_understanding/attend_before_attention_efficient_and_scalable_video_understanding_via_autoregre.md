---
title: >-
  [Paper Note] Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing
description: >-
  [CVPR2026][Video Understanding][video redundancy removal] This paper proposes AutoGaze—a lightweight autoregressive module with only 3M parameters—that operates **before** the ViT to select the minimal set of patches in a multi-scale manner, eliminating spatiotemporal redundancy and achieving 4×–100× token compression and up to 19× ViT speedup, enabling MLLMs to scale to 1K-frame 4K-resolution video.
tags:
  - CVPR2026
  - Video Understanding
  - video redundancy removal
  - autoregressive gazing
  - multi-scale patch selection
  - visual token compression
  - long video high-resolution understanding
date: 2026-05-08
content_hash: e62dacf268c80603
---

# Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing

**Conference**: CVPR2026  
**arXiv**: [2603.12254](https://arxiv.org/abs/2603.12254)  
**Code**: [autogaze.github.io](https://autogaze.github.io/)  
**Area**: Video Understanding  
**Keywords**: video redundancy removal, autoregressive gazing, multi-scale patch selection, visual token compression, long video high-resolution understanding

## TL;DR

This paper proposes AutoGaze—a lightweight autoregressive module with only 3M parameters—that operates **before** the ViT to select the minimal set of patches in a multi-scale manner, eliminating spatiotemporal redundancy and achieving 4×–100× token compression and up to 19× ViT speedup, enabling MLLMs to scale to 1K-frame 4K-resolution video.

## Background & Motivation

**Substantial spatiotemporal redundancy in video**: Static backgrounds repeat across consecutive frames, yet existing MLLMs process every pixel of every frame uniformly, wasting significant computation.

**Limitations of Prior Work**: Existing token compression methods (e.g., ToMe, LongVU, STORM) prune or merge tokens only inside the ViT or between the ViT and LLM; the ViT still processes the full video, forming an efficiency bottleneck.

**Key Challenge**: Heuristic pruning methods based on attention scores underperform learned approaches, while methods involving search and reasoning incur large overhead that limits scalability.

**Goal**: Real-world applications (surveillance, autonomous driving, robotics) demand processing of minute-long 4K video, yet existing models cannot scale to high spatiotemporal resolution due to computational cost.

**Key Insight**: Human vision selectively fixates on moving objects and detail regions via saccades, skipping static backgrounds to achieve efficient real-time scene understanding.

**Gap in benchmarks**: Existing benchmarks (VideoMME, MLVU, etc.) target long video but not high resolution, making it impossible to evaluate models on high-resolution long video.

## Method

### Overall Architecture

AutoGaze is a 3M-parameter lightweight module (convolutional encoder + autoregressive Transformer decoder) placed **before** the ViT. Given a video, AutoGaze encodes frames sequentially and autoregressively decodes patch indices:

1. **Frame encoding**: A convolutional encoder extracts features from the current frame.
2. **Autoregressive gazing**: The decoder, conditioned on current-frame features and historical frame/selected-patch information, sequentially outputs patch indices (vocabulary: $\{1, \ldots, V\}$).
3. **Automatic stopping**: An auxiliary head predicts whether the currently selected patches can reconstruct the frame below a threshold; once the predicted loss falls below a user-defined threshold $\varepsilon$, gazing for the current frame terminates.
4. **Multi-scale gazing**: The vocabulary spans multiple patch scales, allowing the model to select coarse scales for low-detail regions and fine scales for high-detail regions.
5. **Multi-token prediction**: Multiple patch indices are emitted simultaneously via multiple heads, accelerating inference.

### Key Designs

- **Reconstruction objective**: A customized VideoMAE (block-causal attention) serves as the reconstructor; reconstruction quality is measured by a weighted sum of pixel reconstruction loss and perceptual loss.
- **Cross-frame information propagation**: When decoding frame $t$, the model can attend to features from frames $1 \ldots t$ and selected patches from frames $1 \ldots t-1$, avoiding redundant selection of repeated patches.
- **Arbitrary resolution/duration inference**: The video is tiled into $16 \times 224 \times 224$ spatiotemporal tiles; AutoGaze runs independently on each tile and results are merged.
- **ViT adaptation**: Patches at different scales are embedded separately and concatenated before being fed into the ViT, enabling a standard image ViT to accept multi-scale patch inputs; the image ViT is also extended to a video ViT by concatenating 16-frame tokens into a single sequence.

### Loss & Training

**Stage 1: NTP Pre-training**

- Greedy search collects approximately 250K near-optimal gazing sequences (800K videos, 16 frames at 224 resolution).
- Standard next-token prediction cross-entropy loss trains patch index prediction.
- An $\ell_2$ loss supervises per-step reconstruction loss prediction.

**Stage 2: RL Post-training**

- A simplified on-policy GRPO algorithm is adopted, with reconstruction loss as the reward.
- The advantage function $A$ is the discounted sum of negative future-frame reconstruction losses, normalized within the group.
- This further calibrates reconstruction loss prediction.

**Loss functions**:

- Pre-training: $L_{NTP} = -\sum_{t,k} \log \pi_\theta(\tilde{p}_k^t \mid X^{1:t}, \tilde{p}_{1:k-1}^t)$ + $\ell_2$ loss for reconstruction loss prediction
- Post-training: $L_{GRPO} = -\sum_{t,k} \frac{\pi_\theta(p_k^t)}{\pi_{\theta_{detached}}(p_k^t)} A_k^t$ + reconstruction loss prediction supervision

## Key Experimental Results

### Main Results

| Model | Max Frames | Max Resolution | VideoMME (w/o sub) | VideoMME (w/ sub) | MLVU | HLVid |
|---|---|---|---|---|---|---|
| GPT-4o | - | - | 71.9 | 77.2 | 64.6 | 49.3 |
| Qwen2.5-VL-7B | 48 | 896 | 65.1 | 71.6 | 70.2 | 48.1 |
| VideoChat-Flash | 10000 | 448 | 65.3 | 69.7 | 74.7 | 46.6 |
| NVILA-8B-Video | 256 | 448 | 64.2 | 70.0 | 70.1 | 42.5 |
| **+ AutoGaze** | **1024** | **3584** | **67.0** | **71.8** | **71.6** | **52.6** |

NVILA + AutoGaze achieves 67.0% on VideoMME and improves HLVid from 42.5% to 52.6% (+10.1%), surpassing GPT-4o's 49.3%.

### Comparison with Token Compression Methods

At 128 frames and 6.25% selection ratio:

| Method | ViT Latency | LLM Latency | VideoMME |
|---|---|---|---|
| No compression | 2.20s | 1.42s | 53.4 |
| ToMe | 2.23s | 0.11s | 51.5 |
| STORM | 2.18s | 0.15s | 52.7 |
| LongVU | 2.17s | 0.12s | 52.2 |
| **AutoGaze** | **0.55s** | **0.10s** | **52.3** |

AutoGaze is the only method that substantially reduces ViT latency (4× speedup) while maintaining the lowest LLM-side latency.

### Ablation Study

**Training pipeline ablation**: NTP pre-training alone yields a gazing ratio of 0.102; RL post-training alone yields 0.209; combining both achieves 0.094 (optimal).

**Model design ablation**: Multi-token prediction with $k=10$ achieves the best efficiency–quality trade-off (0.193s latency, gazing ratio 0.094); multi-scale gazing reduces the gazing ratio from 0.220 to 0.094, a 2.3× efficiency gain.

### Key Findings

- **Gazing preference for motion regions**: Patches with larger optical flow are selected with higher probability across all scales.
- **Fine-grained scales correspond to high-detail regions**: Patches selected at fine scales exhibit higher Laplacian variance (correlation $\rho = 0.12$, $p < 0.001$).
- **Strong OOD generalization**: AutoGaze robustly tracks changing regions on out-of-distribution videos such as CCTV footage, robotic manipulation, and style-transferred content.
- **~1% patches suffice for 30FPS 4K video**: Redundancy increases with higher FPS and resolution.

## Highlights & Insights

- Removing redundancy **before the ViT** genuinely resolves the efficiency bottleneck, rather than optimizing only the LLM side.
- With only 3M parameters, the module incurs minimal overhead and is plug-and-play.
- The autoregressive framework unifies patch selection and stopping decisions without manual threshold tuning.
- The two-stage training (NTP + RL) is concise and effective; the RL stage yields an additional ~10% improvement over NTP pre-training alone.
- The paper introduces HLVid, the first high-resolution long-video QA benchmark (268 QA pairs, 5-minute 4K video).
- The method scales to 1K frames at 4K resolution and outperforms GPT-4o on HLVid.

## Limitations & Future Work

- The reconstruction objective is a proxy task and may not fully align with downstream semantic understanding goals.
- Integration is validated only on NVILA-8B; generalizability to other MLLMs (e.g., Qwen2.5-VL, InternVL) remains unverified.
- Multi-scale gazing requires modifications to the ViT's patch embedding and positional encoding, introducing non-trivial invasiveness to existing models.
- HLVid contains only 268 QA pairs; its scale is limited, and evaluation reliability warrants validation at larger scale.
- Tiled processing may introduce information discontinuities at tile boundaries.

## Related Work & Insights

- **Video understanding MLLMs**: NVILA, Qwen2.5-VL, LongVILA, VideoChat-Flash, and others scale to long video but are constrained to low resolution.
- **Spatial token compression**: ToMe (token merging), VisionZip, FastV (attention pruning)—all operate only inside the model.
- **Spatiotemporal token compression**: STORM (spatiotemporal routing), LongVU (cross-frame deduplication), FastVID—similarly do not reduce ViT input.
- **Adaptive tokenization**: FlexTok, AnyRes, and related methods flexibly allocate token counts, but the tokenizer itself incurs significant overhead.
- **Video representation learning**: Self-supervised methods such as V-JEPA2 and VideoMAE provide the reconstruction capability foundation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Autoregressive patch selection before the ViT is a genuinely novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Behavioral analysis, efficiency comparison, SOTA comparison, ablation studies, and a new benchmark are all included.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rich figures and tables, well-motivated throughout.
- Value: ⭐⭐⭐⭐⭐ — Addresses the core efficiency bottleneck of video MLLMs with strong practical impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

<!-- RELATED:END -->
