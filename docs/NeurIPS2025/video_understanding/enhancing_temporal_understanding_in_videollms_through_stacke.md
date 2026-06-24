---
title: >-
  [Paper Note] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders
description: >-
  [Video Understanding] This paper proposes STAVEQ2, which inserts parameter-efficient Stacked Temporal Attention (STA) modules into the Vision Encoder to address fundamental architectural deficiencies in existing Video-LLMs for fine-grained temporal understanding (e.g., distinguishing "pulling from left to right" vs. "pulling from right to left"), achieving up to 5.5% improvement on VITATECS/MVBench/Video-MME.
tags:
  - "Video Understanding"
date: 2026-05-08
content_hash: 75f41992aac5671a
---

# Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders

## Basic Information
- **arXiv**: 2510.26027
- **Conference**: NeurIPS 2025
- **Authors**: Ali Rasekh, Erfan Bagheri Soula, Omid Daliran, Simon Gottschalk, Mohsen Fayyaz
- **Institutions**: Leibniz University Hannover / L3S Research Center, Microsoft
- **Code**: https://alirasekh.github.io/STAVEQ2/

## TL;DR
This paper proposes STAVEQ2, which inserts parameter-efficient Stacked Temporal Attention (STA) modules into the Vision Encoder to address fundamental architectural deficiencies in existing Video-LLMs for fine-grained temporal understanding (e.g., distinguishing "pulling from left to right" vs. "pulling from right to left"), achieving up to 5.5% improvement on VITATECS/MVBench/Video-MME.

## Background & Motivation
Existing Video-LLMs exhibit fundamental deficiencies in temporal understanding:
- **Qwen2-VL**: The Vision Encoder contains only spatial attention, delegating temporal understanding entirely to the LLM.
- **InternVideo2-Chat**: Despite joint spatiotemporal attention, it still fails to reliably distinguish temporally directional actions.
- Experiments show: On SSv2-T10 (temporally opposing action pairs), Qwen2-VL 7B achieves only 21.91% zero-shot, and InternVideo2 only 30.60%.
- In-context learning not only fails to help but degrades performance — indicating an architectural deficiency rather than a data problem.

## Core Problem
How can the temporal understanding of Video-LLMs be improved by enhancing the temporal modeling capability of the Vision Encoder, without modifying the LLM?

## Method

### STAVEQ2 Architecture
A temporal attention layer is inserted after the spatial attention in each transformer block of Qwen2-VL's ViT.

**Spatial Attention** (original): Self-attention among $N$ patches within each frame:
$$S_t^{(m)} = A_t^{(m)} V_t^{(m)} + X_t^{(m-1)}$$

**Temporal Attention** (newly added): Self-attention for each patch across $T$ frames:
$$Z_i^{(m)} = A_i'^{(m)} V_i'^{(m)} + Y_i^{(m)}$$
where $Y_i^{(m)} = [S_{1,i}^{(m)}, \ldots, S_{T,i}^{(m)}]^\top$

Final output: $X^{(m)} = \text{MLP}(\text{LN}(Z^{(m)})) + Z^{(m)}$

### Key Designs
1. **Parameter Efficiency**: The number of temporal attention heads is only 1/4 of the spatial heads (head scale = 0.25), substantially reducing parameters.
2. **1D RoPE**: Temporal attention uses 1D RoPE (vs. 2D RoPE for spatial) to encode temporal positions.
3. **Zero Initialization**: The output projection layer is initialized to zero, making the initial state equivalent to the original model.
4. **Two-Stage Training**:
    - Stage 1: All parameters frozen; only the temporal attention blocks and LayerNorm are trained.
    - Stage 2: LoRA adapters are introduced for joint training of the entire model.
5. **Full-Layer Deployment**: Applying STA to all 32 transformer blocks yields the best performance.

## Key Experimental Results

### SSv2 Action Recognition (Vision-only)

| Model | SSv2 Acc. |
|---|---|
| InternVideo2 1B | 77.1% |
| InternVideo2 6B | 77.5% |
| **InternVideo2 1B + STA** | **78.0% (+0.5%)** |

→ A 1.3B model surpasses a 6B model!

### InternVideo2-Chat + STA (SSv2-T10)

| Method | Acc. |
|---|---|
| InternVideo2-Chat 8B | 84.17% |
| + STA | **95.18% (+11.01%)** |

### STAVEQ2 on Video-LLM Benchmarks

| Model | VITATECS Dir. | MVBench | Video-MME (wo/w sub) |
|---|---|---|---|
| Qwen2-VL 7B | 86.6 | 67.0 | 63.3 / 69.0 |
| Qwen2.5-VL 7B | 80.0 | 69.6 | 65.1 / 71.6 |
| **STAVEQ2 7B** | **87.6** | **70.1** | **66.8 / 71.8** |
| Qwen2-VL 72B | 87.8 | 73.6 | 71.2 / 77.8 |
| **STAVEQ2 72B** | **90.1** | **74.5** | **73.9 / 79.9** |
| GPT-4o | – | – | 71.9 / 77.2 |

→ STAVEQ2 72B surpasses GPT-4o on Video-MME (+2.0/+2.7).

### Cross-Model Generalization
- STAVEQ2.5 (Qwen2.5-VL + STA): Further improvements observed.
- VideoRoPE + STA: Complementary gains.
- InternVideo2.5-Chat + STA: MVBench improves from 75.7 to 76.8.

## Highlights & Insights
1. **Thorough Problem Analysis**: Systematically demonstrates that temporal understanding failures are architectural rather than data-related (zero-shot vs. ICL vs. fine-tuning comparisons).
2. **Simple yet Effective**: Only lightweight temporal attention is stacked within the ViT; the LLM remains unchanged.
3. **Broad Generalization**: Effective across Qwen2-VL, Qwen2.5-VL, InternVideo2, and VideoRoPE.
4. **New SOTA**: New state-of-the-art on SSv2 action recognition (1.3B surpasses 6B); Video-MME surpasses GPT-4o.
5. **Revival of Divided Space-Time Attention**: Validates the value of TimeSformer-style divided attention within Video-LLMs.

## Limitations & Future Work
1. Due to resource constraints, the approach is validated only via fine-tuning rather than pre-training from scratch.
2. The largest model evaluated is 72B, despite already outperforming many larger models.
3. STA introduces additional inference latency (one extra temporal attention layer per block).
4. The quality of the WebVid-QA dataset may limit training effectiveness.

## Related Work & Insights
- **vs. Qwen2-VL**: Qwen2-VL relies entirely on the LLM for temporal understanding; STAVEQ2 demonstrates this is insufficient.
- **vs. InternVideo2**: Even joint spatiotemporal attention fails to resolve fine-grained temporal directionality — dedicated divided attention is necessary.
- **vs. ST-LLM**: ST-LLM delegates spatiotemporal modeling to the LLM; STAVEQ2 7B outperforms it by 15.2 points on MVBench.
- **vs. TG-Vid**: TG-Vid employs temporal gating with limited effectiveness and low efficiency; STAVEQ2 exceeds it by 13.7 points.
- **vs. FastVID**: FastVID focuses on efficiency (token pruning), while STAVEQ2 focuses on capability (enhanced temporal modeling) — the two are complementary.

**Further Insights:**
- **The Vision Encoder is the bottleneck**: Temporal understanding cannot be entirely delegated to the LLM — temporal information should be encoded before tokens are fed into the LLM.
- **Divided vs. Joint Space-Time Attention**: Further validates that TimeSformer-style divided attention is more controllable than joint attention in Video-LLM settings.
- **Complementarity with Eyes Wide Open**: Eyes Wide Open handles temporal KV cache management for streaming video, while STAVEQ2 performs encoder-level temporal modeling — the two approaches can be combined.

## Rating
- **Novelty**: ★★★☆☆ — Divided space-time attention is a known method; the contribution lies in its application to the encoder of Video-LLMs.
- **Technical Depth**: ★★★★☆ — Problem analysis is rigorous and ablation studies are thorough.
- **Experimental Thoroughness**: ★★★★★ — 4 models × multiple benchmarks × ablations × attention visualization.
- **Writing Quality**: ★★★★☆ — The motivation analysis section is particularly convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)
- [\[CVPR 2026\] Enhancing Video Vision Language Model with Hippocampal Sensing](../../CVPR2026/video_understanding/enhancing_video_vision_language_model_with_hippocampal_sensing.md)

</div>

<!-- RELATED:END -->
