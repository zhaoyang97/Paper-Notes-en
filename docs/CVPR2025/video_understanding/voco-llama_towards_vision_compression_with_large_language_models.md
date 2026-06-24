---
title: >-
  [Paper Note] VoCo-LLaMA: Towards Vision Compression with Large Language Models
description: >-
  [CVPR 2025][Video Understanding][Vision Token Compression] This paper proposes VoCo-LLaMA, the first method that leverages the LLM's own capacity to compress vision tokens. By inserting VoCo tokens between vision and text tokens and modifying the attention mask to achieve attention distillation, it achieves a 576x compression rate with a single token while preserving 83.7% of the performance.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Vision Token Compression"
  - "Large Language Models"
  - "Attention Distillation"
  - "KV Cache Reuse"
date: 2026-05-08
content_hash: 83feb4d59a4d608f
---

# VoCo-LLaMA: Towards Vision Compression with Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2406.12275](https://arxiv.org/abs/2406.12275)  
**Code**: None  
**Area**: Video Understanding / Vision-Language Models  
**Keywords**: Vision Token Compression, Large Language Models, Attention Distillation, KV Cache Reuse, Video Understanding

## TL;DR

This paper proposes VoCo-LLaMA, the first method that leverages the LLM's own capacity to compress vision tokens. By inserting VoCo tokens between vision and text tokens and modifying the attention mask to achieve attention distillation, it achieves a 576x compression rate with a single token while preserving 83.7% of the performance.

## Background & Motivation

Vision-Language Models (VLMs) have achieved great success in multimodal tasks but suffer from bottlenecks involving **limited context windows** and **high computational costs for high-resolution/video inputs**. For instance, LLaVA-1.6 requires 2,880 vision tokens to process a 672×672 resolution image, occupying more than half of the context length. As the number of input images or video frames increases, the context window available for text is further compressed.

Existing vision compression methods (e.g., Q-Former, Re-sampler, average pooling) utilize **external modules** to compress vision tokens and then force the LLM to understand the compressed tokens. This "external compression, internal learning" paradigm harbors a fundamental issue: the way the LLM understands vision tokens is decoupled from the compression learning process, leading to severe visual information loss under high compression rates.

The core innovation of VoCo-LLaMA is: **letting the LLM perform vision compression itself**. By introducing Vision Compression (VoCo) tokens during the visual instruction tuning phase and modifying the attention mask so that text tokens can only access visual information indirectly through VoCo tokens, the LLM naturally learns to distill visual understanding into the transformer activations of the VoCo tokens. This ensures that compression and understanding share the same set of model parameters and paradigms.

## Method

### Overall Architecture

The input sequence is $(\mathcal{V}, VoCo, \mathcal{T}) = (V_0, ..., V_n, VoCo, T_0, ..., T_m)$. During training, a two-stage information flow is implemented via attention masking: VoCo tokens can attend to all vision tokens, but text tokens can only attend to VoCo tokens (and cannot see the original vision tokens). Inference consists of a two-step forward pass: the first step compresses vision tokens into a VoCo cache, and the second step completes the task using the VoCo cache + text tokens.

### Key Designs

1. **Attention Distillation Compression**:
    - Function: Allows the LLM itself to distill information from vision tokens into compact VoCo tokens.
    - Mechanism: Modifying the attention mask $M_{ij}$: the attention from text token $i \in \mathcal{T}$ to vision token $j \in \mathcal{V}$ is set to False (disallowing direct interaction), text-to-VoCo is set to True, and VoCo-to-vision remains True as in causal attention. The optimization objective is to minimize the KL divergence $E_{\mathcal{V},\mathcal{T}}[D_{KL}(p_{LM_o}(y|\mathcal{V},\mathcal{T}) \| p_{VoCo-LLaMA})]$, where $p_{VoCo-LLaMA} = p_{LM}(y|LM(\mathcal{V}, VoCo), \mathcal{T})$. In practice, this is extremely simple to implement as it only requires modifying the attention mask matrix.
    - Design Motivation: The compression paradigm of external compression modules (e.g., Q-Former) is inconsistent with the LLM's understanding paradigm, leading to information loss; allowing the LLM to compress by itself ensures that compression and understanding utilize the same model and paradigm.

2. **VoCo Cache Reuse**:
    - Function: Caches the compressed visual representations to support multi-task reuse for the same image.
    - Mechanism: Inference is divided into two phases: the first phase inputs [vision tokens, VoCo tokens] and compresses visual information into the KV Cache of the VoCo tokens; the second phase inputs [text tokens] and simply loads the VoCo Cache. The VoCo Cache of the same image can be reused across different tasks. Compared to caching the KV Cache of full vision tokens, storage overhead is reduced by 99.8%.
    - Design Motivation: Avoids re-processing a large number of vision tokens for every query, achieving true computational efficiency gains.

3. **Temporal Modeling Extension**:
    - Function: Extends VoCo-LLaMA from image compression to video understanding.
    - Mechanism: A video is split into multiple frame segments, with each segment independently compressed into a VoCo Cache $Cache_t = LM(\mathcal{V}_t, VoCo_t)$. The VoCo Caches of all frames are then concatenated into a temporal sequence $\mathcal{F} = \{Cache(VoCo_1), ..., Cache(VoCo_k)\}$. Further training on top of this enables the model to focus on temporal correlations, learning $p(y|\mathcal{F}, \mathcal{T})$. Since only a single VoCo token is used per frame, approximately 200 times more video frames can be processed given the same context length.
    - Design Motivation: Video frame sequences are extremely long, and processing all raw vision tokens directly would quickly exceed the LLM context length.

### Loss & Training

- **Training Loss**: KL divergence distillation + standard autoregressive language modeling (SFT).
- **Training Data**: LLaVA-filtered CC3M (alignment phase) + multi-task instruction data (VoCo compression phase) + WebVid + Video-ChatGPT QA (video phase).
- **Model Configuration**: CLIP-ViT-L vision encoder + Vicuna-7B LLM.
- **Training Tricks**: Gradient checkpointing to reduce VRAM usage.

## Key Experimental Results

### Main Results

Image benchmarks (compressing 576 vision tokens down to 1 VoCo token):

| Method | Token Count | GQA | MMB | MME | POPE | VQAv2 | Avg. Retention Rate |
|------|---------|-----|-----|-----|------|-------|----------|
| Upper Bound (No Compression) | 576 | 61.1 | 64.0 | 1487 | 85.0 | 77.7 | 100% |
| **VoCo-LLaMA** | **1** | **57.0** | **58.8** | **1323** | **81.4** | **72.3** | **83.7%** |
| Avg Pool + Linear | 1 | 52.9 | 55.5 | 1210 | 79.1 | 65.0 | 64.1% |
| Q-Former | 1 | 51.1 | — | — | — | — | <64% |

### Ablation Study

| Configuration | Description |
|------|------|
| Multiple VoCo tokens (1→4→16) | More tokens improve quality but decrease the compression rate. |
| No attention mask modification | Text directly attends to vision tokens, failing to force VoCo to learn compression. |
| External compression only (Q-Former) | Performance drops significantly, demonstrating the advantage of LLM's intrinsic compression. |

### Key Findings

- While achieving a 576x compression rate, the method retains 83.7% of the performance, significantly outperforming Q-Former (~64%) and average pooling (64.1%).
- Inference efficiency: KV Cache storage is reduced by 99.8%, FLOPs drop by 94.8%, and inference latency decreases by 69.6%.
- For video understanding, VoCo compression enables processing of ~200x more video frames, outperforming prior methods on the Video-ChatGPT benchmark.
- VoCo Cache reuse eliminates the necessity of redundant encoding for multi-task queries on the same image.

## Highlights & Insights

- **The concept of 'letting the LLM perform compression itself' is highly inspiring**: since the LLM has already learned to comprehend vision tokens, it is naturally best suited to determine which information is core. This intrinsic compression is more natural and suffers from less information loss compared to forced external compression modules.
- **Extremely simple implementation**: The core of the method lies in modifying the attention mask matrix—simply setting the text-to-vision attention to False. It requires no new modules, additional parameters, or architectural changes.

## Limitations & Future Work

- Currently only validated on 7B-scale models; the effectiveness on larger models (e.g., 70B+) remains unverified.
- Extreme compression (down to 1 token) still incurs an approximate 16% performance drop, which significantly affects precise visual reasoning tasks.
- Temporal modeling for videos relies on naive cache concatenation, lacking an explicit temporal attention mechanism.
- Dynamic compression rates based on image complexity—to adaptively select the number of VoCo tokens—have not yet been explored.

## Related Work & Insights

- **vs Q-Former (BLIP-2)**: Q-Former utilizes external learnable queries to compress vision tokens, which is decoupled from the LLM's understanding paradigm. VoCo-LLaMA relies on the LLM itself to compress, exceeding the retention rate of Q-Former by around 20 percentage points.
- **vs Chat-UniVi**: Chat-UniVi uses average pooling and a linear layer for compression. While simple, this induces heavy information loss, whereas VoCo-LLaMA exploits the full attention mechanism of the LLM for compression.
- **vs Text Compression Methods (AutoCompressor, ICAE)**: These share similar ideas but are applied to the textual domain, whereas VoCo-LLaMA is the first work to introduce this paradigm to the visual modality.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first to let the LLM compress vision tokens itself, offering a simple yet highly effective approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple benchmarks, comprehensive ablation studies, efficiency analysis, and extensions to video.
- Writing Quality: ⭐⭐⭐⭐ Clear method descriptions and concise mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ Maintaining 83.7% performance at a 576x compression rate holds significant value for enhancing VLM efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] LLAVIDAL: A Large Language Vision Model for Daily Activities of Living](llavidal_a_large_language_vision_model_for_daily_activities_of_living.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

</div>

<!-- RELATED:END -->
