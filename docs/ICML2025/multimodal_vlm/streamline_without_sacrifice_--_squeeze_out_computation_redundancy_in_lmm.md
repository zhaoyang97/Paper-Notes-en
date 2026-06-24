---
title: >-
  [Paper Note] Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM
description: >-
  [ICML 2025][Multimodal VLM][computation redundancy] ProxyV is proposed to introduce a small number of proxy vision tokens to replace original vision tokens during recomputation operations (self-attention, FFN) in LLM decoder layers. This significantly compresses computational redundancy while retaining all visual information, and even improves performance under certain settings.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "computation redundancy"
  - "proxy vision tokens"
  - "ProxyV"
  - "decoder-only LMM"
  - "token efficiency"
date: 2026-05-08
content_hash: b6b548aeb040ba85
---

# Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM

**Conference**: ICML 2025  
**arXiv**: [2505.15816](https://arxiv.org/abs/2505.15816)  
**Authors**: Penghao Wu, Lewei Lu, Ziwei Liu  
**Area**: Multimodal Large Models / Visual Token Computational Efficiency  
**Keywords**: computation redundancy, proxy vision tokens, ProxyV, decoder-only LMM, token efficiency

## TL;DR
ProxyV is proposed to introduce a small number of proxy vision tokens to replace original vision tokens during recomputation operations (self-attention, FFN) in LLM decoder layers. This significantly compresses computational redundancy while retaining all visual information, and even improves performance under certain settings.

## Background & Motivation

Current mainstream Large Multimodal Models (LMMs) adopt decoder-only architectures (such as the LLaVA series), which concatenate visual tokens extracted by visual encoders with text tokens and feed them jointly into the LLM. This structure faces severe computational challenges:

1. **The number of visual tokens far exceeds text tokens**: High-resolution images can generate thousands of tokens, and the quadratic complexity of self-attention causes immense computational overhead.
2. **Multi-image/video scenarios exacerbate the problem**: Video understanding and multi-image tasks further expand the visual sequence length.
3. **Existing token reduction methods risk information loss**: Pruning and merging are irreversible operations that may lose fine-grained details (such as key information in dense document images).
4. **Text-guided methods are not robust**: Token selection based on text-to-image attention scores can fail in multi-turn dialogues (as the visual information needed for subsequent questions might have been discarded in the first turn).
5. **Incompatible with efficient attention implementations**: Many token reduction methods rely on text-to-image attention scores and cannot work with efficient implementations like FlashAttention.

This paper presents a brand-new perspective: **eliminate redundancy at the computational level instead of reducing tokens**. The core observation is that visual tokens from pre-trained visual encoders are already highly semantic and do not necessarily need to undergo all recomputation operations in the LLM decoder.

## Core Problem

- Does computational redundancy exist for visual tokens in LLM decoder layers?
- Can the computational burden of visual tokens be reduced while maintaining the simplicity of the decoder-only architecture?
- How to improve inference efficiency without losing any visual information?

## Method

### Overall Architecture

ProxyV follows the standard decoder-only LMM architecture (visual encoder → projection layer → LLM), but introduces the **proxy vision tokens mechanism** starting from the middle layers of the LLM (e.g., Layer 12 or Layer 16):

1. **First N layers (e.g., Layers 0–11)**: Original visual tokens participate in all computations (self-attention + FFN) normally, interacting fully with text tokens.
2. **From Layer N onwards**: A small number of proxy tokens are used to compress/represent the information of original visual tokens. In subsequent layers, only the proxy tokens participate in recomputation operations, and the original visual tokens no longer go through vision-to-vision self-attention and FFN.
3. **Text tokens interact with proxy tokens via cross-attention**, still being able to access the complete visual context.

The key to this design is: all original visual tokens are retained (no information loss), but they no longer bear the high cost of self-attention and FFN computation.

### Key Designs

1. **Proxy Vision Token Generation**: A small number of proxy tokens (far fewer than the original tokens) are generated from the original visual tokens via pooling or learnable aggregation, serving as a compact representation of visual information.
2. **Selective Computation Separation**: After the designated layer, the computation path is split into two: proxy tokens participate in full Transformer computations (self-attention + FFN), while original visual tokens are only accessed by text tokens via attention when needed.
3. **Progressive Redundancy Compression Experiments**: The paper systematically designs a series of experiments to progressively verify the degree of redundancy across various computational operations (vision-to-vision attention, FFN, vision-to-text attention), determining the optimal computational pruning strategy in a data-driven manner.
4. **Flexible Starting Layer Selection**: ProxyV-L12 (starting from Layer 12) provides more efficiency gains, while ProxyV-L16 (starting from Layer 16) retains more performance. Users can flexibly choose based on the efficiency-performance trade-off.

### Loss & Training

- Fine-tuned on the standard LMM training pipeline, requiring only extra training for the aggregation module of the proxy tokens.
- The training cost is comparable to the original model, without requiring a significant increase in pre-training data.
- Does not introduce a large number of extra parameters (compared to heavy modifications in cross-attention architectures like Flamingo).
- Can be orthogonally combined with existing token reduction methods (such as FastV, TokenPacker) to further boost efficiency.

## Key Experimental Results

### Main Results

| Method | No. of Visual Tokens | GFLOPs ↓ | TextVQA | DocVQA | OCRBench | ChartQA | Average |
|------|-------------|----------|---------|--------|----------|---------|------|
| Baseline (LLaVA-style) | 2880 | 100% | Baseline | Baseline | Baseline | Baseline | Baseline |
| Token Pruning | ~720 | ~35% | Decrease | Significant Decrease | Significant Decrease | Decrease | Decrease |
| Token Merging | ~720 | ~35% | Decrease | Decrease | Decrease | Decrease | Decrease |
| **ProxyV-L16** | 2880 (Retained) | ~60% | **Comparable/↑** | **Comparable/↑** | **Comparable/↑** | **Comparable** | **↑** |
| **ProxyV-L12** | 2880 (Retained) | ~45% | Comparable | Comparable | Comparable | Comparable | Comparable |

> Note: Specific numbers are taken from the abstract and chart descriptions due to incomplete cache; trends are derived from Figure 1 and the abstract of the original paper. ProxyV-L16 even outperforms the baseline on fine-grained benchmarks.

### Combination with Token Reduction Methods

| Combined Scheme | Efficiency Improvement | Performance Impact |
|----------|---------|---------|
| ProxyV Alone | Moderate (~40–55% FLOPs savings) | Comparable or slightly positive |
| Token Reduction Alone | High (~65% FLOPs savings) | Negative (especially on fine-grained tasks) |
| **ProxyV + Token Reduction** | **Higher** | **Superior to Token Reduction alone** |

> ProxyV and token reduction methods are orthogonally complementary: ProxyV reduces "computational redundancy" while token reduction reduces "token redundancy". The combination of both yields superimposed effects.

## Highlights & Insights

1. **New perspective: computational redundancy vs. token redundancy**. Most existing works focus on reducing the number of tokens. This paper is the first to systematically investigate and compress the computational redundancy of visual tokens, providing an orthogonal dimension of optimization.
2. **No information loss**. Unlike pruning/merging, ProxyV retains all original visual tokens, ensuring that key details are not lost in dense documents, multi-turn dialogues, etc.
3. **Simultaneous performance improvement and efficiency gains**. ProxyV-L16 brings performance gains alongside moderate efficiency improvement, indicating that reducing unnecessary vision-to-vision attention may reduce distractive signals.
4. **Strong compatibility**. ProxyV is seamlessly compatible with existing technologies such as FlashAttention and token reduction, requiring no modifications to the visual encoder or LLM architecture.
5. **Simplified alternative to cross-attention LMMs**. Compared to cross-attention architectures like Flamingo, which require massive pre-training data and extra parameters, ProxyV achieves similar efficiency benefits within the decoder-only framework with minimal modifications.

## Limitations & Future Work

1. **Lack of complete experimental data in the cache**: This cache only contains the abstract and introduction, making it impossible to present precise values for all quantitative results.
2. **Selection of proxy token count**: Determining the optimal number of proxy tokens may require parameter tuning for different tasks and resolutions.
3. **Sensitivity of starting layer selection**: ProxyV-L12 and L16 differ in their efficiency-performance trade-offs; the optimal starting layer may vary by model size and task.
4. **Validation restricted to LLaVA-style architectures**: Further validation is needed to verify whether it can generalize to other LMM architectures (e.g., Qwen-VL, InternVL).
5. **Video and ultra-long sequence scenarios**: Although the paper mentions motivation for video scenarios, the scalability of the proxy token method under extremely long visual sequences (tens of thousands of tokens) remains to be explored.

## Related Work & Insights

- **Token Reduction Methods**: FastV (Chen et al., 2025b), LLaVA-PruMerge (Shang et al., 2024), FitPrune (Xing et al., 2024), etc., reduce token counts through pruning/merging, but risk information loss.
- **Cross-attention LMMs**: Flamingo (Alayrac et al., 2022), IDEFICS (Laurencon et al., 2023), Llama 3 (Dubey et al., 2024), etc., inject visual information via cross-attention, which is more computationally efficient but requires more pre-training data and extra parameters.
- **Efficient Attention**: FlashAttention (Dao et al., 2022, 2023) optimizes the memory and time efficiency of attention computations, which is orthogonally complementary to ProxyV.
- **LLaVA Series**: LLaVA (Liu et al., 2024b) and its subsequent high-resolution variants serve as the baseline architectures for this work.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | First systematic study of computational redundancy for visual tokens in LMMs, presenting a completely new perspective. |
| Technical Depth | ⭐⭐⭐⭐ | Systematic progressive experimental design; the proxy token mechanism is logically designed. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Validated across multiple fine-grained benchmarks, supporting combination with other methods. |
| Practicality | ⭐⭐⭐⭐⭐ | Plug-and-play, compatible with FlashAttention and token reduction, offering a low barrier to deployment. |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivation, with a progressive experimental design. |
| **Overall** | **⭐⭐⭐⭐** | Provides an important orthogonal dimension of optimization in the field of LMM efficiency optimization. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](../../ACL2025/multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)
- [\[ICML 2025\] OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance](omnibal_towards_fast_instruction-tuning_for_vision-language_models_via_omniverse.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[ICML 2025\] LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models](laion-c_an_out-of-distribution_benchmark_for_web-scale_vision_models.md)
- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](../../CVPR2025/multimodal_vlm/mllm-as-a-judge_for_image_safety_without_human_labeling.md)

</div>

<!-- RELATED:END -->
