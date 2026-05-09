---
title: >-
  [Paper Note] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization
description: >-
  [CVPR 2026][Multimodal VLM][Post-training quantization] This paper proposes Quant Experts (QE), a token-aware adaptive quantization error compensation framework based on Mixture of Experts (MoE). By partitioning important channels into token-independent and token-dependent groups, QE employs shared experts and routed experts to perform global and local quantization error reconstruction respectively, achieving significant accuracy recovery on VLMs ranging from 2B to 72B parameters.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Post-training quantization
  - mixture of experts
  - vision-language models
  - channel importance
  - low-rank reconstruction
date: 2026-05-08
content_hash: 49a4586892bde31e
---

# Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization

**Conference**: CVPR 2026
**arXiv**: [2602.24059](https://arxiv.org/abs/2602.24059)
**Code**: None
**Area**: Multimodal VLM / Model Compression
**Keywords**: Post-training quantization, mixture of experts, vision-language models, channel importance, low-rank reconstruction

## TL;DR

This paper proposes Quant Experts (QE), a token-aware adaptive quantization error compensation framework based on Mixture of Experts (MoE). By partitioning important channels into token-independent and token-dependent groups, QE employs shared experts and routed experts to perform global and local quantization error reconstruction respectively, achieving significant accuracy recovery on VLMs ranging from 2B to 72B parameters.

## Background & Motivation

1. **State of the Field**: Post-training quantization (PTQ) is the dominant approach for compressing large-scale vision-language models, reducing computation and memory overhead by mapping weights and activations to low-bit representations. Existing methods such as SmoothQuant, AWQ, and GPTQ primarily rely on statically identifying sensitive/outlier channels and compensating quantization errors globally.

2. **Limitations of Prior Work**: These methods assume channel importance is fixed, estimating a static set of channel scaling factors or low-rank reconstruction matrices from calibration data and applying them uniformly across all input tokens. However, in multimodal settings, the distribution of channel importance varies substantially across modalities and even across tokens within the same modality.

3. **Root Cause**: The authors empirically identify two key observations: (1) the positions of important channels shift significantly across modalities and tokens — even within the same modality, semantic and contextual differences cause activation distributions to vary, leading to dynamic migration of important channels; (2) the occurrence frequency of important channels follows a highly skewed distribution — only a small fraction of channels appear stably across most tokens (token-independent), while the majority of important channels activate only for specific tokens (token-dependent).

4. **Paper Goals**: To simultaneously compensate for globally stable channel errors and dynamically varying local channel errors during quantization.

5. **Starting Point**: Partition important channels into two groups based on occurrence frequency, and handle each with a different type of expert — shared experts for global errors and routed experts for token-dependent local errors.

6. **Core Idea**: Introduce the MoE paradigm into quantization error compensation, achieving token-level adaptive quantization error reconstruction through a combination of shared and routed experts.

## Method

### Overall Architecture

Given calibration data, QE first estimates the occurrence frequency distribution of important channels per layer, partitioning them into token-independent channels (high-frequency) and token-dependent channels (low-frequency). Two types of experts are then used to handle each group: a Shared Expert (SE) reconstructs the global quantization error of token-independent channels via a low-rank adapter; Routed Experts (REs) are trained per cluster by applying co-occurrence clustering on token-dependent channels, with one routed low-rank adapter per cluster. During inference, the shared expert provides fixed global error compensation, while a router dynamically selects the optimal routed expert for local error compensation based on the input token.

### Key Designs

1. **Channel Dependence Partitioning**

   - **Function**: Partition important channels into a globally stable group and a locally dynamic group.
   - **Mechanism**: For each token $x_t$ in the calibration data, the top-$k$ important channels are identified via $\mathcal{C}_t = \text{Top-}k(|x_t| \odot \mathbf{w})$, where $\mathbf{w}$ denotes the row-wise mean of the weight matrix. The occurrence frequency $f_c$ of each channel is then computed; channels are sorted in descending order of frequency, with the top $k$ designated as token-independent channels $\mathcal{C}_s$ and the subsequent $N_r \cdot k$ as token-dependent channels $\mathcal{C}_r$.
   - **Design Motivation**: Leveraging statistical occurrence frequency to distinguish global from local important channels avoids the computational overhead of per-token processing.

2. **Shared Expert (SE)**

   - **Function**: Compensate for global quantization errors caused by token-independent important channels.
   - **Mechanism**: Token-independent channels are exempted from direct quantization. A whitened SVD decomposition is applied to derive a low-rank adapter $(\mathbf{L}_{SA}, \mathbf{L}_{SB})$ that reconstructs the weights of these channels. Channel scaling is additionally employed to suppress activation outliers, reducing activation quantization error. The shared expert is applied uniformly to all tokens.
   - **Design Motivation**: Token-independent channels are important for the majority of tokens; a unified global compensation is sufficient to effectively recover this portion of error, consistent with prior works such as ASER.

3. **Routed Experts (REs) + Co-occurrence Clustering**

   - **Function**: Adaptively compensate for local quantization errors caused by token-dependent channels.
   - **Mechanism**: A co-occurrence matrix $\mathcal{O}$ is first constructed over token-dependent channels, then converted to a similarity matrix via Normalized Pointwise Mutual Information (NPMI), and spectral clustering is applied to partition the channels into $N_r$ clusters. Each cluster corresponds to one routed expert with a weighted SVD-based low-rank adapter. During inference, a lightweight router $\mathbf{R}$ estimates the residual error of each expert based on the absolute values of the input token and activates the expert with the smallest estimated error.
   - **Design Motivation**: Channels that co-occur frequently tend to appear together in semantically similar tokens. Clustering enables each expert to specialize in compensating errors for one cohesive group of channels, which is far more efficient than per-token customization. Using residual error as the routing criterion ensures the most suitable expert is selected for each token.

### Loss & Training

The shared and routed experts in QE are constructed directly via SVD without gradient-based training. An optional lightweight fine-tuning strategy is provided: only the routed experts and router parameters are fine-tuned while all other parameters are frozen, with layer-wise optimization rather than end-to-end training, to further improve quantization accuracy.

## Key Experimental Results

### Main Results

W4A6 quantization results on Qwen2VL-2B (average accuracy across 11 multimodal benchmarks):

| Method | Avg. Accuracy | vs RTN | vs LQER | vs MBQ |
|--------|--------------|--------|---------|--------|
| FP16 (full precision) | 62.97 | - | - | - |
| RTN | 53.62 | - | - | - |
| LQER | 55.92 | +2.30 | - | - |
| MBQ | 54.73 | +1.11 | - | - |
| **QE** | **58.74** | **+5.12** | **+2.82** | **+4.01** |

W4A6 quantization results on Qwen2VL-72B:

| Method | MMMU | OCRBench | ScienceQA | TextVQA | VizWiz |
|--------|------|----------|-----------|---------|--------|
| FP16 | 61.44 | 78.70 | 91.22 | 82.26 | 76.27 |
| MBQ | 52.67 | 69.70 | 86.32 | 76.08 | 67.99 |
| **QE** | **58.11** | **76.60** | **90.33** | **79.27** | **73.91** |

### Ablation Study

Component ablation under Qwen2VL-2B W4A6:

| Configuration | MMMU | ScienceQA |
|---------------|------|-----------|
| REs only | 34.56 | 68.72 |
| SE only | 35.22 | 69.61 |
| SE + random routing | 35.89 | 70.00 |
| SE + random clustering | 35.33 | 69.71 |
| **QE (SE + REs)** | **36.89** | **70.85** |

### Key Findings

- Both shared and routed experts are indispensable: removing either component leads to a notable performance drop, confirming their complementary roles.
- Co-occurrence clustering outperforms random clustering by approximately 1%, demonstrating that NPMI-based spectral clustering captures meaningful inter-channel associations.
- Adaptive routing outperforms random routing by approximately 0.8%, validating the necessity of input-conditioned expert selection.
- Performance improves consistently as the number of routed experts increases from 2 to 16 (average accuracy from 67.08 to 68.06), with diminishing returns.
- The improvement of QE scales with model size: on the 72B model under W4A6, QE yields a 5.09% gain, nearly recovering full-precision performance.

## Highlights & Insights

- **Applying MoE to quantization error compensation is a genuinely novel idea**: Prior work either performs global compensation or distinguishes modalities, whereas QE is the first to achieve adaptive compensation at the token granularity — a framework that is also readily extensible.
- **The co-occurrence clustering design is elegant**: Using NPMI and spectral clustering to discover co-occurrence patterns among token-dependent channels maps the intractable token space onto a finite set of experts, serving as an efficient approximation to the otherwise infeasible strategy of customizing a compensation scheme per token.
- **Extremely low overhead**: The additional FLOPs and parameter count from the low-rank adapters are negligible relative to the original linear layers, and the adapters are constructed solely during the calibration phase without end-to-end training.

## Limitations & Future Work

- The router design is relatively simple (using absolute mean to estimate residual error); more sophisticated routing strategies such as gating networks could be explored.
- Evaluation is limited to the Qwen2VL and InternVL2 families; assessment on a broader range of architectures (e.g., LLaVA, Phi-Vision) is lacking.
- The binary partition of token-independent and token-dependent channels is hard-coded; soft grouping along a continuous frequency spectrum warrants exploration.
- The effectiveness of QE under extreme low-bit settings (e.g., W2) remains unknown.

## Related Work & Insights

- **vs. LQER/ASER**: These methods use a single global low-rank adapter to compensate for all quantization errors. QE decomposes this into global and local components, yielding higher accuracy.
- **vs. MBQ**: MBQ adopts modality-aware processing (handling image and text tokens separately). QE further refines this to the token level, improving over MBQ by 4%+.
- **vs. SmoothQuant/AWQ**: These methods balance activation and weight distributions using fixed scaling factors. QE provides a more flexible adaptive compensation mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of applying MoE to quantization error compensation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models and settings from 2B to 72B, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived, with a complete observation→method logical chain.
- Value: ⭐⭐⭐⭐ Highly practical and applicable to large-scale VLM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_moe_dynamic_expert_skipping.md)

<!-- RELATED:END -->
