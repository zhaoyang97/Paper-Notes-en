---
title: >-
  [Paper Note] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
description: >-
  [CVPR2026][Multimodal VLM][MoE acceleration] This paper proposes MoDES, the first training-free expert skipping framework for MoE multimodal large language models. By leveraging Globally Modulated Local Gating (GMLG) and Dual-Modal Thresholding (DMT), MoDES adaptively skips redundant experts, retaining over 97% of original performance while skipping 88% of experts, and achieving 2.16× prefill speedup.
tags:
  - CVPR2026
  - Multimodal VLM
  - MoE acceleration
  - expert skipping
  - multimodal large language models
  - training-free
  - inference efficiency
date: 2026-05-08
content_hash: ddaaab8ee1fbc9f2
---

# MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

**Conference**: CVPR2026  
**arXiv**: [2511.15690](https://arxiv.org/abs/2511.15690)  
**Code**: [ModelTC/MoDES](https://github.com/ModelTC/MoDES)  
**Area**: Multimodal VLM  
**Keywords**: MoE acceleration, expert skipping, multimodal large language models, training-free, inference efficiency

## TL;DR

This paper proposes MoDES, the first training-free expert skipping framework for MoE multimodal large language models. By leveraging Globally Modulated Local Gating (GMLG) and Dual-Modal Thresholding (DMT), MoDES adaptively skips redundant experts, retaining over 97% of original performance while skipping 88% of experts, and achieving 2.16× prefill speedup.

## Background & Motivation

**Inference bottleneck in MoE MLLMs**: MoE multimodal large language models (e.g., Qwen3-VL-MoE-30B) reduce computation via sparse activation, yet each token still interacts with multiple activated experts, resulting in non-trivial inference overhead.

**Failure of existing expert skipping methods**: Methods such as NAEE, MC-MoE, and DiEP are designed for unimodal LLMs; when directly applied to MLLMs, accuracy degrades by more than 10% at an 83% skipping rate.

**Uneven layer-wise contribution (Insight i)**: Shallow-layer experts contribute far more to the final output than deep-layer ones—errors introduced in shallow layers are amplified by subsequent layers. However, existing methods make skipping decisions solely based on intra-layer routing probabilities, ignoring global layer-level importance.

**Modality-specific behavioral differences (Insight ii)**: Text tokens and visual tokens exhibit significantly different update magnitudes in FFN layers. Visual tokens are more orthogonal to FFN weights (angles approaching 90°), and are therefore less affected by FFN transformations, exhibiting higher redundancy.

**Lack of modality-aware skipping strategies**: Prior work applies a uniform threshold across all modalities without accounting for the distinct characteristics of text and visual tokens, leading to suboptimal skipping decisions.

**High cost of threshold search**: Brute-force search for dual-modal thresholds requires $\mathcal{O}(ND^2)$ time complexity, taking days to complete for 20–30B parameter models.

## Method

### Overall Architecture

MoDES is a training-free inference acceleration framework consisting of two core modules: Globally Modulated Local Gating (GMLG), which computes an importance score for each expert, and Dual-Modal Thresholding (DMT), which makes adaptive skipping decisions based on token modality.

### Globally Modulated Local Gating (GMLG)

To address uneven layer-wise contribution, GMLG combines global layer-level importance with local routing probabilities:

$$s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$$

- $\pi_i^{(l)}$: local routing probability (softmax-normalized) of the $i$-th expert in layer $l$
- $\alpha^{(l)}$: global modulation factor obtained via offline calibration, measuring the impact on the final output of skipping all experts in layer $l$

$\alpha^{(l)}$ is computed as the mean KL divergence between the output distributions of the original model and the model with all experts in layer $l$ skipped, evaluated on a calibration set $\mathcal{C}$:

$$\alpha^{(l)} = \frac{1}{N}\sum_{j=1}^{N}\mathcal{D}_{\text{KL}}(\text{prob}_j \| \text{prob}_j^{(l)})$$

The calibration phase uses 1,024 samples from the GQA dataset and is computed offline, incurring no additional overhead during inference.

### Dual-Modal Thresholding (DMT)

To handle modality-specific behavioral differences, DMT assigns separate skipping thresholds $\tau_t$ and $\tau_v$ for text and visual tokens respectively:

$$\{\text{Expert}_i^{(l)} \mid s_i^{(l)} < \tau_t \cdot \mathbb{I}_t + \tau_v \cdot \mathbb{I}_v\}$$

Experts whose importance scores fall below the threshold corresponding to their token modality are skipped. Visual tokens, being more redundant, are typically assigned a higher skipping threshold.

### Frontier Search Algorithm

To efficiently find the optimal $(\tau_t, \tau_v)$, the problem is formulated as minimizing KL divergence subject to a target skipping rate $\rho$. Exploiting the monotonicity of $f$ and $g$ with respect to the thresholds, a dual-pointer strategy identifies the optimal solution on the Pareto frontier in $\mathcal{O}(ND)$ time—approximately 45× faster than brute-force search at $\mathcal{O}(ND^2)$—reducing search time from days to within a few hours.

## Experiments

### Main Results: Comparison on 13 Benchmarks with Kimi-VL-A3B-Instruct

| Method | Skip Rate | ChartQA | MME | MMBench | LVB | VMMMU | Avg.(%) |
|--------|-----------|---------|-----|---------|-----|-------|---------|
| Default k=6 | 0% | 89.48 | 2207 | 83.16 | 63.13 | 49.33 | 100.00 |
| DiEP | 83% | 78.31 | 2071 | 76.28 | 52.41 | 43.81 | 87.58 |
| MC-MoE | 83% | 80.25 | 2063 | 73.42 | 54.39 | 44.02 | 88.32 |
| **MoDES** | **83%** | **84.20** | **2162** | **81.44** | **62.60** | **47.11** | **96.25** |

### Cross-Model Generalization: Qwen3-VL-MoE-30B at 88% Skip Rate

| Method | ChartQA | MME | MMBench | VMMMU | Avg.(%) |
|--------|---------|-----|---------|-------|---------|
| MC-MoE | 71.43 | 2168 | 75.42 | 37.41 | 86.66 |
| DiEP | 70.51 | 2074 | 73.21 | 34.79 | 85.30 |
| **MoDES** | **78.84** | **2403** | **85.57** | **46.56** | **97.33** |

At an aggressive 88% skipping rate, MoDES outperforms the strongest baseline MC-MoE by 10.67 percentage points.

### Ablation Study

| Configuration | ChartQA | MME | MMBench | LVB | VMMMU |
|---------------|---------|-----|---------|-----|-------|
| Single-threshold baseline | 76.74 | 1956 | 65.48 | 54.67 | 40.33 |
| +GMLG | 79.28 | 2107 | 75.19 | 60.02 | 43.87 |
| +DMT | 82.94 | 2081 | 79.42 | 61.16 | 45.08 |
| +GMLG+DMT (full) | **84.20** | **2162** | **81.44** | **62.60** | **47.11** |

(83% skip rate, Kimi-VL-A3B-Instruct.) Both GMLG and DMT yield significant and independent contributions, with larger gains at higher skipping rates.

### Key Findings

- **Inference speedup**: MoDES achieves 2.16× prefill and 1.26× decoding speedup on Qwen3-VL-MoE-30B.
- **Compatibility with quantization**: MoDES combined with 2.5-bit quantization retains 94.43% of original performance on Qwen3, compared to 89.58% for MC-MoE.
- **Skipping pattern visualization**: Deep layers exhibit far higher skipping rates than shallow layers; visual tokens are skipped at a much higher rate than text tokens, validating both core insights.
- **Calibration data robustness**: Replacing the calibration set with COCO or VMMMU yields negligible performance differences.
- **Search efficiency**: Frontier search achieves ~45× speedup over brute-force search; total overhead (calibration + search) for 20–30B models is within 20 minutes to 4 hours.

## Highlights & Insights

- First work to systematically analyze uneven layer-wise contribution and modality-specific behavioral differences in MoE MLLMs, with both insights supported by thorough empirical evidence.
- GMLG elegantly combines offline global calibration with online local routing, incurring no additional overhead during inference.
- DMT replaces a uniform threshold with modality-aware dual thresholds, with a clear and logical design motivation.
- The frontier search algorithm exploits monotonicity to reduce complexity from $\mathcal{O}(ND^2)$ to $\mathcal{O}(ND)$, offering strong practical utility.
- Experiments span 3 model families × 13 benchmarks, achieving less than 3% accuracy loss when skipping 88% of experts.

## Limitations & Future Work

- Only text and visual modalities are addressed; the framework has not been extended to additional modalities such as audio.
- $\alpha^{(l)}$ operates at the layer level and does not differentiate global importance among individual experts within the same layer.
- Evaluation is limited to image/video understanding tasks; generative tasks (e.g., image captioning quality) are not thoroughly assessed.
- Decoding-stage speedup is modest (~1.2×), as decoding is memory-bound and processes only text tokens.
- The frontier search relies on a monotonicity assumption that, while reasonable in practice, lacks rigorous theoretical guarantees.

## Related Work & Insights

- **NAEE** [Lu et al.]: Skips minor experts based on routing probability ratios, relying solely on intra-layer information.
- **MC-MoE** [Huang et al., 2024]: Extends NAEE with attention-aware expert protection and mixed-precision quantization.
- **DiEP** [Bai et al., 2025]: Differentiable expert pruning that jointly considers routing probabilities and expert similarity for skipping decisions.
- All aforementioned methods are designed for unimodal LLMs and transfer poorly to MLLMs. MoDES is the first to propose a global and modality-aware skipping strategy tailored to multimodal settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — Both insights are convincing, and the GMLG+DMT combination is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 model families × 13 benchmarks × multiple skipping rates, with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with coherent motivation→method→experiment flow.
- Value: ⭐⭐⭐⭐ — Directly applicable to MoE MLLM deployment, with a concise and efficient design.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)

<!-- RELATED:END -->
