---
title: >-
  [Paper Note] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE
description: >-
  [ICLR 2026][Model Compression][MoE pruning] This paper proposes MoNE (Mixture-of-Novices-and-Experts), which identifies redundant experts via joint evaluation of access frequency and output variance, and replaces them with their output mean vectors ("novices"). MoNE achieves more effective and robust compression than existing pruning methods across 5 MoE models, with an average accuracy drop of only 0.14 at a 25% pruning ratio.
tags:
  - ICLR 2026
  - Model Compression
  - MoE pruning
  - expert redundancy
  - novice replacement
  - structured compression
  - access frequency
date: 2026-05-08
content_hash: adfdc2701905a422
---

# MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE

**Conference**: ICLR 2026  
**arXiv**: [2507.00390](https://arxiv.org/abs/2507.00390)  
**Code**: [GitHub](https://github.com/zxgx/mode-pd)  
**Area**: Model Compression  
**Keywords**: MoE pruning, expert redundancy, novice replacement, structured compression, access frequency

## TL;DR
This paper proposes MoNE (Mixture-of-Novices-and-Experts), which identifies redundant experts via joint evaluation of access frequency and output variance, and replaces them with their output mean vectors ("novices"). MoNE achieves more effective and robust compression than existing pruning methods across 5 MoE models, with an average accuracy drop of only 0.14 at a 25% pruning ratio.

## Background & Motivation
MoE architectures scale model capacity through sparse activation, but deployment requires retaining all experts in memory—e.g., all 64 experts (including unactivated ones) on GPU—incurring substantial overhead. Structured pruning can directly reduce expert count to lower memory costs.

Existing methods exhibit three-dimensional instability:

**Cross-architecture instability**: Layer pruning (Angular) and channel pruning (FLAP) do not account for the sparse computation characteristics of MoE.

**Cross-calibration-data instability**: Different calibration data sources lead to large performance fluctuations.

**Cross-sample-size instability**: Performance varies significantly across 100/500/1000 calibration samples.

The core problem is that existing expert pruning relies primarily on access frequency, which does not fully characterize redundancy. An expert with low frequency but highly varying outputs may carry critical discriminative information; conversely, an expert with moderate frequency but highly stable outputs can be replaced by a constant.

Core Idea: Redundancy = low frequency × low variance. Redundant experts are replaced by their output mean ("novice") rather than simply removed, minimizing output discrepancy.

## Method

### Overall Architecture
MoNE proceeds in three steps: ① evaluate each expert's access frequency and output variance on a calibration set; ② combine the two metrics into a redundancy score and identify the most redundant expert subset; ③ replace redundant experts with novices (output mean vectors). The router remains unchanged; replaced experts can still be selected but produce a constant output.

### Key Designs
1. **Variance Redundancy $\phi^{var}$**:

    - Function: Measures the stability of an expert's output over the calibration set.
    - Mechanism: Computes the L2 norm of the unbiased variance estimate of expert $E_i$'s outputs when selected: $\phi_i^{var} = \left\|\sqrt{\frac{\sum(E_i(\mathbf{x}) - \overline{E_i})^2 \cdot \mathbb{I}(E_i \in \mathcal{S}_k)}{\sum\mathbb{I} - 1}}\right\|_2$
    - Design Motivation: High-variance experts provide more discriminative information and cannot be replaced by a constant; low-variance experts produce stable outputs and can be approximated by their mean with minimal error.

2. **Frequency Redundancy $\phi^{freq}$**:

    - Function: Measures the router's reliance on an expert.
    - Mechanism: $\phi_i^{freq} = \frac{\sum G_i(\mathbf{x}) \cdot \mathbb{I}(E_i \in \mathcal{S}_k)}{\sum \mathbb{I}(E_i \in \mathcal{S}_k)}$, i.e., the average routing score when the expert is selected.
    - Design Motivation: Low-frequency experts contribute little to model output, but frequency alone is insufficient—some high-frequency experts also produce stable outputs.

3. **Novice Replacement**:

    - Function: Replaces pruned experts with a constant vector.
    - Mechanism: Redundancy score $\phi = \phi^{var} \cdot \phi^{freq}$; low-scoring experts are marked for pruning. The novice $N_i = \overline{E_i}$ (mean output over the calibration set) is the closed-form optimal solution minimizing L2 discrepancy.
    - Design Motivation: The novice is an unbiased estimate of the output mean; $N_i$ requires no computation with input tokens and is stored as a single $d$-dimensional vector.

### Loss & Training
MoNE is a training-free post-processing method. A single forward pass over the calibration set suffices to collect the necessary statistics for pruning.

## Key Experimental Results

### Main Results (25% Pruning, OLMoE-7B, 100-sample Zyda2)

| Method | Arc-c | Arc-e | BoolQ | COPA | MMLU | OBQA | PIQA | RTE | WinoG | Avg |
|--------|-------|-------|-------|------|------|------|------|-----|-------|-----|
| Original | 49.23 | 76.89 | 70.09 | 85.0 | 53.54 | 44.4 | 79.76 | 71.84 | 68.90 | 66.63 |
| Angular (layer pruning) | 32.76 | 61.91 | 61.71 | 74.0 | 23.13 | 37.6 | 71.65 | 53.07 | 55.09 | 52.33 |
| FLAP (channel pruning) | 40.53 | 67.55 | 62.69 | 78.0 | 41.16 | 37.8 | 74.81 | 61.37 | 60.93 | 58.32 |
| MC-SMoE (expert merging) | 35.67 | 54.92 | 63.49 | 73.0 | 29.04 | 30.6 | 67.19 | 55.23 | 65.75 | 52.77 |
| RS (frequency pruning) | 25.85 | 43.01 | 59.08 | 74.0 | 29.63 | 36.2 | 66.16 | 56.68 | 59.98 | 50.07 |
| **MoNE** | **42.32** | **64.81** | **67.19** | **85.0** | **40.13** | **40.8** | **78.07** | **64.62** | **66.46** | **61.04** |

### Ablation Study

| Dimension | Prior Methods | MoNE | Notes |
|-----------|--------------|------|-------|
| Cross-architecture (5 models) | Large variance | Consistently superior | OLMoE/Moonlight/DS-V2/Qwen2-57B/Qwen3-30B |
| Zyda2 vs. C4 | Significant difference | Small difference | Calibration data robustness |
| 100/500/1000 samples | Large variance | Stable | Sample size robustness |
| Qwen2-57B-A14B @ 25% | Large accuracy drop | Only 0.14 drop | Advantage more pronounced on larger models |

### Key Findings
- MoNE achieves only a 0.14 accuracy drop on Qwen2-57B at 25% pruning, outperforming the best baseline by up to 2.72 points.
- Frequency and variance are complementary: three consistent expert categories emerge across tasks—high-frequency & high-variance (blue), high-variance only (red), and high-frequency only (green).
- RS (frequency-only) performs worst, validating the insufficiency of a single metric.
- Novice (constant vector) replacement outperforms complete deletion: it preserves a knowledge estimate and reduces activated parameters for a subset of tokens.

## Highlights & Insights
- The "novice" replacement concept is elegant and effective—substituting an entire MLP with a single vector at near-zero computational and memory cost.
- The frequency × variance redundancy metric is well-motivated and demonstrates strong cross-task consistency.
- The training-free, weight-update-free design makes MoNE highly practical for real-world deployment.
- The robustness analysis across three dimensions is a standout contribution that addresses a gap in existing evaluation practice.

## Limitations & Future Work
- Evaluation is limited to zero-shot settings; complex tasks (code, math, reasoning) may require additional fine-tuning.
- Novices are fixed constants and do not vary with input—potentially introducing larger error for highly data-dependent experts.
- Performance degradation remains notable at high pruning ratios (50%).
- Adaptive novices with per-layer learning (e.g., low-rank novices) remain unexplored.

## Related Work & Insights
- **vs. MC-SMoE**: Novice replacement is simpler and more effective than expert merging.
- **vs. RS**: Adding the variance dimension yields consistent improvements; frequency alone is insufficient.
- **vs. Angular/FLAP**: MoE-specific pruning is more suitable than general structured pruning.

## Rating
- Novelty: ⭐⭐⭐⭐ The "novice" replacement concept is original, and the dual-metric redundancy measure is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive robustness analysis across 5 models × 2 calibration sources × 3 sample sizes.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, though notation is somewhat heavy.
- Value: ⭐⭐⭐⭐⭐ Practically valuable for MoE deployment; the method is simple and efficient.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](steering_moe_llms_via_expert_deactivation.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)

<!-- RELATED:END -->
