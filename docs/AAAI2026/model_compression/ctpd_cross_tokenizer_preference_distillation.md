---
title: >-
  [Paper Note] CTPD: Cross Tokenizer Preference Distillation
description: >-
  [AAAI 2026][Model Compression][Knowledge Distillation] This paper proposes Cross-Tokenizer Preference Distillation (CTPD), the first unified framework supporting preference distillation across heterogeneous tokenizers. T…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Preference Alignment"
  - "Cross-Tokenizer"
  - "DPO"
  - "Language Models"
date: 2026-05-08
content_hash: 6612c31b49dfe4a7
---

# CTPD: Cross Tokenizer Preference Distillation

**Conference**: AAAI 2026
**arXiv**: [2601.11865](https://arxiv.org/abs/2601.11865)  
**Code**: [Available](https://github.com/dinhtruongng/CTPD)  
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Preference Alignment, Cross-Tokenizer, DPO, Language Models

## TL;DR

This paper proposes Cross-Tokenizer Preference Distillation (CTPD), the first unified framework supporting preference distillation across heterogeneous tokenizers. Through three key innovations—Aligned Span Projection, cross-tokenizer importance weighting, and Teacher-Anchored Reference—CTPD achieves substantial improvements over existing methods on multiple benchmarks.

## Background & Motivation

Aligning LLMs with human preferences has become a central research topic. Methods such as DPO have demonstrated strong results on large models; however, smaller models struggle to align directly due to limited representational capacity. Knowledge distillation (KD) offers a promising alternative—performing the costly alignment on a large teacher model and then transferring the aligned behavior to a smaller student.

White-box distillation, however, faces the **cross-tokenizer problem**: teacher and student models typically employ different tokenizers, making their logit distributions incompatible and precluding direct token-level knowledge transfer. Existing cross-tokenizer distillation works (ULD, DSKD, Multi-Level OT) are designed for pretraining or fine-tuning scenarios and **are not applicable to preference alignment**. Only one prior work has studied preference distillation, but it is restricted to the simplified setting where teacher and student share the same tokenizer.

The core insight of CTPD is that, despite syntactic differences between teacher and student tokens, both ultimately encode **the same natural language substring**. Character-level alignment can therefore establish precise correspondences across heterogeneous tokenizers.

## Method

### Overall Architecture

The CTPD pipeline consists of three stages:

1. **SFT Stage**: Teacher and student are each fine-tuned on instruction-tuning data.
2. **Teacher Contrastive Model Training**: DPO is used to train a positive model $\pi^+$ and a negative model $\pi^-$ from the teacher, which are used for subsequent importance weight estimation.
3. **CTPD Preference Distillation**: The student is trained on preference data with precomputed aligned span weights, using the SFT teacher as the reference model.

### Key Designs

**1. Aligned Span Projection**

The core mechanism partitions teacher and student token sequences into a series of **aligned spans**—each span pair corresponds to an identical character interval in the original string.

Formally, a teacher token subsequence $\{t_i, ..., t_j\}$ and a student token subsequence $\{s_k, ..., s_l\}$ form an aligned span if and only if their decoded characters cover exactly the same start and end positions in the original string $S$.

This mechanism allows any signal from the teacher (log-probabilities, importance weights, etc.) to be aggregated within a span and projected onto the corresponding student tokens, **without introducing any learnable parameters**.

**2. Cross-Tokenizer Importance Weighting**

TIS-DPO is extended to the cross-tokenizer setting. The token-level reward is decomposed into span-level rewards:

$$r(p^t | x, p^{<t}) = \sum_i r(y_{t_i} | x, y_{t_{<i}})$$

Based on Theorem 1 (span-level label noise), it is proven that significant fluctuations in span-level rewards indicate label noise in the preference data. Importance sampling is used to convert the expectation under the ideal noise-free distribution $\mathcal{D}^*$ into a weighted expectation under the true distribution $\mathcal{D}$:

$$w_t = k \cdot \exp\left(\mu \cdot \text{clamp}\left(\log\frac{\pi^+(p^t | x, p^{<t})}{\pi^-(p^t | x, p^{<t})}, L, U\right)\right)$$

The DPO positive/negative models of the teacher are used to estimate these weights, distilling the teacher's fine-grained reward judgment into the student.

**3. Teacher-Anchored Reference**

Conventional DPO uses the student itself (the SFT checkpoint) as the reference model $\pi_\text{ref}$. CTPD innovatively adopts the **teacher model** as the reference.

Theoretical basis: In the DPO gradient, the reference model modulates training via the weight $\lambda$. A stronger reference model provides better sample weighting, guiding the policy toward the correct optimization direction. Through the aligned span projection mechanism, the student can approximate the teacher's log-probabilities in its own token space, enabling a teacher-anchored DPO objective under heterogeneous tokenizers.

### Loss & Training

The final CTPD loss function is:

$$\mathcal{L}_\text{CTPD} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta(r(x, y_w) - r(x, y_l))\right)\right]$$

where $r(x, y) = \sum_{i=1}^T w_i \log \frac{\pi_\theta(p_i | x, p_{<i})}{\pi_\text{ref}(p_i | x, p_{<i})}$ and $\pi_\text{ref}$ is the teacher model.

Training hyperparameters:
- Data: UltraFeedback Binarized (63k+ preference pairs)
- SFT stage: lr = $4 \times 10^{-6}$
- Teacher DPO: lr = $2 \times 10^{-6}$, $\beta = 0.3$
- CTPD: lr = $1 \times 10^{-6}$, $\beta = 0.1$
- Weight clamp range $[L, U] = [-0.5, 1.5]$
- 8× NVIDIA H100-80GB, global batch size 16

## Key Experimental Results

### Main Results

**Table 1: Distillation Results for Qwen-2.5-14B → Llama-3.1-8B**

| Method | HellaSwag | ARC | MMLU | TruthfulQA | Winogrande | GSM8k | Average |
|--------|-----------|-----|------|------------|------------|-------|---------|
| Teacher | 84.34 | 67.06 | 79.74 | 58.51 | 80.58 | 84.23 | 75.74 |
| Student | 81.99 | 57.59 | 65.48 | 45.19 | 77.43 | 50.27 | 62.99 |
| DPO | 82.42 | 60.84 | 65.26 | 52.16 | 78.31 | 54.87 | 65.64 |
| TIS-DPO | 81.08 | 61.92 | 66.73 | 53.86 | 79.05 | 54.31 | 66.16 |
| DSKD | 79.24 | 58.19 | 64.82 | 51.77 | 74.82 | 50.11 | 63.16 |
| **CTPD** | 82.25 | **63.92** | 66.65 | **55.22** | **79.29** | **57.47** | **67.42** |

**Table 2: Distillation Results for Qwen-2.5-7B → Llama-3.2-1B**

| Method | HellaSwag | ARC | TruthfulQA | Winogrande | GSM8k | Average |
|--------|-----------|-----|------------|------------|-------|---------|
| Student | 65.59 | 39.33 | 37.66 | 62.75 | 6.82 | 40.67 |
| TIS-DPO | 66.23 | 40.92 | 43.49 | 64.34 | 9.13 | 42.60 |
| **CTPD** | **67.30** | 40.61 | **46.34** | **64.50** | **9.72** | **43.26** |

### Ablation Study

**Comparison of Weight Estimation Strategies (Llama-3.1-8B)**:

| Strategy | Average |
|----------|---------|
| CTPD (Origin) | **67.42** |
| Average weight | 65.47 |
| Student estimate | 65.88 |
| Teacher-student estimate | 64.51 |
| Random weight | 54.80 |

**Reference Model Selection**: Using the student as the reference model reduces the average score to 65.27, substantially below the 67.42 achieved with the teacher reference, validating the effectiveness of the teacher-anchored reference.

### Key Findings

- CTPD outperforms TIS-DPO by +1.26 (14B→8B) and +0.66 (7B→1B) on average
- The largest gains are observed on GSM8k (+3.16) and TruthfulQA (+2.85), which require reasoning and factual accuracy
- Conventional KD methods (DSKD, ULD, Multi-Level OT) underperform even direct DPO in the preference distillation setting
- Random weights cause severe performance degradation (67.42→54.80), demonstrating the critical role of importance weight estimation

## Highlights & Insights

1. **First work to address cross-tokenizer preference distillation**: fills a clear and well-defined research gap.
2. **Elegant character-level alignment design**: Aligned Spans use the original string as an anchor, naturally resolving tokenizer incompatibility with zero additional parameters.
3. **Theoretical analysis of reference models as reweighting mechanisms**: DPO gradient analysis reveals that the reference model fundamentally acts as a sample weight controller, providing theoretical grounding for using the teacher as the reference.
4. **Rigorous experimental design**: ablation studies systematically verify the contribution of each component.

## Limitations & Future Work

- Experiments are conducted only on the Qwen → Llama model combination; generalization to other architecture pairs remains unverified.
- The computation of aligned spans may introduce additional overhead on long sequences; time complexity is not reported.
- The current importance weight estimation requires separately training positive and negative DPO models from the teacher, increasing pipeline complexity.
- Evaluation is limited to English benchmarks; cross-lingual scenarios with greater tokenizer divergence are not explored.
- Distillation from much larger models (70B+) is not investigated.

## Related Work & Insights

- Compared to preference alignment methods such as DPO and TIS-DPO, CTPD incorporates teacher signals within an end-to-end framework.
- Compared to KD methods such as ULD, DSKD, and Multi-Level OT, CTPD is specifically designed for preference distillation.
- The aligned span mechanism could potentially be applied to other scenarios requiring cross-tokenizer signal transfer, such as cross-model evaluation and federated learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to address cross-tokenizer preference distillation with a clearly defined problem formulation.
- **Technical Depth**: ⭐⭐⭐⭐ — Theoretical derivations based on importance sampling are complete; span-level noise analysis is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Six benchmarks, diverse baselines, and detailed ablations.
- **Value**: ⭐⭐⭐⭐ — Unlocks preference transfer between heterogeneous models, with practical implications for model compression and deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)
- [\[NeurIPS 2025\] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](../../NeurIPS2025/model_compression/universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](../../NeurIPS2025/model_compression/orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](distilling_cross-modal_knowledge_via_feature_disentanglement.md)

</div>

<!-- RELATED:END -->
