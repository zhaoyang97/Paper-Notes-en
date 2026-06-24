---
title: >-
  [Paper Note] Efficient Quantification of Multimodal Interaction at Sample Level
description: >-
  [ICML 2025][Multimodal VLM][Multimodal interaction quantification] Proposes the LSMI (Lightweight Sample-wise Multimodal Interaction) estimator, achieving precise and efficient **sample-wise** quantification of multimodal interactions (redundancy, uniqueness, and synergy) on real-world continuous distribution data for the first time, and demonstrates its practical value in data partitioning, knowledge distillation, and model ensemble.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Multimodal interaction quantification"
  - "Partial Information Decomposition"
  - "Sample-wise estimation"
  - "Redundancy/Uniqueness/Synergy"
  - "Entropy estimation"
date: 2026-05-08
content_hash: 2d97a34df903119d
---

# Efficient Quantification of Multimodal Interaction at Sample Level

**Conference**: ICML 2025  
**arXiv**: [2506.17248](https://arxiv.org/abs/2506.17248)  
**Code**: [GeWu-Lab/LSMI_Estimator](https://github.com/GeWu-Lab/LSMI_Estimator)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal interaction quantification, Partial Information Decomposition, Sample-wise estimation, Redundancy/Uniqueness/Synergy, Entropy estimation

## TL;DR

Proposes the LSMI (Lightweight Sample-wise Multimodal Interaction) estimator, achieving precise and efficient **sample-wise** quantification of multimodal interactions (redundancy, uniqueness, and synergy) on real-world continuous distribution data for the first time, and demonstrates its practical value in data partitioning, knowledge distillation, and model ensemble.

## Background & Motivation

Multimodal information consists of three basic interactions: **redundancy** (information shared across modalities), **uniqueness** (information exclusive to a single modality), and **synergy** (emergent information only present when modalities are combined). Understanding these interactions is crucial for analyzing the information dynamics of multimodal systems.

Limitations of prior work:

1. The **Partial Information Decomposition (PID)** framework mainly defines interactions for discrete distributions, making it difficult to scale directly to continuous distributions.
2. Methods based on distribution optimization (e.g., PID-Batch) can only quantify interactions at the **entire dataset level**, with high computational overhead and an inability to provide fine-grained sample-wise analysis.
3. Existing pointwise PID methods lack efficient and practical solutions for continuous distributions.

Core motivation: Interaction patterns vary significantly across different samples (e.g., musical instrument images and sounds are highly redundant, while "tickling" requires multimodal synergy for recognition). Sample-wise analysis offers a more granular understanding and stronger interpretability.

## Method

### Overall Architecture

The core idea of LSMI is to downscale the multimodal information decomposition problem from the "distribution level" to the "sample-wise level." By defining a reasonable pointwise redundancy metric combined with a lightweight entropy estimation model, it efficiently computes the four interaction values $r, u_1, u_2, s$ for each sample.

Overall pipeline (Algorithm 1):

1. **Input**: Bimodal data $x_1, x_2$ and target $y$; pretrained discriminative models $p(y|x_1,x_2), p(y|x_1), p(y|x_2)$.
2. **Train entropy estimators** $h_{\theta_1}, h_{\theta_2}$ to perform entropy estimation on the data distributions of the two modalities, respectively.
3. **Compute sample-wise entropy** $h(x_1), h(x_2)$ and conditional entropy $h(x_1|y), h(x_2|y)$.
4. **Compute redundancy components** $r^+, r^-$, to obtain the redundancy $r = r^+ - r^-$.
5. **Compute pointwise mutual information** $i(x_1;y), i(x_2;y), i(x_1,x_2;y)$, and solve for $u_1, u_2, s$ using the decomposition equations.
6. **Output**: The $r, u_1, u_2, s$ for each sample.

### Key Designs

#### 1. Redundancy-Based Pointwise Interaction Framework

Extending distribution-level decomposition equations to the pointwise (event-level) setting:

$$i(x_1;y) = r + u_1, \quad i(x_2;y) = r + u_2$$
$$i(x_1, x_2; y) = r + u_1 + u_2 + s$$

Key challenge: Four unknowns $r, u_1, u_2, s$ with only three equations require an extra constraint to determine the redundancy $r$.

#### 2. Resolving Negative Mutual Information via Information Component Decomposition

Directly defining redundancy using pointwise mutual information $i(x;y)$ is problematic: pointwise mutual information can be negative (when $x$ provides misleading information about $y$), which violates the monotonicity required by the redundancy decomposition framework.

**Solution**: Decompose mutual information into two non-negative components:

$$i(x;y) = i^+(x;y) - i^-(x;y)$$

where:
- $i^+(x;y) = h(x) = -\log p(x)$ (self-information/surprisal, always non-negative)
- $i^-(x;y) = h(x|y) = -\log p(x|y)$ (conditional self-information, always non-negative)

Both components satisfy monotonicity and can be decomposed for redundancy over the lattice structure separately.

#### 3. Component-Wise Redundancy Definition

Define redundancy on each component using a minimum operation (set-theoretic intuition: redundancy should not exceed the information from any single source):

$$r^+(x_1;x_2;y) = \min(i^+(x_1;y),\; i^+(x_2;y))$$
$$r^-(x_1;x_2;y) = \min(i^-(x_1;y),\; i^-(x_2;y))$$

Final redundancy:

$$r(x_1;x_2;y) = r^+(x_1;x_2;y) - r^-(x_1;x_2;y)$$

Once $r$ is determined, $u_1, u_2, s$ are uniquely determined by the decomposition equations.

#### 4. Lightweight Entropy Estimation (KNIFE)

The KNIFE differential entropy estimator is adopted to model $p_\theta(x)$ with learnable parameters $\theta$:

$$\mathbb{E}[h_\theta(x)] = \mathbb{E}[h(x)] + D_{KL}(p(x) \| p_\theta(x)) \geq H(X)$$

By minimizing the KL divergence, the parameters are optimized to obtain a tight upper bound on entropy. The negative component is calculated via:

$$i^-(x_m;y) = h_{\theta_m}(x_m) - h(y) - \log p(y|x_m), \quad m \in \{1,2\}$$

### Loss & Training

- **Entropy Estimator Training**: Minimize $\mathbb{E}[h_\theta(x)]$ (i.e., minimize the KL divergence between the estimated and true distributions).
- **Discriminative Models**: Pretrain single-modality models $p(y|x_1), p(y|x_2)$ and the multimodal model $p(y|x_1,x_2)$ using standard classification losses.
- The entire estimation process **does not require joint distribution modeling**, only modality-wise entropy mappings $\mathcal{X}_m \to \mathbb{R}^n$. The complexity is much lower than the joint distribution modeling $\mathcal{X}_1 \times \mathcal{X}_2 \times \mathcal{Y} \to \mathbb{R}^n$ required by PID-Batch.

## Key Experimental Results

### Main Results

**Synthetic Data Validation (Circuit Logic):**

| Method | XOR: R | XOR: S | OR: R | OR: U₁ | XOR+NOT: U₂ | XOR+NOT: S |
|------|--------|--------|-------|--------|-------------|-----------|
| PID-CVX | 0.000 | 0.692 | 0.210 | 0.001 | 0.338 | 0.346 |
| PID-Batch | 0.000 | 0.690 | 0.200 | 0.018 | 0.257 | 0.381 |
| **LSMI** | **0.000** | **0.691** | **0.215** | **0.001** | **0.336** | **0.347** |
| GT | 0.000 | 0.693 | 0.215 | 0.000 | 0.347 | 0.347 |

LSMI is highly consistent with the Ground Truth on all logical tasks, with errors far smaller than PID-Batch.

**Real Dataset Interaction Estimation (Consistency with Human Judgment):**

| Dataset | Estimation Method | R | U₁ | U₂ | S |
|--------|---------|------|------|------|------|
| KS | LSMI | 3.28 | 0.11 | 0.00 | 0.03 |
| KS | PID-Batch | 3.16 | 0.02 | 0.19 | 0.01 |
| Food-101 | LSMI | 4.19 | 0.34 | 0.00 | 0.08 |
| CMU-MOSEI | LSMI | 0.02 | 0.12 | 0.01 | 0.24 |

LSMI achieves Pearson correlation coefficients of **0.98 (redundancy)** and **0.95 (text uniqueness)** with human annotations on Food-101.

**Temporal Efficiency Comparison:**

| Dataset | LSMI (s) | PID-Batch (s) | Speedup |
|--------|----------|---------------|--------|
| KS | 454.4 | 1700.5 | 3.7× |
| CREMA-D | 667.1 | 3124.4 | 4.7× |
| UCF-101 | 426.1 | 5876.5 | 13.8× |
| Food-101 | 501.5 | 21928.0 | 43.7× |

The larger the number of classes, the more significant the efficiency advantage of LSMI (43.7x faster on Food-101).

### Ablation Study

**Impact of Fusion Stage on Learned Interactions (KS Dataset, Hierarchical Transformer):**

| Fusion Layer $l$ | R | U₁ | U₂ | S | Total Information |
|-----------|------|------|------|------|--------|
| 0 (Earliest fusion) | 1.238 | 0.737 | 0.000 | 1.445 | 3.420 |
| 2 | 1.975 | 1.093 | 0.000 | 0.355 | 3.423 |
| 4 (Latest fusion) | 2.335 | 0.907 | 0.000 | 0.181 | 3.423 |

**Impact of Domain Shift (ID vs OOD):**

| Dataset | Setting | R | U₁ | U₂ | S | Total Information |
|--------|------|------|------|------|------|--------|
| UCF | ID | 3.319 | 1.289 | 0.000 | 0.006 | 4.614 |
| UCF | OOD | 2.511 | 0.504 | 0.053 | 0.698 | 3.766 |
| KS | ID | 2.371 | 0.031 | 0.730 | 0.300 | 3.432 |
| KS | OOD | 1.864 | 0.083 | 0.386 | 0.559 | 2.892 |

### Key Findings

1. **Early fusion promotes synergy, late fusion promotes redundancy**: When the fusion layer $l=0$, synergy $S=1.445$ is significantly higher than redundancy $R=1.238$; when $l=4$, redundancy $R=2.335$ is much higher than synergy $S=0.181$, while the total information volume remains nearly unchanged.
2. **OOD data depends more on synergy**: The proportion of synergistic information increases significantly in OOD scenarios, suggesting that the model relies more on cross-modal complementarity when handling unfamiliar data.
3. **Category-level interaction patterns align with human cognition**: Musical instrument categories (e.g., playing organ, playing accordion) show high redundancy; vision-related categories (e.g., grassland, snowy land) tend to exhibit visual uniqueness; audio-related categories (e.g., blowing nose) lean toward auditory uniqueness; complex recognition tasks (e.g., tickling) rely on synergy.

## Highlights & Insights

1. **Solid theoretical contribution**: Through information component decomposition ($i^+, i^-$), the work elegantly bypasses the issue of violated monotonicity caused by potentially negative pointwise mutual information, defining a well-founded sample-wise redundancy metric.
2. **High practical value**: Three downstream applications demonstrate the utility of sample-wise interaction estimation:
    - **Redundancy-guided data partitioning**: Fine-tuning ImageBind on high-redundancy subsets improves multimodal alignment quality, while low-redundancy subsets aid weak-modality learning.
    - **Interaction-guided knowledge distillation**: Choosing distillation strategies based on the relative magnitudes of $r, u, s$ (redundancy/uniqueness $\to$ feature distillation, synergy $\to$ output distillation) outperforms direct distillation.
    - **Interaction-guided model ensemble**: Even adding lower-accuracy models enhances performance because different models focus on different interaction patterns.
3. **High efficiency**: With no joint distribution modeling required, the computational complexity is independent of the number of classes, running 43.7x faster than PID-Batch on Food-101 (101 classes).

## Limitations & Future Work

1. **Two-modality limitation**: The theoretical framework is based on bimodal PID; scenarios with more modalities ($\geq 3$) can only adopt a pairwise analysis strategy, lacking a unified high-order interaction decomposition.
2. **Dependence on pretrained model quality**: Interaction estimation relies on how well the discriminative model approximates the true distribution; model underfitting will impair estimation accuracy.
3. **Semantic interpretation of negative redundancy**: A substantial number of negative information values appear in label-noise experiments, whose physical interpretation requires further investigation.
4. **Unexplored dynamic fusion**: Future work can investigate how to dynamically leverage sample-wise interaction information during training to adaptively adjust fusion strategies.

## Related Work & Insights

- **PID Theory** (Williams & Beer, 2010; Bertschinger et al., 2014): Provides the foundational framework for interaction decomposition.
- **PID-Batch** (Liang et al., 2023b): The first interaction estimation method applied to complex real-world datasets, but restricted to the distribution level.
- **KNIFE** (Pichler et al., 2022): Provides an efficient tool for differential entropy estimation, serving as the computational basis for LSMI.
- **Insights**: The sample-wise interaction estimation concept can be extended to more fields—such as multimodal data cleaning (identifying conflicting-information samples), curriculum learning (ordering training samples by interaction complexity), and active learning (selecting samples with the most diverse interaction patterns for annotation).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to achieve sample-wise multimodal interaction quantification on real-world data.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Validated across synthetic + real data, covering precision, efficiency, and application aspects.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivation, though symbol-dense and requires careful reading.
- Value: ⭐⭐⭐⭐⭐ — Provides new analytical tools and practical guidance for multimodal learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Information-Theoretic Decomposition for Multimodal Interaction Learning](../../CVPR2026/multimodal_vlm/information-theoretic_decomposition_for_multimodal_interaction_learning.md)
- [\[ACL 2025\] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](../../ACL2025/multimodal_vlm/towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](../../CVPR2025/multimodal_vlm/docopilot_improving_multimodal_models_for_document-level_understanding.md)
- [\[ACL 2025\] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](../../ACL2025/multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](../../ACL2025/multimodal_vlm/moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)

</div>

<!-- RELATED:END -->
