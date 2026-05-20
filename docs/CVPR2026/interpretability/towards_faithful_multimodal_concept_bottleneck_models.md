---
title: >-
  [Paper Note] Towards Faithful Multimodal Concept Bottleneck Models
description: >-
  [CVPR 2026][Interpretability][Concept Bottleneck Models] This paper proposes f-CBM — the first faithful multimodal Concept Bottleneck Model framework — which mitigates unintended information leakage in concept representa…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "Leakage Mitigation"
  - "KAN Network"
  - "Multimodal Classification"
date: 2026-05-08
content_hash: b740c4d6b0f21005
---

# Towards Faithful Multimodal Concept Bottleneck Models

**Conference**: CVPR 2026
**arXiv**: [2603.13163](https://arxiv.org/abs/2603.13163)  
**Code**: To be confirmed  
**Area**: Interpretability
**Keywords**: Concept Bottleneck Models, Interpretability, Leakage Mitigation, KAN Network, Multimodal Classification

## TL;DR
This paper proposes f-CBM — the first faithful multimodal Concept Bottleneck Model framework — which mitigates unintended information leakage in concept representations via a differentiable leakage loss, and improves concept detection accuracy using a Kolmogorov-Arnold Network (KAN) prediction head, achieving an optimal Pareto frontier across task accuracy, concept detection, and leakage reduction.

## Background & Motivation
**Background**: Concept Bottleneck Models (CBMs) provide interpretability by routing predictions through a layer of human-understandable concepts. While extensively studied in vision and NLP settings, they remain largely unexplored in multimodal scenarios.

**Limitations of Prior Work**: The faithfulness of CBMs faces two challenges: (a) insufficient concept detection accuracy, and (b) leakage in concept representations — specifically, task-concept leakage (CTL; concepts encoding task-relevant signals beyond their intended semantics) and inter-concept leakage (ICL; unintended mutual information encoded across different concepts).

**Key Challenge**: Existing methods treat concept detection and leakage mitigation as separate problems, such that improving one dimension tends to compromise task accuracy. Independent training protocols reduce leakage but hurt performance; residual connections absorb missing information but undermine interpretability.

**Goal**: Simultaneously achieve accurate concept detection, minimal leakage, and competitive task accuracy in a multimodal setting.

**Key Insight**: Preliminary analysis reveals that CTL and ICL are strongly positively correlated, and that concepts with higher detection accuracy exhibit lower leakage — suggesting that jointly optimizing concept detection and CTL can indirectly reduce ICL.

**Core Idea**: Apply differentiable mutual information estimation as a leakage regularizer during training, replace the linear prediction head with a KAN layer to enhance expressive capacity, and jointly optimize all three objectives.

## Method

### Overall Architecture
The input consists of image-text pairs. CLIP visual and text encoders extract features that are concatenated as $z=[f^v(x^v)\|f^t(x^t)]$, then mapped to concept activations via the concept bottleneck layer $\Phi^C$, and finally passed through a KAN layer $\Phi^{\text{kan}}$ to produce the final classification prediction.

### Key Designs

1. **Differentiable Leakage Loss**:

    - **Function**: Explicitly minimizes concept-task leakage (CTL) during training.
    - **Mechanism**: Mutual information is approximated via kernel density estimation (KDE): $\hat{I}(x;y) = N^{-1}\sum_i \log[\hat{p}(x_i|y_i)/\hat{p}(x_i)]$, using Gaussian kernels to maintain gradient flow. The leakage loss is defined as $\mathcal{L}_{\text{leak}} = [\frac{\hat{I}(\hat{c}_i;y)-\hat{I}(c_i;y)}{H(y)}]^2$.
    - **Design Motivation**: Prior CTL metrics rely on discrete binning, which destroys gradient information and prevents backpropagation. KDE-based estimation preserves differentiability, and the squared formulation enables bidirectional gradient signals — encouraging retention of true concept information while penalizing excess leakage.

2. **KAN Prediction Head (Kolmogorov-Arnold Network Layer)**:

    - **Function**: Replaces the conventional linear layer connecting concept activations to final predictions.
    - **Mechanism**: $\Phi_o^{\text{kan}}(x) = s_o \times \sum_{i=1}^{N}\phi_{i,o}(x)$, where each $\phi_{i,o}$ is a linear combination of first-order trigonometric basis functions $\sum_m c_{i,o,m} \cdot B_m(x)$.
    - **Design Motivation**: Insufficient expressive capacity in linear layers may force the concept layer to encode additional information to compensate, thereby inducing leakage. KAN provides stronger nonlinear mapping, allowing the concept layer to focus on accurate concept detection. A single-layer KAN preserves interpretability — the response curve for each concept can be directly visualized.

3. **Cosine-Annealed Leakage Loss Weight**:

    - **Function**: Gradually increases the leakage loss weight $\alpha$ from 0 to 1.
    - **Mechanism**: The model first learns concept detection in early training, with leakage penalization introduced progressively thereafter.
    - **Design Motivation**: Premature introduction of leakage constraints may interfere with the concept learning phase.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{\text{cls}} + \tilde{\lambda}\mathcal{L}_C + \tilde{\lambda}_{\text{leak}}\alpha\mathcal{L}_{\text{leak}}$, with auxiliary losses dynamically normalized via running mean and $\alpha$ scheduled via cosine annealing. The CLIP backbone is fine-tuned at lr=1e-5; linear layers use a cosine annealing schedule starting at 0.1 or 0.01.

## Key Experimental Results

### Main Results (N24News Dataset, CLIP-base)

| Method | %ACC↑ | c-RMSE↓ | CTL↓ | ICL↓ |
|------|-------|---------|------|------|
| Black-box | 98.5 | — | — | — |
| Indep.-CBM | 96.0 | **0.043** | 0.028 | 0.005 |
| Label-free | 98.2 | 1.264 | 0.212 | 0.050 |
| CT-CBM | 98.1 | 0.101 | 0.244 | 0.059 |
| **f-CBM (ours)** | **98.1** | 0.056 | **0.005** | **0.006** |

### Cross-Dataset and Model Scale Results

| Dataset | Backbone | f-CBM ACC | f-CBM CTL | f-CBM ICL |
|--------|----------|-----------|-----------|-----------|
| N24News | CLIP-base | 98.1 | 0.005 | 0.006 |
| N24News | CLIP-large | 98.5 | 0.004 | — |
| CUB-200 | CLIP-base | 93.7 | 0.008 | 0.009 |
| AG News | CLIP-base | 90.6 | 0.005 | 0.006 |

### Key Findings
- f-CBM reduces CTL by approximately 40× compared to Label-free CBM while maintaining comparable task accuracy.
- The KAN layer improves concept detection (c-RMSE reduced from 0.101 to 0.056), indirectly reducing leakage.
- The contributions of the leakage loss and the KAN layer are complementary — using either alone is less effective than combining both.
- The preliminary analysis hypothesis is validated: reducing CTL consistently leads to a simultaneous reduction in ICL.
- f-CBM generalizes to text-only datasets (AG News, DBpedia), demonstrating the versatility of the multimodal framework.

## Highlights & Insights
- **Causal Chain Analysis**: Preliminary experiments identify positive correlations among concept detection accuracy, task-concept leakage, and inter-concept leakage, motivating a strategy of "optimizing two objectives to improve the third" — a paradigmatic example of analysis-driven method design.
- **Differentiable MI Estimation via KDE**: Transforming a discrete leakage metric into a differentiable training objective is a broadly applicable technique for scenarios requiring mutual information constraints during training.
- **Interpretable Application of KAN**: Beyond improving expressive capacity, the per-concept response curves of the KAN layer provide an additional dimension of interpretability, yielding dual benefits.

## Limitations & Future Work
- The computational complexity of KDE estimation is $O(N^2)$, which may become a bottleneck for large concept sets.
- Concept annotation relies on an LLM (Claude 4.5 Sonnet) and CLIP similarity scores, imposing an upper bound on annotation quality.
- Evaluation is primarily conducted on CUB and N24News; validation across additional domains (e.g., medical, legal) would strengthen the claims.
- The cosine annealing schedule for the leakage loss weight is fixed; an adaptive schedule may yield further improvements.

## Related Work & Insights
- **vs. CT-CBM**: CT-CBM uses residual connections to absorb leaked information and removes them post-training to restore interpretability; f-CBM reduces leakage at the source via the leakage loss, representing a more fundamental approach.
- **vs. Independent-CBM**: Independent training achieves the lowest leakage but at the cost of task accuracy; f-CBM approaches the leakage levels of independent training within a joint training framework through the combination of KAN and leakage loss.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of differentiable leakage loss and KAN prediction head is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐ Limited dataset variety; CUB evaluation uses only 15 classes.
- Writing Quality: ⭐⭐⭐⭐ The preliminary analysis section is well-written with clear methodological motivation.
- Value: ⭐⭐⭐⭐ CBM faithfulness is a core issue in explainable AI; the multimodal extension carries practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)

</div>

<!-- RELATED:END -->
