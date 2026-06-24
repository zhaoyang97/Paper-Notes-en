---
title: >-
  [Paper Note] Towards Faithful Multimodal Concept Bottleneck Models
description: >-
  [CVPR 2025][Physics & Scientific Computing][concept bottleneck model] Proposes f-CBM, a faithful multimodal Concept Bottleneck Model framework based on CLIP. By jointly addressing concept detection accuracy and information leakage via a differentiable leakage loss and a Kolmogorov-Arnold Network prediction head, it achieves the optimal trade-off among task accuracy, concept detection, and leakage.
tags:
  - "CVPR 2025"
  - "Physics & Scientific Computing"
  - "concept bottleneck model"
  - "interpretability"
  - "leakage"
  - "KAN"
  - "multimodal XAI"
date: 2026-05-08
content_hash: 54e14c286c711ff6
---

# Towards Faithful Multimodal Concept Bottleneck Models

**Conference**: CVPR 2025  
**arXiv**: [2603.13163](https://arxiv.org/abs/2603.13163)  
**Code**: To be confirmed  
**Area**: Interpretability  
**Keywords**: concept bottleneck model, interpretability, leakage, KAN, multimodal XAI

## TL;DR

Proposes f-CBM, a faithful multimodal Concept Bottleneck Model framework based on CLIP. By jointly addressing concept detection accuracy and information leakage via a differentiable leakage loss and a Kolmogorov-Arnold Network prediction head, it achieves the optimal trade-off among task accuracy, concept detection, and leakage.

## Background & Motivation

**Background**: Concept Bottleneck Models (CBMs) achieve interpretability by routing predictions to a human-interpretable concept layer. While they have been widely studied in the vision domain (e.g., Label-free CBM, CT-CBM), they remain largely unexplored in multimodal scenarios.

**Limitations of Prior Work**: Standard CBMs face two major faithfulness issues: (1) inaccurate concept detection—the CBL may fail to detect concepts correctly; (2) leakage—the concept representations encode unintended extra information (task leakage: concepts encode task-related signals beyond their semantics; inter-concept leakage: mutual information between concepts exceeds their natural correlation).

**Key Challenge**: Existing methods treat concept detection and leakage mitigation as isolated problems, where improving one often sacrifices the other or task accuracy (e.g., Independent-CBM reduces leakage but degrades task accuracy; CT-CBM absorbs leakage using residual connections but reduces interpretability).

**Goal**: To realize a multimodal CBM that simultaneously achieves accurate concept detection, low leakage, and high task accuracy.

**Key Insight**: Through preliminary analysis, it is found that concept detection accuracy is negatively correlated with task leakage, and task leakage is positively correlated with inter-concept leakage. Based on this, a joint optimization strategy is designed.

**Core Idea**: Formulate a positive feedback loop by explicitly reducing leakage with a differentiable leakage loss and improving concept detection with a highly expressive KAN prediction head.

## Method

### Overall Architecture

1. The vision and text encoders of CLIP extract image/text embeddings respectively, and concatenate them as $z = [f^v(x^v) \| f^t(x^t)] \in \mathbb{R}^{2d}$
2. The Concept Bottleneck Layer (CBL) $\Phi^C: \mathbb{R}^{2d} \to \mathbb{R}^{|C|}$ maps the multimodal representation to concept activation scores
3. The KAN prediction layer $\Phi^{\text{kan}}: \mathbb{R}^{|C|} \to \mathcal{Y}$ replaces the traditional linear layer to yield final predictions
4. Training Objective: $\mathcal{L} = \mathcal{L}_{\text{cls}} + \tilde{\lambda} \mathcal{L}_C + \tilde{\lambda}_{\text{leak}} \alpha \mathcal{L}_{\text{leak}}$

### Key Designs

**1. Preliminary Analysis: Interaction of Faithfulness Factors**
- **Function**: Train a baseline mCBM on the N24News dataset to analyze the relationship between concept detection accuracy and leakage.
- **Mechanism**: It is found that (1) concepts with high detection accuracy have significantly lower task leakage (p < 1% t-test); (2) task leakage is strongly and positively correlated with inter-concept leakage (both Pearson and Spearman correlation are significant).
- **Design Motivation**: Based on this, it is hypothesized that simultaneously optimizing concept detection quality and reducing task leakage can naturally reduce inter-concept leakage, implying that explicitly optimizing only two objectives can indirectly improve all three aspects.

**2. Differentiable Leakage Loss**
- **Function**: A differentiable mutual information estimator based on Kernel Density Estimation (KDE) to explicitly minimize Concept-Task Leakage.
- **Mechanism**: 
  $$\mathcal{L}_{\text{leak}} = \left[\frac{\hat{I}(\hat{c}_i; y) - \hat{I}(c_i; y)}{H(y)}\right]^2$$
  Estimating $\hat{I}(x; y) = N^{-1} \sum_i \log[\hat{p}(x_i|y_i) / \hat{p}(x_i)]$ using a Gaussian kernel, where the bandwidth is automatically determined by Scott's rule: $\sigma = 1.06 \cdot \text{std}(x) \cdot N^{-1/5}$. Squaring is used instead of clamping-at-zero to preserve bi-directional gradients.
- **Design Motivation**: Unlike binning methods, KDE maintains differentiability, allowing it to be directly integrated into the training loss. Explicitly reducing task leakage also indirectly reduces inter-concept leakage.

**3. KAN Prediction Layer**
- **Function**: Replaces the traditional linear shift with a Kolmogorov-Arnold Network (KAN) layer, which uses a learnable univariate function on each edge.
- **Mechanism**:
  $$\Phi_o^{\text{kan}}(x) = s_o \times \sum_{i=1}^{N} \phi_{i,o}(x), \quad \phi_{i,o}(x) = \sum_{m=1}^{M} c_{i,o,m} \cdot B_m(x)$$
  where $B_m$ is a degree-1 triangular basis function, and $s_o$ is a learnable scaling factor. A single-layer KAN is used to preserve interpretability.
- **Design Motivation**: The limited expressiveness of a linear layer might force concept representations to encode extra information (a source of leakage). KAN provides sufficient expressiveness so that the concept layer does not need to "cheat", while the learnable functions on each edge can be visualized as response curves to maintain interpretability.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \tilde{\lambda} \mathcal{L}_C + \tilde{\lambda}_{\text{leak}} \alpha \mathcal{L}_{\text{leak}}$$

- $\mathcal{L}_{\text{cls}}$: cross-entropy classification loss
- $\mathcal{L}_C$: MSE concept prediction loss
- $\mathcal{L}_{\text{leak}}$: KDE-based leakage loss
- Each auxiliary loss is dynamically scaled to a comparable range with the classification loss using a running mean.
- $\alpha$ increases from 0 to 1 following a cosine annealing schedule to prevent the leakage loss from interfering with early concept learning.
- The CLIP backbone is fine-tuned with a fixed learning rate of $10^{-5}$, and the linear layer uses cosine annealing ($10^{-1}$ or $10^{-2}$).

## Key Experimental Results

### Main Results

N24News (CLIP-base / CLIP-large):

| Method | %ACC↑ | c-RMSE↓ | CTL↓ | ICL↓ |
|---|---|---|---|---|
| Black-box | 98.5 / 98.5 | - | - | - |
| Indep.-CBM | 97.3 / 97.9 | **0.045** / **0.044** | 0.027 / 0.025 | 0.004 / 0.025 |
| Label-free | 98.1 / 98.3 | 1.806 / 1.723 | 0.388 / 0.271 | 0.130 / 0.061 |
| CT-CBM | 98.3 / 98.5 | 0.296 / 0.125 | 0.377 / 0.281 | 0.136 / 0.085 |
| **f-CBM** | 97.7 / 98.2 | 0.079 / 0.057 | **0.005** / **0.004** | **0.005** / **0.003** |

CUB-200 (CLIP-base / CLIP-large):

| Method | %ACC↑ | c-RMSE↓ | CTL↓ | ICL↓ |
|---|---|---|---|---|
| Black-box | 91.3 / 95.8 | - | - | - |
| **f-CBM** | **79.3** / **85.3** | 0.200 / 0.273 | **0.026** / 0.045 | - / - |

### Ablation Study

f-CBM lies on the Pareto frontier: In the trade-off between concept detection accuracy and aggregate leakage, f-CBM lies on the Pareto frontier formed by other methods (Figure 1), achieving the optimal faithfulness-performance trade-off.

Key component ablation:
- **KAN only (no leakage loss)**: Improves concept detection, but leakage is only partially reduced.
- **leakage loss only (no KAN)**: Significantly reduces leakage, but concept detection is not as good as when KAN is present.
- **Both combined (f-CBM)**: Achieves optimal or near-optimal results across all metrics.

### Key Findings

1. **f-CBM reduces leakage by 1-2 orders of magnitude**: CTL is reduced from ~0.3-0.4 to ~0.003-0.005, and ICL is reduced from ~0.06-0.13 to ~0.002-0.005.
2. **Task accuracy is barely compromised**: f-CBM achieves 97.7-98.2% on N24News, which is close to the 98.5% of the black-box model.
3. **Hypothesis validated**: Explicitly reducing task leakage indeed naturally decreases inter-concept leakage.
4. **Multimodal generality**: f-CBM is effective on text-image datasets (N24News, CUB) as well as text-only datasets (AGNews, DBpedia).
5. **Concept intervention efficacy**: Low leakage makes concept intervention (inference-time concept correction) more reliable, preventing counter-intuitive effects due to reliance on leaked information.

## Highlights & Insights

- The first work to systematically study CBM faithfulness in a multimodal setting.
- A dynamic triangular relationship between concept detection, task leakage, and inter-concept leakage is revealed via preliminary analysis, providing theoretical guidance for methodology design.
- The KDE-based differentiable leakage loss is an elegant design, making the originally non-differentiable mutual information metric directly embeddable in training.
- The KAN layer improves expressiveness while maintaining interpretability (via visualized response curves), addressing the issue where a linear layer "misleads" the concept layer.
- Validates an important intuition: providing a sufficiently expressive prediction head reduces the need for the concept layer to encode extra information.

## Limitations & Future Work

- A significant gap in task accuracy remains on the CUB dataset compared to the black-box model (79.3% vs 91.3%), indicating that the inherent limitations of the concept bottleneck are more pronounced in fine-grained tasks.
- Concept annotations rely on automated labeling via CLIP + sentence transformers, which introduces annotation noise.
- The computational complexity of KDE mutual information estimation increases as the batch size grows.
- Evaluation is limited to classification tasks; other multimodal tasks such as retrieval and VQA remain to be explored.
- Concept set selection still relies on LLM generation, leaving room for improvement in automated quality and optimization.

## Related Work & Insights

- Label-free CBM (Oikarinen et al.) implements unsupervised concept detection using CLIP, but suffers from large concept detection errors (c-RMSE ~1.7).
- CT-CBM mitigates leakage via concept selection and residual connections, but the residual channels degrade interpretability.
- Mahinpei et al. proposed a rigorous metric for leakage (mutual information gain), which is extended to a differentiable training target in this paper.
- Insight: The faithfulness issues of CBM (detection accuracy + leakage) may inherently exist across all concept-based explanation methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of the KDE leakage loss and the KAN prediction head is innovative, and the discovery of the triangular relationship from the preliminary analysis is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 4 datasets, 2 backbones, and compared with various baselines; ablation studies verified the contributions of each component.
- **Writing Quality**: ⭐⭐⭐⭐ The logical flow from preliminary analysis to method design is clear, and the definitions of faithfulness metrics are rigorous.
- **Value**: ⭐⭐⭐⭐ Addresses the core trust issues within CBM interpretability, providing a significant boost to the field of explainable AI (XAI).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)
- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](../../ICML2025/physics/finetuning_stellar_spectra_foundation_models_with_lora.md)
- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](../../ICML2025/physics/large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](../../ICML2025/physics/liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[NeurIPS 2025\] The Platonic Universe: Do Foundation Models See the Same Sky?](../../NeurIPS2025/physics/the_platonic_universe_do_foundation_models_see_the_same_sky.md)

</div>

<!-- RELATED:END -->
