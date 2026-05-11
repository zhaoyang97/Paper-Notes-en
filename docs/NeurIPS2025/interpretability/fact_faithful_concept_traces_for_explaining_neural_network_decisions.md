---
title: >-
  [Paper Note] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions
description: >-
  [NeurIPS 2025][Interpretability][concept explanation] This paper proposes FaCT, an inherently interpretable model combining B-cos transformations and sparse autoencoders (SAE) that **faithfully** decomposes model predict…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "concept explanation"
  - "faithful attribution"
  - "B-cos networks"
  - "sparse autoencoders"
  - "interpretable models"
date: 2026-05-08
content_hash: fa22e8ac809d7927
---

# FaCT: Faithful Concept Traces for Explaining Neural Network Decisions

**Conference**: NeurIPS 2025
**arXiv**: [2510.25512](https://arxiv.org/abs/2510.25512)
**Code**: [https://github.com/m-parchami/FaCT](https://github.com/m-parchami/FaCT)
**Area**: Interpretability / Concept Explanation
**Keywords**: concept explanation, faithful attribution, B-cos networks, sparse autoencoders, interpretable models

## TL;DR
This paper proposes FaCT, an inherently interpretable model combining B-cos transformations and sparse autoencoders (SAE) that **faithfully** decomposes model predictions into concept contributions (Logit = $\sum$ concept contributions) and faithfully visualizes each concept down to the input pixel level (concept activation = $\sum$ pixel contributions). A DINOv2-based C²-score is also introduced to evaluate concept consistency.

## Background & Motivation

**Background**: Deep networks achieve strong performance across diverse tasks, yet understanding their internal mechanisms remains difficult. Attribution methods such as Grad-CAM reveal which input pixels are important but cannot explain the **high-level concepts** the model relies upon.

**Existing Concept Explanation Methods**:
- **Part-prototype networks** and **concept bottleneck models (CBM)**: attempt to construct inherently interpretable models, but the feature extractor itself remains opaque, and concept grounding may be unfaithful to the model.
- **Post-hoc methods such as CRAFT**: decompose model activations into concepts via NMF, but concepts do not directly participate in computing the prediction; approximate methods are then required to estimate concept importance and produce visualizations, which may be unfaithful.

**Restrictive Assumptions in Prior Work**:
- Concepts are class-specific, precluding observation of concepts shared across classes.
- Concepts are assumed to correspond to small image patches or object parts.
- Concepts are drawn from a predefined set.

**Key Challenge**: Existing methods introduce approximations in concept extraction and attribution, resulting in unfaithful explanations. Furthermore, metrics for evaluating concept consistency rely on manually annotated part masks, which have limited coverage and assume each concept corresponds to an annotated part.

## Core Problem

How can a model be designed such that its concept explanations are **faithful to the model's decisions by construction**—i.e., the contribution of each concept to the output is exactly computable (summing to the logit), and the visualization of each concept in input space is exact (summing to the concept activation value), rather than relying on approximations?

## Method

### Overall Architecture

FaCT consists of two core components:

1. **B-cos transformation layers**: replace standard ReLU layers to achieve dynamic-linear transformations.
2. **Bias-free sparse autoencoder (SAE)**: extracts sparse concept representations at an intermediate layer.

#### B-cos Transformation

Standard ReLU layer: $f^{\text{Standard}}(x) = \text{ReLU}(\mathbf{W}x + \mathbf{b})$

The B-cos transformation removes the bias and applies row-normalized weights $\hat{\mathbf{W}}$ with a cosine nonlinearity:

$$f^{\text{B-cos}}(x; B) = (\hat{\mathbf{W}} x) \odot |c(\hat{\mathbf{W}}; x)|^{B-1} = \tilde{\mathbf{W}}(x) x$$

Key property: a sequence of B-cos transformations collapses into a **dynamic linear transformation** of the input:

$$f_{1 \to n}^{\text{B-cos}}(x) = \tilde{\mathbf{W}}_{1 \to n}(x) \cdot x$$

This means that for any input $x$, the model can produce an explanation $\tilde{\mathbf{W}}_{1 \to n}(x)$ that faithfully reproduces the logit.

#### Bias-free Sparse Autoencoder

At intermediate layer $l$, features $F = f_{1 \to l}(I)$ are encoded into a sparse concept activation tensor by the SAE:

$$\mathbf{U} = \text{Encoder}(F) = \text{ReLU}(\text{conv}(\mathbf{W}, F))$$

$$\breve{F} = \text{conv}(\mathbf{V}, \mathbf{U})$$

The model uses the reconstructed features $\breve{F}$ to compute the final logit:

$$L^{\text{FaCT}} = f_{l \to n}(\breve{F})$$

**Key design**: the SAE contains no bias terms, making the encoding process also dynamic-linear and guaranteeing faithful attribution from concepts back to the input.

### Faithful Concept Contributions (Logit Decomposition)

Because $f_{l \to n}$ is composed of B-cos layers, the logit can be exactly decomposed into per-concept contributions:

$$L_c^{\text{FaCT}} = \sum_{k}^{K} \text{Contribution}_k^c$$

where $\text{Contribution}_k^c = \sum_{i,j}^{H,W} \tilde{\mathbf{W}}(\mathbf{U})_{i,j,k} \cdot \mathbf{U}_{i,j,k}$.

This is an **exact equality**, not an approximation: the sum of all concept contributions equals the logit value. This stands in sharp contrast to methods such as CRAFT and VCC, where concepts do not directly participate in computing the logit, necessitating post-hoc approximate importance measures.

### Faithful Input-Level Visualization

Analogously, each concept's activation can be exactly expressed as a dynamic linear combination of input pixels:

$$\text{Concept Activation}_k = \sum_{i,j,c}^{H_0,W_0,3} [\tilde{\mathbf{W}}(I) \cdot I]_{i,j,c}$$

Every concept thus admits a **pixel-level exact visualization** in input space, as opposed to approximate cropping or upsampled heatmaps.

### C²-Score: Concept Consistency Metric

Existing evaluation approaches rely on manually annotated part masks (e.g., PartImageNet), but suffer from three shortcomings: (1) coverage is limited to a small number of categories; (2) cross-class shared concepts are not supported; (3) the annotation granularity may not match the concepts learned by the model.

FaCT proposes the C²-score:
1. Extract high-resolution features for each image using DINOv2 + LoftUp.
2. For each concept $k$ and image $I$, weight the DINOv2 features by the concept attribution to obtain a concept embedding $\mathcal{E}^k(I)$.
3. Compute weighted cosine similarity to measure consistency:

$$\text{Consistency}^k = \sum_{(I,J) \in \mathcal{D}^2, I \neq J} S^{k,I} S^{k,J} \cos(\mathcal{E}^k(I), \mathcal{E}^k(J))$$

4. Subtract a random baseline to remove bias: $\text{C}^2\text{-score} = \frac{1}{K}\sum_K \text{Consistency}^k - \text{Consistency}^{rand}$

Advantages of the C²-score: class-agnostic, requires no manual annotation, accounts for the spatial distribution of attributions, and supports both shared and class-specific concept sets.

## Key Experimental Results

### Experimental Setup
- Dataset: ImageNet
- Architectures: B-cos ResNet-50, B-cos DenseNet-121, B-cos ViT c-S
- SAE configurations: TopK ∈ {8, 16, 32}, total concept count $K$ ∈ {8192, 16384}
- SAEs trained at multiple layers (early / middle / late)

### Performance Retention
- ImageNet accuracy drop < 3%, while concept consistency improves substantially.
- C²-score for DenseNet Block 3/4 improves from 0.11 to 0.39.

### Concept Consistency (C²-Score Comparison)

| Method | C²-score |
|--------|----------|
| B-cos channels | 0.09 |
| CRP | — (below FaCT) |
| CRAFT | — (below FaCT) |
| **FaCT** | **0.37** |

FaCT achieves substantially higher concept consistency than all baseline methods.

### Concept Deletion Experiment
- Concepts are removed in descending order of contribution; the logit and accuracy drop induced by FaCT's Eq. 9 (faithful contributions) is far steeper than that of post-hoc methods such as Saliency and Sobol.
- In early layers (Block 2/4) in particular, removing a small number of concepts causes a sharp accuracy drop, validating the faithfulness of the contribution measure.

### User Study (38 Participants)
- FaCT concept interpretability ratings are substantially higher than the B-cos channel baseline at both early and late layers.
- Input-level visualization significantly improves interpretability, with an average improvement of approximately 0.5/5 points for early-layer concepts.
- Spearman correlation between C²-score and user ratings: all 38 participants show positive correlation; 33/38 show moderate or higher correlation (> 0.4).

## Highlights & Insights

1. **Faithful by design**: concept contributions sum exactly to the logit, and concept visualizations sum exactly to activation values—these are mathematical equalities, not approximations.
2. **Cross-class shared concepts**: concepts are shared across all classes (e.g., a "wheel" concept appears in both school bus and bicycle classes), providing a unified conceptual basis that facilitates analysis of misclassifications.
3. **Cross-layer concept hierarchy**: concepts can be extracted at different layers, forming a hierarchy from low-level textures to high-level semantics.
4. **Concept diversity**: no fixed spatial size is assumed; concepts range from small local regions (helmets) to large spatial extents (wood grain textures).
5. **C²-score evaluation metric**: leverages general-purpose foundation model features to assess concept consistency, eliminating the need for manual annotation.
6. **Misclassification analysis**: the shared concept basis enables analysis of misclassification causes—e.g., when a basketball is misclassified as a volleyball, the contributions of shared concepts such as "ball" and "jersey" can be examined.

## Limitations & Future Work

1. **Dependence on B-cos architecture**: requires replacing standard layers with B-cos transformations; the method cannot be directly applied to arbitrary pre-trained models.
2. **Accuracy drop**: although less than 3%, this may be unacceptable in certain applications.
3. **Large number of concepts**: with $K$ at 8,192 or 16,384, browsing and understanding all concepts imposes a significant cognitive burden on users.
4. **SAE training instability**: issues with "dead" concepts (never activated) and "always-active" concepts (activated on > 60% of data) are observed.
5. **No textual concept labels**: although CLIP-Dissect can assist with naming, this is not part of the method itself.
6. **Evaluation primarily on ImageNet**: CUB results are provided in the appendix, but validation on additional domains (medical imaging, remote sensing, etc.) is lacking.

## Related Work & Insights

| Method | Concept Faithfulness | Input Visualization | Shared Concepts | Evaluation |
|--------|---------------------|--------------------|--------------------|------------|
| CRAFT | Approximate (NMF) | Approximate (upsampling) | ❌ Class-specific | Annotated IoU |
| CRP | Approximate | Approximate | ✅ | Annotated IoU |
| Part-Prototype | Unfaithful | Patch similarity | ❌ | Annotated IoU |
| CBM | Unfaithful | None | ❌ | Predefined set |
| **FaCT** | **Exact equality** | **Pixel-level exact** | **✅ Shared** | **C²-score** |

**Broader Implications**:

1. **SAE + interpretable architecture paradigm**: combining SAEs (originally developed for understanding LLM features) with B-cos interpretable architectures is a promising research direction that may generalize to video, 3D, and other modalities.
2. **Faithful vs. approximate trade-off**: this work clearly demonstrates the gap between "faithful by design" and "post-hoc approximation," carrying methodological significance for the explainable AI community.
3. **Foundation models as evaluation tools**: using DINOv2 features as a substitute for manual annotation to evaluate concept consistency is a transferable idea applicable to other evaluation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (the B-cos + SAE combination is novel; C²-score is a valuable contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (multiple architectures, multiple layers, user study, ablation study, concept deletion, misclassification analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear structure, rigorous mathematical derivations, excellent visualizations)
- Value: ⭐⭐⭐⭐ (substantive advance for explainable AI, though B-cos dependence limits generality)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)
- [\[NeurIPS 2025\] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)

</div>

<!-- RELATED:END -->
