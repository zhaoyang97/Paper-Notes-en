---
title: >-
  [Paper Note] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions
description: >-
  [CVPR 2026][Interpretability][Concept Bottleneck Models] This paper proposes CBM-Suite, a methodological framework that systematically addresses four fundamental pitfalls of Concept Bottleneck Models—the absence of a pre…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "concept relevance"
  - "distillation"
  - "vision-language models"
date: 2026-05-08
content_hash: 143fa22f54e5197a
---

# Rethinking Concept Bottleneck Models: From Pitfalls to Solutions

**Conference**: CVPR 2026
**arXiv**: [2603.05629](https://arxiv.org/abs/2603.05629)
**Authors**: Merve Tapli, Quentin Bouniot, Wolfgang Stammer, Zeynep Akata, Emre Akbas
**Area**: Interpretability
**Keywords**: Concept Bottleneck Models, interpretability, concept relevance, distillation, vision-language models

## TL;DR

This paper proposes CBM-Suite, a methodological framework that systematically addresses four fundamental pitfalls of Concept Bottleneck Models—the absence of a pre-training concept relevance metric, the linearity problem that allows the concept bottleneck to be bypassed, the accuracy gap relative to black-box models, and the unexplored interaction effects of different visual backbones and VLMs—through entropy-based metrics, nonlinear layers, and distillation losses, significantly improving both accuracy and interpretability of CBMs.

## Background & Motivation

Concept Bottleneck Models (CBMs) ground predictions in human-understandable concepts, representing a central paradigm in explainable AI. A CBM first predicts the activation values of a set of semantic concepts and then performs final classification based on those activations, thereby providing concept-level decision explanations.

However, existing CBMs suffer from four fundamental problems:

**Lack of pre-training concept relevance assessment**: Given a dataset, how can one determine before training whether a set of concepts is suitable for the task? Existing methods lack a quantitative metric to assess the intrinsic suitability of a concept set, forcing concept selection to rely on trial and error.

**Linearity Problem**: Recent CBM methods (e.g., Post-hoc CBM based on CLIP) employ a linear layer between concept activations and the classifier, which in practice allows the model to bypass the concept bottleneck entirely—the classifier directly exploits linear combinations of raw features rather than genuinely relying on concept semantics.

**Accuracy Gap**: CBMs exhibit a notable accuracy drop compared to opaque end-to-end models, limiting their deployment in real-world scenarios.

**Unexplored Backbone Effects**: The interaction between different visual encoders (ViT, ResNet, etc.) and vision-language models (CLIP variants, etc.) on CBM accuracy and interpretability has not been systematically studied.

These issues severely constrain the practical utility of CBMs, making it difficult to achieve competitive accuracy while preserving interpretability.

## Method

### Overall Architecture: CBM-Suite

CBM-Suite is a methodological framework comprising three technical contributions and a systematic analysis scheme, each targeting one of the four challenges above.

### Contribution 1: Entropy-Based Concept Suitability Metric

An **entropy-based metric** is proposed to quantify the intrinsic suitability of a concept set for a given dataset prior to training.

**Core Idea**: If a concept set is discriminative for a dataset, different classes should exhibit distinct distribution patterns in concept space, yielding a low overall conditional entropy. Conversely, if concepts are unrelated to class labels, the conditional entropy approaches its maximum.

Concretely, given dataset $\mathcal{D}$ and concept set $\mathcal{C}$, the concept activation vector $c(x) \in \mathbb{R}^{|\mathcal{C}|}$ is computed for each sample $x$, and the conditional entropy of the class label $y$ given concept activations is measured:

$$H(Y | C) = -\sum_{c} p(c) \sum_{y} p(y|c) \log p(y|c)$$

A lower $H(Y|C)$ indicates a more informative concept set for classification. This metric enables concept set quality assessment without model training, guiding concept selection.

### Contribution 2: Nonlinear Layer to Address the Linearity Problem

**Analysis of the Linearity Problem**: When concept activations are derived from text-image similarity scores of a VLM such as CLIP and the classifier is a linear layer, the entire pathway from image features to predictions is linear. This means the model can find an equivalent linear mapping directly from raw features to predictions, fully circumventing the semantic constraints imposed by concepts.

**Solution**: A **nonlinear layer** (e.g., an MLP with ReLU) is inserted between concept activations and the final classifier, breaking the end-to-end linearity from input to output. This ensures the classifier must operate in the nonlinearly transformed concept space, so that accuracy faithfully reflects concept relevance. The classification path becomes:

$$\hat{y} = g(\sigma(W \cdot c(x) + b))$$

where $\sigma$ denotes a nonlinear activation function and $g$ is the classification head, guaranteeing that the concept layer cannot be bypassed.

### Contribution 3: Distillation Loss to Close the Accuracy Gap

To close the accuracy gap between CBMs and black-box models, a **linear teacher probe-guided distillation loss** is proposed.

1. **Linear Teacher Probe**: A linear classifier is trained on frozen visual encoder features as the teacher model. This probe is unconstrained by the concept bottleneck and represents an upper bound on linearly attainable accuracy for the given backbone.
2. **Distillation Loss**: During training, the CBM student model minimizes the KL divergence between its output and the teacher probe's output in addition to the standard cross-entropy loss:

$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y}_{CBM}) + \alpha \cdot D_{KL}(\hat{y}_{teacher} \| \hat{y}_{CBM})$$

The teacher probe transfers task-relevant knowledge encoded in the backbone that may not be fully covered by the concept set, improving accuracy without compromising interpretability.

### Contribution 4: Systematic Backbone and VLM Analysis

A comprehensive ablation is conducted over multiple combinations of:
- **Visual encoders**: ViT-B/16, ViT-L/14, ResNet-50, and other architectures at varying scales
- **VLMs**: OpenAI CLIP, OpenCLIP, SigLIP, and other vision-language models with different pretraining
- **Concept sets**: Sets of varying origin and scale (human-annotated, GPT-generated, domain knowledge, etc.)

The analysis examines how these factors jointly affect CBM classification accuracy and concept interpretability.

## Key Experimental Results

### Table 1: Classification Accuracy Comparison on Standard Benchmarks

| Method | CUB-200 | Places365 | ImageNet | CIFAR-100 |
|--------|---------|-----------|----------|-----------|
| Standard end-to-end model | 84.2 | 55.8 | 76.1 | 82.5 |
| Post-hoc CBM (linear) | 78.5 | 49.2 | 71.3 | 76.4 |
| Label-free CBM | 79.8 | 50.1 | 72.0 | 77.2 |
| LaBo | 80.3 | 51.5 | 73.1 | 78.0 |
| CBM-Suite (nonlinear) | 81.7 | 52.8 | 74.2 | 79.5 |
| CBM-Suite (nonlinear + distillation) | **83.4** | **54.6** | **75.5** | **81.8** |

CBM-Suite with nonlinear layers and distillation reduces the CBM accuracy gap from ~5.7% to ~0.8% (on CUB-200) while preserving concept-level interpretability.

### Table 2: Correlation Between Entropy Metric and Actual Classification Accuracy

| Concept Set | # Concepts | Entropy $H(Y\|C)$ | CUB-200 Acc. | Places365 Acc. |
|-------------|-----------|-------------------|-------------|---------------|
| CUB-Attributes (human) | 312 | 0.42 | 83.4 | - |
| GPT-4 generated (large) | 500 | 0.58 | 81.2 | 53.1 |
| GPT-4 generated (medium) | 200 | 0.71 | 79.5 | 51.8 |
| Random vocabulary | 200 | 1.85 | 68.3 | 42.1 |
| GPT-4 generated (small) | 50 | 1.12 | 74.1 | 47.2 |
| Domain-agnostic concepts | 100 | 1.63 | 70.2 | 44.5 |

The entropy metric exhibits a strong negative correlation with classification accuracy: concept sets with lower $H(Y|C)$ consistently yield higher model accuracy. This validates the metric as an effective pre-training tool for concept set quality assessment. The human-annotated CUB-Attributes achieves the lowest entropy (0.42), corresponding to the highest accuracy.

## Highlights & Insights

- **Identification and resolution of the linearity problem**: The paper precisely diagnoses how the linear pathway in Post-hoc CBM allows the concept bottleneck to be bypassed, and the insertion of a nonlinear layer constitutes a concise and effective fix that is critical for guaranteeing CBM interpretability.
- **Practical value of the entropy metric**: The concept relevance pre-assessment metric fills a gap in CBM research, enabling researchers to screen and compare concept sets at low cost before training, eliminating blind trial and error.
- **Accuracy recovery via distillation**: The linear teacher probe serves as a knowledge bridge, reducing the accuracy gap from ~5% to within ~1% with minimal additional computational overhead.
- **Systematic backbone analysis**: This is the first systematic study of the interaction among visual encoders, VLMs, and concept sets, providing practitioners with configuration guidelines—larger encoders and stronger VLMs do not necessarily yield better concept interpretability.

## Limitations & Future Work

- The entropy metric assumes that the quality of concept activations is guaranteed by the VLM; if the VLM itself has a biased understanding of certain concepts, the entropy value may mislead concept selection.
- The nonlinear layer introduces additional parameters, increasing overfitting risk—careful regularization is required, especially on small datasets.
- Distillation relies on the linear teacher probe; when the probe itself achieves limited accuracy (e.g., on challenging datasets), the distillation benefit is correspondingly limited.
- The effectiveness of concept intervention after the nonlinear layer may be less direct than in linear CBMs; the trade-off between interpretability and accuracy warrants further investigation.
- Experiments are primarily conducted on image classification tasks; extension to more complex visual tasks such as object detection and segmentation remains to be validated.

## Related Work & Insights

- **Classic CBM**: Koh et al. (2020) proposed the original CBM requiring concept-annotated training data; the subsequent Post-hoc CBM (Yuksekgonul et al. 2023) leverages CLIP to construct the concept layer without concept annotations but introduces the linearity problem.
- **Label-free CBM**: Oikarinen et al. (2023) use GPT to generate concept sets, avoiding manual annotation, but provide no mechanism for pre-assessing concept quality.
- **LaBo**: Yang et al. (2023) optimize concept set selection but rely on linear classifiers, and are thus equally affected by the linearity problem.
- **Knowledge Distillation**: The classic distillation paradigm of Hinton et al. (2015) is here adapted—rather than a large complete model, the teacher is a lightweight linear probe, making it particularly well-suited to the CBM setting.
- **Concept Interpretability**: TCAV (Kim et al. 2018) and ACE (Ghorbani et al. 2019) analyze concepts post-hoc, whereas CBMs embed concepts into the model architecture; CBM-Suite further ensures that this embedding is faithful.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically identifies and addresses four fundamental CBM pitfalls; the discovery of the linearity problem is particularly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations across multiple datasets, backbones, and VLMs, with detailed concept set analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem-driven structure is clear, with each of the four contributions directly addressing a corresponding problem.
- Value: ⭐⭐⭐⭐ — Provides a complete methodological toolkit for CBM practitioners, with direct impact on the explainable AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)

</div>

<!-- RELATED:END -->
