---
title: >-
  [Paper Note] Dynamic Multimodal Prototype Learning in Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][test-time adaptation] This paper proposes ProtoMM, a training-free multimodal prototype learning framework that models prototypes as discrete distributions over textual descriptions and visual…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "test-time adaptation"
  - "CLIP"
  - "Multimodal Prototype"
  - "optimal transport"
  - "Zero-Shot Classification"
date: 2026-05-08
content_hash: 2334ce366a546f04
---

# Dynamic Multimodal Prototype Learning in Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2507.03657](https://arxiv.org/abs/2507.03657)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: test-time adaptation, CLIP, Multimodal Prototype, optimal transport, Zero-Shot Classification

## TL;DR

This paper proposes ProtoMM, a training-free multimodal prototype learning framework that models prototypes as discrete distributions over textual descriptions and visual particles. By leveraging optimal transport to dynamically update multimodal prototypes, ProtoMM achieves state-of-the-art performance across 15 zero-shot benchmarks.

## Background & Motivation

Pre-trained vision-language models (e.g., CLIP) demonstrate strong zero-shot classification performance, yet the ambiguity inherent in category names limits the discriminability of text prototypes:

**Lexical ambiguity**: For example, "sword lily" and "blackberry lily" both contain the word "lily" and exhibit a cosine similarity as high as 0.67, making them difficult to distinguish.

**Semantic ambiguity**: For example, "laptop" and "desktop computer" share no common tokens yet yield a cosine similarity of 0.69, as both belong to the computer category.

Existing methods (TPT, TDA, AWT, etc.) construct prototypes solely from the text modality, overlooking the complementary discriminative cues that visual information can provide. Incorporating the visual features of test images into the prototypes can substantially reduce ambiguity—the authors demonstrate that as the test stream progresses, the KL divergence between multimodal prototypes and the true distribution decreases from 18.7 to 9.5.

## Method

### Overall Architecture

ProtoMM consists of two modules:
1. **Distributed Feature Construction**: Models images and prototypes as discrete distributions.
2. **Multimodal Prototype Learning**: Dynamically updates prototypes via optimal transport.

### Key Designs

1. **Distributed Feature Construction**:

    - The test image is augmented into $N$ views (random cropping/flipping/scaling), represented as a distribution $P_t = \sum_{n=1}^{N} a_t^n \delta_{\mathbf{x}_t^n}$.
    - An LLM (GPT-3.5) generates $M$ textual descriptions per class, which are expanded into $S$ visual particles.
    - The multimodal prototype is defined as: $Q_c = \sum_{m=1}^{M} w_c^m \delta_{\mathbf{z}_c^m} + \sum_{s=1}^{S} w_c^{M+s} \delta_{\mathbf{e}_c^s}$
    - Visual particles are initialized as the mean of textual description features and updated dynamically throughout the test stream.
    - Importance weights are computed based on negative entropy: augmentations with high confidence (i.e., those well-matched to the prototype) receive higher weights.

2. **Optimal Transport-Based Prediction**:

    - A cost matrix $\mathbf{C}_{tc}$ is constructed as the cosine distance between visual augmentations and multimodal prototypes.
    - The entropy-regularized OT problem is solved via the Sinkhorn algorithm.
    - The prediction probability is: $p(y_t=c|\mathbf{x}_t) \propto \exp(-d_{\text{OT}}(P_t, Q_c))$
    - The OT distance measures inter-distribution discrepancy more accurately than pointwise cosine similarity.

3. **Dynamic Prototype Update**:

    - For high-quality samples with prediction confidence $\geq \tau$, augmentation scores $\Theta_t = \mathbf{T}_{tc} \mathbf{w}_{y_t}$ are computed using the transport plan $\mathbf{T}_{tc}$.
    - The top-$S$ augmentations with the highest scores are selected as candidates.
    - The visual cache is updated via weighted moving average: $\mathbf{e}_c^s \leftarrow \frac{w_t^{M+s} \mathbf{e}_c^s + \theta_t^{(s)} \mathbf{x}_t^{(s)}}{w_t^{M+s} + \theta_t^{(s)}}$
    - As the test stream progresses, the multimodal prototypes continuously accumulate richer visual priors.

### Loss & Training

This method is **training-free** and requires no gradient updates or optimization. Key hyperparameters:
- Number of augmentations: $N = M = 50$
- Number of visual particles: $S = 25$
- Confidence threshold: $\tau = 0.8$

## Key Experimental Results

### Main Results (11-dataset cross-domain benchmark, ViT-B/16)

| Method | Aircraft | Caltech | Cars | DTD | EuroSAT | Flower | Food | Pets | SUN | UCF | ImageNet | Avg |
|--------|----------|---------|------|-----|---------|--------|------|------|-----|-----|----------|-----|
| CLIP | 23.22 | 93.55 | 66.11 | 45.04 | 50.42 | 66.99 | 82.86 | 86.92 | 65.63 | 65.16 | 68.34 | 64.93 |
| TDA | 23.91 | 94.24 | 67.28 | 47.40 | 58.00 | 71.42 | 86.14 | 88.63 | 67.62 | 70.66 | 69.51 | 67.71 |
| AWT | 29.22 | 95.40 | 69.80 | 55.56 | 58.40 | 75.07 | 85.54 | 92.23 | 70.00 | 70.70 | 71.26 | 70.28 |
| **ProtoMM** | **31.02** | **95.70** | **69.92** | **56.38** | 56.11 | **77.40** | 85.89 | 91.90 | **70.78** | **71.76** | **72.01** | **70.70** |

### Ablation Study

| Module | Eq.(5) Dist. Feature | Eq.(10) Proto. Learning | ImageNet (RN50) | ImageNet (ViT) | Caltech (ViT) |
|--------|:---:|:---:|:---:|:---:|:---:|
| CLIP baseline | ✗ | ✗ | 59.81 | 68.34 | 93.55 |
| + Dist. Feature | ✔ | ✗ | 60.82 | 68.98 | 94.65 |
| + Multimodal Proto. | ✔ | ✔ | **63.76** | **72.01** | **95.70** |

**OOD Generalization (ViT-B/16)**:

| Method | ImageNet | ImageNet-A | ImageNet-V2 | ImageNet-R | ImageNet-S | OOD Avg |
|--------|----------|------------|-------------|------------|------------|---------|
| DOTA | 70.68 | 61.19 | 64.41 | 81.17 | 51.33 | 64.52 |
| DPE | 71.91 | 59.63 | 65.44 | 80.44 | 52.26 | 64.44 |
| **ProtoMM** | **72.01** | 64.02 | **65.93** | **80.87** | **51.97** | **65.69** |

### Key Findings

- Gains from multimodal prototypes are most pronounced on fine-grained datasets (Flowers +2.3%) and OOD settings.
- Performance stabilizes after 40+ visual augmentations and 40+ textual augmentations.
- The confidence threshold $\tau = 0.8$ is optimal; values that are too high (insufficient update samples) or too low (noisy updates) both degrade performance.
- Transport plan heatmaps show that the model effectively focuses on target object regions.
- Inference time is comparable to TDA, and substantially faster than gradient-based methods such as TPT and DiffTPT.

## Highlights & Insights

- **Core insight of multimodal prototypes**: Text prototypes are inherently ambiguous (semantically similar class names are hard to distinguish), while visual features provide complementary cues. As the test stream progresses, multimodal prototypes become increasingly accurate—an online learning paradigm.
- **Elegance of optimal transport**: Conventional methods treat images and prototypes as single points for similarity computation; OT treats them as distributions and measures inter-distribution distances, better handling multi-view augmentations and multi-description prototypes.
- **Training-free design**: No backpropagation or parameter fine-tuning is required, resulting in high inference efficiency.
- The contributions of distributed feature construction and prototype updating are independently verifiable, yielding clean ablation results.

## Limitations & Future Work

- The method relies on the quality of LLM-generated textual descriptions; different LLMs may produce descriptions of varying quality.
- Visual particles are updated via moving average, which may lead to concept drift over long test streams.
- When the number of categories is large (e.g., ImageNet with 1,000 classes), the computational overhead of OT may increase.
- Validation is limited to classification tasks; extension to downstream tasks such as detection and segmentation remains unexplored.
- Hyperparameters such as the threshold $\tau$ and top-$S$ must be set in advance.

## Related Work & Insights

- **AWT**: The closest prior work, which also formulates image-text distance as an OT problem, but restricts prototype alignment to the text modality. ProtoMM extends this by introducing visual particles to achieve multimodal prototypes.
- **TDA / DOTA**: Training-free methods based on positive-negative caches, but requiring logit fusion and being sensitive to fusion hyperparameters.
- **Sinkhorn algorithm**: The core tool for efficient OT solving, which naturally supports parallel computation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of multimodal prototypes, OT, and dynamic updating is creative, though each individual component is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 15 datasets, dual backbones, comprehensive ablations, OOD evaluation, inference time analysis, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated with intuitive illustrations (distribution evolution and transport heatmaps).
- **Value**: ⭐⭐⭐⭐ A strong contribution to training-free TTA; the multimodal prototype paradigm offers meaningful inspiration for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [\[ICCV 2025\] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning](openvision_a_fully-open_cost-effective_family_of_advanced_vision_encoders_for_mu.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)

</div>

<!-- RELATED:END -->
