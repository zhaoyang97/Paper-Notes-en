---
title: >-
  [Paper Note] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery
description: >-
  [ICLR 2026][Multimodal VLM][Generalized Category Discovery] This paper proposes SpectralGCD, which represents images as semantic mixtures over a CLIP concept dictionary (i.e., cross-modal similarity vectors), employs spectral filtering to automatically select task-relevant concepts, and incorporates forward-reverse knowledge distillation to preserve semantic quality. The method achieves multimodal state-of-the-art across six benchmarks at a computational cost comparable to unimodal approaches.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Generalized Category Discovery
  - CLIP
  - Cross-modal Representation
  - Spectral Filtering
  - Concept Dictionary
  - Knowledge Distillation
date: 2026-05-08
content_hash: f33c9aa0b83cd029
---

# SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery

**Conference**: ICLR 2026
**arXiv**: [2602.17395](https://arxiv.org/abs/2602.17395)

**Code**: [GitHub](https://github.com/miccunifi/SpectralGCD)

**Area**: Category Discovery / Multimodal Learning
**Keywords**: Generalized Category Discovery, CLIP, Cross-modal Representation, Spectral Filtering, Concept Dictionary, Knowledge Distillation

## TL;DR

This paper proposes SpectralGCD, which represents images as semantic mixtures over a CLIP concept dictionary (i.e., cross-modal similarity vectors), employs spectral filtering to automatically select task-relevant concepts, and incorporates forward-reverse knowledge distillation to preserve semantic quality. The method achieves multimodal state-of-the-art across six benchmarks at a computational cost comparable to unimodal approaches.

## Background & Motivation

**Generalized Category Discovery (GCD):** GCD aims to discover novel categories from unlabeled data using a small annotated subset of known categories. Unlike Novel Category Discovery (NCD), the unlabeled set in GCD contains both known (Old) and unknown (New) classes simultaneously, making it more reflective of real-world scenarios.

**Limitations of Prior Work:** Unimodal methods such as SimGCD train parametric classifiers directly on visual features, which are prone to overfitting spurious visual cues (e.g., background) associated with old classes, resulting in an imbalanced Old/New performance trade-off — strong on old classes but weak on new ones.

**Computational Cost of Multimodal Methods:** TextGCD and GET significantly improve performance by incorporating CLIP's textual modality, but TextGCD requires LLM-generated descriptions and separately trained image/text encoders, while GET requires training an inversion network — both incurring far greater computational overhead than unimodal methods.

**Insufficient Cross-modal Fusion:** Existing multimodal GCD methods feed visual and textual modalities into separate classifiers independently, failing to fully exploit CLIP's inherent cross-modal alignment capability.

**Inspiration from Probabilistic Topic Models:** Drawing an analogy to LDA where "a document is a mixture of topics," this work proposes that "an image is a mixture of semantic concepts," using CLIP image-text similarity scores directly as a unified cross-modal representation.

**Practical Deployment Requirements:** In real-world settings, the discovery pipeline must be re-run periodically as new data arrives, making computational efficiency critical and the high cost of multimodal methods a practical barrier.

## Method

### Overall Architecture: Two-Stage Pipeline

SpectralGCD consists of two stages: (1) a spectral filtering stage, in which a frozen, strong teacher model automatically selects task-relevant concepts from a large concept dictionary; and (2) a training stage, in which a parametric classifier is trained on the filtered cross-modal representations, augmented with forward-reverse knowledge distillation to maintain semantic quality.

### Key Design 1: Cross-modal Sufficient Representation

For each image $x_i$ and concept dictionary $\bar{\mathcal{C}} = \{c_j\}_{j=1}^M$, CLIP image-text cosine similarities are computed as:

$$z_{\theta,\phi}(x_i; \bar{\mathcal{C}}) = \left[\frac{f_\theta(x_i)^\top g_\phi(c_j)}{\|f_\theta(x_i)\| \|g_\phi(c_j)\|} \cdot \frac{1}{\tau} \;\Big|\; c_j \in \bar{\mathcal{C}}\right] \in \mathbb{R}^M$$

This representation approximates a **sufficient representation** — if class labels depend only on semantic concepts, then $p(y|x) = p(y|z(x;\mathcal{C}))$, and a classifier trained on this representation can make optimal predictions without access to the raw image. The representation is linearly projected as $u_i = W^\top z_{\theta,\phi}(x_i; \bar{\mathcal{C}})$ before being fed into classifier $p_i = L_\psi(u_i)$.

### Key Design 2: Spectral Filtering

A frozen teacher (ViT-H/14) computes cross-modal representations over the entire dataset. After softmax normalization, a covariance matrix is constructed:

$$G = \frac{1}{N-1} \sum_{i=1}^N (q_i - \mu)(q_i - \mu)^\top \in \mathbb{R}^{M \times M}$$

Eigendecomposition of $G$ yields:
- **Noise filtering:** The top $k^*$ principal components whose cumulative explained variance reaches $\beta_e$ are retained.
- **Concept importance selection:** A concept importance vector $s = \sum_{i=1}^{k^*} \lambda_i v_i^2$ is computed, and the concept subset $\hat{\mathcal{C}}$ whose cumulative importance reaches $\beta_c$ is selected.

The core intuition is that softmax amplifies foreground concepts while suppressing background noise; combined with CLIP's object-centric bias, the leading eigenvectors of the covariance matrix naturally concentrate on task-relevant object semantics — analogous to term-frequency weighting in Latent Semantic Analysis (LSA).

### Key Design 3: Forward-Reverse Knowledge Distillation

During training, the student's cross-modal representations may drift semantically due to joint optimization. Bidirectional distillation from a frozen teacher is applied:

$$\mathcal{L}_{\text{kd}} = \underbrace{-\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \sigma(\hat{z}_i^*) \log \sigma(\hat{z}_i)}_{\text{Forward Distillation}} + \underbrace{-\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \sigma(\hat{z}_i) \log \sigma(\hat{z}_i^*)}_{\text{Reverse Distillation}}$$

Forward distillation aligns the student's distribution toward the teacher's, while reverse distillation penalizes the student for assigning probability mass to concepts the teacher deems irrelevant. Together, they enforce tighter student-teacher alignment. Teacher representations can be precomputed, ensuring training efficiency.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{c}} + \mathcal{L}_{\text{kd}}$$

where $\mathcal{L}_{\text{cls}}$ comprises supervised and unsupervised classification losses, and $\mathcal{L}_{\text{c}}$ comprises supervised and unsupervised contrastive losses. Only the last transformer block of ViT-B/16 is fine-tuned.

## Key Experimental Results

### Main Results: Comprehensive Comparison with SOTA (Accuracy %)

| Method | Type | CUB All | CUB New | Cars All | Cars New | Aircraft All | IN-100 All |
|--------|------|---------|---------|----------|----------|-------------|-----------|
| SimGCD | Unimodal | 60.3 | 57.7 | 53.8 | 45.0 | 54.2 | 83.0 |
| SelEx | Unimodal | 73.6 | 72.8 | 58.5 | 50.3 | 57.1 | 83.1 |
| DebGCD | Unimodal | 66.3 | 63.5 | 65.3 | 57.4 | 61.7 | 85.9 |
| GET | Multimodal | 77.0 | 76.4 | 78.5 | 74.5 | 58.9 | 91.7 |
| TextGCD | Multimodal | 76.6 | 74.7 | 86.9 | 86.7 | 50.8 | 88.0 |
| **SpectralGCD** | **Multimodal** | **79.2** | **78.5** | **89.1** | **87.4** | **63.0** | **93.4** |

### Ablation Study: Distillation Strategy (Stanford Cars)

| Distillation Loss | Spearman ρ | All Accuracy |
|-------------------|-----------|-------------|
| FD + RD | 0.665±0.09 | **89.1** |
| FD only | 0.639±0.11 | 86.0 |
| RD only | 0.611±0.11 | 87.5 |
| No distillation | 0.487±0.15 | 77.4 |

### Ablation Study: Dictionary Robustness (Stanford Cars / CIFAR-100)

| Method | Dictionary | Cars All | CIFAR100 All |
|--------|-----------|----------|-------------|
| TextGCD* | OpenImagesV7 | 78.1 | 82.6 |
| TextGCD* | Tags | 86.2 | 84.3 |
| SpectralGCD | OpenImagesV7 | 85.8 | 84.9 |
| SpectralGCD | Tags | **89.1** | **86.1** |

## Key Findings

- **Cross-modal representations substantially improve novel class performance:** Compared to purely visual features, the cross-modal representation yields large gains on New classes, effectively alleviating overfitting to spurious cues of old classes. SpectralGCD achieves 78.5% on CUB New, substantially outperforming SimGCD's 57.7%.

- **Smaller student surpasses larger teacher:** Despite the student model (ViT-B/16) being significantly smaller than the teacher (ViT-H/14), SpectralGCD exceeds the teacher's zero-shot performance on multiple benchmarks (e.g., +6.6 points on ImageNet-100), demonstrating that the method's contribution outweighs the effect of model scale.

- **Spectral filtering is especially critical for fine-grained datasets:** On Stanford Cars (196 classes, 200–450 concepts selected), spectral filtering yields substantial improvements; the effect is more moderate on CIFAR-100 (100 classes, 1,000–4,000 concepts selected).

- **Both forward and reverse distillation are indispensable:** Without distillation, All accuracy is only 77.4%; combining FD and RD raises it to 89.1%. The Spearman correlation increases from 0.487 to 0.665, confirming that distillation effectively preserves student-teacher representational consistency.

- **Training efficiency is comparable to unimodal methods:** On CUB, SpectralGCD's training time is comparable to that of unimodal SimGCD, far below GET (3,121 seconds of preparation) and TextGCD.

## Highlights & Insights

- **The "image as concept mixture" analogy is elegant:** The transfer from probabilistic topic models to visual concept representation is natural and well-motivated, providing a clear theoretical grounding.

- **Unified cross-modal representation vs. independent modalities:** Rather than processing visual and textual modalities separately before fusion, the method directly uses image-text similarity scores as a unified representation — concise and effective.

- **Information-theoretic foundation of spectral filtering:** Eigendecomposition of the covariance matrix admits a PCA/LSA interpretation — concept selection is not a black box but a mathematically grounded information distillation procedure.

- **Efficiency and performance without trade-off:** Teacher representations are precomputed once, the text encoder is frozen, and only the last transformer block is fine-tuned — making the approach practical for real-world deployment.

## Limitations & Future Work

- **Dependence on teacher model and dictionary:** SpectralGCD's performance is sensitive to teacher quality and the coverage of the concept dictionary. If the teacher lacks domain knowledge or the dictionary fails to cover key concepts, performance degrades. Table 4 shows that using ViT-B/16 as teacher yields only 72.7% on CUB All, far below the 79.2% achieved with ViT-H/14.

- **Dataset-level rather than instance-level concept dictionaries:** The current approach uses a global concept dictionary at the dataset level without adapting to individual images, potentially missing concepts that are locally salient but globally insignificant.

- **Requirement for known number of categories:** Like most GCD methods, the number of categories $K$ must be specified in advance, which may be difficult to determine in practice.

- **Threshold sensitivity in spectral filtering:** Although the default values $\beta_e=0.95, \beta_c=0.99$ perform well across most datasets, optimal values may vary by dataset.

## Related Work & Insights

- **vs. TextGCD:** TextGCD (Tags+Attributes) achieves 86.9% All on Stanford Cars; SpectralGCD reaches 89.1% (+2.2) using Tags alone. TextGCD additionally requires LLM-generated attribute descriptions and separate image/text classifiers, whereas SpectralGCD's unified cross-modal representation is both simpler and more efficient.

- **vs. GET:** GET converts image features into text tokens via an inversion network before extracting textual representations, requiring 3,121 seconds to train the inversion network in the preparation stage. SpectralGCD's spectral filtering takes only 194 seconds, while outperforming GET by 1.7 points on ImageNet-100 (93.4 vs. 91.7).

- **vs. SimGCD:** SimGCD exemplifies the unimodal parametric classifier paradigm with high training efficiency but is constrained by purely visual representations. SpectralGCD achieves comparable training efficiency while substantially improving performance on New classes through cross-modal representations (CUB: 78.5 vs. 57.7).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of cross-modal concept mixture representation and spectral filtering is novel; the analogy from topic models to visual GCD is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six benchmarks × multiple baselines × efficiency analysis × extensive ablations (distillation, dictionary, teacher, student, threshold, data split).
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical motivation grounded in sufficient representations is clear; the method description is well-structured with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Advances both the performance and efficiency frontiers of GCD, with broad implications for multimodal representation learning and concept selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](../../CVPR2026/multimodal_vlm/multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)
- [\[CVPR 2026\] SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](../../CVPR2026/multimodal_vlm/ssr2gcd_rate_reduction_category_discovery.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICLR 2026\] BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models](biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
