---
title: >-
  [Paper Note] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for GCD
description: >-
  [ICLR 2026][Multimodal VLM][Generalized Category Discovery] SpectralGCD represents images as CLIP cross-modal image-text similarity vectors (i.e., mixtures of semantic concepts), employs spectral filtering to automatically select task-relevant concepts, and applies forward-backward knowledge distillation to preserve semantic quality. The method achieves a new multimodal GCD state of the art across six benchmarks at a training cost comparable to unimodal approaches.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Generalized Category Discovery
  - CLIP
  - Cross-modal Representation
  - Spectral Filtering
  - Knowledge Distillation
date: 2026-05-08
content_hash: 35067fd4e85a8d19
---

# SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery

**Conference**: ICLR 2026
**arXiv**: [2602.17395](https://arxiv.org/abs/2602.17395)
**Code**: [https://github.com/miccunifi/SpectralGCD](https://github.com/miccunifi/SpectralGCD)
**Area**: Multimodal VLM
**Keywords**: Generalized Category Discovery, CLIP, Cross-modal Representation, Spectral Filtering, Knowledge Distillation

# SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for GCD

## TL;DR

SpectralGCD represents images as CLIP cross-modal image-text similarity vectors (i.e., mixtures of semantic concepts), employs spectral filtering to automatically select task-relevant concepts, and applies forward-backward knowledge distillation to preserve semantic quality. The method achieves a new multimodal GCD state of the art across six benchmarks at a training cost comparable to unimodal approaches.

## Background & Motivation

1. **Generalized Category Discovery (GCD)**: The goal is to leverage a small set of labeled data from known categories to simultaneously recognize known classes and discover novel unknown classes in unlabeled data—a setting closer to real-world deployment than Novel Category Discovery.
2. **Overfitting in unimodal methods**: Methods such as SimGCD train parametric classifiers on image features and tend to overfit to known classes (Old) under label scarcity, yielding poor performance on novel classes (New), as the model exploits task-irrelevant visual cues such as background.
3. **Low efficiency of existing multimodal methods**: TextGCD relies on LLM-generated descriptions, a frozen teacher for image-text matching, and modality-specific classifiers; GET trains a textual inversion network. Both treat visual and textual modalities independently, substantially increasing training cost.
4. **Insufficient cross-modal fusion**: Existing multimodal methods do not exploit the inherent cross-modal relationships in CLIP, instead feeding features from each modality into separate or shared classifiers.
5. **Noise in concept dictionaries**: Using large-scale general-purpose dictionaries inevitably introduces many task-irrelevant concepts that degrade representation quality.
6. **Efficiency as a practical constraint**: In real-world scenarios, the discovery process must be executed repeatedly as new unlabeled data arrives, making training efficiency a critical requirement.

## Method

### Mechanism: Cross-modal Representation

Inspired by probabilistic topic models, each image is represented as a mixture of semantic concepts. For each concept $c_j$ in a large-scale concept dictionary $\bar{\mathcal{C}} = \{c_j\}_{j=1}^M$, the cosine similarity between the CLIP image encoder and text encoder is computed:

$$z_{\theta,\phi}(x_i; \bar{\mathcal{C}}) = \left[\frac{f_\theta(x_i)^\top g_\phi(c_j)}{\|f_\theta(x_i)\| \|g_\phi(c_j)\|} \cdot \frac{1}{\tau} \;\middle|\; c_j \in \bar{\mathcal{C}}\right] \in \mathbb{R}^M$$

This cross-modal representation is analogous to a Concept Bottleneck Model, where each dimension reflects the association strength between a concept and the image. A linear projection $W$, a parametric classifier $L_\psi$, and a contrastive MLP $\mathcal{M}$ are trained on top of this representation. Only the last Transformer block of CLIP ViT-B/16 is fine-tuned; the text encoder is kept frozen.

### Spectral Filtering

**Purpose**: Automatically select a task-relevant concept subset from the large-scale general dictionary, removing noisy concepts.

1. A frozen strong teacher (CLIP ViT-H/14) computes cross-modal representations over the entire dataset, which are softmax-normalized to obtain $q_i$.
2. An $M \times M$ cross-modal covariance matrix $G$ is computed and subjected to eigendecomposition.
3. **Noise filtering**: The top $k^*$ principal components are retained by cumulative explained variance with threshold $\beta_e = 0.95$.
4. **Concept importance selection**: A concept importance vector $s = \sum_{i=1}^{k^*} \lambda_i v_i^2$ is computed, and a compact dictionary $\hat{\mathcal{C}}$ is obtained by applying a cumulative importance threshold $\beta_c = 0.99$.

Softmax normalization amplifies foreground concepts and suppresses common ones; combined with CLIP's object bias, the leading eigenvectors naturally concentrate on discriminative semantics.

### Forward-Backward Knowledge Distillation

Joint optimization of the student image encoder and classifier causes semantic drift in the cross-modal representation. To mitigate this, the method introduces:

- **Forward distillation $\mathcal{L}_{fd}$**: The student's softmax distribution is aligned with the teacher's to maintain semantic consistency.
- **Backward distillation $\mathcal{L}_{rd}$**: The teacher's distribution is aligned with the student's, penalizing the student for assigning probability mass to concepts the teacher deems irrelevant.

The combination tightly aligns student and teacher cross-modal representations. Teacher representations can be precomputed, incurring no additional inference overhead.

### Overall Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{c}} + \mathcal{L}_{\text{kd}}$$

where $\mathcal{L}_{\text{cls}}$ comprises supervised and unsupervised classification losses, $\mathcal{L}_{\text{c}}$ is the contrastive loss, and $\mathcal{L}_{\text{kd}}$ is the sum of forward and backward distillation losses.

## Key Experimental Results

### Main Results: Comparison on Six Benchmarks (Table 1)

| Method | Dictionary | CUB All | Cars All | Aircraft All | CIFAR-10 All | CIFAR-100 All | IN-100 All |
|--------|------------|---------|----------|-------------|-------------|--------------|------------|
| SimGCD (unimodal) | — | 60.3 | 53.8 | 54.2 | 97.1 | 80.1 | 83.0 |
| GET | InversionNet | 77.0 | 78.5 | 58.9 | 97.2 | 82.1 | 91.7 |
| TextGCD | Tags+Attr | 76.6 | 86.9 | 50.8 | 98.2 | 85.7 | 88.0 |
| **SpectralGCD** | **Tags** | **79.2** | **89.1** | **63.0** | **98.5** | **86.1** | **93.4** |

SpectralGCD with a Tags-only dictionary comprehensively outperforms TextGCD, which uses Tags+Attributes, and surpasses GET on all six benchmarks.

### Ablation Study: Distillation Components (Table 2, Stanford Cars)

| Distillation | Spearman ρ | All Accuracy |
|--------------|-----------|--------------|
| FD + RD | 0.665 | **89.1** |
| FD only | 0.639 | 86.0 |
| RD only | 0.611 | 87.5 |
| None | 0.487 | 77.4 |

Joint forward-backward distillation improves All accuracy from 77.4% to 89.1% and Spearman correlation from 0.487 to 0.665.

### Training Efficiency

On CUB, the spectral filtering preprocessing stage of SpectralGCD takes 194 seconds, and the training stage is comparable in cost to unimodal SimGCD. By contrast, GET's preprocessing requires 3,121 seconds, and TextGCD's training stage is substantially slower.

## Highlights & Insights

- **Unified cross-modal representation**: The method departs from the paradigm of independently processing two modalities, directly training a classifier on CLIP image-text similarities—achieving both semantic interpretability and efficiency.
- **Automatic concept selection via spectral filtering**: Task-relevant concepts are selected automatically through covariance matrix eigendecomposition, without relying on LLM-generated descriptions or manual annotation.
- **Small student outperforms large teacher**: The ViT-B/16 student surpasses the zero-shot performance of the ViT-H/14 teacher on multiple benchmarks (e.g., +6.6 pt on IN-100).
- **Training efficiency**: Training time is close to unimodal levels, far below that of multimodal methods such as GET and TextGCD.

## Limitations & Future Work

- Dictionary choice still matters: Tags vs. OpenImages-v7 yield varying performance across datasets, and selecting the optimal dictionary requires domain expertise.
- Teacher model quality is a bottleneck: A stronger teacher (DFN-5B pretrained) further improves performance, implying the method depends on large-scale pretrained CLIP.
- Validation is limited to classification benchmarks; applicability to downstream tasks such as detection and segmentation remains unexplored.
- Spectral filtering is an offline one-time operation; if the data distribution shifts continuously, it must be re-executed.

## Related Work & Insights

- **Unimodal GCD**: SimGCD (parametric classifier + self-distillation), PromptCAL (visual prompting), SelEx (hierarchical semi-supervised k-means), DebGCD (debiased learning).
- **Multimodal GCD**: CLIP-GCD (feature concatenation), TextGCD (LLM descriptions + modality-independent classifiers), GET (textual inversion network).
- **Concept Bottleneck Models**: CBM projects inputs onto interpretable concept activations; SpectralGCD adopts an analogous idea in the context of unsupervised discovery.
- **Knowledge distillation**: The forward + backward KD scheme is drawn from Wang et al. 2025b to ensure semantic consistency in cross-modal representations.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Introducing topic model intuitions into GCD; the combination of cross-modal representation and spectral filtering is original.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Six benchmarks, multiple ablations, efficiency comparisons, and analyses of dictionaries, teachers, and students.
- ⭐⭐⭐⭐ **Value**: High training efficiency, open-source code, and reliance only on general-purpose dictionaries without requiring LLMs.
- ⭐⭐⭐ **Writing Quality**: Dense mathematical notation, but overall logic is clear and figures are intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](../../CVPR2026/multimodal_vlm/multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)
- [\[CVPR 2026\] SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](../../CVPR2026/multimodal_vlm/ssr2gcd_rate_reduction_category_discovery.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ICLR 2026\] BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models](biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)

</div>

<!-- RELATED:END -->
