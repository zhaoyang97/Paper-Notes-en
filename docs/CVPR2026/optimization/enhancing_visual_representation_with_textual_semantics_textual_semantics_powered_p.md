---
title: >-
  [Paper Note] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning
description: >-
  [CVPR 2026][Optimization][Federated Learning] To address the problem that existing federated prototype learning methods destroy inter-class semantic relations, this paper proposes FedTSP, which leverages pre-trained language models to construct textual prototypes that preserve semantic structure, achieving significant performance gains and faster convergence in heterogeneous federated learning.
tags:
  - CVPR 2026
  - Optimization
  - Federated Learning
  - Prototype Learning
  - Semantic Relations
  - Pre-trained Language Models
  - Data Heterogeneity
date: 2026-05-08
content_hash: 67b72c017067c210
---

# Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning

**Conference**: CVPR 2026
**arXiv**: [2503.13543](https://arxiv.org/abs/2503.13543)
**Code**: [GitHub](https://github.com/XinghaoWu/FedTSP)
**Area**: Optimization
**Keywords**: Federated Learning, Prototype Learning, Semantic Relations, Pre-trained Language Models, Data Heterogeneity

## TL;DR

To address the problem that existing federated prototype learning methods destroy inter-class semantic relations, this paper proposes FedTSP, which leverages pre-trained language models to construct textual prototypes that preserve semantic structure, achieving significant performance gains and faster convergence in heterogeneous federated learning.

## Background & Motivation

Federated Prototype Learning (FedPL) is an effective strategy for handling data heterogeneity in federated learning. The core idea is to have clients collaboratively construct global prototypes and align local features with them. Existing methods (e.g., AlignFed, FedTGP) typically pursue maximizing inter-class distances among prototypes to enhance discriminability, but this approach has an overlooked drawback: enlarging inter-class distances inevitably destroys the semantic relationships between classes.

For example, "horse" and "dog" are semantically similar animal categories, so their prototype distance should be smaller than that between "horse" and "truck." However, prototypes uniformly distributed on a hypersphere cannot preserve such hierarchical semantic structure. The authors verify this finding quantitatively using Spearman correlation coefficients and a semantic gap metric.

Learning semantic relations directly from limited and heterogeneous client data is difficult. Nevertheless, pre-trained language models (PLMs) such as BERT have already captured rich semantic relations from large-scale text corpora. This motivates the core idea of this paper: can textual semantic knowledge be injected into federated learning prototypes so that inter-class relations are preserved even under heterogeneous data?

## Method

### Overall Architecture

Input: client image data → LLM generates class descriptions → PLM encodes them as textual prototypes → trainable prompts align modalities → client local features align with textual prototypes → Output: personalized models for each client.

### Key Designs

1. **LLM-Generated Multi-View Textual Descriptions**:
   - Function: Generate rich semantic descriptions for each class.
   - Mechanism: ChatGPT or a similar LLM is used to generate $k=3$ fine-grained descriptions from different perspectives for each class, following the template "A photo of {CLASS}: {description}."
   - Design Motivation: Hand-crafted prompts (e.g., "A photo of a {CLASS}") differ only in class name, providing minimal semantic context and introducing ambiguity (e.g., "apple" may refer to the fruit or the company).

2. **Trainable Prompts for Modality Alignment**:
   - Function: Bridge the modality gap between PLM text features and client image features.
   - Mechanism: Trainable embedding vectors are inserted into the text embedding sequence, replacing the first $m$ positions, and aligned with client image prototypes via the InfoNCE loss.
   - Design Motivation: PLMs such as BERT are not exposed to image data during pre-training, leading to modality mismatch when used directly.

3. **Contrastive Learning-Based Feature Alignment**:
   - Function: Transfer the semantic structure of textual prototypes to client models.
   - Mechanism: A contrastive learning loss (rather than L2 distance) is used to align local features with textual prototypes, focusing on the relative similarity ordering among classes rather than absolute distances.
   - Design Motivation: PLM-generated prototypes exhibit a high baseline similarity (even the least similar classes share a similarity of 0.73), so L2 alignment would mislead the model into treating unrelated classes as similar.

### Loss & Training

- **Server side**: InfoNCE loss updates the trainable prompts to align textual prototypes with aggregated image prototypes.
- **Client side**: Cross-entropy classification loss + contrastive alignment loss (with temperature parameter $\tau$ controlling sensitivity to relative similarity).
- **Privacy-preserving extension**: Gaussian noise is added to text embeddings via a differential privacy (DP) mechanism to satisfy $(\varepsilon, \delta)$-DP guarantees.

## Key Experimental Results

### Main Results

| Dataset | Metric | FedTSP-BERT | Prev. SOTA | Gain |
|---------|--------|-------------|------------|------|
| CIFAR-10 ($\alpha=0.1$) | Acc | 87.52% | 86.80% (FedKD) | +0.72% |
| CIFAR-100 ($\alpha=0.1$) | Acc | 46.08% | 42.82% (FedMRL) | +3.26% |
| TinyImageNet ($\alpha=0.1$) | Acc | 34.82% (CLIP) | 32.79% (FedKD) | +2.03% |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|------------|---------|
| Contrastive learning vs. L2 alignment | +2–3% | Contrastive learning better handles high baseline similarity |
| LLM descriptions vs. hand-crafted templates | +1–2% | Fine-grained descriptions provide richer semantic context |
| CLIP vs. BERT | Comparable | BERT, despite lacking image pre-training, can bridge the gap via trainable prompts |

### Key Findings

- FedTSP yields more significant improvements under high heterogeneity ($\alpha=0.1$), indicating that textual prototypes are more robust to heterogeneous data.
- FedTSP-BERT achieves larger gains in Top-5 accuracy, demonstrating the effectiveness of semantic relations: even when misclassified, predictions tend to fall within semantically related classes.
- The privacy-preserving variant incurs negligible performance degradation when $\varepsilon \geq 1$.

## Highlights & Insights

- The first work to introduce PLM/LLM semantic knowledge into federated prototype learning, offering a novel perspective.
- Identifies and quantifies the semantic relation degradation caused by existing methods.
- FedTSP is compatible with different PLMs (e.g., CLIP and BERT) and does not rely on CLIP's vision-language alignment.
- Capable of handling both data heterogeneity and model heterogeneity simultaneously.

## Limitations & Future Work

- The server is required to deploy a PLM, increasing server-side computational cost.
- The quality of LLM-generated descriptions depends on the unambiguity of class names.
- Larger-scale datasets (e.g., ImageNet) and more diverse PLM architectures remain unexplored.
- The privacy-preserving extension only considers class-name privacy and does not cover broader privacy scenarios.

## Related Work & Insights

- **vs. FedProto/FedTGP**: These methods aggregate prototypes from client data or maximize inter-class distances, destroying semantic relations. FedTSP constructs prototypes from the text modality, naturally preserving semantic structure.
- **vs. CLIP-based FL**: CLIP-based methods aim to enhance CLIP itself, whereas FedTSP transfers semantic knowledge to lightweight client models without relying on CLIP.
- **vs. FedETF/FedNH**: These methods use fixed ETF or uniformly distributed classifiers as prototypes, which cannot encode semantic relations.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce PLM semantic knowledge into federated prototype learning, with a distinctive perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple datasets, heterogeneity settings, PLM variants, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; visualizations are intuitive; semantic alignment and gap metrics are elegantly designed.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for exploiting language model semantic knowledge in federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[CVPR 2026\] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)

</div>

<!-- RELATED:END -->
