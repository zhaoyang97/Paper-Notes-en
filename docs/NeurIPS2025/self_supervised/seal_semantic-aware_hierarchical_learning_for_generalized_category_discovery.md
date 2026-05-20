---
title: >-
  [Paper Note] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Generalized Category Discovery] This paper proposes SEAL, a framework that leverages naturally occurring semantic hierarchies (rather than manually constructed abstract hierarchie…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Generalized Category Discovery"
  - "Hierarchical Learning"
  - "Contrastive Learning"
  - "Semantic Hierarchy"
  - "Cross-Granularity Consistency"
date: 2026-05-08
content_hash: c69c65a5ef9843e7
---

# SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery

**Conference**: NeurIPS 2025
**arXiv**: [2510.18740](https://arxiv.org/abs/2510.18740)  
**Code**: [Available](https://visual-ai.github.io/seal/)  
**Area**: Self-Supervised Learning / Open-World Recognition
**Keywords**: Generalized Category Discovery, Hierarchical Learning, Contrastive Learning, Semantic Hierarchy, Cross-Granularity Consistency

## TL;DR

This paper proposes SEAL, a framework that leverages naturally occurring semantic hierarchies (rather than manually constructed abstract hierarchies) to guide generalized category discovery. Through hierarchically semantic-guided soft contrastive learning and a cross-granularity consistency module, SEAL achieves state-of-the-art performance on fine-grained benchmarks.

## Background & Motivation

Generalized Category Discovery (GCD) aims to classify all unlabeled images—whether belonging to known or unknown categories—given partially labeled data. This is more challenging than Novel Category Discovery (NCD), as unlabeled data contains samples from both known and unknown categories simultaneously.

The core of GCD lies in transferring knowledge from known to unknown categories. Hierarchical information has been shown to be beneficial for this purpose, but existing methods exhibit notable limitations:

- **InfoSieve** constructs abstract hierarchies implicitly via binary trees (through shared binary code prefixes)
- **CiPR** builds abstract hierarchies through iterative merging of data partitions
- **TIDA** relies on manually defined upper and lower hierarchies with hyperparameter-controlled category counts

The fundamental problem with these approaches is that abstract hierarchies may introduce noise and errors. As illustrated in Figure 1(a), "Siberian Tiger" and "Bengal Cat" may be erroneously merged, "Red Fox" may be incorrectly split, and visually similar classes such as "Basset Hound" and "Beagle" are easily confused.

The paper poses the central question: **Can naturally occurring, semantically grounded taxonomic hierarchies serve as more reliable guidance?** This idea draws from biological taxonomy, where known specimen labels guide the classification of new specimens. From an information-theoretic perspective, the paper further proves that semantic hierarchical labels provide tighter mutual information bounds.

## Method

### Overall Architecture

SEAL builds upon the SimGCD baseline and introduces three key components:
1. **Semantic-aware multi-task framework**: performs category discovery simultaneously across multiple semantic levels
2. **Cross-Granularity Consistency module (CGC)**: enforces mutual consistency across predictions at different granularities
3. **Hierarchically semantic-guided soft contrastive learning**: assigns varying weights to negative samples based on semantic proximity

### Key Designs

1. **Semantic-Aware Hierarchical Learning**:

    - Defines $H$ semantic levels with corresponding labels $\mathbf{y}_1, \ldots, \mathbf{y}_H$ (from coarse to fine)
    - Shares an image encoder $\mathcal{F}$, followed by a projection layer $\phi$ that disentangles features into per-granularity sub-representations: $\mathbf{z} = \phi(\mathcal{F}(\mathbf{x})) = [\mathbf{z}_1; \mathbf{z}_2; \ldots; \mathbf{z}_H]$
    - Coarse-grained branches reuse fine-grained features with gradient blocking: $\hat{\mathbf{z}}_i = [\mathbf{z}_1; \cdots; \mathbf{z}_h; \Gamma(\mathbf{z}_{h+1}); \cdots; \Gamma(\mathbf{z}_H)]$
    - Semantic hierarchies are derived from natural classification systems (e.g., family–genus–species in biology) and require no manual construction
    - An independent GCD classifier is trained at each level

2. **Cross-Granularity Consistency Self-Distillation (CGC)**:

    - Addresses inconsistencies in multi-level classification (e.g., an instance predicted as "Shiba Inu" at the fine level but as "cat" at the coarse level)
    - Defines a dynamic transfer matrix $M_h \in \mathbb{R}^{n_H \times n_h}$ encoding the mapping from fine-grained to coarse-grained classes
    - Known classes: fixed one-hot vectors; unknown classes: uniformly initialized and momentum-updated during training
    - Consistency loss: $\mathcal{L}_{cgc} = \sum_{h=1}^{H-1} D_{KL}(p(\mathbf{x}_i|\boldsymbol{\theta}_h) | p(\mathbf{x}_i|\boldsymbol{\theta}_H) \times M_h)$
    - Core idea: fine-grained predictions, after mapping through the transfer matrix, should align with coarse-grained predictions

3. **Hierarchically Semantic-Guided Soft Contrastive Learning**:

    - Motivation: standard contrastive learning treats all non-positive samples as equivalent negatives, ignoring semantic relatedness
    - Computes pairwise similarity matrices $S_h$ within each batch at every semantic level, then aggregates them into a hierarchical similarity $\tilde{S}_h$
    - Generates semantic-aware soft labels: $\tilde{Y}_{\text{soft}_h} = (1-\lambda_s) \cdot \mathbf{I} + \lambda_s \cdot \tilde{S}_h$
    - Replaces hard 0/1 labels with soft labels in the contrastive loss, so semantically similar samples receive smaller negative weights
    - Adopts a hybrid similarity metric: $\text{sim}(\mathbf{z}_i, \mathbf{z}_k') = \lambda_c \mathbf{z}_i \cdot \mathbf{z}_k'^{\top} - (1-\lambda_c)\|\frac{\mathbf{z}_i}{\|\mathbf{z}_i\|} - \frac{\mathbf{z}_k'}{\|\mathbf{z}_k'\|}\|_2$
    - $\lambda_c$ decays linearly during training: angular metric first (coarse), then distance metric (fine)—a curriculum learning strategy

### Loss & Training

Total loss: $\mathcal{L}_{all} = \sum_h^H (\mathcal{L}_{\text{soft}_{rep}}^h + \mathcal{L}_{cls}^h) + \mathcal{L}_{cgc}$

- $\mathcal{L}_{\text{soft}_{rep}}^h$: soft contrastive learning loss at each level
- $\mathcal{L}_{cls}^h$: classification loss at each level (ground truth for labeled data; sharpened pseudo-labels with entropy regularization for unlabeled data)
- End-to-end single-stage training, 200 epochs, cosine learning rate schedule

## Key Experimental Results

### Main Results: SSB Benchmark (DINOv1 backbone)

| Method | CUB All | CUB New | Cars All | Cars New | Aircraft All | Avg |
|--------|---------|---------|----------|----------|-------------|-----|
| SimGCD | 60.3 | 57.7 | 53.8 | 45.0 | 54.2 | 56.1 |
| SPTNet | 65.8 | 65.1 | 59.0 | 49.3 | 59.3 | 61.4 |
| DebGCD | 66.3 | 63.5 | 65.3 | 57.4 | 61.7 | 64.4 |
| **SEAL** | **66.2** | 63.2 | **65.3** | **58.5** | **62.0** | **64.5** |

### Ablation Study (Oxford-Pet dataset, inferred from paper structure)

| Component | Effect |
|-----------|--------|
| Semantic hierarchical multi-task learning | Information-theoretically proven to provide tighter mutual information bounds (Equations 4–6) |
| CGC module | Momentum-updated dynamic transfer matrix effectively handles unknown categories |
| Soft contrastive learning | Reduces erroneous penalization of semantically similar samples compared to hard negative contrastive learning |
| Hybrid similarity metric | Curriculum learning strategy (angular → distance) yields consistent improvements |
| Gradient controller $\Gamma$ | Prevents training bias from propagating into coarse-grained branches |

### Key Findings

- SEAL achieves state-of-the-art or near state-of-the-art results across all fine-grained benchmarks, with notable gains on Aircraft (+0.3) and Cars New (+1.1)
- Performance improves further with a DINOv2 backbone (higher CUB All accuracy)
- Hierarchical information yields greater improvements on new (unknown) categories than on old (known) categories, confirming that semantic hierarchies facilitate knowledge transfer
- Natural semantic hierarchies are more reliable than manually constructed abstract hierarchies—erroneous class merging or splitting does not occur

## Highlights & Insights

- **Solid information-theoretic motivation**: the superiority of hierarchical labels is rigorously proven via mutual information decomposition (Equations 4–6), going beyond the intuitive notion that "hierarchies are useful"
- **Elegant design of the dynamic transfer matrix in CGC**: fixed mappings for known classes and momentum updates for unknown classes gracefully handle the incomplete label setting inherent to GCD
- **Semantic guidance in soft contrastive learning** is intuitively sound—species within the same family but different genera should not be pushed as far apart as entirely unrelated species
- The analogy to biological taxonomic practice strengthens the motivation and suggests broader applicability

## Limitations & Future Work

- Obtaining semantic hierarchical labels for each dataset may be non-trivial in certain domains
- The method assumes the total number of categories $K$ is known (though existing estimation methods can be applied)
- Advantages over baselines may be less pronounced on general coarse-grained datasets (e.g., CIFAR, ImageNet) compared to fine-grained ones
- The momentum update strategy for the transfer matrix may be sensitive to initialization, and convergence has not been rigorously analyzed

## Related Work & Insights

- SEAL extends SimGCD while preserving its architectural simplicity and end-to-end training convenience
- Hierarchical contrastive learning has been successfully applied in closed-set classification; SEAL is the first to introduce semantic hierarchies into the GCD setting
- SEAL and HypCD (which implicitly models hierarchies via hyperbolic embeddings) are complementary—SEAL uses explicit hierarchical labels while HypCD relies on geometric structure
- Superclass and coarse-grained information are also commonly used in zero-shot learning; SEAL's approach is extensible to related tasks

## Rating

- Novelty: ⭐⭐⭐⭐ — First to leverage natural semantic hierarchies in GCD; information-theoretic motivation is original
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across SSB, Oxford-Pet, and Herbarium19
- Writing Quality: ⭐⭐⭐⭐ — Theory and methodology are clearly presented with intuitive figures
- Value: ⭐⭐⭐⭐ — Makes a substantive contribution to the GCD community with generalizable method design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](../../CVPR2026/self_supervised/omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] Disentangling Hyperedges through the Lens of Category Theory](disentangling_hyperedges_through_the_lens_of_category_theory.md)
- [\[NeurIPS 2025\] Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition](foundation_models_for_scientific_discovery_from_paradigm_enhancement_to_paradigm.md)
- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)

</div>

<!-- RELATED:END -->
