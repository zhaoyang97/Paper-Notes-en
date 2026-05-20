---
title: >-
  [Paper Note] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
description: >-
  [NeurIPS 2025][LLM Pretraining][Object Binding] By defining the IsSameObject predicate and designing quadratic probes, this work demonstrates that large-scale pretrained ViTs — particularly DINO and CLIP — naturally deve…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Object Binding"
  - "Vision Transformer"
  - "IsSameObject"
  - "Self-Supervised Learning"
  - "Probing Analysis"
date: 2026-05-08
content_hash: 86ce22475e934b6a
---

# Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?

**Conference**: NeurIPS 2025
**arXiv**: [2510.24709](https://arxiv.org/abs/2510.24709)  
**Code**: [GitHub](https://github.com/liyihao0302/vit-object-binding)  
**Area**: LLM Pretraining
**Keywords**: Object Binding, Vision Transformer, IsSameObject, Self-Supervised Learning, Probing Analysis

## TL;DR

By defining the IsSameObject predicate and designing quadratic probes, this work demonstrates that large-scale pretrained ViTs — particularly DINO and CLIP — naturally develop object binding capabilities. This signal is encoded in a low-dimensional subspace and actively guides the attention mechanism, challenging the cognitive science community's view that ViTs lack binding ability.

## Background & Motivation

**Object binding** is a core concept in cognitive science: the brain integrates low-level features (color, shape, motion, etc.) distributed across different cortical regions into unified object representations. This capability supports efficient object storage, compositional memory, and reasoning in humans.

This problem carries significant importance yet remains underexplored in AI:

**Skepticism from cognitive science**: Researchers argue that ViTs lack a mechanism for dynamically and flexibly grouping features, lack recurrent connections for iterative refinement, and as purely connectionist models cannot perform genuine symbolic processing.

**Limitations of object-centric learning**: Methods such as Slot Attention enforce binding via external modules, but introduce additional scalability and training challenges.

**Core question**: Can ViTs acquire object binding capabilities purely through large-scale pretraining, without explicit architectural inductive biases?

The authors' key insight is that the quadratic nature of self-attention provides a computational basis for ViTs to represent "whether two patches belong to the same object."

## Method

### Overall Architecture

The **IsSameObject** predicate is defined as follows: for two token embeddings $(x_i^{(\ell)}, x_j^{(\ell)})$ at layer $\ell$, determine whether they belong to the same object:

$$\text{IsSameObject}(x_i^{(\ell)}, x_j^{(\ell)}) = \phi(x_i^{(\ell)}, x_j^{(\ell)}), \quad \phi: \mathbb{R}^d \times \mathbb{R}^d \to [0,1]$$

Key research hypotheses:
- Whether the IsSameObject encoding is **linear** or fundamentally **quadratic**
- Whether the signal is a **pairwise relation** or a **pointwise mapping** (mapping to an object ID first, then comparing)
- Whether the model relies on **category labels** or **object instances** to distinguish objects
- Whether the signal resides in **a few specialized dimensions** or is **distributed across many dimensions**

### Key Designs

Four probe architectures are designed to test the above hypotheses:

**1. Linear probe**:
$$\text{IsSameObject}_{lin}(x,y) = \sigma(Wx + Wy + b), \quad W \in \mathbb{R}^{1 \times d}$$

**2. Diagonal quadratic probe** (specialized dimensions):
$$\text{IsSameObject}_{diag}(x,y) = \sigma(x^\top W y + b), \quad W \text{ is a diagonal matrix}$$

**3. Full quadratic probe** (distributed):
$$\text{IsSameObject}_{quad}(x,y) = \sigma(x^\top W_1^\top W_2 y + b)$$

where $W_1, W_2 \in \mathbb{R}^{k \times d}$, $k \ll d$, and $W_2 = SW_1$ ($S$ is a signed diagonal matrix) to ensure symmetry.

**4. Object category/instance probe** (pointwise): maps embeddings to probability distributions and computes their inner product.

**Decomposition of the binding signal**: Each token embedding is assumed to decompose into a feature component and a binding component:

$$h^{(\ell)}(x_t) = f^{(\ell)}(x_t, c) + b^{(\ell)}(x_t)$$

where $f$ encodes attributes such as texture and shape, and $b$ encodes information about which other tokens belong to the same object. The trained quadratic probe can be interpreted as projecting activations onto the IsSameObject subspace.

### Loss & Training

Probes are trained on the ADE20K dataset using cross-entropy loss to classify same-object vs. different-object patch pairs. The baseline accuracy is 72.6% (always predicting "different"), reflecting the class imbalance in which most patch pairs belong to different objects.

Ablation design:
- **Uninformed ablation**: randomly shuffles the binding vectors $b(x_i)$
- **Informed ablation (injection)**: injects IsSameObject signals using ground-truth instance masks

## Key Experimental Results

### Main Results

**IsSameObject decoding accuracy across models**:

| Model | Peak Accuracy | Above Baseline (pp) | Peak Layer (0–1) |
|-------|--------------|---------------------|-----------------|
| DINOv2-Small | 86.7% | +14.1 | 1.00 |
| DINOv2-Base | 87.5% | +14.9 | 0.82 |
| DINOv2-Large | **90.2%** | **+17.6** | 0.78 |
| DINOv2-Giant | 88.8% | +16.2 | 0.77 |
| Supervised (ViT-L) | 84.2% | +11.6 | 0.39 |
| CLIP (ViT-L) | 82.9% | +10.3 | 0.65 |
| MAE (ViT-L) | 76.3% | +3.7 | 0.13 |

**Probe comparison** (DINOv2-Large): Full quadratic > Diagonal quadratic > Object instance > Object category > Linear

### Ablation Study

**Effect of IsSameObject ablation on downstream tasks** (DINOv2-Large, layer 18):

| Metric | Original | Shuffle 50% | Shuffle 100% | Inject α=0.5 | Inject α=0 |
|--------|----------|-------------|-------------|--------------|------------|
| Semantic Seg. mIoU | 44.14% | 41.03% | 39.20% | 44.91% | 43.59% |
| Instance Seg. mIoU | 35.14% | 31.39% | 28.19% | 36.37% | 37.02% |
| DINO Loss | 0.6182 | 0.6591 | 0.6749 | — | — |

**Attention correlation**: Attention weights in intermediate layers exhibit a positive correlation with IsSameObject scores (Pearson r = 0.163–0.201), indicating that the model actively leverages binding signals to allocate attention.

### Key Findings

1. **Binding is learned, not architecturally inherent**: DINO, CLIP, and supervised ViTs all exhibit strong binding signals, whereas MAE shows almost none (+3.7 pp), demonstrating that binding ability depends on the specific pretraining objective.
2. **The signal is quadratic and distributed**: The full quadratic probe significantly outperforms linear and diagonal probes, consistent with the quadratic form of self-attention.
3. **Layer-wise evolution**: Early and middle layers progressively identify local objects; deep layers shift toward category-based grouping, and positional information is discarded at depth.
4. **Ablation confirms causality**: Shuffling the binding signal degrades segmentation performance and increases pretraining loss; injecting ground-truth signals improves instance segmentation.
5. **Low-dimensional subspace**: IsSameObject is encoded in a low-dimensional projection space, where different object instances are linearly separable along the first few principal components.

## Highlights & Insights

1. **Bridging cognitive science and deep learning**: The work connects the psychological concept of object binding to emergent behaviors in ViTs, providing evidence for human-like cognitive capabilities in AI systems.
2. **Pretraining objective determines binding ability**: The comparative experiments reveal an important source of inductive bias — DINO's contrastive learning requires consistency across augmented views, naturally promoting object-level feature learning, whereas MAE's reconstruction objective does not require this capability.
3. **Brain-like hierarchical organization**: The pattern in which intermediate ViT layers attend to local objects and deep layers attend to semantic categories echoes the retinotopic organization of the brain's ventral visual stream.
4. **Rethinking Slot Attention**: Addressing the binding problem may not require external modules; instead, strengthening ViTs' intrinsic binding mechanisms through tailored training objectives or minimal architectural modifications may suffice.

## Limitations & Future Work

1. The assumption that patch embeddings decompose into "feature" and "binding" components is overly simplistic and requires further empirical validation.
2. A causal relationship between object binding and downstream task performance has not been established.
3. Downstream evaluation is limited to segmentation tasks; other tasks such as visual reasoning remain to be examined.
4. Only patch-level binding is studied; more general forms of binding (e.g., attribute binding) are not explored.
5. The mechanism explaining why MAE does not develop binding signals lacks sufficient depth.

## Related Work & Insights

- **Slot Attention**: An explicit object-centric approach that enforces binding by having learnable slots compete for token features; this work demonstrates that such capability can emerge naturally.
- **Feng & Steinhardt (2023)**: Discover that binding in language models is realized through low-dimensional binding-ID encodings; the present work extends this finding to the visual domain.
- **Dai et al. (2024)**: Analyze binding representations in LLMs and find that attributes are linked to subjects via low-dimensional codes.
- **DINO/DINOv2**: The emergent properties of self-supervised ViTs (e.g., attention maps corresponding to salient regions) are further extended here to reveal object binding capability.
- This work has important implications for multimodal understanding: if ViTs inherently encode "which parts belong together," VLMs can exploit this to improve compositional understanding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Poses an entirely new research question from a cognitive science perspective; the IsSameObject formulation is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Cross-model, cross-probe, and ablation analyses are comprehensive, though downstream validation is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Concepts are clearly defined, argumentation is logically rigorous, and the connection between cognitive science and AI is well articulated)
- Value: ⭐⭐⭐⭐ (Deepens understanding of ViT representations and provides important guidance for object-centric learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] One Prompt Fits All: Universal Graph Adaptation for Pretrained Models](one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](ricl_temporal_credit.md)

</div>

<!-- RELATED:END -->
