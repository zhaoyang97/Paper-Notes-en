---
title: >-
  [Paper Note] Flexible Concept Bottleneck Model
description: >-
  [AAAI 2026][Interpretability][Concept Bottleneck Model] This paper proposes the Flexible Concept Bottleneck Model (FCBM), which introduces a hypernetwork to dynamically generate concept weights and a sparsemax module wit…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Concept Bottleneck Model"
  - "Hypernetwork"
  - "Sparse Activation"
  - "VLM"
date: 2026-05-08
content_hash: 5dcf6bf7f871fd9e
---

# Flexible Concept Bottleneck Model

**Conference**: AAAI 2026
**arXiv**: [2511.06678](https://arxiv.org/abs/2511.06678)  
**Code**: [https://github.com/deepopo/FCBM](https://github.com/deepopo/FCBM)  
**Area**: Interpretability
**Keywords**: Concept Bottleneck Model, Interpretability, Hypernetwork, Sparse Activation, VLM

## TL;DR
This paper proposes the Flexible Concept Bottleneck Model (FCBM), which introduces a hypernetwork to dynamically generate concept weights and a sparsemax module with a learnable temperature parameter, enabling dynamic adaptation of the concept pool—including complete replacement. FCBM achieves accuracy comparable to state-of-the-art baselines with a similar number of effective concepts across five public datasets, and requires only a single epoch of fine-tuning to adapt to an entirely new concept set.

## Background & Motivation

### State of the Field
Concept Bottleneck Models (CBMs) enhance the interpretability of neural networks by introducing an intermediate concept layer—the model first predicts human-understandable concepts and then performs final task prediction based on these concepts. VLM-based CBMs have further leveraged LLMs to automatically generate concept sets and CLIP for automated annotation, substantially reducing reliance on expert labeling.

### Limitations of Prior Work
The central limitation of existing VLM-based CBMs lies in their **fixed concept sets**:

**High retraining cost**: Introducing new concepts or updating existing ones (e.g., newly discovered biomarkers in the medical domain) requires end-to-end retraining of the entire model.

**Rapid iteration of foundation models**: VLM backbones (e.g., CLIP) are frequently updated; changes in the underlying semantic representations necessitate realignment of concept embeddings.

**Limited flexibility**: A fixed concept pool cannot accommodate preferences for different concept subsets across varying deployment scenarios.

### Root Cause
VLM-based CBMs exploit the powerful alignment capabilities of VLMs to automatically construct concept pools; however, the concept-to-label mapping (linear layer) remains fixed at both training and inference time. This means that the number of concepts $m$ and the shape of the weight matrix are coupled—any change in $m$ requires retraining.

### Starting Point
A hypernetwork is employed to map the textual features of concepts to weights, making weight generation **independent of the number of concepts**. Sparsemax is additionally applied to enforce sparsity and preserve interpretability.

## Method

### Overall Architecture
FCBM comprises four core components:
1. A two-stage learning framework (concept predictor + label predictor)
2. LLM-generated concept sets with CLIP feature extraction
3. A hypernetwork for dynamic concept weight generation
4. A sparsemax module with a learnable temperature parameter

### Key Designs

#### 1. **Two-Stage Learning Framework**
- **Stage 1 (Concept Predictor Training)**: Optimizes the mapping $g$ to maximize the cubed cosine similarity between CLIP-encoded image features and concept text features:
  $$g^* = \arg\min_g \sum_{j=1}^m [-\text{sim}(\mathbf{c}_{:,j}, \mathbf{q}_{:,j})]$$
- **Stage 2 (Label Predictor Training)**: Optimizes the hypernetwork $h$ by minimizing cross-entropy loss:
  $$h^* = \arg\min_h \sum_{i=1}^N \text{CE}(g^* \circ \omega(\mathbf{x}_i) \cdot \mathring{h}(\mathbf{t}), \mathbf{y}_i)$$
- Here $\mathring{h}(\mathbf{t}) \triangleq \mathcal{S}_{\max}^\tau(h(\mathbf{t}))$ denotes the sparse weights produced after sparsemax processing.

#### 2. **Hypernetwork**
- The mapping $h: \mathbb{R}^d \rightarrow \mathbb{R}^n$ projects textual feature dimensions to class dimensions.
- **Core advantage**: The parameter scale of $h$ is **independent of the number of concepts** $m$, enabling it to handle concept sets of arbitrary size.
- The output $h(\mathbf{t}) \in \mathbb{R}^{m \times n}$ shares the same shape as a conventional linear projection, and can be interpreted as the contribution weight of each concept toward each class.
- **Inference-time feature distribution alignment**: When a new concept set $\mathbf{t}'$ is introduced, distribution consistency is ensured through statistical normalization:
  $$\tilde{\mathbf{t}}' \triangleq \frac{\sigma_\mathbf{t}}{\sigma_{\mathbf{t}'}}(\mathbf{t}' - \overline{\mathbf{t}'}) + \bar{\mathbf{t}}$$
  $$\tilde{h}(\mathbf{t}') \triangleq \frac{\sigma_{h(\mathbf{t})}}{\sigma_{h(\tilde{\mathbf{t}}')}}\left(h(\tilde{\mathbf{t}}') - \bar{h}(\tilde{\mathbf{t}}')\right) + \bar{h}(\mathbf{t})$$
- **Design Motivation**: The weight matrix dimensions of a fixed linear layer $f$ are bound to $m$ and cannot accommodate changes in concept count. By dynamically generating weights from textual features, the hypernetwork naturally decouples the number of concepts from model parameters.

#### 3. **Sparsemax with Learnable Temperature**
- Standard sparsemax produces sparse outputs (unlike softmax, which yields entirely nonzero outputs), allowing the model to focus on the most relevant concepts.
- A **learnable temperature parameter** $\tau$ is introduced to dynamically control the degree of sparsity: higher $\tau$ activates fewer concepts, while lower $\tau$ considers more.
- Temperature gradient derivation: $\frac{\partial \mathcal{L}}{\partial \tau} = \sum_{i \in P(\mathbf{s})} \frac{1}{|P(\mathbf{s})|} \cdot \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{s}}_i}$
- **Design Motivation**: Hypernetwork outputs are generally non-sparse; using them directly would compromise interpretability. The learnable temperature enables the model to automatically balance predictive accuracy and concept activation sparsity.

### Concept Generation and CLIP Features
- An LLM (GPT-3 / DeepSeek-V3 / GPT-4o) generates concepts using three prompt types: discriminative features, commonly associated features, and superclasses.
- CLIP encodes both images and concept texts into a shared feature space of dimension $d$.
- The CLIP-derived feature matrix is $\mathbf{c} = \mathbf{z} \cdot \mathbf{t}^\top \in \mathbb{R}^{N \times m}$.

## Key Experimental Results

### Main Results (Prediction Accuracy at NEC ≈ 30)

| Backbone | Method | CIFAR10 | CIFAR100 | CUB | Places365 | ImageNet |
|----------|--------|---------|----------|-----|-----------|---------|
| ResNet50 | Standard (non-sparse) | 88.55 | 70.19 | 71.00 | 53.28 | 73.14 |
| ResNet50 | LF-CBM | 86.16 | 64.62 | 56.91 | 48.88 | 66.03 |
| ResNet50 | CF-CBM | 85.42 | 64.31 | 64.23 | 46.39 | 65.95 |
| ResNet50 | **FCBM (Ours)** | **85.59** | **64.77** | 63.46 | **49.13** | **66.34** |
| ViT-L/14 | Standard (non-sparse) | 98.02 | 86.99 | 85.22 | 55.66 | 84.11 |
| ViT-L/14 | LF-CBM | 97.18 | 81.98 | 75.44 | 50.51 | 79.70 |
| ViT-L/14 | CF-CBM | 96.35 | 82.33 | 79.56 | 48.55 | 79.16 |
| ViT-L/14 | **FCBM (Ours)** | **97.21** | **83.63** | **80.52** | **51.39** | **80.62** |

### Ablation Study (Zero-Shot Generalization of Individual Modules, ViT-L/14)

| Method | CIFAR10 Train | CIFAR10 DS Zero-Shot | CIFAR100 Train | CIFAR100 DS Zero-Shot | ImageNet Train | ImageNet DS Zero-Shot |
|--------|--------------|---------------------|---------------|----------------------|---------------|----------------------|
| Hard Truncation | 97.27 | 78.78 | 65.15 | 23.61 | 75.22 | 15.07 |
| FCBM w/o Temperature | 89.05 | 75.58 | 62.42 | 38.54 | 49.13 | 23.65 |
| **FCBM (Full)** | **97.21** | **94.89** | **83.63** | **62.27** | **80.62** | **51.70** |

### Key Findings
1. **Accuracy on par with SOTA**: FCBM surpasses all baselines on more than half of the five datasets and matches them on the remainder.
2. **Zero-shot concept generalization**: After replacing the entire concept set (generated by DeepSeek-V3 or GPT-4o), the model recovers most of its performance with only one epoch of fine-tuning.
3. **Sparsity analysis**: Increasing NEC from 30 to full yields only marginal accuracy gains that quickly plateau, confirming the effectiveness of sparse concept selection.
4. **Hard truncation yields the worst zero-shot performance**: Hard truncation occasionally performs adequately on the training concept set but generalizes most poorly to new concepts.
5. **Learnable temperature is critical**: Removing the learnable temperature significantly degrades the model's ability to regulate sparsity and leads to a marked drop in accuracy.

## Highlights & Insights
- **First solution for dynamic concept adaptation**: FCBM is the first approach to enable seamless replacement of the entire concept pool without retraining the full model.
- **Elegant application of hypernetworks**: Generating weights directly from textual features naturally resolves the problem of variable concept counts.
- **Statistical normalization as a generalization technique**: The distribution alignment formulation for training/inference time is an elegant engineering solution.
- **Concept contribution visualization**: The "campus" class example on Places365 clearly demonstrates the semantic equivalence of different concept sets.
- **Application prospects**: Particularly suited for rapidly evolving knowledge domains, such as medical biomarker updates and VLM backbone switching.

## Limitations & Future Work
- Zero-shot generalization still exhibits a substantial performance gap on fine-grained categories (e.g., CUB bird species) and specialized domains.
- Validation is limited to classification tasks; extension to detection, segmentation, and other visual tasks remains unexplored.
- The hypernetwork introduces additional inference overhead, as a forward pass is required for each concept.
- Only CLIP is evaluated as the VLM backbone; other VLMs (e.g., SigLIP, EVA-CLIP) are not explored.
- Concept generation still relies on LLM prompt engineering, and different prompting strategies may produce significant variation.
- When the semantic gap between old and new concept sets is large, statistical normalization may be insufficient to bridge the distributional discrepancy.

## Related Work & Insights
- Label-free CBM (Oikarinen et al., 2023): The first work to use GPT-3 for automatic concept set generation; FCBM builds upon this by addressing the limitation of fixed concept sets.
- VLG-CBM (Srivastava et al., 2024): Introduced the NEC metric to quantify the number of effective concepts; FCBM adopts the same metric to ensure fair comparison.
- OpenCBM (Tan et al., 2024): Supports flexible addition and removal of concepts at test time, but cannot replace the entire concept pool.
- Hypernetworks (Ha et al., 2016): The classical paradigm of using one network to generate weights for another; FCBM introduces this idea into the CBM framework to achieve dynamic concept mapping.
- Insight: The combination of hypernetworks and sparse selection is generalizable to other tasks requiring dynamic feature or attribute mapping.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of hypernetwork and sparsemax is inventive, though the core mechanism is not overly complex.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Five datasets with multiple ablation groups, though downstream application validation is lacking.)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are clear and the overall structure is complete.)
- Value: ⭐⭐⭐⭐ (Addresses practical deployment challenges of CBMs with well-defined application scenarios.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](partially_shared_concept_bottleneck_models.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)

</div>

<!-- RELATED:END -->
