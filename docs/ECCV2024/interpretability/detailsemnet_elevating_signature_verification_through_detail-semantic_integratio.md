---
title: >-
  [Paper Note] DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration
description: >-
  [ECCV 2024][Interpretability][Offline Signature Verification] DetailSemNet is proposed for offline signature verification, which decouples features into detail and semantic branches via a Detail-Semantics Integrator, and introduces EMD-based local structural matching to achieve SOTA performance on multiple multilingual signature datasets.
tags:
  - "ECCV 2024"
  - "Interpretability"
  - "Offline Signature Verification"
  - "Local Structural Matching"
  - "Feature Decoupling"
  - "Earth Mover's Distance"
  - "Detail-Semantic Integration"
date: 2026-05-08
content_hash: 0f5c29538050a1be
---

# DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration

**Conference**: ECCV 2024  
**arXiv**: [2511.16364](https://arxiv.org/abs/2511.16364)  
**Code**: [https://github.com/nycu-acm/DetailSemNet_OSV](https://github.com/nycu-acm/DetailSemNet_OSV)  
**Area**: Interpretability  
**Keywords**: Offline Signature Verification, Local Structural Matching, Feature Decoupling, Earth Mover's Distance, Detail-Semantic Integration

## TL;DR

DetailSemNet is proposed for offline signature verification, which decouples features into detail and semantic branches via a Detail-Semantics Integrator, and introduces EMD-based local structural matching to achieve SOTA performance on multiple multilingual signature datasets.

## Background & Motivation

**Background**: Offline Signature Verification (OSV) is a crucial technique in forensic document examination, widely applied in banking and commercial fields. Although deep learning methods have made significant progress in OSV recently, critical challenges remain.

**Limitations of Prior Work**:

**Over-reliance on global features**: Existing methods compare similarity using global features, ignoring differences in local stroke structures and stylistic details in signatures. Global representations destroy the spatial structure of images and lose discriminative local information. As shown in Fig. 1, signatures from two different individuals can be highly similar overall but exhibit distinct differences at the patch level.

**Transformers naturally suppress high-frequency information**: Multi-Head Self-Attention modules indiscriminately suppress high-frequency signals, leading to the loss of prominent detailed information. Experiments show that traditional Transformer models mainly learn low-frequency patterns and fail to exploit high-frequency details even when the input is rich in them, showing almost no reduction in EER as high-frequency details increase.

**Key Challenge**: The essence of signature verification lies in comparing subtle stylistic features hidden within strokes rather than the overall content of the signature. However, existing methods lack the capability for fine-grained comparison of local structures, and the backbones (especially Transformers) lose critical high-frequency details during feature extraction.

**Core Idea**: Design a multi-branch network to separately extract and handle detail (high-frequency) and semantic (low-frequency) features, and replace simple global distance metrics with local patch-level structural matching.

## Method

### Overall Architecture

The model processes image pairs consisting of a reference signature R and a query signature Q. The input is first preprocessed into binary images and foreground masks. After Patch Embedding, they enter a four-stage feature extraction backbone (each stage includes Patch Embedding + multiple DSI modules). The output token features $f^\mathcal{R}$ and $f^\mathcal{Q}$ are used to calculate the global distance $dis_{global}$ and structural distance $dis_{struct}$. The final integrated distance is:

$$dis = \lambda_0 \times dis_{global} + dis_{struct}$$

### Key Designs

1. **Detail-Semantics Integrator (DSI)**: The core feature enhancement module, which decouples the input feature $X$ into semantic and detail components:

    - **Semantic features** $Sem[X]$: Extracts the low-frequency component via local average pooling.
    - **Detail features** $Det[X] = X - Sem[X]$: Obtains the high-frequency component via subtraction.
    - **SemanticsAttend Branch**: Applies an attention module on $Sem[X]_{proj}$ to extract contextual semantic features $Y_{Sem}$.
    - **SalientConv Branch**: Uses a max pooling filter followed by convolutional layers to process half of the detail features, retaining salient features.
    - **DetailConv Branch**: Employs two consecutive convolutional layers to process the other half of the detail features, extracting fine high-frequency information.
   
   The outputs of the three branches are concatenated along the channel dimension and fused via a residual convolutional layer. **Design Motivation**: Convolution is adept at detecting high-frequency details (compared to attention), so the detail component is designated to convolution. Attention is superior at global context aggregation, making it suitable for processing the semantic component. This cooperative design empowers the model with both detail awareness and semantic understanding.

2. **Structural Matching**: After filtering out background tokens using foreground masks, pairwise cosine distances between local embeddings are calculated:

$$d_{ij} = 1 - \frac{r_i^T q_j}{\|r_i\| \|q_j\|}$$

This forms a distance matrix $D$. Then, the optimal matching flow $F^*$ is solved via Earth Mover's Distance (EMD) to obtain the structural distance:

$$dis_{struct} = \frac{\sum_{i}\sum_{j} d_{ij} f_{ij}^*}{\sum_{i}\sum_{j} f_{ij}^*}$$

The EMD is efficiently solved using the Sinkhorn algorithm with entropy regularization. **Design Motivation**: EMD allows many-to-many matching and is robust to outliers, making it more robust than Hausdorff distance and better at capturing subtle differences than the simple Chamfer distance.

3. **Foreground Mask Filtering**: Signature images typically contain large areas of blank background. After resizing the input image to the feature map size, global threshold binarization is performed to generate a foreground mask $Mask$. This filters out non-informative background tokens, leaving only meaningful stroke-containing tokens for matching.

### Loss & Training

Double-Margin Contrastive Loss is employed during training:

$$Loss_{DM} = y \cdot \max(0, dis - m)^2 + (1-y) \cdot \max(0, n - dis)^2$$

where $y=1$ corresponds to positive sample pairs (genuine-genuine), $y=0$ corresponds to negative sample pairs (genuine-forgery), and $m < n$ are margin parameters.

## Key Experimental Results

### Main Results

| Dataset | Metric | DetailSemNet | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| BHSig-H (Hindi) | EER↓ | **2.07%** | 3.39% (TransOSV) | -1.32% |
| BHSig-H | Acc↑ | **98.24%** | 96.61% (TransOSV) | +1.63% |
| BHSig-B (Bengali) | EER↓ | **2.11%** | 3.96% (CaC) | -1.85% |
| BHSig-B | Acc↑ | **98.19%** | 96.04% (CaC) | +2.15% |
| CEDAR (English) | EER↓ | **0.58%** | 1.75% (SDINet) | -1.17% |
| CEDAR | Acc↑ | **99.53%** | 96.16% (AVN) | +3.37% |

### Cross-Dataset Zero-Shot Transfer (EER%)

| Train→Test | SigNet | CaC | TransOSV | **Ours** |
|-----------|--------|-----|----------|---------|
| BHSig-H→BHSig-B | 39.35 | 14.66 | 18.66 | **7.46** |
| BHSig-H→CEDAR | 40.43 | 29.49 | - | **14.05** |
| BHSig-B→BHSig-H | 35.43 | 30.41 | 17.17 | **15.91** |
| CEDAR→BHSig-H | 44.39 | 39.08 | - | **16.35** |
| CEDAR→BHSig-B | 35.85 | 38.07 | - | **8.40** |

### Ablation Study

| SM | DetailConv | SalientConv | BHSig-H EER↓ | CEDAR EER↓ | ChiSig EER↓ |
|----|-----------|-------------|-------------|-----------|-------------|
| ✗ | ✗ | ✗ | 4.70 | 3.41 | 12.47 |
| ✓ | ✗ | ✗ | 4.67 | 1.99 | 10.69 |
| ✗ | ✓ | ✓ | 2.62 | 1.74 | 7.00 |
| ✓ | ✓ | ✗ | 2.51 | 1.09 | 8.65 |
| ✓ | ✗ | ✓ | 2.72 | 2.10 | 6.36 |
| ✓ | ✓ | ✓ | **2.07** | **0.58** | **5.85** |

### Backbone Comparison

| Backbone | BHSig-H EER↓ | BHSig-B EER↓ |
|----------|-------------|-------------|
| PVT | 4.62 | 2.72 |
| Swin | 4.24 | 10.27 |
| DAT | 4.94 | 20.09 |
| BiFormer | 4.38 | 8.66 |
| **Ours (DSI)** | **2.07** | **2.11** |

### Key Findings

1. **Incremental contributions of the three modules**: The dual branches of DSI (DetailConv + SalientConv) contribute the most, reducing the EER from 4.70% to 2.62%; adding SM further reduces it to 2.07%.
2. **High-frequency information is critical**: As high-frequency details in test images increase, DetailSemNet's EER consistently decreases from 6.52% to 0.58%, whereas traditional Transformers saturate at 3.41%.
3. **Strong cross-lingual generalization**: In cross-dataset zero-shot transfer, DetailSemNet significantly outperforms all comparison methods, indicating that local structural matching learns language-agnostic stroke features.
4. **SM works best in the final stage**: Applying Structural Matching at stage 4 (EER 2.09%) outperforms applying it at stage 3 (3.47%).

## Highlights & Insights

- Precise problem insight: Frequency analysis experiments intuitively demonstrate the issue of Transformers suppressing high-frequency information, providing sound justification for the DSI design.
- Elegant feature decoupling: Achieves frequency decoupling via simple average pooling and subtraction, with minimal computational overhead.
- Local structural matching significantly enhances interpretability: Visualizing the matching flow shows how the model aligns patches, making the decision rationale human-understandable.
- Cross-lingual generalization capability validates that the method learns low-level stroke structures rather than language-specific patterns.

## Limitations & Future Work

- Solving EMD (even with Sinkhorn acceleration) still incurs computational overhead when the number of tokens is very large.
- The foreground mask employs simple threshold binarization, which may require more refined segmentation methods in complex backgrounds.
- Evaluation is limited to offline signature verification; could it be extended to related tasks such as document authentication and handwriting identification?
- The potential of utilizing temporal information in online signature verification remains unexplored.
- The training data scale is relatively small (24-30 samples per person), indicating room for improvement in data efficiency.

## Related Work & Insights

- TransOSV [Li et al.] first introduced Transformer to OSV but overlooked the loss of high-frequency information.
- The recurrent observation approach of CaC [Lu et al.] offers a new comparison strategy but still relies on global features.
- Local matching methods from the Re-ID field (such as BPB, PAT, etc.) can be drawn upon, although OSV demands higher sensitivity to fine-grained differences.
- The frequency decoupling concept of DSI can be generalized to other tasks that simultaneously require attention to both global semantics and local details (e.g., fine-grained recognition, defect detection).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of DSI's frequency decoupling design and local structural matching represents a significant innovation in the OSV domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four multilingual datasets + cross-dataset transfer + detailed ablation studies + backbone comparisons + visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivational analysis with rich and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Achieves significant SOTA improvements on the OSV task, with good interpretability and cross-lingual generalization capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Visual Composition through Improved Semantic Guidance](../../CVPR2025/interpretability/learning_visual_composition_through_improved_semantic_guidance.md)
- [\[ICLR 2026\] Conjuring Semantic Similarity](../../ICLR2026/interpretability/conjuring_semantic_similarity.md)
- [\[CVPR 2026\] VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](../../CVPR2026/interpretability/viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)
- [\[ICML 2026\] Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning](../../ICML2026/interpretability/formal_concept_lattices_are_good_semantic_scaffolds_for_concept-based_learning.md)
- [\[ACL 2026\] On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models](../../ACL2026/interpretability/on_emergent_social_world_models_--_evidence_for_functional_integration_of_theory.md)

</div>

<!-- RELATED:END -->
