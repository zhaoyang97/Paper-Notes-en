---
title: >-
  [Paper Note] Eliminating Feature Ambiguity for Few-Shot Segmentation
description: >-
  [ECCV 2024][Segmentation][Few-Shot Segmentation] This work proposes AENet, a plug-and-play network that eliminates feature ambiguity by mining discriminative query foreground regions to enhance foreground-foreground matching in cross-attention, consistently improving the performance of existing few-shot segmentation methods (e.g., +3.0% 1-shot mIoU for SCCAN on PASCAL-5i).
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Few-Shot Segmentation"
  - "Feature Ambiguity"
  - "Cross-Attention"
  - "Discriminative Prior Mask"
  - "Plug-and-Play Network"
date: 2026-05-08
content_hash: bc1b296e107b7437
---

# Eliminating Feature Ambiguity for Few-Shot Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.09842](https://arxiv.org/abs/2407.09842)  
**Code**: Yes ([https://github.com/Sam1224/AENet](https://github.com/Sam1224/AENet))  
**Area**: Image Segmentation  
**Keywords**: Few-Shot Segmentation, Feature Ambiguity, Cross-Attention, Discriminative Prior Mask, Plug-and-Play Network

## TL;DR

This work proposes AENet, a plug-and-play network that eliminates feature ambiguity by mining discriminative query foreground regions to enhance foreground-foreground matching in cross-attention, consistently improving the performance of existing few-shot segmentation methods (e.g., +3.0% 1-shot mIoU for SCCAN on PASCAL-5i).

## Background & Motivation

Few-Shot Segmentation (FSS) aims to segment query images containing arbitrary target classes using a few annotated support samples. The core idea is to learn class-agnostic patterns: identifying regions in the query features that are similar to the support foreground (FG) features and classifying them as foreground.

**Evolution of Prior Work**:
- **Prototype-based methods** (e.g., PFENet, BAM): These compress support FG features into prototypes and compare them with query features. However, prototype compression leads to **information loss** and **spatial structure destruction**.
- **Cross-attention methods** (e.g., CyCTR, SCCAN, HDMNet): These utilize cross-attention for pixel-level matching between query and uncompressed support FG features, selectively activating features in the query that share the same class as the support FG.

**Core Problem — Feature Ambiguity**:

The authors are the first to identify an overlooked critical issue in cross-attention methods. Due to the **large receptive fields** of deep backbones (such as ResNet50 Block4), the extracted FG/BG pixel features inevitably mix with surrounding BG/FG features, especially in boundary regions between FG and BG. This leads to:

- **FG features contaminated by BG**: Query FG pixels contain mixed features of both FG (target object) and BG (background objects). For example, bird pixel features also contain fence features.
- **BG features contaminated by FG**: Support BG pixels might also contain FG information (e.g., a person at the boundary), leading to spurious high similarity with the query FG.
- **Matching failure**: Since the query FG and support FG features are contaminated by different classes of BG features, their mutual similarity decreases, lowering the cross-attention scores. Consequently, the query FG fails to sufficiently aggregate support FG information.

**Intuitive Validation**: This problem can be directly observed through the visualization of prior masks: large BG areas are erroneously activated in the FG prior $M_{Prior}^{FG}$ (due to mixing with FG features, making them similar to the support FG), while the support BG can also match the query FG in the BG prior $M_{Prior}^{BG}$.

**Key Insight**: Suppress **ambiguous regions** that are simultaneously similar to both support FG and BG (signifying heavy contamination by BG features), retain the most discriminative query FG regions, and use these "pure" FG features to recalibrate the contaminated query and support features.

## Method

### Overall Architecture

AENet is a plug-and-play network consisting of two core modules:

- **Prior Generator (PG)**: A parameter-free prior mask generation module used to locate discriminative query FG regions.
- **Ambiguity Eliminator (AE)**: A module utilizing discriminative FG features to correct query and support features.

AENet can be plugged into any cross-attention-based FSS method. Taking SCCAN as an example: the original PMA module is replaced by PG, and an AE module is inserted before each SCCA block.

### Key Designs

1. **Prior Generator (PG)**: Utilizes high-level features $F_Q^h$, $F_S^h$, and the support mask $M_S$ to generate discriminative prior masks.

    - First, support FG and BG prototypes are obtained via global average pooling: $P_S^{FG} = GAP(F_S^h, M_S)$ and $P_S^{BG} = GAP(F_S^h, 1-M_S)$.
    - Cosine similarities between query features and the two prototypes are computed and normalized to obtain $M_{Prior}^{FG}$ and $M_{Prior}^{BG}$.
    - **Key Operation — Truncated Subtraction**: $M_{Prior}^{Disc} = ReLU(M_{Prior}^{FG} - M_{Prior}^{BG})$
    - **Design Motivation**: Regions that are highly similar to both support FG and BG (i.e., ambiguous regions) yield values close to 0 or negative after subtraction, which are then truncated by ReLU. The remaining highly activated regions are the truly discriminative FG regions with minimal BG contamination.
    - Finally, $M_{Prior}^{FG}$ and $M_{Prior}^{Disc}$ are concatenated as the final prior mask, where the former provides coarse FG localization and the latter offers discriminative anchors.
    - **Memory Advantage**: The computational complexity for each calculation is $HW \times 1$, which is significantly lower than the $HW \times HW$ in PFENet/SCCAN.

2. **Ambiguity Eliminator (AE)**: Actively corrects features using discriminative FG regions.

    - The mid-level query features $F_Q$ are projected into $K$ and $V$ via linear layers, while support features $F_S$ are projected into $Q$.
    - PG computes the discriminative mask $M^{Disc}$, which is supervised with an auxiliary BCE loss: $\mathcal{L}_{aux} = BCE(M^{Disc}, M_Q)$.
    - The discriminative query FG prototype is extracted via matrix multiplication: $P_Q^{FG} = Softmax(M^{Disc}) \otimes V$.
    - Compute the cosine similarity between support and query FG prototypes: $\alpha = (Cosine(P_S^{FG}, P_Q^{FG}) + 1) / 2$.
    - A weighted fusion generates the comprehensive FG prototype: $P^{FG} = \alpha \cdot P_S^{FG} + (1-\alpha) \cdot P_Q^{FG}$.
    - $P^{FG}$ is expanded, concatenated with query/support features, and corrected through a linear layer: $F_* = Linear(F_* \| P^{FG})$.
    - **Design Motivation**: $P^{FG}$ integrates the purest support and query FG information. Concatenating it with the original contaminated features increases the proportion of FG information in the mixed features, thereby enhancing FG-FG matching.

3. **Transformer Wrapper**: The AE module is wrapped by Transformer blocks, and the corrected query and support features are output and fed into subsequent cross-attention blocks.

### Loss & Training

Taking SCCAN as an example, the total loss is:

$$\mathcal{L} = Dice(\hat{M}_Q, M_Q) + \lambda \cdot \frac{1}{N} \sum_{i=1}^{N} BCE(M_i^{Disc}, M_Q)$$

where $\lambda=1$ and $N$ represents the number of attention blocks. The main loss is the Dice loss (keeping the original baseline unchanged), and the auxiliary loss is the BCE loss applied to the discriminative mask output from each AE module.

## Key Experimental Results

### Main Results

Results on PASCAL-5i with ResNet50 backbone (mIoU%):

| Method | 1-shot Mean | Gain | 5-shot Mean | Gain |
|------|------------|------|------------|------|
| CyCTR | 64.2 | - | 65.6 | - |
| CyCTR + AENet | **69.0** | +4.8 | **72.6** | +7.0 |
| SCCAN | 66.8 | - | 70.3 | - |
| SCCAN + AENet | **69.8** | +3.0 | **74.1** | +3.8 |
| HDMNet | 69.4 | - | 71.8 | - |
| HDMNet + AENet | **70.3** | +0.9 | **74.2** | +2.4 |

Results on COCO-20i with ResNet50 backbone (mIoU%):

| Method | 1-shot Mean | Gain | 5-shot Mean | Gain |
|------|------------|------|------------|------|
| CyCTR | 40.3 | - | 45.6 | - |
| CyCTR + AENet | **47.0** | +6.7 | **52.4** | +6.8 |
| SCCAN | 46.3 | - | 53.9 | - |
| SCCAN + AENet | **49.4** | +3.1 | **56.7** | +2.8 |
| HDMNet | 49.6 | - | 55.3 | - |
| HDMNet + AENet | **51.3** | +1.7 | **57.1** | +1.8 |

### Ablation Study

Component ablation (PASCAL-5i, ResNet50, 1-shot):

| PG | AE | BAM | Mean mIoU | Gain |
|----|----|----|-----------|------|
| ✗ | ✗ | ✗ | 66.8 | baseline |
| ✓ | ✗ | ✗ | 67.8 | +1.0 |
| ✗ | ✓ | ✗ | 67.9 | +1.1 |
| ✓ | ✓ | ✗ | 68.3 | +1.5 |
| ✓ | ✓ | ✓ | **69.8** | +3.0 |

Importance of subtraction in AE:

| AE Configuration | Mean mIoU | Description |
|--------|-----------|------|
| No AE | 66.8 | baseline |
| $M^{FG}$ (without subtraction) | 66.9 | Using only FG info is largely ineffective |
| $M^{Disc}$ (with subtraction) | **67.9** | Subtraction operation is the key |

### Key Findings

- **AENet achieves larger gains on more challenging datasets**: CyCTR achieves a 6.7% gain on COCO-20i, far exceeding the 4.8% on PASCAL-5i. This is because COCO images contain more small objects and complex backgrounds, making the feature ambiguity issue more severe.
- **Subtraction is the core operation**: Correcting features with only FG information without subtraction provides almost no improvement (66.8% $\rightarrow$ 66.9%), because the model learns class-specific decoupling patterns, failing to generalize to novel classes. Subtraction provides class-agnostic guidance.
- **The optimal loss weight is $\lambda=1$**: Even with $\lambda=0$ (no auxiliary supervision), the mIoU already reaches 69%+, indicating that the feature correction based on the discriminative mask is intrinsically effective.

## Highlights & Insights

- The negative impact of feature ambiguity on cross-attention matching in FSS is identified for the first time, providing a precise problem definition.
- The subtraction in PG is remarkably simple (parameter-free) yet highly effective, demonstrating the aesthetic of "doing subtraction" in model design.
- The plug-and-play design makes it easy to integrate into various baselines, showing strong practicality.
- The design of weighting query and support FG prototypes via $\alpha$ in AE is clever — dynamically adjusting weights based on their consistency.

## Limitations & Future Work

- When the FG object is extremely small (occupying <5% of the image), the discriminative region may not provide sufficient signals for effective feature correction.
- PG relies on prototype-level FG/BG computation, which might not be fine-grained enough when multiple instances exist in the support image.
- Exploring adaptation to 4D correlation-based methods (such as HSNet, VAT) is worth investigating.
- Multi-scale discriminative mask fusion could be considered to exploit the complementarity of different feature levels.

## Related Work & Insights

- The prior mask concept from PFENet is elegantly extended — from merely computing FG similarity to considering the difference between FG and BG similarities.
- The self-calibrated cross-attention in SCCAN is complementary to AENet — the former handles misaligned BG feature matching, while the latter addresses FG features contaminated by BG.
- The class-agnostic nature of the subtraction operation warrants exploration in other few-shot tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to identify the feature ambiguity problem; the subtraction operation is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three baselines $\times$ two datasets $\times$ two backbones $\times$ detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem statement and thorough visualization validation.
- **Value**: ⭐⭐⭐⭐ The plug-and-play design is highly practical, yielding consistent improvements across multiple baselines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[ECCV 2024\] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [\[ECCV 2024\] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures](representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)

</div>

<!-- RELATED:END -->
