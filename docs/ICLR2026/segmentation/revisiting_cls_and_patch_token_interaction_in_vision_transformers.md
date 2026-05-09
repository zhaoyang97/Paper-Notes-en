---
title: >-
  [Paper Note] Revisiting [CLS] and Patch Token Interaction in Vision Transformers
description: >-
  [ICLR 2026][Segmentation][Vision Transformer] This paper systematically analyzes the interaction friction between the global [CLS] token and local patch tokens in Vision Transformers. It reveals that normalization layers implicitly differentiate between the two token types, and proposes specialized processing paths in normalization layers and early QKV projections. With only an 8% parameter increase, the method achieves over 2 mIoU improvement in segmentation while preserving classification accuracy.
tags:
  - ICLR 2026
  - Segmentation
  - Vision Transformer
  - "CLS] token"
  - patch token
  - normalization layer
  - dense prediction
date: 2026-05-08
content_hash: 706dff0cf48986ff
---

# Revisiting [CLS] and Patch Token Interaction in Vision Transformers

**Conference**: ICLR 2026
**arXiv**: [2602.08626](https://arxiv.org/abs/2602.08626)
**Code**: N/A
**Area**: Image Segmentation / Vision Transformer
**Keywords**: Vision Transformer, [CLS] token, patch token, normalization layer, dense prediction

## TL;DR
This paper systematically analyzes the interaction friction between the global [CLS] token and local patch tokens in Vision Transformers. It reveals that normalization layers implicitly differentiate between the two token types, and proposes specialized processing paths in normalization layers and early QKV projections. With only an 8% parameter increase, the method achieves over 2 mIoU improvement in segmentation while preserving classification accuracy.

## Background & Motivation
Vision Transformers (ViT) have emerged as powerful, scalable, and general-purpose visual representation learners. In the standard ViT architecture, a learnable [CLS] token is prepended to the patch token sequence to aggregate global information for classification. Despite the fundamentally different semantic roles of [CLS] and patch tokens—the former capturing global features and the latter encoding local spatial information—both are processed **identically** throughout the model: through the same attention layers, the same FFN, and the same normalization layers.

This uniform treatment introduces a fundamental friction:

**Global–Local Competition**: The [CLS] token must aggregate global semantics from all patches, while each patch token must retain its local spatial information. These two objectives may interfere with each other within shared attention computations.

**Implicit Bias of Normalization Layers**: Standard LayerNorm/RMSNorm normalizes the entire token sequence jointly, yet the statistical properties (mean and variance) of [CLS] and patch tokens may differ fundamentally. Unified normalization may prevent either type from achieving its optimal representation.

**Degraded Dense Prediction Performance**: When ViT is applied to dense prediction tasks such as segmentation and detection—which require high-quality patch representations—the aforementioned friction degrades patch representation quality.

Core Observation: By analyzing the behavior of normalization layers in ViT, the authors find that normalization layers already **implicitly** differentiate between [CLS] and patch tokens, with systematic differences in normalization statistics between the two types. Given that implicit differentiation already exists, the paper asks whether **explicit specialization** can amplify this effect to simultaneously improve both global and local representations.

## Method

### Overall Architecture
The proposed modifications constitute a **surgical refinement** of the existing ViT architecture—introducing separate processing only at specific critical modules rather than altering the overall structure:
- **Input**: Standard ViT architectures (e.g., ViT-B/16, ViT-L/14)
- **Modified Components**: Normalization layers (LayerNorm) and early QKV projections
- **Output**: [CLS] token (for classification) + patch tokens (for dense prediction)
- **Design Principle**: Minimize parameter overhead (~8%) with zero additional computational cost

### Key Designs

1. **Implicit Token Differentiation in Normalization**: This constitutes the paper's core analytical contribution. The authors examine the behavior of normalization layers in ViTs pretrained under various strategies (supervised, DINO, DINOv2, MAE, etc.):

    - **Observation 1**: Within standard LayerNorm, the normalization statistics (mean and variance) of [CLS] and patch tokens exhibit systematic differences—the statistics of [CLS] typically deviate from the average of patch tokens.
    - **Observation 2**: This discrepancy becomes more pronounced in deeper layers, indicating that the representation spaces of the two token types progressively diverge with network depth.
    - **Observation 3**: The normalization operation effectively "flattens" this existing discrepancy, potentially suppressing the independent development of optimal representations for each type.
    - **Conclusion**: Since the two token types are fundamentally distinct, assigning each its own independent normalization parameters (or even independent normalization statistics) may be more effective.

2. **Specialized Normalization Paths**: Based on the above findings, independent normalization processing is designed for [CLS] and patch tokens:

    - Within the normalization layer of each Transformer block, the token sequence is partitioned into the [CLS] portion and the patch portion.
    - The [CLS] token uses its own normalization parameters (independent scale and shift).
    - Patch tokens use a separate set of normalization parameters.
    - This allows [CLS] to develop a normalization scale optimized for global aggregation, while patches develop one optimized for local detail preservation.
    - Critically, this modification introduces no additional computational overhead—the complexity of the normalization operation remains unchanged after parameter separation.

3. **Early QKV Projection Specialization**: Beyond normalization layers, the authors find that introducing specialization in the QKV (Query-Key-Value) projections of attention layers is similarly beneficial:

    - In the early layers only (rather than all layers), separate QKV projection matrices are used for [CLS] and patch tokens.
    - This enables the Query of [CLS] in early layers to specialize in "how to query for global information," while the Query of patch tokens specializes in "how to interact with neighboring patches to maintain spatial coherence."
    - The motivation for applying this modification to early layers is that representation differentiation is not yet pronounced in early layers; specialized QKV projections help establish separate representation pathways earlier. In deeper layers, tokens are already well-differentiated and shared QKV suffices.
    - Parameter analysis: A separate, small QKV projection matrix is added for [CLS] in each modified layer; since there is only one [CLS] token, the parameter overhead is negligible.

### Loss & Training
The proposed modifications integrate seamlessly into any ViT pretraining or fine-tuning pipeline:
- Modifications can be applied during **pretraining** (e.g., training a specialized ViT within the DINOv2 framework) so that the model learns differentiated representations from the outset.
- Alternatively, modifications can be introduced during **fine-tuning** to adapt an already pretrained standard ViT.
- The loss function follows the original training framework (e.g., DINO's self-distillation loss, MAE's reconstruction loss).
- No additional loss terms or hyperparameters are introduced.

## Key Experimental Results

### Main Results
On standard segmentation benchmarks, the specialization modifications yield consistent and significant improvements:

| Task / Dataset | Metric | Standard ViT | Specialized ViT | Gain |
|---|---|---|---|---|
| Semantic Segmentation (ADE20K) | mIoU | baseline | +2+ mIoU | > 2 mIoU |
| Semantic Segmentation (other benchmarks) | mIoU | baseline | consistent gain | > 2 mIoU |
| Image Classification (ImageNet) | Top-1 Acc | baseline | on par or slightly higher | no classification loss |

Key conclusion: A gain of over 2 mIoU is a substantial improvement in segmentation, while classification accuracy is maintained (or marginally improved), demonstrating that specialization does not trade classification performance for segmentation gains.

### Ablation Study

| Configuration | Segmentation | Classification | Param. Increase | Note |
|---|---|---|---|---|
| Normalization specialization only | significant gain | on par | ~4% | core contribution |
| QKV specialization only | moderate gain | on par | ~4% | complementary contribution |
| Normalization + QKV | best | on par or slightly higher | ~8% | two components are complementary |
| All-layer QKV vs. early-layer only | early-layer superior | — | — | deep layers benefit from shared QKV |
| Different model scales | consistent gain | consistent | — | effective across ViT-S/B/L |
| Different training frameworks | consistent gain | consistent | — | effective across supervised/self-supervised |

### Key Findings
- **Normalization layers are the primary source of friction**: Specialization applied solely to normalization layers recovers the majority of the performance gain, confirming that unified normalization is a critical bottleneck in the interaction between the two token types.
- **Early QKV specialization provides complementary gains**: Adding early QKV specialization on top of normalization specialization yields further improvement, indicating that attention computation is also a source of friction.
- **Only early layers require QKV specialization**: The marginal benefit of QKV specialization diminishes in deeper layers, suggesting that as network depth increases, the representation pathways of the two token types are already sufficiently differentiated through normalization specialization.
- **Generalizes across model scales and training frameworks**: The method proves effective across settings ranging from ViT-S to ViT-L and from supervised training to DINOv2/MAE, demonstrating its generality as an architectural improvement.
- **High parameter efficiency**: An 8% parameter increase yields over 2 mIoU in segmentation improvement with no additional inference FLOPs.

## Highlights & Insights
- **Analysis-driven design**: Rather than empirically testing various modifications, the paper starts from a statistical analysis of normalization layers, identifies the implicit differentiation phenomenon, and then designs targeted specialization accordingly.
- **Principle of minimal intervention**: Separation is introduced only in normalization and early QKV projections—the smallest set of modifications that produces the largest impact. Other components (FFN, residual connections) remain shared, avoiding over-engineering.
- **Win-win: preserved classification with improved dense prediction**: This demonstrates that the classification performance of standard ViT does not depend on high-quality patch representations (classification uses only [CLS]), whereas dense prediction is critically dependent on patch quality—hence specialization yields far greater gains for segmentation than for classification.
- **Normalization layers as an "information bottleneck": a new perspective**: The paper reveals a frequently overlooked fact—normalization layers are not merely training stabilizers; they also implicitly shape the representation development pathways of different token types.

## Limitations & Future Work
- Validation is currently limited to segmentation and classification; performance on other dense prediction tasks (e.g., depth estimation, optical flow) and detection remains unexplored.
- No in-depth analysis is provided on how the geometry of the two token types' representations changes after specialization—e.g., isotropy of the representation space, clustering structure, etc.
- The method does not directly apply to ViT variants without a [CLS] token (e.g., architectures using mean pooling only).
- Whether analogous token-type friction exists in non-ViT Transformer architectures (e.g., Swin Transformer with window attention) has not been explored.
- Although an 8% parameter increase is small in relative terms, it represents a non-trivial absolute count for very large-scale models (e.g., ViT-G), and effectiveness at extreme scales remains to be verified.
- Theoretical analysis explaining why normalization layers—rather than FFN or attention—constitute the critical bottleneck is absent.

## Related Work & Insights
- **DINOv2**: The proposed specialization scheme can be directly integrated into the DINOv2 pretraining pipeline to enhance the dense prediction capability of pretrained models.
- **ViT-Adapter**: Improves dense prediction via external adapter modules, contrasting with the paper's "internal specialization" approach—one adds external modules while the other introduces internal differentiation.
- **Register Tokens**: Recent work adds extra semantics-free tokens to ViT to absorb noisy attention artifacts, complementing the role of [CLS] and potentially related to the friction issue identified in this paper.
- **Layer-by-layer, module-by-module (concurrent work)**: Analyzes ViT internal modules from the perspective of OOD linear probing, complementing the module-level analytical viewpoint of this paper.
- Directions inspired by this work: Could patch tokens be further subdivided into specialized groups—e.g., foreground versus background patches?

## Rating
- Novelty: ⭐⭐⭐⭐ (The discovery of implicit differentiation in normalization layers is insightful, though the specialization solution itself is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive ablation studies; validated across multiple scales and frameworks)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear exposition; the logical chain from motivation → analysis → design → validation is complete)
- Value: ⭐⭐⭐⭐⭐ (8% parameter overhead for 2+ mIoU; highly practical and readily adoptable by the ViT community)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)
- [\[ICLR 2026\] Locality-Attending Vision Transformer](locality-attending_vision_transformer.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](../../CVPR2026/segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)

<!-- RELATED:END -->
