---
title: >-
  [Paper Note] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification
description: >-
  [CVPR 2026][Interpretability][Interpretable Classification] This paper proposes DINO-QPM, a lightweight interpretability adapter that transforms the complex…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Interpretable Classification"
  - "DINOv2"
  - "Quadratic Programming"
  - "Visual Foundation Models"
  - "Feature Sparsification"
date: 2026-05-08
content_hash: 1d9ec5d76fcbb8b6
---

# DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

**Conference**: CVPR 2026
**arXiv**: [2604.07166](https://arxiv.org/abs/2604.07166)
**Code**: [https://github.com/RobertZimm/DINO-QPM](https://github.com/RobertZimm/DINO-QPM)
**Area**: Model Interpretability
**Keywords**: Interpretable Classification, DINOv2, Quadratic Programming, Visual Foundation Models, Feature Sparsification

## TL;DR

This paper proposes DINO-QPM, a lightweight interpretability adapter that transforms the complex, high-dimensional features of a frozen DINOv2 backbone into contrastive, class-agnostic interpretable representations. Through quadratic programming for sparse feature selection and class-level feature assignment, the method simultaneously surpasses DINOv2 linear probing in accuracy and all comparable methods in interpretability on CUB-2011 and Stanford Cars.

## Background & Motivation

Visual foundation models such as DINOv2 excel as feature extractors, yet their complex, high-dimensional representations pose substantial challenges for interpretability. Existing approaches suffer from the following limitations:

**Post-hoc explanation methods are unreliable**: Attention maps, Grad-CAM, and similar techniques are external approximations that do not faithfully reflect the model's decision process; attention maps are decoupled from downstream tasks and frequently omit information critical for classification.

**End-to-end interpretable models are resource-intensive**: Methods such as prototype networks require full backbone fine-tuning, incurring prohibitive computational cost.

**Frozen-backbone methods lack accuracy**: Post-hoc Concept Bottleneck Models (Post-hoc CBMs) rely on textual concept supervision and cannot provide direct spatial localization, with accuracy typically lagging behind full fine-tuning approaches.

**Prototype model interpretability can be misleading**: Their similarity computations do not necessarily align with human cognition.

Core Problem: Can one construct a lightweight adapter on top of a fully frozen DINOv2 backbone that converts its powerful but entangled features into sparse, spatially localizable, and globally interpretable class representations?

## Method

### Overall Architecture

The DINO-QPM pipeline:
1. A frozen DINOv2 backbone extracts patch embeddings $\boldsymbol{F}^{\text{froz}} \in \mathbb{R}^{N_p \times D}$.
2. An MLP projects patch embeddings into a task-specific feature space $\boldsymbol{F} \in \mathbb{R}^{N_p \times N_f}$.
3. Average pooling produces a feature vector $\boldsymbol{f} = \text{AvgPool}(\boldsymbol{F}) \in \mathbb{R}^{N_f}$.
4. A Binary Low-Dimensional Decision layer (BLDD) performs sparse feature assignment for classification.

Key design choice: **the CLS token is discarded in favor of patch embeddings exclusively**. This ensures that each feature has a corresponding spatial feature map, enabling high-fidelity spatial localization.

### Key Designs

1. **MLP Feature Transformation**:

    - Function: Maps DINOv2's $D$-dimensional patch representations to $N_f$ task-specific features.
    - Core Idea: The BLDD layer is a binary sparse matrix with no representational transformation capacity; the MLP therefore performs the necessary feature transformation upstream.
    - Ablation Finding: The number of MLP layers has little effect on the dense model, but is critical for the sparse QPM — QPM with multiple MLP layers can outperform the dense model by nearly 10%.

2. **Quadratic Programming Feature Selection (QP)**:

    - Function: Selects $N_f^* = 50$ features from $N_f$ candidates and assigns $N_f^c = 5$ features per class.
    - Core Idea: Three objectives are jointly optimized — maximizing the correlation between each class and its assigned feature activations, minimizing similarity among selected features, and maximizing bias to prioritize local features.
    - Design Motivation: Feature selection via mathematical programming rather than learning ensures diversity, contrastiveness, and compactness of the feature set.

3. **Average Pooling for Spatial Localization**:

    - Function: Replaces standard CLS token aggregation with average pooling.
    - Core Idea: Each dimension of the feature vector is the spatial average of the corresponding feature map, so feature maps can be directly upsampled to the original image resolution as saliency maps.
    - Experimental Validation: On DINOv2 with register tokens, using patch embeddings alone achieves 88.3% accuracy, surpassing the CLS-based approach (87.6%).

4. **Feature Map Sparsity Loss $\mathcal{L}_{\text{L1-FM}}$**:

    - Function: Applies L1 regularization to feature maps.
    - Core Idea: Compels feature activations to concentrate on object regions relevant to classification, suppressing background noise and spatial diffusion.
    - Surprising Finding: This loss not only improves Plausibility but also substantially increases classification accuracy, indicating a strong positive correlation between the two.

### Loss & Training

Three-stage training procedure:
1. **Dense Training**: Train with cross-entropy $\mathcal{L}_{\text{CE}}$, feature diversity loss $\mathcal{L}_{\text{div}}$, and L1 sparsity loss.
2. **Quadratic Programming**: Solve the QP to determine the feature selection vector $\boldsymbol{s}$ and sparse weights $\boldsymbol{W}^{\text{sparse}}$.
3. **Sparse Fine-tuning**: Fix $\boldsymbol{W}^{\text{sparse}}$ and fine-tune only on the selected features.

Total loss: $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{L1-FV}} \mathcal{L}_{\text{L1-FV}} + \lambda_{\text{L1-FM}} \mathcal{L}_{\text{L1-FM}}$

## Key Experimental Results

### Main Results

| Method | CUB Acc↑ | CARS Acc↑ | CUB Plausib.↑ | Contrast.↑ |
|------|---------|----------|--------------|-----------|
| DINOv2 CLS Linear Probe | 87.9 | 91.7 | 42.6 | 59.2 |
| Dense $\boldsymbol{F}^{\text{froz}}$ | 78.1 | 92.9 | 32.7 | 84.5 |
| ResNet50 QPM | 82.9 | 92.1 | 82.9 | 93.6 |
| DINO-SLDD | 84.6 | 92.9 | 78.0 | 93.0 |
| DINO-QSENN | 85.4 | 93.3 | 86.0 | 94.4 |
| **DINO-QPM (Ours)** | **88.3** | **94.0** | **95.0** | **100.0** |

DINO-QPM surpasses the non-interpretable DINOv2 linear probe in accuracy (88.3 vs. 87.9), while Plausibility improves dramatically from 42.6 to 95.0.

### Ablation Study

| Configuration | CUB Acc (%) | Note |
|------|-----------|------|
| CLS + no register | 87.3 | CLS token lacks spatial localization |
| CLS + register | 87.6 | Register improves CLS |
| Patch + no register | 83.3 | Patch representations degrade without register |
| **Patch + register** | **88.3** | Register tokens are critical |

| Backbone Size | CUB Acc (%) | Patch Contextualization↑ |
|---------|-----------|------------------------|
| DINO ViT-B/16 | 37.1 | 8.9 |
| DINOv2 ViT-S/14 Reg | 83.4 | 42.9 |
| DINOv2 ViT-B/14 Reg | 88.3 | 43.9 |
| DINOv2 ViT-L/14 Reg | 86.5 | 2.2 |

### Key Findings

1. **Register tokens are essential**: Without register tokens, the spatial information quality of patch embeddings deteriorates, causing approximately 5% accuracy drop. Register tokens prevent patches from acting as "artifact tokens" that store global context.
2. **Plausibility and accuracy are strongly correlated**: The $\mathcal{L}_{\text{L1-FM}}$ loss jointly improves both metrics, indicating that directing model attention toward object regions is intrinsically beneficial for classification.
3. **The compactness–accuracy trade-off is minimal**: Reducing features per class from 5 to 4 (Compact variant) incurs negligible accuracy loss.
4. **Illustrative bird classification case**: When distinguishing Brewer's Blackbird from Rusty Blackbird, the model automatically localizes to the beak region — precisely the discriminative cue used by ornithological experts.

## Highlights & Insights

1. **Non-interpretable models are not necessarily more accurate**: DINO-QPM, operating under the extreme sparsity constraint of 50 total features with only 5 per class, outperforms linear probing that uses 768 features.
2. **Counter-intuitive architectural choice**: Discarding the CLS token — the standard choice in classification literature — yields superior results, as globally interpretable representations constructed directly from local evidence are both more interpretable and more effective than opaque pre-aggregated representations.
3. **Exceptional training efficiency**: Because the backbone is fully frozen, patch embeddings can be precomputed, reducing per-epoch training time to approximately 6 seconds.
4. **Well-designed Plausibility metric**: The use of dilated masks to handle patch boundary effects prevents unfair penalization of activations on precise object contours.

## Limitations & Future Work

1. Validation is limited to fine-grained classification benchmarks (CUB-2011, Stanford Cars); generalization to generic image classification remains to be tested.
2. Performance degrades with the ViT-L backbone, suggesting that adapter design may need to be tailored to different backbone scales.
3. The quadratic programming feature selection is performed once and does not adapt dynamically during training, potentially limiting the discovery of optimal feature combinations.
4. The fixed allocation of 5 features per class does not account for varying class complexity, which may warrant adaptive feature budgets.

## Related Work & Insights

- **QPM / ChiQPM**: End-to-end trainable interpretable models based on quadratic programming; this work transfers the paradigm to frozen backbones.
- **Post-hoc CBM**: Achieves interpretable classification via textual concepts, but depends on external language models and lacks spatial localization.
- **ProtoViT / Zhu et al.**: Prototype-based interpretable methods that require backbone fine-tuning for prototype clustering.
- Insight: The paradigm of frozen backbone + lightweight adapter as an efficient interpretable solution merits further exploration across a broader range of visual tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Transferring QPM to frozen visual foundation models is a natural yet effective contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Ablations are comprehensive, metrics are rigorously designed, and cross-backbone validation is complete.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear, and visualizations are outstanding (the bird feature localization case study is particularly compelling).
- Value: ⭐⭐⭐⭐ — Provides a strong tool for interpretable classification with frozen foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](../../NeurIPS2025/interpretability/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](language_models_can_explain_visual_features_via_steering.md)
- [\[CVPR 2026\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](../../ACL2026/interpretability/llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)

</div>

<!-- RELATED:END -->
