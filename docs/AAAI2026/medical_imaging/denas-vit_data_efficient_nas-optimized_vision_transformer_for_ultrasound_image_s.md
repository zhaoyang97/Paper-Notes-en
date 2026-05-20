---
title: >-
  [Paper Note] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation
description: >-
  [AAAI 2026][Medical Imaging][Neural Architecture Search] DeNAS-ViT is proposed as the first method to apply NAS at the token level within ViT for optimizing multi-scale feature extraction in ultrasound image segmentation…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Neural Architecture Search"
  - "Vision Transformer"
  - "Ultrasound Segmentation"
  - "Semi-Supervised Learning"
  - "Token-Level Search"
date: 2026-05-08
content_hash: 6746cc8b6ab7b581
---

# DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation

**Conference**: AAAI 2026
**arXiv**: [2407.04203](https://arxiv.org/abs/2407.04203)  
**Code**: Unavailable  
**Area**: Medical Imaging
**Keywords**: Neural Architecture Search, Vision Transformer, Ultrasound Segmentation, Semi-Supervised Learning, Token-Level Search

## TL;DR
DeNAS-ViT is proposed as the first method to apply NAS at the token level within ViT for optimizing multi-scale feature extraction in ultrasound image segmentation. A NAS-constrained semi-supervised learning framework is designed incorporating network independence loss, hierarchical contrastive loss, and staged optimization, achieving state-of-the-art performance under limited annotation.

## Background & Motivation

**Background**: Ultrasound image segmentation is critical for cardiac disease diagnosis. Deep learning methods (UNet++, TransUNet, EfficientViT) have shown progress but rely on manually designed architectures. NAS can automate architecture optimization; however, existing NAS methods search only at the module level (selecting between convolution or Transformer blocks), neglecting finer-grained operations within modules.

**Limitations of Prior Work**:
- Manual architecture design yields limited gains when prior knowledge is insufficient
- Ultrasound annotation data is scarce, whereas NAS inherently requires large amounts of data
- Module-level NAS search has low precision and tends to select the most complex operations (e.g., ViT), rendering the search meaningless

**Key Challenge**: NAS requires abundant data to search for optimal architectures, yet ultrasound segmentation suffers from precisely this scarcity of labeled data — how can the conflicting data demands of the two be reconciled?

**Goal**: Design an efficient token-level NAS framework combined with a data-efficient semi-supervised learning strategy.

**Key Insight**: Apply NAS prior to the attention computation in ViT — searching for optimal multi-scale token representations (rather than entire modules) — while using NAS-guided semi-supervised learning to reduce dependence on labeled data.

**Core Idea**: Token-level NAS search within ViT + NAS-constraint-driven semi-supervised co-training.

## Method

### Overall Architecture
- **Efficient NAS-ViT Module**: Performs NAS search over Q/K/V tokens to find optimal multi-scale representations
- **NAS Backbone**: Encoder (hierarchical NAS + NAS-ViT cells) + Decoder (NAS cells)
- **NAS-based SSL**: Two networks sharing the NAS backbone architecture but with independent parameters perform co-training, augmented by independence loss and hierarchical contrastive loss

### Key Designs

1. **Efficient NAS-ViT Module**:

    - **Function**: Searches for optimal multi-scale token representations prior to attention computation
    - **Mechanism**: Applies partial channel connections and continuous relaxation for differentiable search over Q/K/V tokens: $\{Q'/K'/V'\} = (1-P) \odot \{Q/K/V\} + \sum_{O_i} \frac{\exp(\alpha_i)}{\sum_j \exp(\alpha_j)} O_i(P \odot \{Q/K/V\})$
    - **Design Motivation**: DAST treats the entire ViT layer as a candidate operation, resulting in high computational cost and a degenerate search (the model consistently selects the most complex ViT variant). Token-level search provides finer granularity and reduces parameter overhead.

2. **Hierarchical NAS Backbone**:

    - Encoder: Cell-level search (NAS-ViT handling multi-scale tokens) + layer-level search (softmax-weighted aggregation across paths of different resolutions)
    - Decoder: U-shaped structure with independent NAS cell search at each layer
    - Six resolution scales ($r=1,2,4,8,16,32$) to capture multi-scale features

3. **NAS-based Constraint-Driven Semi-Supervised Learning**:

    - **Network Independence Loss**: Encourages the two networks to learn complementary representations, measured by the cosine similarity of convolutional layer weights
    - **Hierarchical Contrastive Loss**: Computes uncertainty over multi-resolution outputs of decoder NAS cells, aligning lower-quality features toward higher-quality features
    - **Staged Optimization**: Sequentially introduces different constraints to ensure training stability

### Loss & Training
$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{uns} + \mathcal{L}_{ind} + \mathcal{L}_{con}$. Architecture parameters $\alpha, \beta, \gamma$ and network weights $\theta$ are optimized via DARTS-style bilevel optimization.

## Key Experimental Results

### Main Results (3 datasets, 100% labels)

| Method | HMC-QU DSC | CAMUS DSC | CETUS DSC |
|--------|-----------|-----------|-----------|
| UNet++ | 0.899 | 0.919 | 0.952 |
| nnU-Net | 0.908 | 0.922 | 0.958 |
| TransFuse | 0.903 | 0.923 | 0.957 |
| Se2NAS† | 0.907 | 0.920 | 0.955 |
| **DeNAS-ViT† (Ours)** | **~0.920** | **~0.930** | **~0.962** |

(† denotes SSL usage). DeNAS-ViT outperforms all baselines across all datasets, including both NAS and SSL methods.

### Ablation Study
- Efficient NAS-ViT vs. DAST (module-level search): fewer parameters with superior performance
- Contribution of each SSL constraint: independence loss and contrastive loss each contribute approximately 0.5–1% DSC improvement
- Remains competitive under 10% labeled data

### Key Findings
- Token-level search is more efficient and generalizes better than module-level search
- Combining NAS and SSL yields greater gains than either approach alone
- The method also performs well on non-ultrasound datasets, demonstrating generalizability

## Highlights & Insights
- The granularity choice of **token-level NAS search** is elegant — finer than module-level yet less expensive than pixel-level, operating precisely at the Q/K/V level within ViT
- The **natural synergy between NAS and SSL** — architectures produced by NAS naturally provide the diversity required for co-training, and the two paradigms mutually reinforce each other
- The **staged optimization strategy** reflects important engineering insight — directly joint-optimizing all objectives tends to be unstable in practice

## Limitations & Future Work
- The NAS search process itself is computationally expensive; search time is not thoroughly discussed in the paper
- Validation is limited to 2D ultrasound; extension to 3D volumetric data remains unexplored
- The semi-supervised component involves numerous hyperparameters (loss weights, stage transition timing)

## Related Work & Insights
- **vs. EfficientViT**: EfficientViT uses fixed multi-scale convolutions for token processing; DeNAS-ViT employs NAS to search for optimal token representations — a more flexible approach
- **vs. DAST**: In DAST's module-level search, ViT is consistently selected; DeNAS-ViT's token-level search avoids this degeneracy
- **vs. Se2NAS**: Se2NAS also combines NAS and SSL but without additional constraints; DeNAS-ViT's independence and contrastive losses yield superior performance

## Rating
- Novelty: ⭐⭐⭐⭐ Token-level NAS + unified NAS-SSL framework, first applied to ultrasound segmentation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets + 12 baselines + comprehensive ablation + generalization validation
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear; method description is systematic
- Value: ⭐⭐⭐⭐ Provides an effective NAS+SSL solution for data-scarce medical image segmentation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](../../CVPR2026/medical_imaging/medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](../../CVPR2026/medical_imaging/visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)

</div>

<!-- RELATED:END -->
