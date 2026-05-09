---
title: >-
  [Paper Note] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration
description: >-
  [CVPR 2026][Self-Supervised Learning] This paper proposes Mine-JEPA, the first in-domain self-supervised learning (SSL) pipeline for side-scan sonar (SSS) mine classification. Built upon SIGReg regularization loss, sonar-adapted augmentation strategies, and ImageNet initialization, Mine-JEPA pretrained on only 1,170 unlabeled sonar images surpasses DINOv3—a foundation model pretrained on 1.7 billion images.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - side-scan sonar
  - mine classification
  - in-domain pretraining
  - SIGReg
date: 2026-05-08
content_hash: 965afcb7c08d9601
---

# MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration

**Conference**: CVPR 2026
**arXiv**: [2604.00383](https://arxiv.org/abs/2604.00383)
**Code**: None
**Area**: Self-Supervised Learning / Underwater Sonar Imagery
**Keywords**: Self-supervised learning, side-scan sonar, mine classification, in-domain pretraining, SIGReg

## TL;DR

This paper proposes Mine-JEPA, the first in-domain self-supervised learning (SSL) pipeline for side-scan sonar (SSS) mine classification. Built upon SIGReg regularization loss, sonar-adapted augmentation strategies, and ImageNet initialization, Mine-JEPA pretrained on only 1,170 unlabeled sonar images surpasses DINOv3—a foundation model pretrained on 1.7 billion images.

## Background & Motivation

Side-scan sonar (SSS) is widely used for seabed surveying and mine detection, forming a core technology in mine countermeasures (MCM). The field faces three major challenges:

**Extreme data scarcity**: Public datasets contain only 1,170 sonar images (with 668 annotated targets), making annotation acquisition exceedingly difficult.

**Large domain gap**: Sonar images are formed from acoustic echoes, differing fundamentally from RGB natural images in imaging mechanism and texture statistics—color information is nearly meaningless, and the key cues lie in acoustic highlight/shadow regions and seabed texture.

**Large models are not always sufficient**: While it is intuitive to transfer large-scale pretrained models such as DINOv3, the domain specificity of sonar imagery renders this strategy unreliable.

**Core research question**: In an extreme low-data sonar setting, *can carefully designed in-domain SSL replace large-scale general-purpose foundation models?*

## Method

### Overall Architecture

Mine-JEPA follows a three-stage pipeline:

**Stage 1 (Data Preparation)** → Sliding window extraction (stride 64) of 96×96 patches from 1,170 SSS images, yielding approximately 153K unlabeled patches.
**Stage 2 (In-Domain SSL Pretraining)** → ViT pretraining using SIGReg loss + SSS-adapted augmentations + ImageNet-1K initialization.
**Stage 3 (Probe Evaluation)** → Classification evaluation with frozen/fine-tuned backbone and linear/MLP head.

### Key Designs

1. **SIGReg Self-Supervised Loss (Core SSL Objective)**

    - Derived from the LeJEPA framework; requires no teacher–student architecture, momentum encoder, or large batch size.
    - Combines an invariance loss and a distributional regularization loss: $\mathcal{L} = (1-\lambda)\mathcal{L}_{inv} + \lambda\mathcal{L}_{sig}$
    - **Invariance loss**: encourages different augmented views of the same patch to converge to similar representations.
      - $\mathcal{L}_{inv} = \frac{1}{NV}\sum_{i}\sum_{v}\|z_{i,v} - \bar{z}_i\|_2^2$
    - **SIGReg loss**: regularizes the embedding distribution toward a standard normal via Epps–Pulley goodness-of-fit statistics computed through random projections.
      - Prevents representational collapse without requiring negative pairs or an EMA teacher.
      - Complexity $O(N)$, well-suited for small-data, small-batch settings.
    - **Design Motivation**: In the extreme low-data regime of only 1,170 source images, the simplest and most stable SSL objective is required.

2. **Ultra-Low Projection Dimensionality ($d=16$)**

    - This is not an implementation detail but an **intentional structural bottleneck**.
    - A low-dimensional projection space regularizes representation learning and reduces overfitting when data is limited.
    - Contrast: SimCLR uses 128; VICReg uses 2048.
    - **Design Motivation**: Serves as implicit regularization in small-data settings.

3. **SSS-Adapted Augmentation Strategy (Key Domain Adaptation)**

    - **Removed augmentations**: hue/saturation jitter, solarization, grayscale conversion (meaningless or harmful for sonar).
    - **Retained augmentations**: horizontal flip, random resized crop (scale 0.5–1.0), Gaussian blur.
    - **Added augmentations**: vertical flip, random rotation (±15°)—reflecting directional invariance in sonar scanning geometry.
    - Dataset statistics used for normalization.
    - **Experimental evidence**: Using natural image augmentations yields a macro-F1 as low as 0.312 (worse than no pretraining at 0.557); sonar-adapted augmentations recover F1 to 0.725.

4. **Initialization Strategy**

    - The ViT backbone is initialized from **ImageNet-1K pretrained weights**; the projection head is randomly initialized.
    - **DINOv3 initialization is not used**—experiments show that in-domain SSL starting from DINOv3 actually **degrades performance by 10–13 percentage points**.
    - **Design Motivation**: ImageNet provides useful low-level visual priors (edges, textures) without over-specializing in the way DINOv3 does, thereby preserving room for domain adaptation.

5. **Mixed Data Composition (Real+Syn)**

    - In addition to approximately 153K real patches, approximately 256K synthetic sonar patches (grayscale) are incorporated, totaling approximately 409K patches.
    - Real and synthetic patches are normalized by their respective statistics and concatenated for training.
    - Regularization-based SSL (SIGReg, VICReg) benefits from heterogeneous data, whereas contrastive/distillation-based methods degrade under the same setting.

### Loss & Training

- SSL loss: SIGReg $= (1-\lambda)\mathcal{L}_{inv} + \lambda\mathcal{L}_{sig}$, with $\lambda=0.1$
- 4 views, batch size 1024, AdamW (weight decay 0.05)
- Learning rate $1.4 \times 10^{-3}$, 1-epoch warmup + cosine decay, 100 epochs
- Four probe modes for evaluation: linear / mlp (frozen backbone), finetune / ft\_mlp (fine-tuned backbone)

## Key Experimental Results

### Main Results

**3-Class Classification (BG / MILCO / NOMBO)**

| Method | Init | SSL Data | macro-F1 | NOMBO F1 | Acc |
|--------|------|----------|----------|----------|-----|
| Random Init | Random | — | 0.557 | 0.439 | 58.3% |
| IN1K Only | IN1K | — | 0.739 | 0.647 | 76.3% |
| DINOv3 | DINOv3 | — | 0.810 | 0.700 | 83.4% |
| SimCLR | IN1K | Real+Syn | 0.801 | 0.693 | 82.6% |
| VICReg | IN1K | Real+Syn | 0.800 | 0.676 | 82.8% |
| BYOL | IN1K | Real+Syn | 0.693 | 0.539 | 72.8% |
| **Mine-JEPA** | **IN1K** | **Real+Syn** | **0.820** | **0.734** | **83.8%** |

**Binary Classification (Mine vs. Non-mine) + Model Scale Comparison**

| Method | Init | Params | 3-class F1 | 2-class F1 | MILCO Recall |
|--------|------|--------|-----------|-----------|-------------|
| DINOv3 | DINOv3 | 21.5M | 0.810 | 0.922 | 88.1% |
| Mine-JEPA (ViT-S) | IN1K | 21.6M | **0.820** | **0.935** | 90.9% |
| Mine-JEPA (ViT-Tiny) | IN1K | **5.5M** | 0.814 | **0.935** | **91.4%** |

### Ablation Study

**Cumulative Effect of Initialization, Augmentation, and Data Composition**

| Configuration | Init | SSL Data | macro-F1 | Δ |
|---------------|------|----------|----------|---|
| Random Init (no pretraining) | Random | — | 0.557 | — |
| Natural image augmentation SSL | Random | Real* | 0.312 | −24.5%p |
| DINOv3 + SIGReg | DINOv3 | Real | 0.706 | −10.4%p vs DINOv3 |
| DINOv3 + SIGReg | DINOv3 | Real+Syn | 0.677 | −13.3%p vs DINOv3 |
| SSS augmentation SIGReg | Random | Real | 0.725 | baseline |
| + IN1K init | IN1K | Real | 0.756 | +3.1%p |
| + $\lambda$ tuning | IN1K | Real | 0.799 | +4.3%p |
| + Real+Syn data | IN1K | Real+Syn | **0.820** | +2.1%p |

**SSL Method Comparison and Data Scalability**

| SSL Method | Loss Type | proj dim | Real only | Real+Syn | Δ |
|-----------|-----------|---------|----------|---------|---|
| SIGReg | Regularization | 16 | 0.799 | **0.820** | +2.1%p |
| VICReg | Regularization | 2048 | 0.774 | 0.800 | +2.6%p |
| SimCLR | Contrastive | 128 | 0.806 | 0.801 | −0.5%p |
| BYOL | Distillation | 256 | 0.774 | 0.693 | −8.1%p |

### Key Findings

1. **In-domain SSL surpasses large-scale foundation models**: Mine-JEPA (ViT-S) pretrained on only 1,170 sonar images outperforms DINOv3 pretrained on 1.7 billion images.
2. **Domain adaptation of strong models leads to degradation**: Continuing in-domain SSL from DINOv3 reduces performance by 10–13 percentage points, challenging the intuition that a stronger backbone is always better.
3. **Augmentation strategy is a prerequisite for in-domain SSL**: Applying natural image augmentations to sonar data drops F1 from 0.557 to 0.312—worse than no pretraining.
4. **Regularization-based SSL is more robust**: SIGReg and VICReg benefit from heterogeneous synthetic data, while SimCLR remains flat and BYOL degrades significantly.
5. **ViT-Tiny is highly competitive**: With only 5.5M parameters (one-quarter of DINOv3), it achieves comparable performance, making it suitable for resource-constrained platforms such as AUVs.

## Highlights & Insights

1. **Deep insight into "small but specialized" vs. "large but general"**: The most important finding is not the specific design of Mine-JEPA, but the demonstration that, under extreme domain shift, a carefully adapted small model outperforms a large unadapted one.
2. **Counterintuitive conclusion on initialization**: A moderately strong ImageNet initialization serves as a better starting point for in-domain SSL than the stronger DINOv3, because over-specialized representations lack the plasticity required for domain adaptation.
3. **Positive signal on heterogeneous data utility**: Modality-heterogeneous data—RGB real patches and grayscale synthetic patches—can still be effectively exploited under regularization-based SSL.
4. **Minimalist design philosophy**: Projection dimension of 16, a single hyperparameter $\lambda$, and no momentum encoder—in small-data settings, the most effective SSL is not the most complex.

## Limitations & Future Work

1. **Single dataset**: Experiments are conducted on a single public sonar dataset (1,170 images); the generalizability of the findings requires validation on additional data.
2. **Small test set**: The 3-class evaluation uses only 110 test samples, resulting in high variance.
3. **Patch-level classification only**: The method is not extended to sliding-window detection or semantic segmentation, both of which are required in practical MCM applications.
4. **Synthetic data provenance**: The source of synthetic data in the Real+Syn setting is not described in detail.
5. **Insufficient analysis of DINOv3 degradation**: Only speculative explanations are offered (feature distortion, distribution shift) without quantitative analysis.

## Related Work & Insights

- **Relationship to LeJEPA**: Mine-JEPA builds directly on LeJEPA/SIGReg; its primary contributions lie at the domain adaptation level—augmentation strategy, initialization strategy, and data composition.
- **Medical imaging analogy**: The findings are consistent with the observation in medical imaging that in-domain SSL outperforms general-purpose pretraining (e.g., Models Genesis).
- **Broader implications**: In other extreme domain-shift settings—such as satellite SAR imagery or seismic waveform data—in-domain SSL may similarly outperform off-the-shelf foundation models.
- **Caution for foundation models**: Naively continuing pretraining from a foundation model may be counterproductive; the "plasticity" of the initialization must be assessed.

## Rating

- **Novelty**: ⭐⭐⭐ — The method is an adapted combination of existing techniques; the primary contribution is the experimental insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablations are comprehensive and systematic, covering initialization, augmentation, data composition, and SSL method dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is logically structured with clear conclusions and in-depth discussion.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for data-scarce marine visual tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)
- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)

<!-- RELATED:END -->
