---
title: >-
  [Paper Note] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis
description: >-
  [AAAI 2026][Medical Imaging][X-ray angiography] This paper proposes VasoMIM, a domain-specific self-supervised pre-training framework for X-ray angiograms. It introduces an anatomy-guided masking strategy that prioritizes vessel regions, an anatomical consistency loss to preserve vascular topology in reconstructed images, and a newly constructed XA-170K pre-training dataset — the largest of its kind. VasoMIM comprehensively outperforms both general-purpose and medical SSL met…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "X-ray angiography"
  - "self-supervised learning"
  - "masked image modeling"
  - "vascular anatomy awareness"
  - "foundation model"
  - "vessel segmentation"
  - "stenosis detection"
date: 2026-05-08
content_hash: 759019362a36e1c9
---

# Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis

**Conference**: AAAI 2026
**arXiv**: [2602.11536](https://arxiv.org/abs/2602.11536)  
**Code**: [GitHub](https://github.com/Dxhuang-CASIA/XA-SSL)  
**Area**: Medical Imaging / Self-supervised Pre-training
**Keywords**: X-ray angiography, self-supervised learning, masked image modeling, vascular anatomy awareness, foundation model, vessel segmentation, stenosis detection

## TL;DR
This paper proposes VasoMIM, a domain-specific self-supervised pre-training framework for X-ray angiograms. It introduces an anatomy-guided masking strategy that prioritizes vessel regions, an anatomical consistency loss to preserve vascular topology in reconstructed images, and a newly constructed XA-170K pre-training dataset — the largest of its kind. VasoMIM comprehensively outperforms both general-purpose and medical SSL methods (including DINOv3 pre-trained on 1.69 billion images) across 4 downstream tasks and 6 datasets.

## Background & Motivation

**Background**: Cardiovascular disease is the leading cause of death worldwide, and X-ray angiography serves as the clinical gold standard for diagnosis. Deep learning methods (e.g., UNet, Faster R-CNN) have achieved notable progress in vessel segmentation and stenosis detection, but are severely constrained by the scarcity of annotated data. Self-supervised learning (SSL) offers a promising solution; however, the field lacks dedicated SSL frameworks and large-scale datasets tailored to this modality.

**Limitations of Prior Work**:
   - **Random masking strategies in generic MIM are ill-suited**: Vascular structures in angiograms are extremely sparse, occupying only a small fraction of the image. Random, attention-guided, or loss-guided masking predominantly covers background regions, causing models to learn background reconstruction rather than vessel features.
   - **Pixel-level reconstruction objectives lack semantic discriminability**: MSE loss encourages prediction of low-frequency background textures rather than high-frequency vascular details.
   - **Absence of large-scale datasets**: Unlike chest X-rays (e.g., CheXpert with 220K+ images) or CT, the angiography domain lacks large-scale pre-training datasets.
   - **General visual foundation models (e.g., DINOv3) transfer poorly across domains**: Models pre-trained on natural images lack the anatomical semantics specific to angiography.

**Core Idea**: Inject strong anatomical inductive biases into MIM — enabling the model to identify vascular regions and forcing it to focus on reconstructing them.

## Method

### Overall Architecture (VasoMIM)

Input X-ray angiogram → Frangi filter for vascular anatomy extraction + UNeXt-S segmentor for probability map generation → co-guidance → anatomy-guided masking strategy → ViT encoder + decoder for reconstruction → pixel reconstruction loss $\mathcal{L}_{rec}$ + anatomical consistency loss $\mathcal{L}_{cons}$

### Key Designs

1. **Frangi Filter for Vascular Anatomy Extraction**:

    - Multi-scale Hessian analysis (σ = 1, 2, 3, 4) for tubular structure detection
    - Adaptive percentile thresholding (α = 92) to generate a coarse binary mask
    - Region growing to remove isolated artifacts, yielding the final binary vessel mask $B \in \{0,1\}^{1 \times H \times W}$

2. **Anatomy-Guided Masking Strategy**:

    - **Co-guidance**: Combines the Frangi filter mask $B$ and the UNeXt-S segmentation probability map $M$:
    $G = \eta \cdot B + (1-\eta) \cdot M, \quad \eta=0.5$
    - UNeXt-S compensates for the Frangi filter's tendency to miss low-contrast fine vessels.
    - **Patch-level sampling probability**: $f(g_i) = \frac{\sum_j g_{ij}}{\sum_k \sum_j g_{kj}}$, assigning higher masking probability to patches with greater vessel density.
    - **Weak-to-strong curriculum**: Random masking is blended more heavily in early training stages to avoid premature optimization on difficult targets; anatomical guidance is progressively increased:
    $\beta_e = \beta_0 + \frac{e}{E}(\beta_E - \beta_0)$
    - Per epoch: $\beta_e \gamma N$ patches are sampled via anatomy-guided strategy; $(1-\beta_e)\gamma N$ patches are sampled randomly.

3. **Anatomical Consistency Loss**:

    - Core idea: applying the same segmentor to both the original and reconstructed images should yield consistent outputs:
    $\mathcal{L}_{cons} = \mathcal{L}_{CE}(\mathcal{S}(I), \mathcal{S}(I'))$
    - A lightweight UNeXt-S (only 0.26M parameters) serves as a differentiable proxy, since the Frangi filter is non-differentiable.
    - Encourages the model to learn topologically accurate vascular representations rather than merely reproducing pixel intensities.

4. **Overall Training Objective**:
    $\mathcal{L}_{MIM} = \mathcal{L}_{rec} + \mathcal{L}_{cons}$

### XA-170K Dataset

177,478 X-ray angiogram images collected from 4 public sources:
- CADICA: 42 patients, 6,594 frames
- SYNTAX: 231 patients, 2,943 images
- XCAD: 1,621 images
- CoronaryDominance: 1,574 patients, 160,320 images (primary source)

## Experiments

### Downstream Tasks and Datasets
- **Vessel segmentation**: ARCADE-V, CAXF, XCAV (DSC + clDice)
- **Vessel segment segmentation**: ARCADE-VS (DSC)
- **Stenosis segmentation**: ARCADE-S (DSC)
- **Stenosis detection**: Stenosis (mAP50, mAP75, mAP)

### Main Results: Segmentation Tasks

| Method | Pre-training Data | ARCADE-V DSC | ARCADE-V clDice | XCAV DSC | ARCADE-S DSC | ARCADE-VS DSC | Avg. Rank |
|--------|------------------|-------------|-----------------|----------|-------------|--------------|-----------|
| UNet (scratch) | - | 71.44 | 70.67 | 78.18 | 27.04 | 38.77 | 22.00 |
| MAE | XA-170K | 79.39 | 80.74 | 84.84 | 51.72 | 56.69 | 4.88 |
| DINOv3 | LVD-1698M | 79.36 | 80.90 | 82.76 | 53.57 | 54.36 | 7.25 |
| DeblurringMIM | XA-170K | 79.25 | 80.77 | 85.38 | 51.70 | 56.66 | 4.38 |
| RAD-DINO | LVD-142M+CXR-838K | 78.96 | 80.26 | 84.88 | 51.55 | 54.81 | 6.62 |
| VasoMIM-v1 | XA-170K | 79.90 | 81.57 | 85.80 | 54.52 | 58.03 | 2.12 |
| **VasoMIM** | **XA-170K** | **80.25** | **82.06** | **86.09** | **55.62** | **58.87** | **1.00** |

**Key Findings**:
- VasoMIM achieves the best performance on all metrics with an average rank of 1.00.
- Substantial gains over UNet trained from scratch: ARCADE-S DSC +28.58, ARCADE-VS DSC +20.10.
- **Domain-specific pre-training > general large-scale pre-training**: DINOv3, pre-trained on 1.69 billion natural images, is outperformed by VasoMIM pre-trained on only 170K angiograms.
- VasoMIM further improves upon VasoMIM-v1 (conference version) with statistical significance (p = 1.18×10⁻⁴, paired t-test).

### Stenosis Detection Task

| Method | mAP50 | mAP75 | mAP |
|--------|-------|-------|-----|
| Faster R-CNN (scratch) | 88.37 | 19.01 | 36.63 |
| MAE | 92.30 | 24.28 | 39.69 |
| DINOv3 | 93.89 | 23.60 | 40.90 |
| VasoMIM-v1 | 94.25 | 25.01 | 40.91 |
| **VasoMIM** | **94.91** | **25.72** | **41.07** |

### Ablation Study

**Independent contributions of anatomy-guided masking and anatomical consistency loss**:

| Guidance | $\mathcal{L}_{cons}$ | ARCADE-V DSC | XCAV DSC |
|----------|---------------------|-------------|----------|
| ✗ | ✗ | 79.31 | 84.52 |
| ✗ | ✓ | 79.85 (+0.54) | 85.79 (+1.27) |
| ✓ | ✗ | 79.87 (+0.56) | 85.92 (+1.40) |
| **✓** | **✓** | **80.25 (+0.94)** | **86.09 (+1.57)** |

- Both components are individually effective; their combination yields super-additive gains.
- Anatomy-guided masking provides a larger boost on XCAV (+1.40 vs. +1.27), where vessel structures are more sparse.

**Masking guidance visualization**:
- Baseline (MAE random masking): only 5–10% of masked patches contain vessel structures.
- VasoMIM: the proportion of masked vessel patches progressively increases from ~20% to ~70% over training.
- Co-guidance captures low-contrast vascular branches missed by Frangi filtering alone.

**Comparison with alternative reconstruction targets**: The anatomical consistency loss using UNeXt-S (0.26M parameters) is far more lightweight than DINOv2-based (86M) feature distillation, while achieving comparable or superior performance.

## Highlights & Insights

1. **Domain-knowledge-driven design**: The Frangi filter is a classical vessel analysis tool; its integration as an anatomical prior into the MIM framework exemplifies effective combination of traditional methods and deep learning.
2. **Counter-intuitive "small data beats big data" finding**: 170K domain-specific images outperform 1.69 billion natural images (DINOv3), underscoring the importance of domain adaptation.
3. **Complementarity of co-guidance**: The Frangi filter provides hard edge responses but misses fine vessels, while UNeXt-S provides soft probability maps that compensate for such omissions.
4. **Curriculum from weak to strong guidance**: Avoids difficult optimization in early stages by gradually increasing anatomical masking, reflecting sound engineering intuition.
5. **Elegant design of the anatomical consistency loss**: A lightweight 0.26M segmentor replaces the non-differentiable Frangi filter with minimal computational overhead.
6. **Dataset contribution**: XA-170K, the largest pre-training dataset in this domain, will be made publicly available.

## Limitations & Future Work

1. **Limitations of the Frangi filter**: Sensitivity to noise and potential misclassification of bony structures as vessels (partially mitigated by co-guidance).
2. **Only ViT-B/16 is used as backbone**: Whether larger models (ViT-L/H) yield further gains remains unexplored.
3. **Full fine-tuning for downstream tasks**: Parameter-efficient fine-tuning strategies (e.g., LoRA) and performance under more extreme few-shot settings are not investigated.
4. **Dataset scale remains limited**: Although XA-170K is the largest in this domain, it is still substantially smaller than million-scale pre-training datasets for chest X-rays or CT.
5. **Restricted to coronary angiography**: Generalizability to other angiographic modalities (e.g., cerebrovascular, peripheral vascular) has not been validated.
6. **UNeXt-S trained with Frangi pseudo-labels**: Segmentation quality is bounded by Frangi filter performance, potentially introducing systematic bias.

## Related Work & Insights

- **General SSL**: MAE, SimMIM, DINO, iBOT, I-JEPA, DINOv3
- **Medical SSL**: Model Genesis, LVM-Med, DeblurringMIM, RAD-DINO, CheXWorld, MedDINOv3
- **MIM masking strategies**: AMT (attention-guided), HPM (loss-guided), AnatoMask, HAP (human anatomy prior)
- **Angiogram analysis**: ARCADE dataset, XCAD, supervised methods based on UNet/Faster R-CNN

## Rating ⭐⭐⭐⭐⭐

The method is elegantly designed with clear motivation and effective integration of domain knowledge. The experiments are highly comprehensive — covering 4 downstream tasks, 6 datasets, 20+ baselines, and detailed ablations. The dataset contribution and scaling law analysis further enhance the paper's value. This is a high-quality contribution to the field of self-supervised pre-training for medical image analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[CVPR 2026\] InvCoSS: Inversion-driven Continual Self-supervised Learning in Medical Multi-modal Image Pre-training](../../CVPR2026/medical_imaging/invcoss_inversion-driven_continual_self-supervised_learning_in_medical_multi-mod.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)

</div>

<!-- RELATED:END -->
