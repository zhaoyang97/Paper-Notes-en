---
title: >-
  [Paper Note] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification
description: >-
  [CVPR2025][Medical Imaging][Multiple Instance Learning] This paper proposes the MIL-PF framework, which leverages precomputed features from frozen foundation vision models and employs an ultra-lightweight MIL aggregation head with only ~40k parameters to achieve SOTA performance on mammography classification tasks, significantly reducing training costs.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Multiple Instance Learning"
  - "mammography"
  - "foundation model"
  - "precomputed features"
  - "attention pooling"
date: 2026-05-08
content_hash: e8492484d655d6f1
---

# MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**Conference**: CVPR2025  
**arXiv**: [2603.09374](https://arxiv.org/abs/2603.09374)  
**Code**: Available  
**Area**: Medical Imaging  
**Keywords**: Multiple Instance Learning, mammography, foundation model, precomputed features, attention pooling

## TL;DR

This paper proposes the MIL-PF framework, which leverages precomputed features from frozen foundation vision models and employs an ultra-lightweight MIL aggregation head with only ~40k parameters to achieve SOTA performance on mammography classification tasks, significantly reducing training costs.

## Background & Motivation

### Background

**Background**: Breast cancer is the most common malignant tumor and the leading cause of cancer death among women, and mammography is the primary screening and diagnostic modality.

### Limitations of Prior Work

**Limitations of Prior Work**: Mammograms exhibit extremely high resolutions (up to 4708×5844 pixels) and lack pixel-level annotations, usually possessing only weak breast-level labels.

### Key Challenge

**Key Challenge**: End-to-end fine-tuning of large models in this scenario is computationally prohibitive and impractical, while existing CLIP-style training is limited by insufficient supervision signals and structural constraints of high-resolution inputs.

### Key Insight

**Key Insight**: Frozen general-purpose foundation encoders (such as DINOv2 and MedSigLIP) have already demonstrated outstanding zero-shot generalization capabilities in the out-of-distribution mammography domain, thereby eliminating the need to fine-tune the encoder.

## Method

The MIL-PF pipeline is divided into two stages:

### Stage 1: Feature Precomputation
- Use a frozen foundation encoder $\mathcal{F}$ to generate two types of embeddings for each bag (all views of the same breast):
    - **Global Embeddings** $\mathcal{G}_i = \{\mathcal{F}(I_i^{(n)})\}_n$: Encodes each full image to capture low-frequency global signals such as tissue structure and breast density.
    - **Local Embeddings** $\mathcal{T}_i = \bigcup_n \bigcup_k \{\mathcal{F}(C_i^{(n)(k)})\}$: Partitions the image into a grid of tiles corresponding to the encoder input size (518×518 for DINOv2, 448×448 for MedSigLIP). Pure background tiles are discarded using a heuristic $\mathcal{H}$, retaining only tiles containing breast tissue to capture sparse local lesion signals after encoding.
- The number of tiles $M_i^{(n)}$ per image is variable, depending on the breast size and position.
- Construct an embedding dataset $\mathcal{E} = \{(\mathcal{G}_i, \mathcal{T}_i, y_i)\}_i$. Subsequent training is performed entirely within this fixed representation space.
- Overlapping tiles are not used for the classification task; 75% overlap is used during inference for attention map calculation to improve visualization resolution.

### Stage 2: MIL-PF Head Training
- A late fusion strategy is adopted, where two streams are independently aggregated and then concatenated: $\hat{y}_i = h_\theta(\text{concat}(\mathcal{A}^G_\psi(\mathcal{G}_i), \mathcal{A}^T_\omega(\mathcal{T}_i)))$
- **Global Aggregator** $\mathcal{A}^G$:
    - A two-layer MLP (embedding dimension $\rightarrow$ 16 $\rightarrow$ 8 with ReLU activation) projects high-dimensional embeddings to a compact representation.
    - Subsequently, max pooling is used to aggregate multi-view global features.
    - This branch performs task-relevant high-level feature processing (since the encoder is frozen), and contains most of the parameters.
- **Local Aggregator** $\mathcal{A}^T$:
    - Similarly, a two-layer MLP is first used for task-relevant projection.
    - A Perceiver-style cross-attention mechanism is employed: a single trainable latent vector $z$ acts as the query, while tile embeddings are projected as Key and Value.
    - The weighted summary vector is calculated as $\text{softmax}(zK^T)V$, selectively focusing on task-relevant sparse ROIs.
    - Compared to mean pooling (where signals are diluted by numerous background tiles) and max pooling (which only captures a single most prominent tile), the attention mechanism can focus on multiple independent lesion regions simultaneously.
    - Experiments show that a single latent query is sufficient, and adding more latent queries yields no additional benefits.
- Finally, $h_\theta$ maps the concatenated summary vector to classification predictions.
- The loss function is Binary Cross-Entropy, with only ~40k total trainable parameters.

### Optimization over Multiple Runs
- Since head training is extremely fast (approx. 5-7 minutes per run on a single A100 40GB), 36 independent training runs are performed for each experiment.
- Cross-run variance: max variance of 2% in AUC, and max variance of 11% in Spec@Sens=0.9.
- Selecting the model with the highest validation AUC stably approaches the peak performance on the test set.
- Data is split into 70%/10%/20% (train/validation/test), balanced by the proportion of BI-RADS values, ensuring no patient leakage.

## Key Experimental Results

- **Datasets**:
    - EMBED: ~500,000 mammograms, one of the largest public datasets, featuring highly diverse real-world clinical scenarios.
    - VinDr: A Vietnamese mammography dataset containing mass and calcification annotations.
    - RSNA: A breast cancer screening competition dataset.
- **EMBED BI-RADS Malignancy Classification** (BI-RADS 1 is negative, BI-RADS 4/5/6 is positive):
    - MIL-PF (DINOv2, attn): AUC=0.916, bAcc=0.850, Spec@Sens=0.9=0.762
    - MIL-PF (MedSigLIP, max): AUC=0.918, bAcc=0.845, Spec@Sens=0.9=0.735
    - Strongest baseline FPN-AbMIL: AUC=0.802, Spec@Sens=0.9=0.367
- **RSNA Cancer Detection**: MIL-PF (MedSigLIP, max) AUC=0.925, Spec@Sens=0.9=0.733
- **VinDr Calcification Detection**: MIL-PF (DINOv2, attn) AUC=0.967, bAcc=0.930
- **Ablation Study**: Compared to single-instance learning using only global views, the full MIL-PF improves AUC by up to 5% and Spec@Sens=0.9 by up to 14%.
- **Encoder Selection**: DINOv2 ViT Giant and MedSigLIP significantly outperform MammoCLIP, BiomedCLIP, RADDINO, and DINOv3.
- **Computational Efficiency**: Only ~2M FLOPs per forward pass (per breast), with 0.04-0.05M trainable parameters vs. 1.76-22.89M for baselines.

## Highlights & Insights

1. **Extremely Lightweight**: Achieves SOTA with only ~40k trainable parameters, which is 1/450 of the parameters of the strongest baseline.
2. **Precomputation Paradigm**: Frozen encoder + feature precomputation allows the entire EMBED embedding dataset to fit into a single batch on a single A100, completing training in 5-7 minutes.
3. **Perceiver-style Attention Pooling**: Elegantly solves the sparse ROI aggregation problem, where a single latent query can effectively localize multiple lesion areas.
4. **Breast-level MIL Modeling**: Aligns with the clinical workflow of radiologists who synthesize multi-view information.
5. **Interpretability**: Attention maps can localize lesion regions, which is directionally accurate despite the coarse tile granularity.

## Limitations & Future Work

1. **Limited Detection Resolution**: Due to the large tile size (448×448 / 518×518), the mAP for small lesion detection is extremely low (mAPs≈0.1-1.2), resulting in insufficient localization accuracy for attention maps.
2. **Label Noise**: BI-RADS labels exhibit noise and high inter-radiologist variability, limiting evaluation reliability.
3. **Simplicity of Late Fusion Strategy**: The global and local streams do not model complex interactions, potentially missing cross-scale correlation information.
4. **Underutilized Temporal Information**: Patient historical exams or bilateral symmetry are not considered, which are important diagnostic clues in clinical practice.
5. **Reliance on Encoder Quality**: The performance ceiling depends on the representation capability of the frozen encoder, and generalization to novel encoders is yet to be validated.

## Related Work & Insights

| Method | Characteristics | Comparison with Ours |
|------|------|-----------|
| GMIC [Shen et al.] | Global + local end-to-end training, 14.11M parameters | MIL-PF achieves a 10% higher AUC on EMBED using 1/350 of the parameters |
| FPN-AbMIL [Mourão et al.] | FPN + attention MIL, image-level inference | MIL-PF's breast-level modeling is clinically more aligned, significantly leading on EMBED |
| FPN-SetTrans [Mourão et al.] | Set Transformer aggregation | Higher complexity but inferior performance compared to MIL-PF |
| ES-Attside [Pathak et al.] | Case-level attention, 22.89M parameters | AUC on EMBED is only 0.836 vs. 0.918 for MIL-PF |
| MammoCLIP | Mammography pre-trained CLIP | Generalization on new datasets is inferior to general DINOv2 |

## Rating

- Novelty: ⭐⭐⭐⭐ — The paradigm of frozen encoder + ultra-lightweight MIL head is systematically validated for the first time in mammography.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three large-scale datasets, multi-encoder comparisons, detailed ablation studies, and statistics across 36 runs.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem formulation and coherent methodological derivation.
- Value: ⭐⭐⭐⭐ — Provides an efficient and feasible mammography CAD solution for resource-constrained research groups.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Do Multiple Instance Learning Models Transfer?](../../ICML2025/medical_imaging/do_multiple_instance_learning_models_transfer.md)
- [\[CVPR 2026\] Contrastive Cross-Bag Augmentation for Multiple Instance Learning-based Whole Slide Image Classification](../../CVPR2026/medical_imaging/contrastive_cross-bag_augmentation_for_multiple_instance_learning-based_whole_sl.md)
- [\[CVPR 2026\] Universal-to-Specific: Dynamic Knowledge-Guided Multiple Instance Learning for Few-Shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/universal-to-specific_dynamic_knowledge-guided_multiple_instance_learning_for_fe.md)
- [\[ICLR 2026\] ASMIL: Attention-Stabilized Multiple Instance Learning for Whole-Slide Imaging](../../ICLR2026/medical_imaging/asmil_attention-stabilized_multiple_instance_learning_for_whole-slide_imaging.md)
- [\[ICLR 2026\] Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in Multiple Instance Learning](../../ICLR2026/medical_imaging/mixture_of_mini_experts_overcoming_the_linear_layer_bottleneck_in_multiple_insta.md)

</div>

<!-- RELATED:END -->
