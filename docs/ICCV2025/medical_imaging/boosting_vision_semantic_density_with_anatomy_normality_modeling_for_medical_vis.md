---
title: >-
  [Paper Note] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training
description: >-
  [ICCV 2025][Medical Imaging][vision-language pre-training] This paper proposes ViSD-Boost, which addresses the alignment bias caused by low visual semantic density in medical vision-language pre-training (VLP). The metho…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "vision-language pre-training"
  - "semantic density"
  - "anatomy normality"
  - "VQ-VAE"
  - "zero-shot diagnosis"
date: 2026-05-08
content_hash: 10a5c8a91f687485
---

# Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training

**Conference**: ICCV 2025
**arXiv**: [2508.03742](https://arxiv.org/abs/2508.03742)  
**Code**: [alibaba-damo-academy/ViSD-Boost](https://github.com/alibaba-damo-academy/ViSD-Boost)  
**Area**: Medical Imaging / Vision-Language Pre-training
**Keywords**: vision-language pre-training, semantic density, anatomy normality, VQ-VAE, zero-shot diagnosis

## TL;DR

This paper proposes ViSD-Boost, which addresses the alignment bias caused by low visual semantic density in medical vision-language pre-training (VLP). The method employs disease-level visual contrastive learning to enhance visual semantics and VQ-VAE-based anatomical normality modeling to amplify abnormality signals, achieving 84.9% AUC in zero-shot diagnosis across 54 diseases spanning 15 organs.

## Background & Motivation

Vision-language pre-training (VLP) has achieved remarkable success in natural image domains, but its effectiveness in medical settings remains limited. The authors identify the core bottleneck as the **semantic density gap**:

- **Low signal-to-noise ratio on the visual side**: CT images contain extensive anatomical content, yet diagnostically relevant regions often occupy only a tiny fraction of the image (e.g., bladder stones may account for less than one-thousandth of the volume).
- **High signal-to-noise ratio on the text side**: Diagnostic reports are highly condensed summaries of image observations, rich in diagnosis-relevant semantics.
- **Alignment bias**: Directly aligning low-semantic-density visual representations with high-semantic-density textual representations prevents visual attention from focusing on regions of interest.

This also explains why most medical VLP attempts remain confined to relatively simple 2D chest scenarios—extracting diagnostically relevant visual cues in complex 3D abdominal scenes is substantially more challenging.

## Method

### Overall Architecture

ViSD-Boost builds upon anatomy-wise image-report alignment and comprises two key stages: (1) visual semantic enhancement via disease-level contrastive learning to distinguish normal from abnormal anatomical structures, and (2) visual semantic density boosting via VQ-VAE normality modeling to amplify abnormality signals.

### Key Designs

1. **Anatomy-wise Image-Report Alignment**:

    - A whole-body segmentation model parses each image into organ-level structures: $X_i^I \to \{X_{i,j}^I; j=1,...,M\}$
    - A Qwen LLM decomposes diagnostic reports into anatomy-level sub-reports: $X_i^R \to \{X_{i,j}^R; j=1,...,M\}$
    - ResNet is used for visual feature extraction; a pre-trained BERT serves as the text encoder.
    - Learnable query tokens are appended to visual and text tokens and aggregated via cross-attention.
    - Contrastive learning objective: $\arg\min -\frac{1}{B \times M}\sum_{i,j} \log \frac{e^{\langle Q_{i,j}^I, Q_{i,j}^R \rangle / \tau}}{\sum_k e^{\langle Q_{i,j}^I, Q_{k,j}^R \rangle / \tau}}$

2. **Disease-level Visual Contrastive Learning**:

    - Core idea: normal samples belong to the same category and should be pulled together, while abnormal samples exhibit diverse patterns and should be pushed apart.
    - Organ-level abnormality labels $y \in \{0: \text{normal}, 1: \text{abnormal}\}^{B \times M}$ are automatically extracted from reports using an LLM.
    - Contrastive loss design:
        - Abnormal samples ($y_{i,j}=1$): only different augmented views of the same sample are treated as positive pairs (preventing erroneous attraction between distinct abnormalities).
        - Normal samples ($y_{i,j}=0$): all normal samples of the same organ are treated as mutual positive pairs.
    - A momentum encoder generates features $Q_{i,j}^{I'}$ for positive pairs to prevent degenerate solutions.
    - This step is executed independently prior to VLP training.

3. **Anatomical Normality Modeling**:

    - A VQ-VAE is employed to learn the normal distribution of each anatomical structure in the latent space.
    - Two key innovations:
        - **Multi-distribution learning**: CT images contain dozens of anatomical structures; an anatomy-conditioned token $A_j$ guides the VQ-VAE to reconstruct features specific to each structure.
        - **Latent-space modeling**: Training is performed in the latent space rather than image space, improving efficiency and encoding in a higher-level semantic space.
    - A Transformer-based encoder $\varphi_E$ and decoder $\varphi_D$ are used with a discrete codebook $e \in \mathbb{R}^{M \times K \times C}$.
    - Trained exclusively on normal samples: $\mathbb{I}_{y_{i,j}=0} \cdot (\|f_{i,j}^I - \varphi_D(e_{j,k})\|_2^2 + \beta\|\text{sg}[e_{j,k}] - \varphi_E(f_{i,j}^I; A_j)\|_2^2)$
    - Distribution shift in abnormal samples leads to degraded reconstruction quality → reconstruction error serves as an abnormality indicator.

4. **Abnormality Semantic Perception Module**:

    - The original embedding $f_{i,j}^I$ and the VQ-VAE-reconstructed normal embedding $q_{i,j}^I$ are concatenated.
    - An MLP perceives the difference between the two, amplifying the abnormal component.
    - The output $\hat{f}_{i,j}^I$ replaces the original embedding in VLP training.
    - Design motivation: reconstruction error is directly correlated with diagnostically relevant information.

### Loss & Training

- Three-stage pipeline: (1) disease-level contrastive learning to pre-train the visual encoder; (2) VQ-VAE normality modeling training; (3) VLP alignment training.
- The VQ-VAE codebook is updated via an exponential moving average strategy.
- Hyperparameter $\beta = 0.25$.

## Key Experimental Results

### Main Results (CT-RATE Chest Zero-shot)

| Method | Precision | ACC | F1 | AUC |
|--------|-----------|-----|----|-----|
| CT-CLIP | 32.6 | 66.9 | 70.8 | 73.3 |
| BIUD | 33.8 | 68.1 | 71.6 | 71.3 |
| Merlin | 33.7 | 67.2 | 70.9 | 72.8 |
| fVLM | 37.9 | 71.8 | 75.1 | 77.8 |
| **ViSD-Boost** | **38.7** | **73.1** | **75.9** | **79.0** |

Abdominal scenario (MedVL-CT69K, zero-shot over 54 diseases across 15 organs):

| Method | Mean SE | Mean SP | Mean AUC |
|--------|---------|---------|----------|
| Supervised | 62.0 | 76.2 | 73.3 |
| CLIP | 65.5 | 68.0 | 68.4 |
| fVLM | 67.9 | 72.5 | 74.5 |
| **ViSD-Boost** | **72.4** | **74.5** | **78.5** |

### Ablation Study

| Configuration | Chest AUC | Abdominal AUC | Notes |
|---------------|-----------|---------------|-------|
| Baseline (alignment only) | 77.8 | 74.5 | fVLM baseline |
| + Disease-level contrastive learning | 78.2 | 76.8 | Enhanced normal/abnormal discrimination |
| + VQ-VAE normality modeling | 78.6 | 77.5 | Amplified abnormality signals |
| + Both combined | **79.0** | **78.5** | Full ViSD-Boost |

### Key Findings

- Improvements are especially pronounced in the complex abdominal scenario (+4.0% AUC over fVLM), validating the importance of semantic density boosting for 3D scenes.
- The design of disease-level contrastive learning is critical: clustering normal samples while dispersing abnormal samples outperforms conventional instance-level contrastive learning.
- The VQ-VAE is trained solely on normal samples and detects anomalies naturally via distribution shift, requiring no abnormality annotations.
- Strong performance on the external validation set Rad-ChestCT demonstrates good generalizability.

## Highlights & Insights

- **Semantic density concept**: This work is the first to explicitly formulate the visual semantic density problem in medical VLP, providing a novel theoretical perspective for the field.
- **Elegant use of distribution shift for normality modeling**: Rather than directly learning abnormal features, the method models the distribution of "what normal looks like," allowing abnormalities to emerge naturally through reconstruction error.
- **Automatic annotation via LLM**: Organ-level normal/abnormal labels are automatically extracted from reports using an LLM, eliminating the need for manual annotation.
- **Cross-scenario validation**: State-of-the-art results are achieved in both chest and abdominal CT scenarios, demonstrating the generality of the approach.

## Limitations & Future Work

- The method depends on the quality of the whole-body segmentation model; segmentation errors propagate to all subsequent stages.
- The VQ-VAE codebook size and anatomy-conditioned token design may require scenario-specific tuning.
- The approach has only been validated on CT modality; its effectiveness on X-ray, MRI, and other modalities remains unknown.
- The three-stage training pipeline is relatively complex; end-to-end training strategies are worth exploring.

## Related Work & Insights

- Unlike conventional visual representation enhancement methods, ViSD-Boost approaches the problem from the perspective of semantic density, achieving both disease-level semantics and generalizability.
- The normality modeling concept shares conceptual roots with the anomaly detection literature, but its application within a VLP framework is novel.
- The multi-organ joint modeling design is generalizable to other multi-region medical image analysis tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The semantic density concept is novel, and the idea of amplifying abnormality signals via normality modeling is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across chest and abdominal scenarios with zero-shot assessment over 54 diseases and downstream task transfer validation.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clearly articulated, though some sections are notation-heavy.
- Value: ⭐⭐⭐⭐⭐ Achieving 84.9% zero-shot AUC across 54 diseases in 15 organs demonstrates significant clinical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](../../AAAI2026/medical_imaging/sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)

</div>

<!-- RELATED:END -->
