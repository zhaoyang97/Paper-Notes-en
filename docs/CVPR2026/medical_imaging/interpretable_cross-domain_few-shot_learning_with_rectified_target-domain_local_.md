---
title: >-
  [Paper Note] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment
description: >-
  [CVPR 2026][Medical Imaging][Cross-Domain Few-Shot Learning] This paper identifies and addresses the degradation of local feature alignment in CLIP under cross-domain few-shot learning (CDFSL), and proposes CC-CDFSL…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Cross-Domain Few-Shot Learning"
  - "CLIP"
  - "Local Feature Alignment"
  - "Cycle Consistency"
  - "Interpretability"
date: 2026-05-08
content_hash: 6d66c1db2c990b01
---

# Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.17655](https://arxiv.org/abs/2603.17655)  
**Code**: [CC-CDFSL](https://github.com/zyaz/CC-CDFSL)  
**Area**: Medical Imaging
**Keywords**: Cross-Domain Few-Shot Learning, CLIP, Local Feature Alignment, Cycle Consistency, Interpretability

## TL;DR

This paper identifies and addresses the degradation of local feature alignment in CLIP under cross-domain few-shot learning (CDFSL), and proposes CC-CDFSL, a cycle-consistency-based framework. Through bidirectional T-I-T and I-T-I cyclic paths and a semantic anchor mechanism, CC-CDFSL improves patch-level vision-language alignment while enhancing model interpretability.

## Background & Motivation

- **Background**: Vision-language models such as CLIP provide a strong foundation for cross-domain few-shot learning.
- **Limitations of Prior Work**: After fine-tuning on target domains, models struggle to focus on fine-grained visual cues (e.g., ground-glass opacities and local nodules in chest X-rays). Although CLIP can roughly cover all salient regions in the source domain, local patch-text alignment degrades far more severely than global alignment after domain transfer.
- **Key Challenge**: Quantitative analysis measures global alignment score $\text{A}_g$ and local alignment score $\text{A}_l$, revealing that the drop in $\text{A}_l$ is significantly larger than that in $\text{A}_g$ on cross-domain tasks, confirming that domain shift and data scarcity disproportionately harm local feature alignment.
- **Goal**: This degradation is especially critical in domains requiring fine-grained recognition such as medical diagnosis—subtle textures or density variations indicative of pneumonia appear in only a few patches, yet model heatmaps can only coarsely outline body contours.

## Method

### Overall Architecture

Three modules are appended on top of CLIP fine-tuning: Semantic Anchor (SA) augmentation stage → T-I-T cycle consistency → SA contraction stage → I-T-I cycle consistency. The entire framework is incorporated as regularization losses added to the standard cross-entropy loss.

### Key Designs

1. **T-I-T Cycle Consistency (Text-to-Image-to-Text)**: For each text feature $\mathbf{T}_j$, the most similar patch feature is selected from all patch features: $\mathbf{L}_j^* = \mathbf{L}_{\arg\max_i \mathbf{D}_{j,i}^{txt}}$. This patch is then mapped back to the text space to retrieve the most similar text $\mathbf{T}_j^{rec}$, enforcing $\mathbf{T}_j \approx \mathbf{T}_j^{rec}$. Loss: $\mathcal{L}_{\text{cyc\_txt}} = 1 - \frac{1}{C}\sum_{j=1}^{C}\text{sim}(\mathbf{T}_j, \mathbf{T}_j^{rec})$. **Design Motivation**: Analogous to cycle consistency in machine translation, this aligns local visual and textual semantics without requiring patch-level annotations.

2. **Semantic Anchor Mechanism (SA)**: *Augmentation stage* — $A$ augmented views are generated per image to expand the patch candidate pool $\mathbf{X}_{aug} \in \mathbb{R}^{((A+1) \cdot M) \times d}$. *Contraction stage* — the top-$k$ most similar patches per class are selected as semantic anchors $\mathbf{X}_{anchor}$, filtering out noisy and semantically irrelevant background regions. **Design Motivation**: The visual modality is information-rich but noisy; augmentation first provides diversity (enlarging the candidate pool for T-I-T), then contraction removes noise (retaining core semantics for I-T-I).

3. **I-T-I Cycle Consistency (Image-to-Text-to-Image)**: For each anchor $\mathbf{x}_n$, the most similar text $t_n$ is retrieved, which is then used to retrieve the most similar patch $\hat{\mathbf{x}}_n$ from the augmented view space, enforcing $\mathbf{x}_n \approx \hat{\mathbf{x}}_n$. The cross-view retrieval strategy enhances robustness to input transformations (rotation, flipping).

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{\text{cyc\_txt}} + \lambda_2 \mathcal{L}_{\text{cyc\_img}}$$

- $\lambda_1 = 3.0$, $\lambda_2 = 2.0$ (determined via grid search on ISIC)
- $k=10$ (number of anchor patches), fixed across all experiments
- ViT-Base/16 CLIP backbone, 100 epochs of fine-tuning, single RTX 4090
- 2-layer MLP to project local patch features into the text feature space

## Key Experimental Results

### Main Results

| Dataset | Task | CLIP-LoRA | CLIP-LoRA + Ours | Gain |
|--------|------|-----------|-----------------|------|
| ISIC (Dermatology) | 5-way 1-shot | 35.23 | 38.13 | +2.90 |
| ChestX (Chest X-ray) | 5-way 1-shot | 21.73 | 22.21 | +0.48 |
| EuroSAT (Satellite) | 5-way 1-shot | 81.49 | 86.07 | +4.58 |
| CropDisease | 5-way 1-shot | 85.11 | 88.91 | +3.80 |
| ISIC | 5-way 5-shot | 50.68 | **54.72** | **+4.04** |
| EuroSAT | 5-way 5-shot | 92.63 | **94.35** | **+1.72** |

### Ablation Study

| Configuration | ISIC | ChestX | EuroSAT | Crop. | Avg. |
|------|------|--------|---------|-------|------|
| Baseline | 50.68 | 24.44 | 92.63 | 96.20 | 65.98 |
| + T-I-T | 51.13 | 25.15 | 93.79 | 96.37 | 66.61 |
| + T-I-T + SA | 54.30 | 25.35 | 94.33 | 96.95 | 67.73 |
| + I-T-I + SA | 53.81 | 25.14 | 93.83 | 97.01 | 67.45 |
| **Full (T-I-T + I-T-I + SA)** | **54.72** | **25.47** | **94.35** | **97.08** | **67.90** |

### Key Findings

- T-I-T cycle contributes more than I-T-I cycle (+0.63 vs. +1.47 avg.), as T-I-T focuses on the most semantically relevant patches and reduces interference.
- The SA mechanism yields significant gains for both cycles (+1.12 and +0.84 avg., respectively).
- Cross-view retrieval > intra-image retrieval > full-image retrieval; augmentation view diversity is the key factor.
- CC-CDFSL functions as a plug-and-play module compatible with multiple PEFT methods including CoOp, CLIP-Adapter, MaPLe, and CLIP-LoRA.
- Improvements are also observed on 11 datasets in base-to-new generalization, particularly on EuroSAT (+3.6%).

## Highlights & Insights

- This work is the first to identify and quantify the phenomenon that local alignment degradation exceeds global alignment degradation in CLIP under CDFSL.
- Introducing cycle consistency from machine translation into VLM local alignment is an elegant self-supervised idea that requires no additional annotations.
- The SA "augment-then-contract" design elegantly balances candidate diversity and noise filtering.
- Interpretability of the T-I-T path: even when reconstructed texts are not perfectly matched, the method reveals pathological regions attended to by the model and cross-class semantic relationships.
- Formulating the method as a regularization term endows it with strong plug-and-play generality.

## Limitations & Future Work

- Gains on ChestX are limited (+0.48 / +1.03), possibly due to the greater semantic complexity of chest radiographs.
- $\lambda_1$ and $\lambda_2$ require tuning on the target domain; optimal hyperparameters may differ across datasets.
- The specific data augmentation strategies used for augmented view generation are not elaborated.
- Validation is limited to ViT architectures; extension to other visual encoders has not been explored.
- Computational overhead analysis is insufficient; the added patch similarity computations may affect training efficiency.

## Related Work & Insights

- The cycle consistency concept from CycleGAN (Liu et al. 2017) is creatively applied to VLM local alignment.
- FG-CLIP (Xie et al. 2025) and related works investigate the insufficient fine-grained capability of CLIP.
- CLIP-LoRA (Zanella & Ben Ayed 2024) serves as the strongest baseline; the proposed method achieves an average improvement of +2.94 (1-shot) over this baseline.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem is precisely identified; applying cycle consistency to VLM local alignment is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 4 datasets, 4 PEFT methods, 2 backbones, with detailed ablations — exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Logically rigorous with rich visualizations; the problem–observation–solution narrative flows smoothly.
- **Value**: ⭐⭐⭐⭐ A versatile plug-and-play framework with significant practical value for few-shot scenarios requiring fine-grained recognition, such as medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy](unsupervised_domain_adaptation_with_target-only_margin_disparity_discrepancy.md)
- [\[AAAI 2026\] MPA: Multimodal Prototype Augmentation for Few-Shot Learning](../../AAAI2026/medical_imaging/mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
