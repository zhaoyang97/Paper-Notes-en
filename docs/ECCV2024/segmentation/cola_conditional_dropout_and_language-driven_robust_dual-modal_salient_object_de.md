---
title: >-
  [Paper Note] CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection
description: >-
  [ECCV2024][Segmentation][Dual-modal Salient Object Detection] This paper proposes the CoLA framework, which introduces two core modules, Language-driven Quality Assessment (LQA) and Conditional Dropout (CD), to simultaneously address two key robustness issues in dual-modal salient object detection for the first time: noisy inputs and missing modalities.
tags:
  - "ECCV2024"
  - "Segmentation"
  - "Dual-modal Salient Object Detection"
  - "Modality Robustness"
  - "CLIP"
  - "Conditional Dropout"
  - "Quality Assessment"
date: 2026-05-08
content_hash: 8602ef94e2ab234f
---

# CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection

**Conference**: ECCV2024  
**arXiv**: [2407.06780](https://arxiv.org/abs/2407.06780)  
**Code**: None  
**Area**: Image Segmentation  
**Keywords**: Dual-modal Salient Object Detection, Modality Robustness, CLIP, Conditional Dropout, Quality Assessment

## TL;DR

This paper proposes the CoLA framework, which introduces two core modules, Language-driven Quality Assessment (LQA) and Conditional Dropout (CD), to simultaneously address two key robustness issues in dual-modal salient object detection for the first time: noisy inputs and missing modalities.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Dual-modal Salient Object Detection (SOD) utilizes RGB images and auxiliary modalities (depth/thermal infrared) to detect the most salient objects in a scene. The high accuracy of existing methods relies on high-quality and complete inputs, but real-world deployments face two core challenges:

1. **Noisy inputs**: Image quality of RGB or auxiliary modalities degrades due to communication failures, etc., leading to undesirable prediction results in traditional methods.
2. **Missing modalities**: Device failures cause a modality to be completely unavailable, leading to a sharp decline in the performance of existing models due to over-reliance on complete inputs.

Existing quality assessment methods either rely on pre-trained networks with fixed parameters (unable to adapt to target datasets) or use imprecise pseudo-labels. Meanwhile, direct dropout methods for handling missing modalities, while effective in missing-modality scenarios, significantly damage performance on complete modalities.

### Goal

**Goal**: 1. How to adaptively assess the quality of inputs from each modality and recalibrate their contributions without requiring additional quality annotations?  
2. How to enhance robustness to missing modalities without sacrificing detection accuracy in complete-modality scenarios?

## Method

CoLA adopts a two-stage training architecture containing four components: a dual-branch encoder, an LQA module, a conditional dropout encoder, and a decoder.

### Stage 1: Language-driven Quality Assessment (LQA)

LQA utilizes a pre-trained CLIP vision-language model for modality quality assessment:

- Dual-modal images are separately fed into the CLIP image encoder to obtain image embeddings $\varepsilon_i \in \mathbb{R}^{1 \times D}$ ($D=512$).
- The text encoder receives a fixed prompt "A photo of high quality." to generate a text embedding $\varepsilon_t$.
- A learnable prompt $\omega$ is introduced and added to the text embedding to achieve parameter-efficient fine-tuning on the target dataset.
- Modality quality scores $\alpha^{m_1}, \alpha^{m_2}$ are calculated using cosine similarity.
- Features from each layer are fused by weighting based on the quality scores: $g_j = g_j^{m_1} \cdot \frac{\alpha^{m_1}}{\alpha^{m_1}+\alpha^{m_2}} + g_j^{m_2} \cdot \frac{\alpha^{m_2}}{\alpha^{m_1}+\alpha^{m_2}}$.

Compared with traditional methods, LQA retains the generalization capability of pre-trained models while adapting to the target dataset.

### Stage 2: Conditional Dropout (CD)

Inspired by conditional control, CD treats missing modalities as conditions to avoid the performance degradation brought by direct dropout:

- Freeze the encoder parameters $\theta$ trained in Stage 1.
- Duplicate the encoder to obtain a trainable copy (parameters $\theta_f$), connected via zero convolution $\mathcal{Z}$ (weights and biases initialized to zero).
- Randomly select inputs from three conditions during training: complete dual-modal $\{m_1, m_2\}$, missing modality two $\{m_1, \phi\}$, or missing modality one $\{\phi, m_2\}$.
- Final features: $g = \mathcal{F}(\rho(\mathcal{M});\theta) + \mathcal{Z}(\mathcal{F}(\rho(\mathcal{M});\theta_f);\theta_z)$.

Zero initialization ensures that Stage 2 training has no impact on the original model in the initial phases, and newly learned features are step-by-step integrated. The frozen first term preserves existing knowledge, while the trainable second term learns more fine-grained single-modality representations.

### Loss & Training

A combination of BCE loss and IoU loss is used: $\mathcal{L}_{total} = \mathcal{L}_{bce}(pred, GT) + \mathcal{L}_{iou}(pred, GT)$.

## Key Experimental Results

### RGB-T Salient Object Detection (VT821/VT1000/VT5000)

| Condition | Metric | TAGFNet | CoLA (Ours) |
|------|------|---------|------------|
| Complete Modalities (VT5000) | $E_m$ / $F_\beta$ | .913 / .819 | **.927 / .843** |
| Missing RGB (VT5000) | $E_m$ / $F_\beta$ | .869 / .742 | **.887 / .774** |
| Missing Thermal (VT5000) | $E_m$ / $F_\beta$ | .895 / .791 | **.913 / .822** |
| Average Performance Drop (VT5000) | $E_m$ / $F_\beta$ | -.031 / -.052 | **-.027 / -.045** |

CoLA achieves the best or second-best performance across all datasets under both complete and missing modality conditions, with the smallest average performance degradation.

### Ablation Study (VT5000)

| Configuration | Complete $S_\alpha$ | Missing RGB $S_\alpha$ | Missing T $S_\alpha$ | Average $S_\alpha$ |
|------|----------------|-------------------|-----------------|----------------|
| Baseline | .859 | .820 | .845 | .841 |
| +LQA | .887 | .828 | .849 | .855 |
| +CD | .880 | .833 | .868 | .860 |
| +LQA+CD | **.892** | **.840** | **.874** | **.869** |

LQA mainly improves complete modality performance (+.028), while CD mainly improves missing modality performance (+.023). The two are complementary.

### Comparison of Quality Assessment Methods

Comparing LQA with BRISQUE, GIE, CLIP-IQA, and CLIP-IQA+, LQA achieves $S_\alpha$ of .888 (second best is .878) on VT821, validating the effectiveness of the learnable prompt fine-tuning strategy.

## Highlights & Insights

1. **The first dual-modal SOD model to simultaneously handle noise and missing modalities**, filling a gap in robustness research in this area.
2. **Intelligently designed LQA**: CLIP is fine-tuned with only a small number of learnable parameters, requiring no quality-labeled data, thus preserving generalization while adapting to the target domain.
3. **The zero-convolution strategy in CD** borrows inspiration from ControlNet, decoupling complete-modality performance from robustness to missing modalities so that they do not interfere with each other.
4. **Plug-and-play property**: As a training scheme, CD can be applied to various dual-modal SOD models, exhibiting good training universality.
5. Encoder and decoder designs are deliberately kept simple to clearly demonstrate the contribution of the core modules.

## Limitations & Future Work

1. Two-stage training increases overall training time; whether it can be combined into end-to-end single-stage training is worth exploring.
2. It only considers complete modality missingness (all-zero input) and does not handle partial spatial damage or progressive degradation.
3. LQA relies on pre-trained CLIP weights, and its applicability to non-natural images (such as medical images) remains to be validated.
4. Quality assessment only produces a global scalar, lacking fine-grained spatial-dimension quality awareness.
5. CD requires duplicating the encoder, which doubles the parameter size during inference and is unfriendly to resource-restricted scenarios.

## Related Work & Insights

- **vs Direct Modality Dropout**: Direct dropout improves missing modality robustness but degrades complete modality performance. CD ensures both using a decoupled design of freeze + copy.
- **vs GIE / BRISQUE Quality Assessment**: GIE uses a fixed pre-trained network and cannot adapt to target data, while BRISQUE is based on handcrafted features and is non-trainable. LQA achieves both generalization and adaptability through prompt learning.
- **vs TAGFNet**: Comparative results with the previous strongest robust method show that CoLA further reduces the average performance degradation by about 30-40%.
- **vs ControlNet**: Transfers the zero-convolution concept of ControlNet from generative models to discriminative SOD tasks to handle missing modalities instead of conditional generation.

## Related Work & Insights

1. The zero-convolution decoupling strategy has wide transferability and can be used in any multi-modal task requiring adaptation to new conditions while maintaining original capabilities.
2. The concept of using CLIP for modality quality assessment can be extended to other multi-modal fusion tasks (such as RGB-D semantic segmentation, multi-modal object tracking).
3. Modeling missing modalities as conditional inputs has potential links to missing data imputation and robust multi-task learning.
4. The prompt learning method of LQA can be further extended to multi-grained quality descriptions, such as "blurry", "dark", "saturated", etc.

## Rating
- Novelty: 4/5 — The first to simultaneously address noise and missing modalities in dual-modal SOD. The design motivations for LQA and CD are clear, and the schemes are novel.
- Experimental Thoroughness: 4/5 — Covers both RGB-T and RGB-D scenarios. The ablation studies and comparison analyses are complete, though efficiency analysis is missing.
- Writing Quality: 4/5 — Clear problem definition, systematic method description, and highly informative diagrams.
- Value: 4/5 — The plug-and-play feature of CD has practical application value, though two-stage training and inference overheads limit real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](../../NeurIPS2025/segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)
- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](../../CVPR2025/segmentation/visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[ICLR 2026\] S3OD: Towards Generalizable Salient Object Detection with Synthetic Data](../../ICLR2026/segmentation/s3od_towards_generalizable_salient_object_detection_with_synthetic_data.md)
- [\[ECCV 2024\] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)

</div>

<!-- RELATED:END -->
