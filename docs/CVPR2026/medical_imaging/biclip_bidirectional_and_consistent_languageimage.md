---
title: >-
  [Paper Note] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Medical segmentation] This paper proposes BiCLIP, a framework that introduces a Bidirectional Multimodal Fusion (BMF) module enabling text and visual features to mutually refine each other in a closed loop, and an Image Augmentation Consistency (IAC) module that enforces consistency of intermediate features under weak/strong perturbations. BiCLIP achieves robust medical image segmentation under extremely label-scarce (1% annotations only) and image-degraded (low-dose CT noise/motion blur) clinical conditions.
tags:
  - CVPR 2026
  - Medical Imaging
  - Medical segmentation
  - vision-language fusion
  - bidirectional fusion
  - cycle consistency
  - robustness enhancement
date: 2026-05-08
content_hash: 20f6266542d6c4d2
---

# BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.00156](https://arxiv.org/abs/2603.00156)
**Code**: None
**Area**: Medical Image Segmentation / Vision-Language Models
**Keywords**: Medical segmentation, vision-language fusion, bidirectional fusion, cycle consistency, robustness enhancement

## TL;DR

This paper proposes BiCLIP, a framework that introduces a Bidirectional Multimodal Fusion (BMF) module enabling text and visual features to mutually refine each other in a closed loop, and an Image Augmentation Consistency (IAC) module that enforces consistency of intermediate features under weak/strong perturbations. BiCLIP achieves robust medical image segmentation under extremely label-scarce (1% annotations only) and image-degraded (low-dose CT noise/motion blur) clinical conditions.

## Background & Motivation

**Background**: Medical image segmentation is a foundational task in computer-aided diagnosis. Multimodal vision-language methods have recently attracted attention by leveraging textual descriptions to enhance semantic understanding, yet their robustness under realistic clinical conditions—including label scarcity and acquisition degradation—remains insufficiently explored.

**Limitations of Prior Work**:
- Existing vision-language segmentation methods predominantly adopt unidirectional fusion—text conditions visual representations—while visual information cannot reciprocally correct text semantics. When image quality degrades, the mismatch between static text conditioning and low-quality images leads to reduced segmentation accuracy.
- Explicit robustness enhancement mechanisms are lacking; learned representations remain fragile under annotation scarcity and appearance variation.
- Methods such as LGA and ARSeg introduce improved fusion strategies but underperform under extreme low-annotation (1%) and clinical degradation conditions.

**Core Idea**: By allowing visual features to iteratively refine text representations in a bidirectional closed loop, and by enforcing augmentation consistency constraints to stabilize representations under perturbation, both semantic alignment and robustness can be addressed simultaneously.

## Method

### Overall Architecture

A 224×224×3 medical image is processed by a lightweight CNN encoder to extract a global visual embedding $\mathbf{i}$, while clinical text is encoded by a frozen CXR-BERT and projected into a compact text embedding $\mathbf{t}$. Both embeddings are fed into the BMF module for bidirectional interaction to generate a pseudo image encoding cross-modal semantics. The pseudo image and the original image are concatenated along the channel dimension and passed into a U-Net for segmentation. During training, the IAC module imposes consistency constraints on intermediate features extracted from weakly and strongly augmented views, promoting augmentation-invariant representation learning.

### Key Designs

1. **Bidirectional Multimodal Fusion (BMF) Module**:
    - **Function**: Establishes a complete bidirectional interaction loop of the form "text → vision → text."
    - **Mechanism**: Text embedding $\mathbf{t}$ and image embedding $\mathbf{i}$ are concatenated into a joint representation $\mathbf{z} = [\mathbf{t}; \mathbf{i}]$, which is passed through an MLP $g_{\text{BMF}}(\cdot)$ to produce a text correction $\Delta\mathbf{t}$. A residual addition yields the refined text $\mathbf{t}' = \mathbf{t} + \Delta\mathbf{t}$. The refined text is decoded by a pseudo image generator to produce $\hat{\mathbf{x}}$, which is then mapped back to text space via an image-to-text head $h(\cdot)$ to obtain $\hat{\mathbf{t}}$.
    - **Cycle Consistency Loss**: $\mathcal{L}_{\text{cycle}} = \|\mathbf{t} - \hat{\mathbf{t}}\|_2^2$, ensuring no information loss along the text → vision → text pathway.
    - **Design Motivation**: In unidirectional fusion, text is static and cannot adapt to visual evidence. The bidirectional closed loop enables text embeddings to be aware of image content, achieving adaptive alignment under degraded imaging conditions.

2. **Image Augmentation Consistency (IAC) Module**:
    - **Function**: Compels the model to learn stable representations across appearance perturbations of varying intensity.
    - **Mechanism**: The concatenated multimodal input $\mathbf{x}_{\text{cat}}$ first undergoes a spatial augmentation $\mathcal{A}_g$ (applied jointly to the image and mask to preserve alignment). The real image portion then receives either weak augmentation $\mathcal{A}_w$ or strong augmentation $\mathcal{A}_s$, while the pseudo image portion is only normalized via $\mathcal{N}_p$ to serve as a stable semantic reference. Both views are passed through the same U-Net to obtain features $\mathbf{f}_w$ and $\mathbf{f}_s$; after global pooling and linear projection through a projection head, consistency is enforced via cosine distance: $\mathcal{L}_{\text{IAC}} = 1 - \frac{\mathbf{p}_w^\top \mathbf{p}_s}{\|\mathbf{p}_w\|_2 \|\mathbf{p}_s\|_2}$.
    - **Design Motivation**: Clinical CT images suffer from degradations such as low-dose noise and motion blur; IAC encourages intermediate representations to remain consistent under such perturbations.

3. **Pseudo Image Generator**:
    - **Function**: Decodes the refined text embedding into a pseudo image with the same spatial resolution as the original image.
    - **Mechanism**: An L1 reconstruction loss $\mathcal{L}_{\text{gen}}$ supervises alignment between the pseudo image and the reference signal.
    - **Design Motivation**: Serving as a visual bridge for cross-modal semantics, the pseudo image encodes joint text-vision semantics and, when concatenated with the original image, provides the U-Net with additional semantic channels.

### Loss & Training

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{\text{gen}}\mathcal{L}_{\text{gen}} + \lambda_{\text{IAC}}\mathcal{L}_{\text{IAC}} + \lambda_{\text{cycle}}\mathcal{L}_{\text{cycle}}$

- $\mathcal{L}_{\text{seg}}$: Composite segmentation loss (Dice + Cross-Entropy).
- $\mathcal{L}_{\text{gen}}$: L1 pseudo image reconstruction loss.
- Training configuration: AdamW optimizer, initial lr = 1×10⁻⁴, cosine annealing warm restart scheduler, batch size = 16, 150 epochs, single NVIDIA RTX 4090.
- Final predictions are derived from the weak augmentation branch only: $\hat{\mathbf{y}} = \sigma(\text{Conv}_{1 \times 1}(\mathbf{f}_w))$.

## Key Experimental Results

### Main Results: Segmentation Performance on Two COVID-19 Chest CT Datasets

| Method | Conference | Text | QaTa-COV19 Dice(%) | QaTa mIoU(%) | MosMedData+ Dice(%) | MosMed mIoU(%) |
|------|------|:---:|:---:|:---:|:---:|:---:|
| U-Net | MICCAI'15 | × | 79.02 | 69.46 | 64.60 | 50.73 |
| nnU-Net | Nature'21 | × | 80.42 | 70.81 | 72.59 | 60.36 |
| LViT | TMI'23 | ✓ | 83.66 | 75.11 | 74.57 | 61.33 |
| RecLMIS | TMI'24 | ✓ | 85.22 | 77.00 | 77.48 | 65.07 |
| EF-UNet | arXiv'25 | ✓ | 90.46 | 82.58 | 80.50 | 67.37 |
| **BiCLIP** | — | ✓ | **90.59** | **82.81** | **80.80** | **67.79** |

### Robustness Evaluation: Performance Comparison Under Extreme Conditions

| Scenario | Condition | BiCLIP Dice(%) | EF-UNet Dice(%) | Gain |
|------|------|:---:|:---:|------|
| Low annotation | 1% data (QaTa) | **74.79** | 66.76 | +8.03 |
| Low annotation | 1% data (MosMed) | **46.49** | 33.68 | +12.81 |
| Low-dose CT | DL-140 (QaTa) | **81.90** | 70.97 | +10.93 |
| Motion blur | K7 (QaTa) | **88.01** | 87.20 | +0.81 |

### Key Findings

- Under full data, the gap between BiCLIP and EF-UNet is marginal (+0.13% Dice), yet advantages are substantial under extreme conditions—BiCLIP leads by 8 points on QaTa at 1% annotations, indicating that BMF's bidirectional alignment effectively compensates for annotation scarcity.
- Under low-dose CT noise (DL-140), BiCLIP surpasses EF-UNet by nearly 11 percentage points, demonstrating the effectiveness of the IAC module in enhancing robustness against acquisition degradation.
- Compared to the vision-only nnU-Net, BiCLIP achieves over 10% Dice improvement on QaTa, validating the complementary value of textual information.
- BMF contributes primarily to accuracy gains, while IAC mainly improves robustness under degradation; the two modules are complementary.

## Highlights & Insights

- The extension of vision-language fusion from a unidirectional to a bidirectional closed loop, regularized by the cycle consistency loss $\|\mathbf{t} - \hat{\mathbf{t}}\|_2^2$, is conceptually elegant and empirically effective.
- The use of pseudo images as a cross-modal bridge is an insightful design: it simultaneously provides U-Net with additional semantic channels and reinforces BMF representation learning through the generative objective.
- Substantial margins over baselines at 1% annotation (+8–13%) demonstrate that textual information can effectively compensate for label scarcity.
- The robustness evaluation is grounded in clinically realistic scenarios (low-dose CT simulating radiation reduction, motion blur simulating patient movement).

## Limitations & Future Work

- Validation is limited to two COVID-19 chest CT datasets; experiments on other modalities (MRI, ultrasound) and anatomical regions are absent, raising questions about generalizability.
- The influence of text source and prompt design on performance is not systematically analyzed, despite considerable variation in clinical report quality and format in practice.
- The pseudo image generator introduces additional parameters and computational overhead, leaving room for lightweight optimization.
- The cycle consistency constraint enforces information preservation along the "text → vision → text" path but does not explicitly constrain the semantic quality of the generated pseudo images.
- No systematic comparison is made against language-guided adaptation strategies built on foundation models such as SAM.

## Related Work & Insights

- **vs. LViT (TMI'23)**: Unidirectional text guidance vs. BiCLIP's bidirectional fusion raises QaTa Dice from 83.66% to 90.59% (+6.93%), demonstrating clear benefits of bidirectional interaction.
- **vs. RecLMIS (TMI'24)**: BiCLIP achieves Dice improvements of 5.37% and 3.32% on the two datasets, respectively.
- **vs. EF-UNet (arXiv'25)**: The gap under full data is negligible, but BiCLIP's advantage is pronounced under extreme conditions—robustness is the key differentiator.
- **Insights**: The bidirectional fusion + cycle consistency paradigm is transferable to report-guided segmentation, multimodal detection, and other cross-modal tasks; the IAC augmentation consistency concept has natural connections to self-supervised and semi-supervised learning.

## Rating

⭐⭐⭐ (3/5)

Overall assessment: Bidirectional fusion and augmentation consistency are individually not novel contributions; their combination is effective in the medical setting but yields limited incremental gains under full data (+0.13% Dice). The core value lies in robustness advantages under extreme conditions (low annotation/degradation). The clinically motivated experimental design is commendable, but conclusions drawn from only two COVID-19 CT datasets have limited generalizability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
