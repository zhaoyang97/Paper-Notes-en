---
title: >-
  [Paper Note] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Medical image segmentation] This paper proposes BiCLIP, a framework that employs Bidirectional Multimodal Fusion (BMF) to refine text representations using visual information…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Medical image segmentation"
  - "vision-language models"
  - "bidirectional multimodal fusion"
  - "augmentation consistency"
  - "low-annotation robustness"
date: 2026-05-08
content_hash: 3e9214b2bca55dfa
---

# BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation

**Conference**: CVPR 2026  
**arXiv**: [2603.00156](https://arxiv.org/abs/2603.00156)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: Medical image segmentation, vision-language models, bidirectional multimodal fusion, augmentation consistency, low-annotation robustness

## TL;DR

This paper proposes BiCLIP, a framework that employs Bidirectional Multimodal Fusion (BMF) to refine text representations using visual information, and Image Augmentation Consistency (IAC) to enforce perturbation-invariant intermediate features. BiCLIP surpasses state-of-the-art methods on COVID-19 CT segmentation while remaining robust with as little as 1% labeled data.

## Background & Motivation

### 1. State of the Field
Medical image segmentation is foundational to computer-aided diagnosis and treatment planning. While vision-only methods such as U-Net have proven successful, they remain highly sensitive to image quality and acquisition conditions. Vision-language approaches (LViT, Cap2Seg, RecLMIS, LGA, etc.) have emerged as a promising paradigm by incorporating textual descriptions as supplementary semantic context.

### 2. Limitations of Prior Work
Existing vision-language segmentation methods almost universally adopt **unidirectional fusion**: text embeddings condition visual representations, but visual information cannot reciprocally refine textual semantics. This asymmetric design reveals weaknesses in two scenarios: (1) **label scarcity**, where static text conditioning is insufficient to compensate for limited supervision; and (2) **acquisition degradation** (low-dose CT noise, motion blur), where visual features are inherently noisy and demand more robust cross-modal interaction.

### 3. Root Cause
Deep interaction between visual and textual features is necessary for robustness, yet naively increasing interaction complexity leads to overfitting and unstable learning, particularly in data-limited medical settings.

### 4. Starting Point
(1) Design a bidirectional fusion loop in which visual evidence reciprocally refines text representations; (2) introduce augmentation consistency regularization to stabilize intermediate features across different perturbations.

## Method

### Overall Architecture

BiCLIP takes a medical image and its clinical text description as input. The text is encoded by a frozen CXR-BERT to produce text embedding $\mathbf{t}$, and the image is processed by a lightweight convolutional encoder to produce visual embedding $\mathbf{i}$. Both are fed into the BMF module for bidirectional fusion, generating a pseudo image that encodes cross-modal semantics. The pseudo image and the original image are concatenated and passed to a U-Net backbone for segmentation prediction; simultaneously, the IAC module enforces feature consistency between weakly and strongly augmented views.

### Key Designs

#### 1. BMF (Bidirectional Multimodal Fusion)

**Function**: Establishes a closed-loop interaction in which visual information reciprocally refines text representations.

**Mechanism**:
- **Forward fusion**: The text embedding $\mathbf{t}$ and image embedding $\mathbf{i}$ are concatenated into a joint representation $\mathbf{z} = [\mathbf{t}; \mathbf{i}]$, which is passed through an MLP $g_{\text{BMF}}(\cdot)$ to predict a residual $\Delta\mathbf{t} = g_{\text{BMF}}(\mathbf{z})$, yielding a refined text embedding $\mathbf{t}' = \mathbf{t} + \Delta\mathbf{t}$.
- **Pseudo image generation**: $\mathbf{t}'$ is transformed into a pseudo image $\hat{\mathbf{x}}$ via a pseudo image generator supervised by ground-truth signals (via an $L_1$ reconstruction loss $\mathcal{L}_{\text{gen}}$), encoding cross-modal semantics in a visual format.
- **Backward loop**: The pseudo image is mapped back to the text space via an image-to-text head $h(\cdot)$ to obtain $\hat{\mathbf{t}}$, with a cycle consistency loss enforced: $\mathcal{L}_{\text{cycle}} = \|\mathbf{t} - \hat{\mathbf{t}}\|_2^2$.

**Design Motivation**: The residual connection preserves the original linguistic structure while injecting visual cues. Cycle consistency ensures semantic coherence of the bidirectional mapping and prevents the refinement from drifting away from the original text semantics. The pseudo image serves as a bridge, materializing cross-modal semantics into a segmentation-ready visual signal.

#### 2. IAC (Image Augmentation Consistency)

**Function**: Constrains intermediate features to remain consistent under augmentations of varying intensity, improving robustness to appearance variation.

**Mechanism**:
- **Input construction**: The pseudo image $\hat{\mathbf{x}}$ and original image $\mathbf{x}$ are concatenated along the channel dimension to form $\mathbf{x}_{\text{cat}}$. Spatial augmentation is first applied jointly to the image and mask to preserve spatial alignment. The real image portion then undergoes weak augmentation $\mathcal{A}_w$ and strong augmentation $\mathcal{A}_s$ respectively, while the pseudo image portion undergoes normalization $\mathcal{N}_p$ as a stable semantic anchor:
    - $\mathbf{x}_w = \text{concat}(\mathcal{A}_w(\mathbf{x}_g^r), \mathcal{N}_p(\mathbf{x}_g^p))$
    - $\mathbf{x}_s = \text{concat}(\mathcal{A}_s(\mathbf{x}_g^r), \mathcal{N}_p(\mathbf{x}_g^p))$
- **Consistency constraint**: Both views are passed through the same U-Net; feature maps $\mathbf{f}_w, \mathbf{f}_s$ are extracted from the final upsampling stage of the decoder and projected to compact embeddings $\mathbf{p}_w, \mathbf{p}_s$ via a lightweight projection head (global pooling + linear layer). The cosine distance is minimized: $\mathcal{L}_{\text{IAC}} = 1 - \frac{\mathbf{p}_w^\top \mathbf{p}_s}{\|\mathbf{p}_w\|_2 \|\mathbf{p}_s\|_2}$
- **Segmentation prediction**: The predicted mask is produced from the weakly augmented branch's feature map via a $1 \times 1$ convolution followed by sigmoid.

**Design Motivation**: The weak/strong augmentation pair constructs two views of varying difficulty; consistency regularization compels the network to learn augmentation-invariant representations, which is especially beneficial under limited data — acting as implicit data augmentation. The pseudo image portion is kept normalized without augmentation to maintain a stable cross-modal semantic anchor.

### Loss & Training

The total training loss is a weighted sum of four terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{\text{gen}}\mathcal{L}_{\text{gen}} + \lambda_{\text{IAC}}\mathcal{L}_{\text{IAC}} + \lambda_{\text{cycle}}\mathcal{L}_{\text{cycle}}$$

- $\mathcal{L}_{\text{seg}}$: Dice + Cross-Entropy segmentation loss
- $\mathcal{L}_{\text{gen}}$: $L_1$ reconstruction loss for the pseudo image
- $\mathcal{L}_{\text{IAC}}$: Augmentation consistency cosine distance loss
- $\mathcal{L}_{\text{cycle}}$: $L_2$ cycle consistency loss for bidirectional fusion

### Training Details
- AdamW optimizer, initial learning rate $1 \times 10^{-4}$, cosine annealing warm restart
- Batch size 16, trained for 150 epochs on a single RTX 4090
- Text encoder: frozen CXR-BERT

## Key Experimental Results

### Main Results (Comparison with SOTA)

| Method | Text | QaTa-COV19 Dice(%) | QaTa-COV19 mIoU(%) | MosMedData+ Dice(%) | MosMedData+ mIoU(%) |
|------|------|---------------------|---------------------|---------------------|---------------------|
| U-Net | × | 79.02 | 69.46 | 64.60 | 50.73 |
| nnU-Net | × | 80.42 | 70.81 | 72.59 | 60.36 |
| LViT | ✓ | 83.66 | 75.11 | 74.57 | 61.33 |
| RecLMIS | ✓ | 85.22 | 77.00 | 77.48 | 65.07 |
| EF-UNet | ✓ | 90.46 | 82.58 | 80.50 | 67.37 |
| **BiCLIP** | ✓ | **90.59** | **82.81** | **80.80** | **67.79** |

### Low-Annotation Robustness (vs. EF-UNet)

| Annotation Ratio | BiCLIP QaTa Dice | EF-UNet QaTa Dice | BiCLIP MosMed Dice | EF-UNet MosMed Dice |
|----------|------------------|--------------------|--------------------|---------------------|
| 25% | 88.78 | 88.78 | 72.18 | 65.63 |
| 10% | 87.14 | 87.84 | 68.29 | 64.24 |
| 5% | 84.92 | 84.87 | 64.71 | 55.48 |
| 1% | **74.79** | 66.76 | **46.49** | 33.68 |

### Noise Robustness (Low-Dose CT Noise, QaTa-COV19 Dice)

| Method | Noise 140 | Noise 120 | Noise 110 |
|------|-----------|-----------|-----------|
| LViT | 70.07 | 68.27 | 67.60 |
| RecLMIS | 66.44 | 64.23 | 62.53 |
| EF-UNet | 70.97 | 67.68 | 65.70 |
| **BiCLIP** | **81.90** | **78.03** | **74.84** |

### Key Findings
- BiCLIP outperforms all image-only and multimodal baselines on both datasets.
- Compared to the strongest multimodal baseline RecLMIS, BiCLIP achieves +5.37% Dice on QaTa-COV19 and +3.32% on MosMedData+.
- The advantage is most pronounced at **1% annotation**: BiCLIP Dice 74.79% vs. EF-UNet 66.76% (+8.03%), with an even larger gap on MosMedData+ (+12.81%).
- Under low-dose CT noise, BiCLIP substantially outperforms all baselines (Noise 140: 81.90% vs. EF-UNet 70.97%, +10.93%).
- Motion blur robustness is comparable to EF-UNet, with a slight advantage on MosMedData+.

## Highlights & Insights
- The **bidirectional fusion loop** is the core innovation: the text→image→text cycle consistency allows visual evidence to reciprocally refine text semantics, yielding greater robustness than unidirectional (text→vision) fusion.
- **The pseudo image as a cross-modal bridge** elegantly materializes abstract cross-modal semantics into a concatenable visual channel — a conceptually clean and practically straightforward design.
- The **weak/strong augmentation consistency** in IAC is concise yet effective, drawing on FixMatch-style consistency regularization and adapting it to multimodal medical segmentation.
- The robustness demonstrated under extreme label scarcity (1%) and strong noise (low-dose CT) is impressive and directly addresses well-defined clinical pain points.

## Limitations & Future Work
- Validation is limited to two closely related COVID-19 CT datasets; generalization across organs and modalities (MRI, X-ray, ultrasound) remains unverified.
- The frozen CXR-BERT text encoder is pretrained on chest X-rays; extending to non-thoracic imaging may require a more general-purpose medical language model.
- The pseudo image generator relies on ground-truth supervision and cannot be directly applied in unsupervised or self-supervised pre-training settings.
- The overall architecture is relatively simple (MLP + U-Net); more expressive cross-modal interaction mechanisms (e.g., cross-attention, prompt tuning) remain unexplored.
- Ablation studies isolating the individual contributions of BMF and IAC are absent.

## Related Work & Insights
- The cycle consistency idea in bidirectional fusion is generalizable to other vision-language tasks (e.g., referring segmentation, VQA), with the key principle being "letting visual feedback refine linguistic representations."
- IAC's weak/strong augmentation consistency can serve as a general-purpose regularization strategy for any multimodal learning scenario with limited labeled data.
- The pseudo image bridge design is worth exploring in 3D medical segmentation (e.g., nnU-Net + text).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of bidirectional fusion loop and augmentation consistency is novel, though individual components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐ Two datasets, low-annotation, and noise robustness experiments are well-covered, but ablation studies and cross-domain validation are lacking.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with well-formulated equations, though the introduction is somewhat lengthy.
- Value: ⭐⭐⭐⭐ Practical value in low-annotation robustness for medical segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)

</div>

<!-- RELATED:END -->
