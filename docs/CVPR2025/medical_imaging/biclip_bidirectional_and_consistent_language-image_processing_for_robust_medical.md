---
title: >-
  [Paper Note] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation
description: >-
  [CVPR 2025][Medical Imaging][Medical Image Segmentation] BiCLIP proposes a bidirectional consistent vision-language segmentation framework. Through bidirectional multimodal fusion (BMF, letting visual features reversely refine text embeddings) and image augmentation consistency (IAC, regularization across weak/strong perturbations), it maintains robust performance on COVID-19 CT segmentation with only 1% of labeled data and shows tolerance to clinical image degradation (noise…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Medical Image Segmentation"
  - "Vision-Language Models"
  - "Bidirectional Fusion"
  - "Semi-Supervised"
  - "Robustness"
date: 2026-05-08
content_hash: ed8c22fcd847ba40
---

# BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.00156](https://arxiv.org/abs/2603.00156)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: Medical Image Segmentation, Vision-Language Models, Bidirectional Fusion, Semi-Supervised, Robustness

## TL;DR
BiCLIP proposes a bidirectional consistent vision-language segmentation framework. Through bidirectional multimodal fusion (BMF, letting visual features reversely refine text embeddings) and image augmentation consistency (IAC, regularization across weak/strong perturbations), it maintains robust performance on COVID-19 CT segmentation with only 1% of labeled data and shows tolerance to clinical image degradation (noise/blur).

## Background & Motivation
**Background**: Vision-language models like CLIP have been introduced to medical image segmentation to guide segmentation using textual priors.

**Limitations of Prior Work**: (1) Existing VL segmentation methods mostly employ unidirectional fusion (text $\rightarrow$ vision), failing to fully leverage visual feedback to text; (2) performance drops sharply under annotation scarcity and image quality degradation; (3) degradations such as noise and motion blur are common in clinical scenarios.

**Key Challenge**: Unidirectional fusion prevents text embeddings from adapting to specific image content, and there is a lack of consistency constraints against input perturbations.

**Goal**: How to achieve robust medical image segmentation under extreme annotation scarcity and image degradation?

**Key Insight**: Bidirectional fusion allows vision to feed back into text + consistency regularization ensures perturbation invariance.

**Core Idea**: Bidirectional vision-text fusion (vision $\rightarrow$ text pseudo-image generator) + weak/strong augmentation consistency regularization.

## Method

### Overall Architecture
Input CT image + text description $\rightarrow$ CLIP vision/text encoders $\rightarrow$ Bidirectional Multimodal Fusion (BMF) module: text $\rightarrow$ vision conventional path + vision $\rightarrow$ text reverse path (via pseudo-image generator) $\rightarrow$ IAC module imposing consistency on features of weak/strong augmented versions $\rightarrow$ segmentation output.

### Training Details
- Vision encoder: A ResNet variant of CLIP ViT-B/16, pre-trained on medical image-text pairs from CXR-BERT
- Segmentation network: U-Net based, with a 4-layer encoder-decoder structure, integrating BMF at each layer
- Optimizer: AdamW, learning rate $1 \times 10^{-4}$ with cosine decay, 1000 warmup steps
- Batch size 16, trained for 200 epochs
- Weak augmentation: Random horizontal flip + minor rotation ($\pm 10^\circ$); Strong augmentation: CutMix + Gaussian noise + contrast variation + elastic deformation

### Key Designs

1. **Bidirectional Multimodal Fusion (BMF)**:

    - **Function**: In addition to standard text-guided visual features, a visual feedback-to-text path is introduced
    - **Mechanism**: Text embeddings $\mathbf{t}$ and image embeddings $\mathbf{i}$ are concatenated and passed through an MLP to predict text refinement $\Delta\mathbf{t} = g_{\text{BMF}}([\mathbf{t};\mathbf{i}])$, followed by a residual update to get $\mathbf{t}' = \mathbf{t} + \Delta\mathbf{t}$. The refined text is processed via a pseudo-image generator to produce a pseudo-image $\hat{\mathbf{x}}$, which is then mapped back into the text space under a closed-loop constraint using a cycle-consistency loss $\mathcal{L}_{\text{cycle}} = \|\mathbf{t} - \hat{\mathbf{t}}\|_2^2$
    - **Design Motivation**: Generic textual descriptions (e.g., "COVID-19 lesion") are not specific enough, requiring image-level contextual information for adaptation. Cycle-consistency ensures the consistency of bidirectional fusion.

2. **Image Augmentation Consistency (IAC)**:

    - **Function**: Forces the feature representations of the weak and strong augmented versions of the same image to remain consistent
    - **Mechanism**: Features extracted from weak augmentation $\mathbf{x}_w$ and strong augmentation $\mathbf{x}_s$ by the U-Net are projected into a compact space, constrained by cosine distance $\mathcal{L}_{\text{IAC}} = 1 - \frac{\mathbf{p}_w^\top \mathbf{p}_s}{\|\mathbf{p}_w\|_2 \|\mathbf{p}_s\|_2}$. The pseudo-image serves as a stable semantic reference and does not undergo augmentation.
    - **Design Motivation**: Enhances robustness against common clinical degradations (low-dose CT noise, motion blur).

### Loss & Training
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{\text{gen}} \mathcal{L}_{\text{gen}} + \lambda_{\text{IAC}} \mathcal{L}_{\text{IAC}} + \lambda_{\text{cycle}} \mathcal{L}_{\text{cycle}}$, where the segmentation loss utilizes Dice + CE, and $\mathcal{L}_{\text{gen}}$ is the L1 reconstruction loss supervising the pseudo-image generation.

## Key Experimental Results

### Main Results

| Dataset | Dice↑ | mIoU↑ | Note |
|--------|-------|-------|------|
| QaTa-COV19 | **90.59%** | **82.81%** | Outperforms all methods including EF-UNet |
| MosMedData+ | **80.80%** | **67.79%** | Outperforms multi-modal methods like RecLMIS and LGA |

Compared with the strongest single-modal baseline nnU-Net, Dice increases by 10%+ on QaTa-COV19 and 8%+ on MosMedData+. Compared with multi-modal methods such as RecLMIS/LGA/MedLangViT, Dice consistently improves by 3-6%.

### Robustness with Low Data Volume

| Training Ratio | BiCLIP Dice | EF-UNet Dice | Note |
|---------|------------|-------------|------|
| 100% | 90.59% | 90.46% | Comparable |
| 25% | - | 88.78% | BiCLIP still superior |
| 5% | - | 84.87% | BiCLIP shows larger advantage |
| 1% | - | Severe drop | BiCLIP maintains meaningful performance |

### Robustness to Image Degradation

| Degradation Type | BiCLIP Dice | Strongest Baseline Dice | Note |
|---------|------------|------------------|------|
| Gaussian noise σ=0.1 | 87.23% | 82.15% | +5.08% |
| Motion blur kernel=15 | 85.91% | 79.32% | +6.59% |
| Low contrast γ=0.5 | 88.76% | 85.41% | +3.35% |

The IAC module contributes the most under degradation conditions—weak/strong augmentation consistency directly enhances tolerance to noise and blur.

## Highlights & Insights
- **Vision $\rightarrow$ Text Reverse Path**: Allows text embeddings to be dynamic rather than fixed, adjusting according to specific images, bridging the semantic gap between generic text and specific images. This is more structured than simple feature concatenation or cross-attention.
- **Practicality in Extremely Low Annotations**: Operates robustly with only 1% labels, which is highly practical for clinical medical scenarios where annotation is expensive.
- **Pseudo-image as a Bridge**: The pseudo-image is simultaneously used for (1) closed-loop cycle-consistency constraints and (2) concatenation with the original image for input into the segmentation network, serving two purposes with one intermediate representation.
- **Robustness to Clinical Degradation**: Specifically evaluates performance on low-dose CT noise and motion blur, which are common issues in real clinic scenarios.

## Limitations & Future Work
- Evaluated on only two COVID-19 CT datasets; generalization to other organs/modalities remains untested.
- Relies on the CXR-BERT text encoder; imaging domains beyond the chest may require different domain-specific encoders.
- Lacks comparison with recent SAM-based medical segmentation methods.
- Pseudo-image generation increases computational overhead, and the actual inference speed is not reported.
- The selection of weak/strong augmentation strategies in IAC may require adjustments for different modalities.

## Related Work & Insights
- **vs. CLIP-Driven Methods**: Most CLIP-based segmentation methods perform only unidirectional fusion (text $\rightarrow$ vision). BiCLIP introduces a vision $\rightarrow$ text path, making text representations adaptive to specific images.
- **vs. FixMatch/Mean Teacher**: IAC is similar to consistency regularization in semi-supervised learning, but integrates the pseudo-image as a stable anchor.
- **vs. RecLMIS/LGA**: Achieves a 3-6% Dice improvement on the same benchmarks, mainly owing to deeper cross-modal interaction through bidirectional fusion.
- **vs. EF-UNet**: Performance is comparable under full annotations (90.59% vs 90.46%), but BiCLIP shows a significant advantage under extremely low annotations (1%).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The bidirectional fusion idea is somewhat novel but not a major breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on only 2 datasets, leading to a relatively small experimental scale.
- **Writing Quality**: ⭐⭐⭐⭐ Clear description of the methodology.
- **Value**: ⭐⭐⭐⭐ Offers practical reference value for medical image segmentation under scant annotations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)

</div>

<!-- RELATED:END -->
