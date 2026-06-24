---
title: >-
  [Paper Note] Unsupervised Exposure Correction
description: >-
  [ECCV 2024][Signal & Communication][Unsupervised Exposure Correction] This paper proposes the first unsupervised exposure correction (UEC) method, which leverages multi-exposure sequences generated freely by ISP pipelines to train images as mutual ground truths. It designs a pixel-level transformation function with only 19K parameters to preserve image details, outperforming supervised SOTA on exposure correction and downstream edge detection.
tags:
  - "ECCV 2024"
  - "Signal & Communication"
  - "Unsupervised Exposure Correction"
  - "Radiometric Modeling"
  - "Pixel-level Color Transformation"
  - "Multi-exposure Sequences"
  - "Edge Detection"
date: 2026-05-08
content_hash: c9970d073d870af1
---

# Unsupervised Exposure Correction

**Conference**: ECCV 2024  
**arXiv**: [2507.17252](https://arxiv.org/abs/2507.17252)  
**Code**: [https://github.com/BeyondHeaven/uec_code](https://github.com/BeyondHeaven/uec_code)  
**Area**: Signal Communication / Image Exposure Correction  
**Keywords**: Unsupervised Exposure Correction, Radiometric Modeling, Pixel-level Color Transformation, Multi-exposure Sequences, Edge Detection

## TL;DR

This paper proposes the first unsupervised exposure correction (UEC) method, which leverages multi-exposure sequences generated freely by ISP pipelines to train images as mutual ground truths. It designs a pixel-level transformation function with only 19K parameters to preserve image details, outperforming supervised SOTA on exposure correction and downstream edge detection.

## Background & Motivation

**Background**: Exposure is a critical factor affecting image quality. Although ISPs can automatically adjust exposure values (EV), post-processing of sRGB images remains essential under non-ideal lighting conditions. Deep learning methods have achieved significant progress in this area, but existing methods face three core challenges.

**Challenge 1: High Annotation Costs**. Supervised methods rely on professional photographers to manually adjust and generate paired data as ground truth, which is complex and time-consuming. Unlike the simple labeling in classification tasks, each image requires elaborate editing and correction. The MSEC dataset is annotated by 5 experts individually, but as shown in the figure, their annotations show prominent differences in color and style.

**Challenge 2: Limited Generalization**. (a) The low efficiency of manual annotation leads to a small scale of datasets available in academia; (b) Manual adjustment inevitably introduces personal stylistic bias—different retouchers have different understandings of "correct exposure", meaning the ground truth itself is noisy.

**Challenge 3: Low-level Feature Degradation**. Existing methods primarily pursue generating visually pleasing images, but the output images often suffer from significant degradation in low-level features such as edges. This makes the enhanced images perform poorly in downstream tasks that rely on low-level features, such as edge detection and semantic segmentation.

**Key Insight**: Acquiring paired data does not necessarily require human intervention. By simulating the ISP pipeline to generate multi-exposure sequences with different EVs on RAW data, images within the same sequence can serve as mutual ground truths—these images differ only in exposure (radiometry), eliminating stylistic bias.

## Method

### Overall Architecture

The UEC framework consists of three core networks: (1) an **exposure feature encoder** $e(\cdot)$ to extract exposure-related features from the image; (2) a **parameter predictor** $d(\cdot, \cdot)$ to calculate the exposure difference between two images and predict the transformation parameters $\lambda$; (3) an **exposure corrector** $f(\cdot)$ to perform pixel-level correction on the image based on the exposure difference. The entire framework is trained end-to-end, and during inference, only one reference image is needed to determine the target exposure.

### Key Designs

#### 1. **Unsupervised Exposure Correction Modeling**

**Function**: Design a self-supervised training paradigm that requires no ground truth.

**Mechanism**: Leverage multi-exposure sequences generated freely from RAW data via the ISP pipeline, allowing images in the sequence to serve as mutual ground truths. Two training principles are proposed:

**Principle 1 — Restoration Supervision**: Sample input $I_1$ and reference $I_2$ from the same multi-exposure sequence, and train the transformation function such that $I_1' = I_2$. This serves as a pretext task to establish the basic mapping:

$$I_1' = f(\Delta E, I_1), \quad \Delta E = d(e(I_1), e(I_2))$$

**Principle 2 — Monopoly Principle**: Address the cross-scene generalization issue. Select two references $J_1, J_2$ ($\text{EV}(J_1) > \text{EV}(J_2)$) from different sequences $J$, transform the same input $I_1$ respectively, and require the outputs to satisfy pixel-wise luminance monotonicity:

$$\forall(x,y), \quad I_{J1}'(x,y) \geq I_{J2}'(x,y)$$

**Design Motivation**: Restoration supervision solves exposure mapping in the same scene, while the monopoly principle ensures cross-scene generalization. Since the transformation calculates exposure differences in latent space instead of pixel space, it only modifies exposure without changing content, making it adaptable to reference images from different scenes.

#### 2. **Pixel-level Exposure Transformation Function**

**Function**: Design an exposure correction operation that preserves image details.

**Mechanism**: Model exposure correction as an interpolation of direct scaling and non-linear adjustment, implemented via $1 \times 1$ convolutions:

$$I_{out}(x,y) = \lambda \times I_{in}(x,y) + (1-\lambda) \times h(I_{in}(x,y))$$

where $\lambda$ is the hybrid weight output by the parameter predictor, and $h(\cdot)$ is a non-linear transformation implemented by a $1 \times 1$ convolutional layer. This process is iterated 3 times to enhance the effect.

**Design Motivation**: Unlike image-to-image translation methods, pixel-level transformations (a) preserve original pixel relationships without introducing artifacts; (b) can flexibly handle inputs of arbitrary resolutions, supporting 4K real-time processing; (c) have an extremely small number of parameters (only 19K), far lower than methods like ECM (182M).

#### 3. **Exposure Feature Encoder**

**Function**: Extract compact features representing the global exposure properties of the image.

**Mechanism**: The encoder consists of two convolutional layers followed by global pooling, generating a 96-dimensional feature representation through a combination of three statistics: maximum, mean, and standard deviation.

**Design Motivation**: These statistics correlate with the global properties of the image (such as contrast, histogram distribution) and can effectively depict exposure features. Calculating exposure differences in latent space rather than pixel space omits low-level feature information, making $d(\cdot,\cdot)$ adaptable across different scenes.

### Loss & Training

Total loss: $L = \alpha_1 \cdot L_{\text{restoration}} + \alpha_2 \cdot L_{\text{monopoly}} + \alpha_3 \cdot L_{\text{semantic}}$

1. **Restoration Loss** (paired within the same sequence): $L_{\text{restoration}} = \frac{1}{CHW}\|I^{\text{out}} - I^{\text{ref}}\|_2$

2. **Monopoly Loss** (paired across sequences): $L_{\text{monopoly}} = \frac{1}{CHW}\text{ReLU}(I^{\text{out2}} - I^{\text{out1}})$, constraining output luminance monotonicity when $\text{EV}(I^{\text{ref1}}) > \text{EV}(I^{\text{ref2}})$

3. **Semantic Preservation Loss** (Total Variation regularization): $L_{\text{semantic}} = \frac{1}{CHW}\|\nabla I^{\text{out}}\|_2$, maintaining spatial coherence

Weight settings: $\alpha_1 = \alpha_2 = 1, \alpha_3 = 0.1$.

**Testing Strategy**: After training, a well-exposed reference image is fixed, and its exposure features are directly used as the target exposure for all test samples, eliminating the need for sample-by-sample selection.

## Key Experimental Results

### Main Results: Exposure Correction on MSEC Dataset

| Method | Supervision Type | PSNR↑ | SSIM↑ |
|------|----------|-------|-------|
| HDRCNN w/PS | Supervised | 17.032 | 0.687 |
| DPED (iPhone) | Supervised | 16.274 | 0.629 |
| DPE (S-FiveK) | Supervised | 17.510 | 0.677 |
| Zero-DCE | Supervised | 12.597 | 0.549 |
| Afifi et al. | Supervised | 19.483 | 0.739 |
| ECM | Supervised | **20.874** | **0.877** |
| **UEC (Ours)** | **Unsupervised** | 18.756 | 0.812 |

UEC achieves performance close to the SOTA supervised method, ECM, under unsupervised conditions.

### Results on Radiometric Calibration Dataset (Ablation: Pure Radiometric Performance)

| Method | Avg PSNR↑ | Avg SSIM↑ |
|------|-----------|-----------|
| ECM (Supervised) | 20.445 | 0.744 |
| **UEC (Unsupervised)** | **20.548** | **0.868** |

On the pure radiometric calibration task, UEC significantly outperforms ECM in SSIM (0.868 vs 0.744) and is slightly superior in PSNR.

### Ablation Study: Downstream Edge Detection Task

| Method | Avg PSNR↑ | Avg F1-Score↑ | Description |
|------|-----------|---------------|------|
| ECM (Supervised) | 16.312 | 0.922 | Image-to-image translation loses details |
| **UEC (Unsupervised)** | **22.665** | **0.969** | Pixel-level transformation preserves low-level features |

UEC improves edge detection PSNR by 6.3dB (+39%) and F1-score from 0.922 to 0.969.

### Generalization Experiment (MSEC Training → LOL Testing)

| Method | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Afifi et al. (MSEC pre-trained) | 14.268 | 0.638 |
| ECM (MSEC pre-trained) | 15.439 | 0.650 |
| ECM (Radiometry pre-trained) | 17.537 | 0.725 |
| **UEC (MSEC pre-trained)** | **18.571** | **0.728** |

UEC exhibits the strongest generalization, with a cross-dataset PSNR improvement of 3.1dB (vs ECM-MSEC).

### Key Findings

1. **Extremely high efficiency**: With only 19K parameters (0.01% of ECM) and a model size of 0.079MB vs 695MB, GPU inference speed is 4.85x faster (1.46ms vs 7.08ms), and CPU speed is 23.4x faster.
2. **Advantages of radiometry separation**: Learning only radiometric changes makes the model more generalizable and avoids the stylistic bias of human annotations.
3. **Effectiveness of the monopoly principle**: The cross-scene luminance monotonicity constraint ensures reasonable exposure transfer.
4. **Low-level feature preservation**: Compared to image translation methods, the pixel-level transformation improves edge detection significantly.

## Highlights & Insights

1. **Intelligent unsupervised design**: The concept of using multi-exposure sequence images as mutual ground truths is simple yet effective, completely eliminating expensive manual annotation.
2. **Ultimate parameter efficiency**: Achieving competitive performance with only 19K parameters (less than one ten-thousandth of ECM) while supporting 4K real-time processing.
3. **Radiometry-chrominance decoupling**: Modifying only the radiometry (luminance) while freezing other ISP post-processing removes the stylistic bias problem.
4. **Focus on downstream tasks**: Instead of only pursuing visual aesthetics, it focuses on the utility of enhanced images in downstream tasks like edge detection, presenting a meaningful new perspective in the field of exposure correction.

## Limitations & Future Work

1. **Only handles radiometric correction**: It does not involve color correction (chromatic adjustment), which limits its performance in scenarios that require color enhancement.
2. **Degradation under extreme exposures**: In the wide range of -2EV to +3EV, severely over-exposed/under-exposed images are difficult to fully restore due to texture loss.
3. **Single reference image dependency**: During testing, a suitable reference image must be chosen. Although the authors claim robustness to reference selection, this remains an additional assumption.
4. **Limitations of evaluation metrics**: PSNR/SSIM may not fully reflect human perception of exposure quality.

## Related Work & Insights

- **Afifi et al. (MSEC)**: Proposed a multi-exposure dataset and a supervised exposure correction method. This paper converts their ISP pipeline design into an unsupervised paradigm.
- **ECM**: The current SOTA supervised method, utilizing image-to-image translation, resulting in a bulky model and lost details.
- **Zero-DCE**: A reference-free method but relies on hand-crafted loss functions, which is not unsupervised.
- **Inspiration**: The concept of radiometry-chrominance decoupling can be extended to other image enhancement tasks. The unsupervised strategy of "using data as mutual labels" is worth exploring in more low-level vision tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first unsupervised exposure correction method, with an ingenious mutual ground truth design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset evaluation + generalization analysis + downstream edge detection task + efficiency comparison.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, with refined expressions of restoration supervision and monopoly principles.
- **Value**: ⭐⭐⭐⭐ — Achieving competitive performance with extremely few parameters, possessing actual deployment value, and providing an inspiring perspective on radiometric correction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)
- [\[ECCV 2024\] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [\[ECCV 2024\] RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)
- [\[ECCV 2024\] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)
- [\[ECCV 2024\] Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)

</div>

<!-- RELATED:END -->
