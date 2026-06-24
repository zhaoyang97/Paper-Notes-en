---
title: >-
  [Paper Note] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution
description: >-
  [ECCV 2024][Image Restoration][Infrared Image Super-Resolution] To address the unique challenges of infrared image super-resolution, this paper proposes the CoRPLE framework. It utilizes the Contourlet transform for multi-scale and multi-directional infrared spectral residual enhancement, and introduces a prompt learning paradigm based on vision-language models to capture the inherent features of infrared images, achieving SOTA performance on infrared SR tasks.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Infrared Image Super-Resolution"
  - "Contourlet Transform"
  - "Prompt Learning"
  - "Multi-scale Multi-directional Decomposition"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: afb21a51dd2d3997
---

# Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution

**Conference**: ECCV 2024  
**Code**: [https://github.com/hey-it-s-me/CoRPLE](https://github.com/hey-it-s-me/CoRPLE)  
**Area**: Image Super-Resolution / Infrared Image Enhancement  
**Keywords**: Infrared Image Super-Resolution, Contourlet Transform, Prompt Learning, Multi-scale Multi-directional Decomposition, Vision-Language Models

## TL;DR

To address the unique challenges of infrared image super-resolution, this paper proposes the CoRPLE framework. It utilizes the Contourlet transform for multi-scale and multi-directional infrared spectral residual enhancement, and introduces a prompt learning paradigm based on vision-language models to capture the inherent features of infrared images, achieving SOTA performance on infrared SR tasks.

## Background & Motivation

**Background**: Image super-resolution (SR) is a critical technique for image enhancement. In recent years, Transformer-based methods (such as SwinIR, DAT, etc.) have achieved great success in visible-light image SR. However, infrared image SR, as a specialized subfield, faces unique and distinct challenges.

**Limitations of Prior Work**: Infrared images are acquired by infrared sensors, and their inherent characteristics include: (1) limited resolution—the pixel density of infrared detectors is much lower than that of visible-light sensors; (2) temperature sensitivity—image quality is heavily affected by target and environmental temperatures; (3) high noise levels—thermal and readout noises are significant; (4) lack of texture details—infrared images primarily reflect thermal radiation, lacking the rich color and texture info present in visible-light images. These characteristics make the direct application of visible-light SR methods to infrared images ineffective, resulting in blurred reconstructed edges and severe loss of detail.

**Key Challenge**: Existing deep-learning-based SR methods are primarily designed for visible-light images. They learn feature mappings in the spatial domain and ignore the unique frequency-domain distribution of infrared images. Key details of infrared images (edge contours, temperature gradients) are mainly concentrated in specific frequency bands and directions, which conventional SR methods fail to capture and enhance effectively.

**Goal**: (1) Design an SR framework specifically tailored to the frequency-domain characteristics of infrared images; (2) utilize multi-scale, multi-directional frequency-domain decomposition to precisely enhance key high-frequency details of infrared images; (3) introduce semantic understanding to guide the infrared image super-resolution process.

**Key Insight**: The Contourlet transform is a multi-scale, multi-directional image decomposition tool that is superior to the wavelet transform in capturing directional edges and contour information. The authors observe that the critical information of infrared images is concentrated precisely in these directional high-frequency subbands, making the Contourlet transform highly suitable as a frequency-domain analysis tool for infrared SR. Meanwhile, leveraging the semantic understanding of vision-language models (VLMs) can generate targeted super-resolution guidance for infrared images via prompt learning.

**Core Idea**: Utilizing multi-scale and multi-directional residuals from the Contourlet transform to accurately enhance the high-frequency details of infrared images, while providing semantic-level optimization guidance via prompt learning with vision-language models.

## Method

### Overall Architecture

The overall pipeline of CoRPLE consists of two core components: (1) the Contourlet Residual Module, which performs Contourlet decomposition on the input low-resolution infrared image to extract multi-scale, multi-directional high-frequency subbands, enhances critical details in these subbands by learning residuals, and then reconstructs them back to the spatial domain via the inverse transform; (2) the Prompt Learning Enhancement Module, which leverages a vision-language model (such as CLIP) to generate semantic prompts for the infrared images and injects these prompt embeddings into the intermediate layers of the SR network, guiding the model to focus on the unique characteristics of infrared images. The outputs of both components are fused in the feature space and passed through an upsampling module to generate the final high-resolution infrared image.

### Key Designs

1. **Contourlet Residual Module**:

    - **Function**: Precisely localize and enhance directional high-frequency information of infrared images in the frequency domain.
    - **Mechanism**: The input image is first decomposed using the Contourlet transform to yield one low-frequency subband and multiple bandpass directional subbands at different scales and directions. Contourlet decomposition first employs a Laplacian pyramid for multi-scale decomposition (yielding subbands of different resolutions) and then utilizes a Directional Filter Bank (DFB) at each scale to further decompose them into multiple directional subbands. The edge and contour information of infrared images is primarily concentrated in these high-frequency directional subbands. The module learns a residual enhancement for each high-frequency subband, predicting the missing high-frequency details using a small CNN. The enhanced subbands are then synthesized back into the spatial domain via the inverse Contourlet transform. Finally, the spatial-domain residual is added to the original feature.
    - **Design Motivation**: Infrared images lack rich color and texture variations, with their key information concentrated on edges and temperature gradients. Compared to conventional wavelets, the Contourlet transform offers anisotropic advantages and better captures these directional edge details, making residual learning highly efficient.

2. **Prompt Learning Enhancement Module**:

    - **Function**: Provide high-level guidance for infrared SR via the semantic understanding of vision-language models.
    - **Mechanism**: A pre-trained CLIP vision encoder is used to extract features from the infrared image, generating embedding vectors that reflect the semantic content and quality of the image. Subsequently, a set of learnable prompt vectors is designed to capture implicit knowledge regarding "what kind of enhancement the infrared image needs" during training. After interacting with the CLIP features, these prompt vectors generate conditional embeddings that are injected as auxiliary conditions into the Transformer layers of the SR network via attention mechanisms. This design allows the model to adaptively adjust its super-resolution strategy based on the specific content of each infrared image.
    - **Design Motivation**: Conventional SR methods apply uniform processing across all images, neglecting differences among various infrared images (e.g., indoor/outdoor, day/night, and varying temperature distributions). Prompt learning provides a lightweight, adaptive mechanism to address this.

3. **Multi-Scale Feature Fusion Backbone**:

    - **Function**: Act as the elemental architecture of the framework to integrate spatial and frequency-domain features.
    - **Mechanism**: Based on the DAT (Dual Aggregation Transformer) architecture, spatial-domain features are extracted across multiple scales. The outputs of the Contourlet Residual Module and the Prompt Learning Module are fused into the backbone network at various scales via skip connections. The upsampling stage employs PixelShuffle for sub-pixel convolution, ultimately generating the high-resolution infrared image.
    - **Design Motivation**: The DAT architecture demonstrates outstanding performance in visible-light SR. Serving as the baseline architecture, it offers powerful spatial feature extraction capability, complementing the frequency-domain features from the Contourlet transform and the semantic features from prompt learning.

### Loss & Training

Training employs a combination of multiple loss functions: (1) $L_1$ pixel loss to ensure pixel-level reconstruction accuracy; (2) perceptual loss to guarantee visual perceptual quality; (3) Contourlet-domain loss—calculating $L_1$ loss on each high-frequency subband to directly optimize reconstruction quality in the frequency domain. A two-stage training strategy is adopted: the backbone network is first pre-trained to acquire baseline SR capabilities, followed by joint training of the Contourlet Residual Module and the Prompt Learning Module for refinement. Data augmentation techniques include random cropping, flipping, and rotation.

## Key Experimental Results

### Main Results

| Dataset | Metrics | Ours (CoRPLE) | Prev. SOTA | Gain |
|---------|------|-------------|----------|------|
| Infrared SR x2 | PSNR | Highest SOTA | DAT / SwinIR, etc. | +0.3 - 0.8 dB |
| Infrared SR x4 | PSNR | Highest SOTA | DAT / SwinIR, etc. | +0.5 - 1.2 dB |
| Infrared SR x2 | SSIM | Highest SOTA | Various visible SR methods | Consistent Improvement |
| Infrared Detection Task | mAP | Significant Improvement | Low-Resolution Baseline | Detection performance improved after SR |
| Infrared Segmentation Task | mIoU | Significant Improvement | Low-Resolution Baseline | Segmentation performance improved after SR |

Notably, the paper evaluates not only conventional PSNR/SSIM metrics but also the positive impact of SR on downstream infrared vision tasks (detection and segmentation).

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| Backbone only | Baseline PSNR | Performance of standard DAT on infrared SR |
| + Contourlet Residual | PSNR improved by ~0.3dB | Verifies the effectiveness of frequency-domain enhancement |
| + Prompt Learning | PSNR improved by ~0.2dB | Verifies the effectiveness of semantic guidance |
| + Both combined | Optimal PSNR | Frequency domain and semantics complement each other |
| Wavelet vs. Contourlet | Contourlet is better | The advantage of directional decomposition is prominent |
| Number of directions | 8 directions is optimal | Excessive directions increase computation with diminishing returns |

### Key Findings

- The advantage of the Contourlet transform in infrared SR is primarily manifested in edge areas, with quantitative analysis showing the most significant PSNR gain (+1-2 dB) in these regions.
- The effectiveness of prompt learning correlates positively with the scene diversity of infrared images—the more diverse the scenes, the more pronounced the adaptive advantage of prompt learning.
- In downstream task evaluations, the mAP improvement in object detection achieved by CoRPLE-reconstructed infrared images far exceeds that of other SR methods.
- The model exhibits a more significant advantage under x4 magnification than x2, indicating that frequency-domain and semantic guidance become more critical in large-scale SR.

## Highlights & Insights

1. **Targeted Design for Infrared SR**: Rather than simply transferring visible-light SR methods to the infrared domain, the authors carry out an in-depth analysis of infrared image properties and design tailor-made solutions.
2. **Clever Application of Contourlet Transform**: Utilizing the anisotropic decomposition capability of the Contourlet transform to accurately enhance the directional edge details of infrared images serves as an excellent practice of frequency-domain methods in the infrared domain.
3. **Introduction of the Prompt Learning Paradigm**: Introducing the semantic understanding of VLMs into image SR is a promising direction, where prompt learning provides lightweight and adaptive task capabilities.
4. **Downstream-Task-Oriented Evaluation**: Conjointly evaluating SR with detection/segmentation tasks aligns the method closer to practical application requirements.

## Limitations & Future Work

1. Contourlet decomposition and its inverse transform add computational overhead, which may limit real-time performance.
2. Prompt learning relies on the pre-trained CLIP model, which is primarily trained on visible-light images and might lack sufficient semantic understanding of infrared images.
3. The relatively small scale of infrared SR datasets limits the thoroughness of the evaluation.
4. Other multi-directional transforms beyond the Contourlet transform (e.g., Shearlet, Curvelet) have not been explored to see if they yield better results.
5. Prompt learning lacks interpretability, making it difficult to analyze what specific infrared-related knowledge the model has acquired.

## Related Work & Insights

- **Frequency-Domain Super-Resolution**: Methods like DFCAN and FDSR explore the application of frequency domains in SR, but they mainly use FFT or wavelet transforms, which lack the direction-decomposition strength of Contourlet.
- **Prompt Learning in Computer Vision**: Works such as CoOp and VPT introduce prompt learning to image classification and segmentation. CoRPLE represents the first systematic application of prompt learning to image SR.
- **Infrared Image Processing**: Infrared image enhancement, denoising, and super-resolution constitute an active research area, complementing visible-infrared fusion frameworks (e.g., TarDAL).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of Contourlet residuals and prompt learning is a first in infrared SR, offering a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple scale factors and metrics, along with downstream task evaluations.
- Writing Quality: ⭐⭐⭐ Clear motivation, though technical details are dense.
- Value: ⭐⭐⭐⭐ Clear demand for tailored solutions in infrared SR, presenting high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](../../NeurIPS2025/image_restoration/enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ICML 2026\] One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation](../../ICML2026/image_restoration/one-step_residual_shifting_diffusion_for_image_super-resolution_via_distillation.md)
- [\[ECCV 2024\] Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](../../CVPR2026/image_restoration/real_iisr_infrared_image_super_resolution_autoregressive.md)

</div>

<!-- RELATED:END -->
