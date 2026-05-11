---
title: >-
  [Paper Note] Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection
description: >-
  [AAAI 2026][Image Restoration][Motion blur] This paper proposes JFD3, an end-to-end dual-branch framework that performs deblurring in the feature domain rather than the image domain…
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "Motion blur"
  - "infrared UAV target detection"
  - "feature-domain deblurring"
  - "frequency structure guidance"
  - "end-to-end joint framework"
date: 2026-05-08
content_hash: 322e936aaad9ac86
---

# Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection

**Conference**: AAAI 2026
**arXiv**: [2511.14371](https://arxiv.org/abs/2511.14371)
**Code**: [IVPLaboratory/JFD3](https://github.com/IVPLaboratory/JFD3)
**Area**: Image Restoration
**Keywords**: Motion blur, infrared UAV target detection, feature-domain deblurring, frequency structure guidance, end-to-end joint framework

## TL;DR
This paper proposes JFD3, an end-to-end dual-branch framework that performs deblurring in the feature domain rather than the image domain, and leverages frequency structure priors to guide the detection network, achieving high-accuracy real-time infrared UAV target detection under motion blur conditions.

## Background & Motivation

- **Practical demand for infrared UAV detection**: Infrared UAV target (IRUT) detection is critical for all-weather surveillance and reconnaissance, yet rapid sensor motion frequently induces severe motion blur.
- **Destructive effect of motion blur on detection**: IRUT inherently exhibits weak signatures embedded in complex backgrounds; motion blur further diminishes target-background contrast and makes discriminative feature extraction more difficult.
- **Limitations of direct detection**: Running detectors directly on blurred images leads to frequent missed and false detections, as blur severely degrades target distinguishability.
- **Bottleneck of sequential pipelines**: Two-stage deblur-then-detect approaches suffer from three issues: (1) deep deblurring networks entail high computational complexity and latency; (2) deblurring optimizes for perceptual quality rather than detection objectives; (3) it may introduce noise detrimental to detection.
- **Limitations of existing joint methods**: Prior work on jointly addressing low-level and high-level vision tasks primarily targets haze degradation in visible-light scenes; infrared motion-blur detection remains largely unexplored.
- **Core insight**: Deblurring should be performed in the feature domain rather than the image domain, allowing the restoration process to directly serve detection objectives, while frequency-domain structural priors compensate for the edge and detail loss caused by blur.

## Method

### Overall Architecture
JFD3 employs a dual-branch architecture during training and a single-branch architecture during inference. The training stage comprises a sharp branch and a blur branch with shared weights; the sharp branch provides feature-level supervision to the blur branch. During inference, only the blur branch is retained for efficient deployment. The base detector is DEIM (D-FINE-N).

### Key Design 1: Feature-Domain Deblurring (FDD) Network
- **Function**: Restores blurred features in the feature domain rather than the image domain, producing semantically enhanced feature representations.
- **Mechanism**: Built upon MIMO-UNet as a lightweight encoder–decoder (base channel count reduced to 2, only 2 residual blocks per stage). Features from the sharp branch serve as supervision targets; an L1 loss aligns the encoder features of the blur branch, while an SSIM loss maintains structural consistency in the decoder.
- **Design Motivation**: Conventional image-domain deblurring incurs large computational overhead and generates redundant visual details. Feature-domain restoration directly adjusts feature distributions and reduces domain shift with only 0.02M additional parameters, making it suitable for real-time deployment.

### Key Design 2: Frequency Structure Guidance Module (FSGM)
- **Function**: Extracts high-frequency structural priors from the deblurring network and injects them between the stem and stage 1 of the detection backbone to compensate for structural detail loss caused by blur.
- **Mechanism**: Composed of two sub-modules — (1) **FFRB**: applies FFT to the structural prior, extracts high-frequency components via learnable high-pass filters, and refines them through spatially-aware channel attention (SCA) and channel-aware spatial attention (CSA); (2) **SPIB**: fuses the refined structural prior into detection feature maps via cross-attention, employing multi-scale dynamic convolution kernels (5×5 and 7×7) for hierarchical structural guidance.
- **Design Motivation**: Boundary structures of small targets degrade severely in blurred infrared images, while high-frequency components in the frequency domain precisely encode edge and texture information. Partial-compression attention avoids discarding critical cues for small targets that global compression would cause.

### Key Design 3: Feature Consistency Self-Supervised (FCSS) Loss
- **Function**: Imposes feature consistency constraints across 4 stages of the dual-branch detection backbone.
- **Mechanism**: Cosine similarity measures the alignment between intermediate features of the sharp and blur branches: $\mathcal{L}_{\text{FCSS}} = \frac{1}{4}\sum_{i=1}^{4}(1 - \cos(\mathbf{F}_C^{(i)}, \mathbf{F}_B^{(i)}))$
- **Design Motivation**: Drives the blur branch to approximate the feature representations of the sharp branch, bridging the representation gap between degraded and clean inputs and improving the feature extraction capability of the blur branch.

## Loss & Training

The total loss is a weighted sum of three terms: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + 0.4\mathcal{L}_{\text{deb}} + 0.2\mathcal{L}_{\text{FCSS}}$. The weight $\lambda_2$ is annealed to 0.01 after 20 epochs, allowing the model to focus on detection accuracy in later training stages. The model is trained for 150 epochs using the AdamW optimizer on an RTX 3090 GPU.

## Key Experimental Results

### Dataset
The IRBlurUAV benchmark is constructed with 30,000 synthetic blurred/sharp image pairs (IRBlurUAV-syn, 8:1:1 split) and 4,118 real blurred images (IRBlurUAV-real, test-only), at a resolution of 640×512.

### Main Results on IRBlurUAV-syn (Table 1)

| Method | Strategy | AP50 | AR50 | AP | AR | Params(M) | FPS |
|--------|----------|------|------|----|----|-----------|-----|
| RT-DETR | Direct | 0.716 | 0.811 | 0.369 | 0.400 | 19.0 | 50.2 |
| D-FINE | Direct | 0.722 | 0.795 | 0.347 | 0.382 | 3.5 | 45.8 |
| DeepRFT+RT-DETR | Separate | 0.673 | 0.749 | 0.284 | 0.329 | 36.1 | 9.6 |
| DREB-Net | Joint | 0.710 | 0.754 | 0.300 | 0.357 | 34.6 | 10.9 |
| **JFD3 (Ours)** | **Joint** | **0.767** | **0.850** | **0.428** | **0.458** | **3.5** | **25.7** |

JFD3 surpasses the strongest baseline RT-DETR by +5.9% in AP, while containing only 3.5M parameters and achieving 25.7 FPS to meet real-time requirements.

### Generalization on IRBlurUAV-real (Table 2)

| Method | AP50 | AR50 | AP | AR |
|--------|------|------|----|----|
| D-FINE | 0.514 | 0.693 | 0.151 | 0.190 |
| DREB-Net | 0.520 | 0.619 | 0.143 | 0.196 |
| **JFD3 (Ours)** | **0.623** | **0.730** | **0.251** | **0.291** |

In real-world blur scenarios, JFD3 leads the second-best method by +10.0% in AP, demonstrating strong generalization capability.

### Ablation Study (Table 3)

| FDD | FSGM | AP50 | AP |
|-----|------|------|----|
| ✗ | ✗ | 0.654 | 0.290 |
| ✓ | ✗ | 0.763 | 0.390 |
| ✓ | ✓ | 0.765 | 0.420 |

FDD contributes the most (+10.0% AP), while FSGM yields a further gain of +3.0% AP; the two modules are complementary.

## Highlights & Insights

- **First end-to-end joint detection framework for infrared motion blur**, filling a gap in this area.
- The **feature-domain deblurring** paradigm is elegantly designed: only 0.02M additional parameters substantially improve detection accuracy, avoiding the high computational cost of image-domain deblurring.
- The frequency structure guidance module applies FFT high-pass filtering with a dual-attention mechanism to refine structural priors, directly addressing edge degradation in small infrared targets.
- The dual-branch training and single-branch inference strategy balances knowledge transfer during training with computational efficiency at inference.
- The IRBlurUAV benchmark (34K+ images) covering both synthetic and real-world scenarios provides a valuable evaluation resource for the community.

## Limitations & Future Work

- Synthetic data simulate blur using simple linear motion trajectories, which may introduce a domain gap with real-world non-uniform motion blur.
- Evaluation is conducted solely on the self-constructed dataset, without cross-domain assessment on other public infrared or visible-light blur datasets.
- Inference FPS is 25.7 on an RTX 3090; whether real-time performance is achievable on embedded platforms remains an open question.
- The shared-weight dual-branch design is strongly dependent on the quality of sharp-branch features; low-quality clean training images may degrade performance.
- The optimal strategy for combining image-domain and feature-domain deblurring has not been fully explored (Table 4 indicates complementarity but does not investigate this thoroughly).

## Related Work & Insights

- **Image deblurring**: Methods such as DeepRFT (CNN), MDT (Transformer), EVSSM (Mamba), and MaIR pursue perceptual quality but incur large computational overhead and are not optimized for detection tasks.
- **Infrared UAV target detection**: MSHNet and PConv are designed for clean images; UniCD addresses non-uniformity degradation but does not handle motion blur.
- **Joint deblurring and detection**: DREB-Net proposes a dual-stream fusion architecture but targets visible-light vehicle detection, without addressing the challenges of texture scarcity and structural degradation characteristic of infrared small targets.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First joint end-to-end optimization of feature-domain deblurring with infrared UAV detection; the frequency guidance module design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on both synthetic and real datasets with comprehensive multi-strategy comparisons and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive comparison figures for the three detection strategies, and detailed formula derivations.
- **Value**: ⭐⭐⭐⭐ — Addresses an unexplored problem in infrared motion-blur detection; the lightweight design has practical engineering deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](../../ICCV2025/image_restoration/generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](../../CVPR2026/image_restoration/real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[AAAI 2026\] TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis](tmdc_a_two-stage_modality_denoising_and_complementation_framework_for_multimodal.md)

</div>

<!-- RELATED:END -->
