---
title: >-
  [Paper Note] A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging
description: >-
  [ECCV 2024][Model Compression][Network Quantization] The first low-bit quantization framework, Q-SCI, specifically designed for Video Snapshot Compressive Imaging (Video SCI) reconstruction. By incorporating a high-quality feature extraction module, a precise video reconstruction module, and query/key distribution shift calibration in the Transformer branch, it achieves a 7.8x theoretical speedup with only a 2.3% performance drop under 4-bit quantization.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Network Quantization"
  - "Video Snapshot Compressive Imaging"
  - "Transformer Quantization"
  - "Low-bit Inference"
  - "Efficient Reconstruction"
date: 2026-05-08
content_hash: e1e24ea706bf8b2d
---

# A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging

**Conference**: ECCV 2024  
**arXiv**: [2407.21517](https://arxiv.org/abs/2407.21517)  
**Code**: [https://github.com/mcao92/QuantizedSCI](https://github.com/mcao92/QuantizedSCI)  
**Area**: Model Compression / Computational Imaging  
**Keywords**: Network Quantization, Video Snapshot Compressive Imaging, Transformer Quantization, Low-bit Inference, Efficient Reconstruction  

## TL;DR

The first low-bit quantization framework, Q-SCI, specifically designed for Video Snapshot Compressive Imaging (Video SCI) reconstruction. By incorporating a high-quality feature extraction module, a precise video reconstruction module, and query/key distribution shift calibration in the Transformer branch, it achieves a 7.8x theoretical speedup with only a 2.3% performance drop under 4-bit quantization.

## Background & Motivation

Video Snapshot Compressive Imaging (Video SCI) utilizes low-speed 2D cameras to compress high-speed scenes into snapshot measurements via coding masks, and then reconstructs high-speed video frames through reconstruction algorithms. High-performing Deep Learning (DL)-based reconstruction methods (such as EfficientSCI, STFormer, etc.) achieve excellent performance, but their parameters and computational overhead remain huge. For example, EfficientSCI-S has 3.78M parameters and 563.87 GFLOPs, making it difficult to deploy on resource-constrained devices like mobile phones and autonomous driving platforms. Network quantization is one of the most direct and effective ways to reduce computational costs. However, quantization in the context of Video SCI reconstruction has not been previously explored. Directly quantizing existing reconstruction networks into low bits leads to severe performance collapse (4-bit direct quantization drops performance by 4.11 dB). Thus, tailor-made quantization strategies must be designed for the specific structures of SCI reconstruction tasks.

## Core Problem

1. **Performance Collapse under Low-Bit Quantization**: Directly quantizing the end-to-end Video SCI reconstruction network (comprising feature extraction, feature enhancement, and video reconstruction modules) into 4-bit causes severe quality degradation. The root cause is that the feature extraction module loses massive amounts of high-quality feature information under low bits.
2. **Distribution Distortion in the Transformer Branch**: After quantization, the activation distributions of query and key in the Transformer shift, leading to distortion in attention weight computation. Crucially, this distribution profile differs from standard Vision Transformers (non-bell-shaped), rendering existing methods (such as Q-ViT) inapplicable.

## Method

The overall methodology involves identifying the root causes of performance collapse through empirical analysis, followed by designing three targeted, lightweight enhancement modules to recover the quality loss caused by quantization.

### Overall Architecture

Q-SCI is built upon an end-to-end Video SCI reconstruction network (using EfficientSCI-S as the backbone). The inputs are the 2D compressed measurements and coding masks, and the output is the reconstructed multi-frame high-speed video. The network comprises three stages: a feature extraction module, a feature enhancement module (ResDNet containing the Transformer branch of CFormer), and a video reconstruction module. Q-SCI introduces targeted improvements at each of these three stages, trading minimal parameter overhead for significant quantization performance recovery.

### Key Designs

1. **High-Quality Feature Extraction Module (FEM)**: Empirical analysis reveals that the feature extraction module is the primary source of performance collapse (quantizing it alone to 4-bit drops performance by 2.22 dB, far exceeding the ~0.5 dB drop from other modules). The reason is that low-bit quantization severely degrades initial feature quality, which subsequent modules cannot compensate for. The proposed solution is straightforward: introduce several 1x1x1 convolutions as shortcut connections (including pixel shuffle for spatial alignment) in the feature extraction module, and keep these shortcut convolutions in 8-bit precision. This maintains a high-quality feature propagation path within the low-bit backbone. This design contributes the largest performance gain (+2.35 dB).

2. **Shifted Transformer Branch (RDM)**: Post-quantization, the query and key distributions in the Transformer shift significantly (e.g., the query mean drifts by 1.207 in the 8-bit model), which distorts attention calculations. Unlike Q-ViT, the quantized distribution in SCI is not bell-shaped, making Q-ViT's method inapplicable. Q-SCI introduces learnable shift biases $\beta_q$ and $\beta_k$ to perform $\tilde{q} = q + \beta_q$ and $\tilde{k} = k + \beta_k$ on query and key, realigning the quantized distribution with the full-precision model. This operation incurs almost no computational overhead and yields a 0.53 dB improvement.

3. **Precise Video Reconstruction Module (VRM)**: Following the same design philosophy as FEM, 8-bit precision 1x1x1 shortcut convolutions are added to the video reconstruction module to guarantee that high-quality features can propagate directly to the final network output, contributing a 0.43 dB improvement.

### Loss & Training

- **Loss Function**: Standard MSE loss, $\mathcal{L}_{MSE} = \frac{1}{T \cdot n_x \cdot n_y} \sum_{t=1}^{T} \|\hat{X}_t - X_t\|_2^2$
- **Quantization Scheme**: Asymmetric quantization for activations and symmetric quantization for weights. During training, step size (scale) and zero-point are optimized as learnable parameters via Quantization-Aware Training (QAT).
- **Training Pipeline**: Initialized with full-precision EfficientSCI-S using the Adam optimizer. The model is first trained with 128x128 crops for 100 epochs (lr=1e-4), then trained with 256x256 inputs for 20 epochs, and finally fine-tuned for another 20 epochs with lr reduced to 1e-5.
- **Variants**: Four quantization levels: 8/4/3/2-bit. The 8-bit variant only adds the shifted Transformer branch. The 4/3/2-bit variants simultaneously incorporate all three proposed modules.

## Key Experimental Results

| Method | PSNR (avg) | SSIM (avg) | Params (M) | OPs (G) |
|------|-----------|-----------|-----------|---------|
| EfficientSCI-S (Full-precision) | 35.51 | 0.970 | 3.78 | 563.87 |
| Q-ViT (8-bit) | 35.17 | 0.967 | 0.95 | 141.04 |
| **Q-SCI (8-bit)** | **35.57** | **0.969** | **0.95** | **140.95** |
| **Q-SCI (4-bit)** | **34.69** | **0.963** | **0.48** | **72.69** |
| Q-SCI (3-bit) | 33.62 | 0.953 | 0.37 | 37.47 |
| Q-SCI (2-bit) | 31.62 | 0.928 | 0.25 | 19.85 |
| BIRNAT | 33.31 | 0.951 | 4.13 | 390.56 |
| RevSCI | 33.92 | 0.956 | 5.66 | 766.95 |
| Dense3D-Unfolding | 35.26 | 0.968 | 61.91 | 3975.83 |

- Q-SCI (8-bit) outperforms full-precision EfficientSCI-S by 0.06 dB, while requiring only 1/4 of the OPs.
- Q-SCI (4-bit) is only 0.82 dB behind full precision (a 2.3% performance gap) and achieves a 7.8x theoretical speedup.
- Q-SCI (4-bit) outperforms BIRNAT by 1.38 dB with only 1/5.4 of its OPs.

### Ablation Study

| Configuration | PSNR | SSIM | Gain |
|------|------|------|------|
| 4-bit Baseline (Direct Quantization) | 31.40 | 0.931 | — |
| + Shifted Transformer (RDM) | 31.93 | 0.929 | +0.53 |
| + High-Quality Feature Extraction (FEM) | 34.28 | 0.959 | +2.35 |
| + Precise Video Reconstruction (VRM) | 34.71 | 0.963 | +0.43 |

- **FEM contributes the most** (+2.35 dB), validating the critical role of high-quality initial features for quantized models.
- Combining all three modules recovers 3.31 dB with only a 3.14% increase in computational overhead.
- Generalization validation: On STFormer-S, FEM yields a +3.23 dB improvement, and VRM provides an additional +0.25 dB.

## Highlights & Insights

- **Pioneering Work**: This is the first network quantization work in the Video SCI field, opening up a new direction for the efficiency optimization of SCI reconstruction.
- **Analysis-Driven Design**: The study first performs systematic performance analysis (module-wise quantization experiments and feature visualization) to precisely locate bottlenecks before designing solutions, presenting a solid and instructive methodology.
- **Simple Yet Effective Shortcut Strategy**: Employing a small number of 8-bit 1x1x1 convolutions as shortcuts dramatically recovers quantization loss. This trick sheds light on the quantization of other low-level vision tasks.
- **Calibrating Transformer Distribution Shifts**: The learnable shift bias method is extremely lightweight and applicable to any quantized networks containing Transformer branches.
- **Robust Generalization**: The Q-SCI framework can be seamlessly transferred to different end-to-end SCI reconstruction methods (e.g., EfficientSCI, STFormer) rather than being bounded to a single architecture.

## Limitations & Future Work

- **Lack of Hardware Deployment Validation**: The paper only reports theoretical speedups (OPs) without latency testing on physical chips or GPUs, leaving the actual hardware acceleration effect unverified.
- **Traditional Quantization Policy**: It adopts standard uniform quantization with QAT, leaving advanced quantization techniques like mixed-precision or layer-wise adaptive bit allocation unexplored.
- **Suboptimal 8-bit Shortcut Assignment**: Fixing the shortcuts to 8-bit is a manual design choice rather than automatically searching for the optimal mixed-precision configuration.
- **Evaluation on Limited Backbones**: Generalization experiments are only conducted on EfficientSCI and STFormer; other architectures (e.g., deep unfolding methods) remain untested.
- **Simplistic Loss Function**: Only MSE loss is used; incorporating perceptual loss or adversarial loss could potentially further enhance visual quality.

## Related Work & Insights

- **vs Q-ViT**: Q-ViT is a generic ViT quantization framework but performs poorly compared to Q-SCI when directly applied to SCI tasks (0.4 dB lower at 8-bit). This is because Q-ViT assumes a post-quantization bell-shaped distribution, which does not hold for the Transformer branches in SCI networks. Q-SCI addresses this via learnable shift instead of Q-ViT's fixed re-parameterization, offering greater flexibility.
- **vs Low-level Quantization Methods (e.g., PAMS, CADyQ, BBCU)**: These methods are designed for tasks like super-resolution or denoising, overlooking the unique feature extraction $\rightarrow$ enhancement $\rightarrow$ reconstruction three-stage pipeline and 3D convolutional properties in SCI. Q-SCI's shortcut design is tailored around these pipeline properties.
- **vs EfficientSCI**: EfficientSCI focuses on lightweight architecture design, whereas Q-SCI approaches high efficiency from the perspective of quantization. The two paradigms are orthogonal and complementary. Q-SCI (8-bit) matches the performance of EfficientSCI-S while requiring only 1/4 of the OPs.

## Rating

- Novelty: ⭐⭐⭐⭐ The first quantization work in SCI, but the core quantization technology itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on both simulation and real-world datasets, validates generalization on two backbones, conducts comprehensive ablation studies; however, lacks physical deployment latency testing.
- Writing Quality: ⭐⭐⭐⭐ Analysis-driven, logical, with high-quality figures and tables.
- Value: ⭐⭐⭐⭐ Pioneers the efficient deployment of SCI models; the proposed shortcut quantization strategy holds practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer](uncertainty-driven_spectral_compressive_imaging_with_spatial-frequency_transform.md)
- [\[ECCV 2024\] Simple Unsupervised Knowledge Distillation With Space Similarity](simple_unsupervised_knowledge_distillation_with_space_similarity.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/model_compression/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](../../NeurIPS2025/model_compression/littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[ICLR 2026\] Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs](../../ICLR2026/model_compression/towards_quantization-aware_training_for_ultra-low-bit_reasoning_llms.md)

</div>

<!-- RELATED:END -->
