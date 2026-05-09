---
title: >-
  [Paper Note] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing
description: >-
  [NeurIPS 2025][Segmentation][Remote sensing foundation models] This paper proposes RoMA — the first self-supervised autoregressive pre-training framework based on the Mamba architecture for remote sensing. By introducing an adaptive rotation encoding strategy and a multi-scale token prediction mechanism, RoMA addresses the challenges of orientation diversity and extreme scale variation inherent in remote sensing imagery, while empirically validating that Mamba follows data and parameter scaling laws in the remote sensing domain.
tags:
  - NeurIPS 2025
  - Segmentation
  - Remote sensing foundation models
  - Mamba
  - autoregressive pre-training
  - rotation invariance
  - multi-scale prediction
date: 2026-05-08
content_hash: 79dfdc58aa9e394b
---

# RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing

**Conference**: NeurIPS 2025
**arXiv**: [2503.10392](https://arxiv.org/abs/2503.10392)
**Code**: [GitHub](https://github.com/VisionXLab/RoMA)
**Area**: Image Segmentation
**Keywords**: Remote sensing foundation models, Mamba, autoregressive pre-training, rotation invariance, multi-scale prediction

## TL;DR

This paper proposes RoMA — the first self-supervised autoregressive pre-training framework based on the Mamba architecture for remote sensing. By introducing an adaptive rotation encoding strategy and a multi-scale token prediction mechanism, RoMA addresses the challenges of orientation diversity and extreme scale variation inherent in remote sensing imagery, while empirically validating that Mamba follows data and parameter scaling laws in the remote sensing domain.

## Background & Motivation

- **State of the Field**: Remote sensing foundation models (RSFMs) have advanced rapidly, predominantly relying on ViT-based self-supervised pre-training paradigms such as MAE. However, the self-attention mechanism in ViT incurs **quadratic complexity**, creating severe computational bottlenecks when processing high-resolution remote sensing images — for instance, 4000×4000 pixel images from the DOTA dataset generate an enormous number of tokens.

- **Limitations of Prior Work**: The Mamba architecture, with its **linear complexity**, has emerged as a promising alternative and has demonstrated efficient inference on remote sensing downstream tasks. Nevertheless, existing remote sensing Mamba applications are confined to small-scale supervised training, failing to harness the potential of large-scale unlabeled remote sensing data.

- **Root Cause**: Applying autoregressive pre-training to remote sensing Mamba presents three unique challenges:

  - **Sparse and unevenly distributed information**: Foreground objects are sparsely distributed against complex backgrounds in remote sensing imagery (e.g., aircraft on airport runways).
  - **Arbitrary orientation**: Objects captured from a nadir viewpoint can appear at any orientation, unlike natural images constrained by gravity.
  - **Extreme scale variation**: Object scales in remote sensing imagery span several orders of magnitude, from buildings to vehicles.

- **Paper Goals**: Furthermore, whether Mamba autoregressive pre-training can yield consistent performance gains with increasing data volume and model scale — analogous to ViT+MAE — remains an unvalidated open question.

## Method

### Overall Architecture

RoMA adopts an autoregressive pre-training paradigm, partitioning images into patch sequences for next-token prediction. The Mamba encoder processes the complete image to compute Key and Value representations for all tokens; learnable Query vectors then interact with the KV representations to compute the prediction loss. Two core innovations are introduced on top of this framework: an adaptive rotation encoding strategy and a multi-scale prediction strategy.

**Why autoregressive pre-training rather than MAE?** MAE's masking operation disrupts the continuity of token sequences upon which Mamba's linear scan depends. Autoregressive models construct sequential dependencies that naturally align with Mamba's token-by-token scanning mechanism.

### Key Designs

1. **Adaptive Rotation Encoding Strategy (ARES)**: Designed to address the challenges of arbitrary object orientation and sparse distribution in remote sensing imagery, ARES operates in five steps: (1) partition the input image $x \in \mathbb{R}^{H \times W \times C}$ into $N = (H \times W)/p^2$ non-overlapping patches; (2) compute a saliency score for each patch using a feature descriptor (e.g., LBP); (3) select the highest-scoring patch $\text{token}_{\text{top}}$; (4) expand a candidate region of size $L \in \{96, 64, 32\}$ centered on this patch, retaining regions whose mean feature value exceeds the global image mean; (5) apply random rotation to the selected region and replace it with the cropped inscribed square. Learnable angle embeddings are introduced as implicit orientation priors to facilitate the learning of rotation-invariant representations.

2. **Multi-scale Prediction Strategy (MSP)**: Since autoregressive methods flatten 2D images into 1D sequences, unidirectional modeling degrades the representation of vertical and long-range spatial relationships in remote sensing imagery. Beyond the standard token-level MSE loss, MSP aggregates neighboring tokens into larger-scale blocks (e.g., $6\times6$ pixels) and performs next-block prediction at these coarser scales. The total loss is:
$$\ell(\theta) = \frac{1}{K-1}\sum_{k=2}^{K}\|\hat{x}_k - x_k\|_2^2 + \frac{\lambda}{N-1}\sum_{n=2}^{N}\|\hat{y}_n - y_n\|_2^2$$
where $K$ is the total number of tokens and $N$ is the number of aggregated blocks. Coarse-scale supervision helps Mamba capture more complete object structures.

3. **Scaling Law Validation**: The scaling behavior of Mamba in the remote sensing domain is systematically investigated: (a) *Data scaling* — Mamba-B is pre-trained across data volumes ranging from 62.5K to 4M samples, with performance improving consistently without apparent saturation; (b) *Model scaling* — four variants (Tiny/Small/Base/Large) are trained, with larger models consistently achieving superior results.

### Loss & Training

Mamba-B is pre-trained on the OpticalRS-4M dataset with input resolution 196×196, patch size 16, AdamW optimizer, cosine learning rate schedule, initial lr=1.5e-4, batch size 256, and 400 training epochs. The multi-scale loss weight is set to $\lambda=0.1$; an aggregation scale of $6\times$ yields the best performance.

## Key Experimental Results

### Main Results

**Performance comparison on three downstream tasks**:

| Method | Backbone | Params | AID Cls. (OA) | UCM Cls. (OA) | OSCD CD (F1) | SpaceNet Seg. (mF1) |
|--------|----------|--------|--------------|--------------|-------------|---------------------|
| MAE | ViT-B | 86M | 84.21 | 52.75 | - | - |
| ARM | Mamba-B | 85M | 81.14 | 50.41 | 47.28 | 77.89 |
| RVSA | ViT-B+RVSA | 86M | 84.06 | 50.86 | 50.28 | 79.56 |
| SatMAE++ | ViT-L | 307M | 85.98 | 55.72 | 53.10 | 79.21 |
| MA3E | ViT-B | 86M | 85.86 | 55.69 | - | - |
| **RoMA** | Mamba-B | 85M | **87.36** | **59.45** | **55.63** | **79.50** |

RoMA with 85M parameters outperforms SatMAE++ (ViT-L) with 307M parameters.

### Ablation Study

| Configuration | AID OA (TR=20%) | AID OA (TR=50%) | Notes |
|--------------|----------------|----------------|-------|
| Baseline (ARM) | 69.59 | 76.80 | Without ARES and MSP |
| +ARES | 71.70 | 78.00 | With rotation encoding, +1.2% |
| +ARES+MSP | **72.69** | **79.16** | Full RoMA, +2.4% |

**High-resolution scalability**:

| Resolution | RoMA-B Mem. (MB) | ViT-B Mem. (MB) | RoMA-B Speed (s/s) | ViT-B Speed (s/s) |
|-----------|-----------------|----------------|-------------------|------------------|
| 1248 | 6526 | 24531 | 11.43 | 4.99 |
| 2048 | 16934 | OOM | 4.37 | OOM |
| 4096 | 66357 | OOM | 1.15 | OOM |

### Key Findings

- Mamba demonstrably follows data and parameter scaling laws in the remote sensing domain.
- At resolution 1248, RoMA-B achieves 2.29× the inference speed of ViT-B while consuming only 26.6% of its memory.
- ViT-B runs out of memory at resolutions above 2048, whereas RoMA scales stably to 4096.
- The choice of feature descriptor (LBP vs. HOG vs. Wavelet) has minimal impact on results, indicating that the core value of ARES lies in the framework design rather than the specific descriptor.
- Incorporating too many scales simultaneously (e.g., 2×+4×+6×) degrades performance; a single scale (6×) is optimal.

## Highlights & Insights

- This work constitutes the first systematic validation that Mamba autoregressive pre-training is viable for remote sensing, filling an important gap in the field.
- The "high-value region rotation" strategy is elegant: rather than randomly rotating the entire image, ARES adaptively identifies information-dense regions for rotation augmentation.
- The multi-scale prediction strategy is simple yet effective, mitigating the loss of 2D spatial relationships caused by 1D sequence modeling.
- The ability of RoMA to surpass larger models with fewer parameters carries strong practical value.

## Limitations & Future Work

- Mamba's advantage is less pronounced on pixel-level tasks (semantic segmentation) compared to classification and detection.
- Training of Mamba-Large is insufficient (only 300 epochs), precluding a full demonstration of large-model potential.
- Autoregressive pre-training is inherently biased toward patch-level object prediction, potentially requiring additional adaptation for fine-grained pixel-level tasks.
- Validation is limited to optical RGB remote sensing data; performance on multispectral/SAR and other modalities remains unknown.

## Related Work & Insights

- Compared to ARM (the pioneering work on Mamba autoregressive pre-training), RoMA achieves significant gains through remote sensing-specific design choices.
- MA3E proposes incorporating angular factors into MIM training; RoMA extends this idea to the autoregressive framework and combines it with adaptive cropping.
- Multimodal foundation models for remote sensing (e.g., SkySense, AnySat) represent an important future direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First autoregressive pre-training framework for remote sensing Mamba; rotation encoding and multi-scale designs are creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Scaling experiments are systematic and ablations are comprehensive, though Large model training is insufficient.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with compelling motivation.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a practical solution for efficient foundation models in the remote sensing domain.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

<!-- RELATED:END -->
