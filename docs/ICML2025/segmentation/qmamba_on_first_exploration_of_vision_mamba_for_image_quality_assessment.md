---
title: >-
  [Paper Note] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment
description: >-
  [ICML2025][Segmentation][State Space Model] This work introduces Vision Mamba (State Space Model) into image quality assessment (IQA) for the first time, proposing the QMamba framework and the StylePrompt lightweight fine-tuning strategy, which outperform CNN and Transformer baselines on various synthetic/realistic/AIGC IQA tasks with lower computational costs.
tags:
  - "ICML2025"
  - "Segmentation"
  - "State Space Model"
  - "Mamba"
  - "Image Quality Assessment"
  - "transfer learning"
  - "prompt tuning"
date: 2026-05-08
content_hash: 1e0238681badd235
---

# QMamba: On First Exploration of Vision Mamba for Image Quality Assessment

**Conference**: ICML2025  
**arXiv**: [2406.09546](https://arxiv.org/abs/2406.09546)  
**Code**: [GitHub](https://github.com/bingo-G/QMamba)  
**Area**: Image Segmentation  
**Keywords**: State Space Model, Mamba, Image Quality Assessment, transfer learning, prompt tuning

## TL;DR

This work introduces Vision Mamba (State Space Model) into image quality assessment (IQA) for the first time, proposing the QMamba framework and the StylePrompt lightweight fine-tuning strategy, which outperform CNN and Transformer baselines on various synthetic/realistic/AIGC IQA tasks with lower computational costs.

## Background & Motivation

Image quality assessment (IQA) aims to measure the subjective perceived quality of images, which is widely applied in scenarios such as image compression, enhancement, and AIGC. Existing IQA backbone networks suffer from inherent limitations:

- **CNN**: Good at learning local translation-invariant features, but lacks long-range dependency modeling capabilities, making global quality perception difficult.
- **Vision Transformer**: Effectively models long-range dependencies through the self-attention mechanism, but its quadratic complexity incurs high computational overhead, making it especially unfriendly to large-scale images.
- **Core Problem**: Can a backbone network be found for IQA that possesses both global modeling capabilities and linear complexity?

Mamba (State Space Model) has shown potential to balance performance and efficiency in high-level tasks like segmentation and classification, but its capabilities in **low-level visual perception** (such as quality assessment) remain unexplored. This paper is the first to answer whether "Mamba can outperform existing backbones in low-level perception tasks".

## Method

### Overall Architecture

QMamba adopts a hierarchical residual structure consisting of multiple network stages. Each stage is composed of a downsampling layer and an enhanced Mamba processing module. Multi-scale representations are constructed layer-by-layer to extract rich perceptual features. Three variants are provided:

- **QMamba-Tiny**: 4 blocks, embedding dimension of 96, 27.99M parameters, 4.47G FLOPs.
- **QMamba-Small**: 15 blocks, embedding dimension of 96, 49.37M parameters, 8.71G FLOPs.
- **QMamba-Base**: 15 blocks, embedding dimension of 128, 87.53M parameters, 15.35G FLOPs.

### Local Scanning

The cross-scanning strategy in the original VMamba flattens 2D images into 1D sequences, which disrupts the spatial continuity of adjacent tokens and hampers the capture of critical **local distortion** information in IQA. QMamba instead employs window-based scanning:

1. Perform horizontal/vertical scanning within local windows.
2. Then perform scanning across windows.
3. Vary the window size with network depth to achieve multi-scale perception.

This hierarchical fixed-window design (LQMamba) avoids inference instability and high computational overhead caused by attention-based dynamic routing, while balance is struck between local details and global context.

### StylePrompt Fine-Tuning Strategy

To improve the transferability of QMamba across different IQA domains, based on the discovery that "domain shifts in IQA are often related to feature statistics or style", StylePrompt is proposed:

**StylePrompt Generation (SPG)**: In each network stage, a set of prompts $P_s \in \mathbb{R}^{N \times 1 \times 1 \times C}$ is learned. By utilizing global pooling of input features and Softmax to predict the weights of each prompt component, they are fused into a style prompt:

$$P_f = \sum_{c=1}^{N} w_s P_s, \quad w_s = \text{Softmax}(\text{Conv}_{1 \times 1}(\text{GAP}(F_i)))$$

**StylePrompt Injection (SPI)**: The fused prompt is passed through a linear layer to generate affine parameters $\gamma_v, \beta_v \in \mathbb{R}^{1 \times 1 \times \hat{C}}$, which adjust the mean and variance of the original features along the channel dimension:

$$\gamma_v = \text{Linear}_\gamma(\text{Conv}(P_f)), \quad \beta_v = \text{Linear}_\beta(\text{Conv}(P_f))$$

$$F_i' = F_i \cdot (1 + \gamma_v) + \beta_v$$

Only about **3.83M parameters (4% of all parameters)** need to be fine-tuned to achieve performance close to full-parameter fine-tuning.

## Key Experimental Results

### Task-Specific IQA (Average PLCC/SRCC across 8 Datasets)

| Method | Parameters | GFLOPs | Average Performance |
|------|--------|--------|----------|
| DEIQT | 24.04M | 5.41G | 0.884 |
| Swin-B | 86.74M | 15.47G | 0.872 |
| ViT-B | 85.80M | 17.58G | 0.854 |
| **QMamba-T** | **27.99M** | **4.47G** | **0.893** |
| **LQMamba-S** | **52.91M** | **8.66G** | **0.895** |
| **LQMamba-B** | **93.79M** | **15.30G** | **0.896** |

### Universal IQA (Joint Training on 6 Datasets)

| Method | GFLOPs | PLCC_Avg | SRCC_Avg |
|------|--------|----------|----------|
| Swin-T | 4.51G | 0.900 | 0.883 |
| DEIQT | 5.41G | 0.895 | 0.873 |
| **LQMamba-T** | **4.44G** | **0.909** | **0.888** |

### Transferable IQA (Transfer Performance of StylePrompt)

| Fine-Tuning Strategy | Parameters | PLCC_Avg | SRCC_Avg |
|----------|--------|----------|----------|
| No fine-tuning | 0 | — | 0.642 |
| Full tuning | 93.79M | — | 0.908 |
| **StylePrompt** | **3.83M** | — | **0.901** |

StylePrompt achieves 99% of full tuning performance using only 4% of the parameters.

### Prompt Strategy Ablation

| Strategy | Parameters | PLCC_Avg | SRCC_Avg |
|------|--------|----------|----------|
| SSF | 6.1M | 0.750 | 0.735 |
| Crossattn_Prompt | 12.17M | 0.806 | 0.772 |
| Conv_Prompt | 28.33M | 0.883 | 0.856 |
| **StylePrompt** | **3.83M** | **0.911** | **0.890** |

## Highlights & Insights

1. **First exploration of Mamba's low-level perception capability**: This work systematically validates the advantages of SSM in IQA tasks. t-SNE visualization shows that QMamba offers significantly better feature separation for different distortion types compared to CNN/ViT/Swin.
2. **Crucial role of local scanning**: For complex datasets such as TID2013 (24 distortion types) and KADID (25 distortion types), the improvements of LQMamba are particularly significant (e.g., TID2013 SRCC: 0.964 vs. 0.949 for QMamba).
3. **Exquisite design of StylePrompt**: Based on the insight that "domain shift $\approx$ feature style shift", efficient transfer is achieved solely by adjusting mean and variance, where 3.83M parameters outperform Conv_Prompt which requires 28.33M parameters.
4. **Obvious efficiency advantage**: QMamba-T surpasses the 17.58G ViT-B and 15.47G Swin-B with only 4.47G FLOPs.

## Limitations & Future Work

1. **Evaluation limited to IQA**: The method has not been extended to Video Quality Assessment (VQA) or Audio Quality Assessment, where the sequential modeling characteristics of SSM should inherently be suitable.
2. **Limited dataset scale**: Labeled data in the IQA field is scarce, and the effect of large-scale pre-training remains unexplored.
3. **Resolution limitation**: Images were uniformly cropped to 224×224 in the experiments, and the effect of using original resolution or multi-resolution inputs was not explored.
4. **StylePrompt only adjusts first- and second-order statistics**: For more complex domain shifts (such as differences in content distribution), richer adaptation strategies may be required.
5. **Lack of comparison with recent Mamba variants**: Such as newer architectures like Mamba-2.

## Related Work & Insights

- **VMamba / LocalMamba**: The foundation for QMamba's backbone design and the source of inspiration for local window scanning.
- **DEIQT**: Reference for training strategies and a strong Transformer-based IQA baseline.
- **SSF (Scale & Shift Feature)**: The predecessor of StylePrompt, but SSF learns fixed affine parameters and lacks input adaptability.
- **Research Insights**: The advantage of Mamba in low-level perception tasks suggests its potential in tasks such as image super-resolution, denoising, and enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce Mamba into IQA, with innovative designs in local scanning + StylePrompt.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Very comprehensive experiments with 10 datasets, three major tasks, and multiple ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, thorough analysis, and convincing t-SNE visualizations.
- Value: ⭐⭐⭐⭐ — Opens up new directions for the application of SSM in low-level visual perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Image Quality Assessment: From Human to Machine Preference](../../CVPR2025/segmentation/image_quality_assessment_from_human_to_machine_preference.md)
- [\[CVPR 2025\] MambaOut: Do We Really Need Mamba for Vision?](../../CVPR2025/segmentation/mambaout_do_we_really_need_mamba_for_vision.md)
- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](../../CVPR2025/segmentation/mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](../../ICCV2025/segmentation/tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)

</div>

<!-- RELATED:END -->
