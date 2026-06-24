---
title: >-
  [Paper Note] Semi-supervised Video Desnowing Network via Temporal Decoupling Experts and Distribution-Driven Contrastive Regularization
description: >-
  [ECCV2024][Earth Science][video desnowing] This paper proposes SemiVDN, the first semi-supervised video desnowing framework. By incorporating a physics-prior-guided temporal decoupling expert module and distribution-driven contrastive regularization, SemiVDN utilizes unlabeled real-world snowy videos to narrow the synthetic-to-real domain gap, outperforming existing methods on both synthetic and real-world datasets.
tags:
  - "ECCV2024"
  - "Earth Science"
  - "video desnowing"
  - "semi-supervised learning"
  - "mixture of experts"
  - "contrastive learning"
  - "atmospheric scattering model"
date: 2026-05-08
content_hash: 2c83d8f8bdfb4793
---

# Semi-supervised Video Desnowing Network via Temporal Decoupling Experts and Distribution-Driven Contrastive Regularization

**Conference**: ECCV2024  
**arXiv**: [2410.07901](https://arxiv.org/abs/2410.07901)  
**Code**: [TonyHongtaoWu/SemiVDN](https://github.com/TonyHongtaoWu/SemiVDN)  
**Area**: Earth Sciences  
**Keywords**: video desnowing, semi-supervised learning, mixture of experts, contrastive learning, atmospheric scattering model

## TL;DR

This paper proposes SemiVDN, the first semi-supervised video desnowing framework. By incorporating a physics-prior-guided temporal decoupling expert module and distribution-driven contrastive regularization, SemiVDN utilizes unlabeled real-world snowy videos to narrow the synthetic-to-real domain gap, outperforming existing methods on both synthetic and real-world datasets.

## Background & Motivation

Snow is a common adverse weather degradation factor in outdoor videos. Snow particles and snow streaks severely impair the visibility of video frames, which subsequently degrades downstream tasks such as autonomous driving. Existing deep learning-based desnowing methods perform well on synthetic benchmarks, but their performance severely degrades in real-world scenarios due to the significant distribution shift (e.g., large differences in snow shapes and motion trajectories) between synthetic and real-world data. Furthermore, acquiring paired real-world snowy data is extremely difficult because weather conditions are highly variable and object/camera alignment is difficult, rendering fully supervised approaches unfeasible.

The core motivation of this paper is to introduce unlabeled real-world snowy videos and train the model in a semi-supervised manner, thereby enhancing the model's generalization capability in real-world scenarios.

## Core Problem

1. **Domain Gap Problem**: The distribution of synthetic snow and real-world snow differs significantly in morphology and motion patterns, leading to poor generalization of models trained on synthetic data.
2. **Lack of Paired Real-world Data**: It is virtually impossible to obtain paired snowy/clean videos in real-world scenarios.
3. **Insufficient Decoupling of Physical Components**: Although previous methods (e.g., SVDNet) utilize the atmospheric scattering model, their decoupling of the snow layer, transmission map, and atmospheric light is not explicit and accurate enough.

## Method

### Overall Architecture

SemiVDN is based on the Mean-Teacher semi-supervised architecture, comprising a student network and a teacher network. The student network consists of an encoder (ConvNeXt backbone), the Prior-guided Temporal Decoupling Experts (PTDE) module, and a decoder. The teacher network updates its weights via Exponential Moving Average (EMA, decay factor $\eta=0.99$).

During training, supervised loss is calculated using labeled synthetic data, while unsupervised loss is calculated using unlabeled real-world data.

### Atmospheric Scattering Model

The model is based on the classical degradation formula: $I_{snow}(x) = J(x)T(x) + A(x)(1-T(x)) + S(x)$, where $J$ is the clean video, $T$ is the transmission map, $A$ is the atmospheric light, and $S$ is the snow map. The network explicitly decomposes these physical components in the feature space.

### Prior-guided Temporal Decoupling Experts (PTDE)

This is the core module of the proposed method, replacing the MLP layers in the Transformer block:

- **Physics Transformer Block**: After the encoder extracts frame features, overlapped patch embedding is performed to obtain token sequences, which are then fed into a two-layer Transformer. The first layer uses a fusion feed-forward network to enhance feature fusion, while the second layer replaces the MLP with Temporal Decoupling Experts.
- **Temporal Decomposition Router**: Computes the temporally adaptive weight $Q_{ij}$, normalized via softmax along the temporal dimension $N_f \cdot m$, to aggregate complementary information between consecutive frames.
- **Expert Networks**: Three experts are established, corresponding to the snow layer (Snow Expert), transmission (Transmission Expert), and atmospheric light (Atmospheric Light Expert). Each expert processes the temporally adaptive tokens weighted by the router.
- **Continuously Differentiable Routing**: Unlike the discrete routing of sparse MoE, this method utilizes a continuously differentiable softmax operation, avoiding token dropping and expert imbalance problems.
- **Prior-guided Recovery Module**: Utilizes the atmospheric scattering formula to perform physical inversion in the feature space: $F'_B = (F'_I - F'_S - (1-F'_T)F'_A) / (F'_T + \beta)$

### Semi-supervised Training Strategy

**Supervised Loss** (applied to labeled synthetic data):

$$\mathcal{L}_{sup} = \mathcal{L}_{pixel} + 0.03 \cdot \mathcal{L}_{perceptual} + 10 \cdot \mathcal{L}_{Frequency}$$

It includes Charbonnier pixel loss, VGG-16 perceptual loss, and Focal Frequency Loss.

**Unsupervised Loss** (applied to unlabeled real-world data):

$$\mathcal{L}_{un} = 2 \cdot \mathcal{L}'_{pixel} + 0.1 \cdot \mathcal{L}_{cl} + 0.1 \cdot \mathcal{L}_{DCP} + 0.5 \cdot \mathcal{L}_{TV}$$

It includes student-teacher consistency loss, perceptual contrastive loss, dark channel prior loss, and total variation loss.

The total loss dynamically adjusts the weights of supervised and unsupervised losses using a Gaussian warm-up function.

### Distribution-driven Contrastive Regularization (DCR)

To narrow the distribution gap between synthetic and real-world snow, a distribution-based contrastive learning strategy is designed:

1. **Physical Component Separation**: Background features and snow layer features are obtained from the student and teacher networks, respectively.
2. **Real-world Snow Distribution Modeling with GMM**: A Gaussian Mixture Model (with $K=3$ components) is used to fit the distribution of real-world snow layer features.
3. **Ultra-positive Sample Selection**: By calculating the KL divergence, the samples closest to the real-world snow distribution are selected from the synthetic snow layer to serve as "ultra-positive samples".
4. **Contrastive Regularization**: Positive samples are constructed using the real-world background and the ultra-positive synthetic snow; negative samples are constructed using the synthetic background and augmented real-world snow; the anchor is defined by the real-world background from the teacher network and the real-world snow from the student network. This forces the network to focus on recovering snow-independent background details.

### Real-world Dataset Realsnow85

This dataset consists of 85 real-world snowy videos (covering city, park, rural, and natural scenes, with various snowfall intensities and illumination conditions), where 60 videos are used for training and 25 for testing.

## Key Experimental Results

On the RVSD synthetic test set:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SVDNet (ICCV2023) | 25.06 | 0.9210 | 0.0842 |
| Restormer (CVPR2022) | 24.34 | 0.8929 | 0.1164 |
| Snowformer (2022) | 24.01 | 0.8939 | 0.1219 |
| **SemiVDN** | **25.68** | **0.9254** | **0.0785** |

On the real-world dataset (no-reference metrics): SemiVDN achieves NIMA of 4.259 and MUSIQ of 51.57, both representing state-of-the-art performance.

Ablation studies demonstrate the contribution of each component:

| Configuration | PSNR | SSIM | NIMA |
|------|------|------|------|
| M1 (Baseline) | 24.41 | 0.9116 | 4.165 |
| M2 (+TDE) | 25.16 | 0.9217 | 4.212 |
| M3 (+TDE+SST) | 25.29 | 0.9237 | 4.239 |
| SemiVDN (+TDE+SST+DCR) | 25.68 | 0.9254 | 4.259 |

The temporal decoupling expert module contributes the most ($+0.75\text{ dB}$), while semi-supervised training and distribution-driven contrastive regularization bring further improvements.

## Highlights & Insights

- **First semi-supervised video desnowing framework**, filling the blank of semi-supervised learning for this task.
- **Physics-prior-driven MoE design** is highly ingenious: the discrete sparse routing of traditional MoE is replaced with a continuously differentiable operation, and experts are mapped one-to-one to physical components ($S$, $A$, $T$), implicitly guided by the atmospheric scattering formula during training.
- **Contrastive strategy using GMM + KL divergence to select ultra-positive samples** is novel and effectively bridges the synthetic-to-real domain gap.
- Collected and open-sourced the Realsnow85 real-world snowy video dataset, providing a valuable resource for future research.
- Comprehensively outperforms 15 SOTA methods while balancing the trade-off between efficiency and performance.

## Limitations & Future Work

- The scale of the real-world dataset is relatively small (only 85 videos), which may not cover all snowy scenarios.
- The frame count is fixed to 3, which might be insufficient for scenarios requiring long-term temporal dependencies.
- The number of GMM components is fixed to 3, without exploring strategies to automatically select the optimal number of components.
- Only snow degradation is considered, without exploring joint removal of multiple adverse weather conditions.
- The improvement in no-reference metrics (NIMA/MUSIQ) is relatively limited, and evaluation in real-world scenarios still lacks more reliable metrics.
- The simplified assumptions of the atmospheric scattering model may not hold in extreme scenarios (such as blizzards or severe occlusions).

## Related Work & Insights

- **vs SVDNet (ICCV2023)**: Both utilize the atmospheric scattering model, but SVDNet's decoupling of physical components relies on ordinary convolutional layers, which easily introduces background noise. SemiVDN explicitly decouples them using expert networks and introduces a temporal router, achieving a $0.62\text{ dB}$ improvement in PSNR.
- **vs Single-image desnowing methods** (JSTASR, HDCW-Net, Snowformer): These lack temporal information usage, falling behind across all performance metrics.
- **vs Semi-supervised image restoration methods** (AECR-Net, S2VD): These methods are not designed for the physical characteristics of video snow degradation, underperforming SemiVDN on both synthetic and real-world data.
- **vs General video restoration methods** (BasicVSR++, IconVSR): Although they possess strong temporal modeling capabilities, they lack priors specific to snow degradation, underperforming dedicated methods significantly.

## Insights & Connections

- The approach of incorporating physical priors into MoE architectures can be generalized to other physical model-based image restoration tasks (dehazing, deraining), with each expert corresponding to a physical component.
- The combination of a semi-supervised strategy and domain-adaptive contrastive learning is applicable to all low-level vision tasks lacking paired real-world data.
- The method of modeling degradation distribution using GMM and selecting optimal positive samples via KL divergence can serve as a general domain-adaptive contrastive learning strategy.
- The continuously differentiable design of the Temporal Decomposition Router is valuable for other video processing tasks requiring temporal aggregation.

## Rating
- **Novelty**: 8/10 — First time introducing semi-supervised learning in video desnowing; the physical prior-driven MoE and distribution-driven contrastive regularization designs are novel.
- **Experimental Thoroughness**: 8/10 — Compared with 15 methods, comprehensive ablation studies, provided evaluations on synthetic and real-world data, and collected a new dataset.
- **Writing Quality**: 7/10 — Clear structure, complete derivation of formulas, but contains many mathematical symbols which could be further improved for readability.
- **Value**: 7/10 — Semi-supervised desnowing is a meaningful research direction, but the scope of application is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](../../AAAI2026/earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)
- [\[NeurIPS 2025\] Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](../../NeurIPS2025/earth_science/reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)
- [\[ICML 2026\] Scaling Laws of Global Weather Models](../../ICML2026/earth_science/scaling_laws_of_global_weather_models.md)
- [\[NeurIPS 2025\] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](../../NeurIPS2025/earth_science/controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)
- [\[ICLR 2026\] Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models](../../ICLR2026/earth_science/task-adaptive_parameter-efficient_fine-tuning_for_weather_foundation_models.md)

</div>

<!-- RELATED:END -->
