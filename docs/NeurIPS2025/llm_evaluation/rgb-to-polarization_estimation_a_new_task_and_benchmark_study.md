---
title: >-
  [Paper Note] RGB-to-Polarization Estimation: A New Task and Benchmark Study
description: >-
  [NeurIPS 2025][LLM Evaluation][polarization estimation] This paper formally defines the novel task of estimating polarization components (S₁/S₂/S₃) from standard RGB images…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "polarization estimation"
  - "Stokes parameters"
  - "RGB images"
  - "benchmark"
  - "diffusion models"
date: 2026-05-08
content_hash: 3585d367d350bfbe
---

# RGB-to-Polarization Estimation: A New Task and Benchmark Study

**Conference**: NeurIPS 2025
**arXiv**: [2505.13050](https://arxiv.org/abs/2505.13050)
**Code**: To be released
**Area**: Computer Vision / Polarization Imaging
**Keywords**: polarization estimation, Stokes parameters, RGB images, benchmark, diffusion models

## TL;DR

This paper formally defines the novel task of estimating polarization components (S₁/S₂/S₃) from standard RGB images, establishes the first systematic benchmark encompassing both restoration-based and generative methods, and finds that pretrained MAE achieves the best overall pixel-level accuracy (PSNR 24.74). Restoration-based methods consistently outperform diffusion-based generative methods, with pretrained weight transfer identified as a critical advantage.

## Background & Motivation

**Background**: Polarization images encode rich physical information inaccessible to standard RGB cameras — including birefringence, surface stress, roughness, and other material properties — making them valuable for CV tasks such as reflection separation, material classification, shadow removal, and 3D reconstruction. However, capturing polarization information requires specialized hardware (polarization cameras or rotating polarizers), which is costly and inconvenient.

**Limitations of Prior Work**: The hardware barrier of polarization imaging severely restricts its widespread adoption. Existing polarization datasets are scarce, and no prior work has attempted to directly estimate polarization information from RGB images; this direction remains entirely unexplored.

**Key Challenge**: RGB images are ubiquitous but lack polarization information. The same RGB appearance may correspond to multiple polarization states, since RGB encodes only intensity and color, not the vectorial properties of light — making this estimation problem inherently ill-posed.

**Goal**: Can a neural network directly infer polarization Stokes components from RGB input? Which class of deep learning methods is better suited to this task? What is the performance gap between pretrained and from-scratch training?

**Key Insight**: An RGB image corresponds to the total intensity component S₀ of the Stokes vector, which is physically related to S₁/S₂/S₃ through surface normals and material properties implicitly encoded in RGB. Pretrained models may have already learned visual priors relevant to polarization.

**Core Idea**: Formally define RGB-to-Polarization as a new task using Stokes parameters as a unified representation, and establish the first systematic benchmark for evaluating a diverse set of deep learning approaches.

## Method

### Overall Architecture

The task is defined as pixel-wise prediction: given an RGB input image $\mathbf{I}_{\text{RGB}} \in \mathbb{R}^{H \times W \times 3}$ (corresponding to total intensity S₀), the model outputs $\mathbf{S} = [S_1, S_2, S_3] \in \mathbb{R}^{H \times W \times 9}$, where each Stokes component is a 3-channel image. All images are resized to 256×256 and normalized to [0,1]. Evaluation employs three complementary metrics — PSNR, SSIM, and LPIPS — computed independently per component and averaged.

### Key Designs

1. **Stokes Parameter Representation**:

    - **Function**: Provides a unified mathematical framework for fully describing the polarization state of light.
    - **Mechanism**: $S_1 = S_0 \cos(2\psi)\cos(2\chi)$ (horizontal/vertical linear polarization), $S_2 = S_0 \sin(2\psi)\cos(2\chi)$ (±45° linear polarization), $S_3 = S_0 \sin(2\chi)$ (circular polarization). Derived metrics include degree of polarization (DoP), degree of linear polarization (DoLP), circular polarization ratio (CoP), and angle of linear polarization (AoLP), providing more intuitive physical visualization.
    - **Design Motivation**: Each component carries distinct material-sensitive information — high S₁ indicates smooth dielectric or metallic surfaces, high S₂ indicates birefringent or fibrous materials, and high S₃ indicates scattering or optically active media — motivating per-component evaluation.

2. **Restoration-Based Baselines**:

    - **Function**: Model polarization estimation as an image restoration/translation task.
    - **Mechanism**: Restormer (4-level hierarchy, 4/6/6/8 Transformer blocks) and Uformer (4-layer encoder + 4-layer decoder) are modified to produce 9-channel outputs and trained with L1 loss. MAE uses pretrained weights to initialize the encoder (24 layers) and decoder (8 layers), with projection layers extended to 9 channels via channel replication and end-to-end fine-tuning.
    - **Design Motivation**: Restoration architectures excel at pixel-accurate prediction, and MAE's self-supervised pretraining provides strong visual prior representations.

3. **Generative Baselines**:

    - **Function**: Model polarization estimation as a conditional image generation task.
    - **Mechanism**: WDiff (6-level U-Net) and DiT (10 Transformer blocks) serve as from-scratch conditional diffusion models, taking RGB as conditioning input to directly generate 9-channel outputs. RealFill and Img2ImgTurbo are fine-tuned from pretrained Stable Diffusion, training separate models for each Stokes component (maintaining 3-channel output structure) with LoRA for lightweight adaptation.
    - **Design Motivation**: This tests whether diffusion models' generative capacity can capture the complex distribution of polarization information and whether pretrained diffusion priors can transfer effectively.

### Loss & Training

- All restoration models are trained with L1 loss supervising the discrepancy between predicted and ground-truth Stokes components.
- Diffusion models use the standard denoising objective.
- Data is drawn from the large-scale RGB-polarization dataset of Jeon et al., with the first 1,000 pairs for training and the last 200 for testing.

## Key Experimental Results

### Main Results

| Method | Type | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ |
|--------|------|-----------|-----------|------------|
| MAE | Restoration (pretrained) | **24.74** | **0.8876** | 0.2684 |
| Uformer | Restoration (from scratch) | 24.34 | 0.8722 | **0.2267** |
| Restormer | Restoration (from scratch) | 23.98 | 0.8730 | 0.2369 |
| DiT | Generative (from scratch) | 23.33 | 0.8409 | 0.2447 |
| Img2ImgTurbo | Generative (pretrained) | 23.33 | 0.8725 | 0.3542 |
| RealFill | Generative (pretrained) | 21.81 | 0.8025 | 0.2654 |
| WDiff | Generative (from scratch) | 13.11 | 0.6822 | 0.3772 |

### Ablation Study

Cross-dataset generalization (models trained on the Jeon dataset):

| Method | Jeon PSNR | Qiu PSNR | Kurita PSNR |
|--------|-----------|----------|-------------|
| MAE | 24.74 | 15.02 | 18.81 |
| Uformer | 24.34 | 14.68 | 18.74 |
| Img2ImgTurbo | 23.33 | 15.65 | 18.78 |
| DiT | 23.33 | 14.74 | 17.86 |

### Key Findings

- **S₁ is the hardest component to estimate**: S₁ consistently achieves the lowest PSNR across all methods (MAE reaches only 22.73 on S₁ vs. 25.94 on S₃), as S₁ is most sensitive to surface orientation and material properties, yielding lower signal-to-noise ratios in natural scenes.
- **Restoration > Generative**: Restoration-based methods comprehensively outperform generative methods on pixel-level and structural metrics. Generative methods struggle with structural consistency; although Img2ImgTurbo achieves relatively high PSNR, its perceptual fidelity is inferior to DiT.
- **Pretrained weights are critical**: MAE leverages large-scale RGB pretraining to achieve optimal PSNR/SSIM, and Img2ImgTurbo's Stable Diffusion priors outperform from-scratch WDiff and DiT.
- **Severe cross-dataset performance drop**: All methods degrade substantially on out-of-domain data (PSNR drops of approximately 6–10), indicating limited generalization capability.

## Highlights & Insights

- **First definition of the RGB-to-Polarization task**: This work shifts polarization estimation from hardware-dependent acquisition to a computational approach, potentially enabling any RGB camera to obtain polarization information — a practically valuable new direction.
- **Complementarity of restoration vs. generation**: Uformer performs better on DoLP reconstruction while DiT performs better on AoLP, suggesting that the two paradigms capture different dimensions of polarization features; hybrid approaches may yield further gains.
- **Absence of physical priors**: All methods are purely data-driven and do not exploit physical constraints among Stokes parameters (e.g., $S_1^2 + S_2^2 + S_3^2 \leq S_0^2$); incorporating such constraints could substantially improve performance.

## Limitations & Future Work

- **Ill-posed nature insufficiently addressed**: The same RGB may correspond to multiple polarization states, and current methods provide no uncertainty estimates — a critical limitation for downstream applications.
- **Limited data scale**: Only 1,000 training images restrict the learning capacity of deep models, particularly diffusion models that typically require much larger datasets.
- **Generic evaluation metrics**: PSNR/SSIM/LPIPS are general image quality metrics; polarization-specific evaluations (e.g., DoLP/AoLP accuracy, material classification accuracy) are absent.
- **Poor cross-domain generalization**: All models degrade substantially on out-of-domain data, limiting practical utility.
- Future directions include incorporating physics-constrained losses, uncertainty estimation, larger-scale data collection, and polarization-specific architectural designs.

## Related Work & Insights

- **vs. Traditional polarization imaging**: Conventional methods rely on hardware (polarization cameras or rotating polarizers); this paper proposes a computational alternative that substantially lowers the acquisition barrier.
- **vs. Image translation tasks** (e.g., depth estimation, normal estimation): The unique challenge of polarization estimation lies in the ambiguity of many-to-one mappings and the complexity of physical constraints.
- **vs. MAE applied to downstream tasks**: This work further validates the strong transferability of self-supervised pretrained ViTs to low-level vision tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First definition of a new task and benchmark construction, though no novel architectural designs are proposed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers a wide range of methods and multiple datasets, but lacks downstream task validation.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough introduction of the physical background of Stokes parameters.
- **Value**: ⭐⭐⭐⭐ Opens a new research direction with benchmark resources and practical application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[NeurIPS 2025\] A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification](a_standardized_benchmark_for_multilabel_antimicrobial_peptide_classification.md)

</div>

<!-- RELATED:END -->
