---
title: >-
  [Paper Note] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining
description: >-
  [ICCV 2025][Image Restoration][event camera deraining] The first point-based event camera deraining framework, leveraging 4D event cloud representation and a Multi-Scale State Space Model (MS3M) to achieve efficient dera…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "event camera deraining"
  - "state space model"
  - "Mamba"
  - "point cloud processing"
  - "spatiotemporal modeling"
date: 2026-05-08
content_hash: 76c5100c27ecf375
---

# PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining

**Conference**: ICCV 2025
**arXiv**: [2505.05307](https://arxiv.org/abs/2505.05307)
**Code**: [https://github.com/softword-tt/PRE-Mamba](https://github.com/softword-tt/PRE-Mamba)
**Area**: Image Restoration / Event Camera
**Keywords**: event camera deraining, state space model, Mamba, point cloud processing, spatiotemporal modeling

## TL;DR

The first point-based event camera deraining framework, leveraging 4D event cloud representation and a Multi-Scale State Space Model (MS3M) to achieve efficient deraining while preserving microsecond-level temporal precision, reaching state-of-the-art performance with only 0.26M parameters.

## Background & Motivation

Event cameras are renowned for their microsecond-level temporal resolution and high dynamic range, yet in rainy scenes they produce dense noise—the rapid motion of raindrops triggers a large number of intensity-change events that overwhelm meaningful scene information. Existing event camera deraining methods face three fundamental tensions:

**Temporal Precision vs. Deraining Quality**: Frame-based methods (e.g., DistillNet) convert event streams into grayscale frames before deraining, sacrificing the native temporal resolution and sparsity of event data.

**Deraining Quality vs. Computational Efficiency**: Although Point Transformer is well-suited for point data, it becomes infeasible at high event rates (>10M events/s) due to quadratic attention complexity.

**Performance Gap between Point-Based and Frame-Based Methods**: Existing point-based methods (PointNet, GNN, SNN) still underperform frame-based counterparts.

Mamba offers a promising solution with its linear complexity and long-range modeling capability; however, adapting it to event deraining is non-trivial—vanilla Mamba is designed for causal language tasks and cannot directly handle asynchronous, sparse, high-temporal-resolution event data.

## Method

### Overall Architecture

PRE-Mamba formulates event deraining as a per-event classification task (akin to denoising) rather than a reconstruction task, thereby reducing computational complexity and avoiding artifacts. The framework comprises four core components: 4D event cloud representation, a Spatiotemporal Decoupling and Fusion module (STDF), a Multi-Scale State Space Model (MS3M), and frequency-domain regularization, organized in a U-Net-style encoder–decoder structure.

### Key Designs

1. **4D Event Cloud Representation**: The raw event stream $\{e_i=(x_i,y_i,t_i,p_i)\}$ is partitioned into fixed temporal windows. Within each window, timestamps are normalized to construct a 3D pseudo-point cloud $p_i=(x_i,y_i,\frac{t_i-t_0}{t_e-t_0})$; a fourth dimension $T_n$ (global window index) is introduced across windows to form the 4D event cloud. This hierarchical dual-scale framework captures intra-window local dynamics via the normalized timestamp $z$ and models inter-window global evolution via the window index $T_n$. Z-order and Hilbert curves are used to serialize the 4D event cloud into SSM-compatible sequences.

2. **Spatiotemporal Decoupling and Fusion Module (STDF)**: Spatial features $f_s$ (extracted via depthwise 1D convolution over coordinates) and temporal features $f_t$ (embedded from the window index) are extracted separately. The key innovation is **temporal-first modulation of spatial features**: intra-window microsecond-level dynamics $f_t^{\text{intra}}$ are extracted by 1D convolution along the $z$ axis, inter-window long-term trends $f_t^{\text{inter}}$ are extracted by 1D convolution over $T_n$, and spatial features are modulated via Hadamard product: $f_s^* = f_s \otimes (f_t^{\text{intra}} + f_t^{\text{inter}})$. The three-branch features are ultimately integrated through residual connections.

3. **Multi-Scale State Space Model (MS3M)**: Comprises dual temporal-scale global modeling and multi-spatial-scale local pathways:

    - **Dual Temporal Scales**: The intra-window branch extracts spatial appearance features $f_s^{\text{intra}}$ via reverse aggregation and 1D convolution; the inter-window branch extracts motion-aware features $f_m^{\text{inter}}$. An adaptive gating mechanism produces weights $f_G$, and after fusion via cross-multiplicative attention the result is passed to the SSM for global feature learning: $f_{\text{dual}} = \text{SSM}(\sigma(f_{\text{fuse}})) \otimes f_G$
    - **Multi-Spatial Scales**: Convolution kernels of varying sizes (1, 3, 5) capture raindrop features at different scales—small kernels capture fine rain streaks, large kernels model diffuse rain patterns. Multi-scale features are fused with the global output via residual addition.

### Loss & Training

Total loss $\mathcal{L} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{fft}$:
- **Cross-Entropy Loss** $\mathcal{L}_{ce}$: per-event binary classification loss.
- **Frequency-Domain Regularization** $\mathcal{L}_{fft}$: Rather than embedding FFT layers directly (prohibitively expensive for millions of events), this term acts as a regularizer that aligns the frequency-domain distribution of predictions and labels. It exploits the microsecond-level resolution of event cameras to capture raindrop dynamics—high-density rainfall exhibits low-frequency continuous patterns, while low-density rainfall exhibits high-frequency sparse patterns.

Training uses the AdamW optimizer with an initial learning rate of $4.8\times10^{-4}$, conducted on 6× RTX A6000 GPUs for 50 epochs, processing 5 temporal windows (each 0.1 s) per iteration.

## Key Experimental Results

### Main Results

Evaluated on the self-constructed EventRain-27K dataset (18K synthetic + 9K real) covering 6 rainfall intensities (5–150 mm/h):

| Method | Type | 5mm SR/NR/DA | 20mm SR/NR/DA | 50mm SR/NR/DA | 150mm SR/NR/DA | Params | Inference Speed |
|--------|------|-------------|---------------|---------------|----------------|--------|-----------------|
| TS | Filter | 0.888/0.265/0.576 | 0.887/0.305/0.596 | 0.883/0.231/0.557 | 0.872/0.243/0.557 | N/A | 1.0× |
| EDnCNN | Learning | 0.968/0.905/0.937 | 0.954/0.904/0.929 | 0.948/0.888/0.918 | 0.929/0.843/0.886 | 614.55K | 1.0× |
| EDformer | Learning | 0.981/0.818/0.899 | 0.962/0.832/0.897 | 0.924/0.844/0.884 | 0.839/0.834/0.836 | 49.80K | 8.09× |
| **PRE-Mamba** | **Learning** | **0.994/0.914/0.954** | **0.978/0.915/0.947** | **0.955/0.911/0.933** | **0.908/0.895/0.902** | **264.63K** | **204.54×** |

PRE-Mamba achieves an average SR/NR/DA of 0.95/0.91/0.93 with only 0.26M parameters, consuming merely 2.66% of EDnCNN's FLOPs while running 204.5× faster.

### Ablation Study

Module ablation on the EventRain-27K validation set:

| Model | STDF | MS3M | $\mathcal{L}_{fft}$ | SR↑ | NR↑ | DA↑ |
|-------|------|------|------|-----|-----|-----|
| M1 (baseline) | ✗ | ✗ | ✗ | 0.9123 | 0.8394 | 0.8759 |
| M2 | ✓ | ✗ | ✗ | 0.8983 | 0.8747 | 0.8865 |
| M3 | ✗ | ✓ | ✗ | 0.8941 | 0.8906 | 0.8923 |
| M4 | ✓ | ✓ | ✗ | 0.9046 | 0.8954 | 0.9000 |
| **Full** | **✓** | **✓** | **✓** | **0.9080** | **0.8949** | **0.9015** |

Ablation over the number of temporal windows: 3 windows DA=0.8268, 5 windows DA=0.9015 (+9.03%), 8 windows DA=0.9330 (+12.84%), with corresponding inference speeds of 1.0×/0.70×/0.52×; 5 windows is selected to balance accuracy and efficiency.

### Key Findings

- STDF and MS3M improve baseline DA by 1.21% and 1.87% respectively; MS3M contributes more, highlighting the importance of deep spatiotemporal feature modeling.
- Filter-based methods are ineffective for rain denoising, as the spatiotemporal patterns of raindrops differ fundamentally from typical noise.
- The method generalizes to snow scenes without architectural modifications.
- Frequency-domain regularization yields additional gains by aligning frequency-domain distributions, exploiting the pronounced spectral differences between rainfall intensities.

## Highlights & Insights

- **Pioneering Contribution**: The first point-based event camera deraining framework, filling a gap in the field.
- **Extreme Efficiency**: 0.26M parameters and 6.23G FLOPs; processing 100K events requires only 0.1 s, making it genuinely suitable for edge deployment.
- **Elegant 4D Representation**: The dual temporal-scale design (intra-window microsecond + inter-window macroscopic) preserves the native temporal precision of event cameras while enabling long-range dependency modeling.
- **Ingenious Frequency-Domain Regularization**: Implemented as a loss term rather than an embedded FFT layer, leveraging frequency-domain information without incurring computational overhead.
- **Dataset Contribution**: EventRain-27K is the first point-based event deraining dataset, encompassing synthetic, artificial, and real-world scenarios.

## Limitations & Future Work

- Although the dataset includes real-world data, ground-truth labels are limited to synthetic and artificial subsets; no ground truth exists for real scenes.
- Validation is conducted solely on EventRain-27K; cross-dataset generalization to other event camera datasets remains untested.
- The number of temporal windows is fixed at 5; adaptive window strategies are not explored.
- Integration effects on downstream tasks (e.g., object tracking, SLAM) are not validated.
- Extension to a unified framework for other adverse weather conditions (fog, dust) remains an open direction.

## Related Work & Insights

- Mamba/SSM is increasingly applied to non-language sequence tasks; this work demonstrates its effectiveness for event data processing.
- MSSM (motion-aware SSM, originally from LiDAR moving object segmentation) is cleverly repurposed here for appearance/motion separation of rain events.
- Event camera deraining is an emerging yet important direction, particularly for adverse-weather robustness in UAV and autonomous driving applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First point-based event deraining framework; 4D event cloud and frequency-domain regularization are novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on synthetic, artificial, real, and snow scenarios with comprehensive ablations; downstream task validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-formatted equations, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Opens a new research direction with extreme efficiency; practically valuable for adverse-weather event camera applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[NeurIPS 2025\] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](../../NeurIPS2025/image_restoration/rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)

</div>

<!-- RELATED:END -->
