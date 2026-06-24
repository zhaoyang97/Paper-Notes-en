---
title: >-
  [Paper Note] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion
description: >-
  [CVPR 2025][Autonomous Driving][Adverse weather LiDAR generation] This paper proposes WeatherGen, the first unified diffusion generation framework for diverse adverse weather LiDAR data. By preserving the physical structure of LiDAR through the Spider Mamba generator and achieving controllable weather generation via a contrastive learning-based controller, it significantly outperforms physics-based simulation methods in both data fidelity and downstream detection performance.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Adverse weather LiDAR generation"
  - "Diffusion model"
  - "Mamba"
  - "Contrastive learning"
  - "Data augmentation"
date: 2026-05-08
content_hash: 2d8388c33c5ec86c
---

# WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2504.13561](https://arxiv.org/abs/2504.13561)  
**Code**: [https://github.com/wuyang98/weathergen](https://github.com/wuyang98/weathergen)  
**Area**: Autonomous Driving / Point Cloud Generation  
**Keywords**: Adverse weather LiDAR generation, Diffusion model, Mamba, Contrastive learning, Data augmentation

## TL;DR

This paper proposes WeatherGen, the first unified diffusion generation framework for diverse adverse weather LiDAR data. By preserving the physical structure of LiDAR through the Spider Mamba generator and achieving controllable weather generation via a contrastive learning-based controller, it significantly outperforms physics-based simulation methods in both data fidelity and downstream detection performance.

## Background & Motivation

**Background**: 3D perception in clear weather has made significant progress in autonomous driving, but LiDAR data in adverse weather (snow, fog, rain) remains extremely scarce. Laser beams undergo scattering and diffraction in adverse weather, producing noise points and missing points, which severely degrades the reliability of perception models.

**Limitations of Prior Work**: (1) Existing simulation methods (FSRL, LSS, LISA) can only simulate a single type of weather using a single physical model, failing to handle multiple weather types in a unified framework. (2) Due to the complexity of optical propagation and incomplete knowledge of sensor parameters, these physical simulators suffer from limited data fidelity and exhibit a significant domain shift from real-world data. (3) The inherent sparsity and ring-like structure of LiDAR data present unique challenges for generative models.

**Key Challenge**: The demand for LiDAR data under adverse weather is high, but the acquisition cost is prohibitive. Existing non-learning physical simulators fail to provide high-fidelity data to train generative models effectively.

**Goal**: To build a unified, high-fidelity multi-weather LiDAR data generation framework capable of generating adverse weather data that boosts downstream performance at low cost.

**Key Insight**: (1) Utilize a learnable map-based data producer to provide pre-training data (addressing the training data scarcity). (2) Design a Mamba generator adapted to the physical structure of LiDAR (addressing the fidelity issue). (3) Use a CLIP-supervised contrastive learning controller to provide discriminative weather control signals (addressing the controllability of unified generation).

**Core Idea**: Project LiDAR data into range maps, and use a DDPM diffusion framework combined with a spider mamba scan specifically designed for the ring structure of LiDAR for denoising generation, integrated with contrastive learning for controllable unified weather generation.

## Method

### Overall Architecture

WeatherGen consists of three core components: (1) Map-based Data Producer (MDP)—converts clear-weather LiDAR range maps into adverse-weather range maps using a learnable mask to provide pre-training data; (2) Spider Mamba Generator (SMG)—a DDPM-based U-Net denoising network that models LiDAR features using spider mamba scans instead of traditional convolutions or self-attention; (3) Contrastive Learning-based Controller (CLC)—uses a weather encoder and a CLIP text encoder with contrastive learning to generate discriminative weather control signals. The training adopts a pre-train-then-fine-tune strategy, while inference only requires the CLC and SMG.

### Key Designs

1. **Map-based Data Producer (MDP)**:

    - **Function**: Generates a large volume of high-quality multi-weather LiDAR data for the pre-training phase.
    - **Mechanism**: After converting clear-weather LiDAR data into a range map $\mathcal{R}_c$, the weather data $\mathcal{R}_w$ is generated in a map-to-map manner. The generation process consists of three parts: (a) a Bernoulli mask $\mathcal{M}_e$ based on a distance threshold $r_w$ to simulate laser attenuation and random point dropouts at long ranges; (b) random noise $\mathcal{R}_n$ to simulate weather-induced noise points; (c) a learnable mask $\mathcal{M}_d$ to adaptively align with the real data distribution. Smaller $r_w$ represents more severe weather conditions. Different weather types employ distinct range attenuation strategies (snow: point dropouts across both near and far distances; fog: mainly far-range attenuation; rain: missing points at far ranges due to wet ground reflections).
    - **Design Motivation**: Unlike pure physical simulators, the MDP incorporates a learnable mask, enabling it to learn real distribution characteristics from data. This is crucial for the "pre-train-then-fine-tune" strategy—first pre-training with a large volume of MDP data, then fine-tuning with a small amount of real adverse weather data.

2. **Spider Mamba Generator (SMG)**:

    - **Function**: Models LiDAR feature interactions during the diffusion denoising process to preserve the physical structure of LiDAR data.
    - **Mechanism**: Unlike Vision Mamba, which scans at the patch level, Spider Mamba scans at the pixel level along the LiDAR beam rings (rows of the range map) and center rays (columns of the range map). This behaves like a spider hunting on a web—moving along radial lines and concentric rings. For the technical implementation, the feature $\mathcal{F}$ is split into 4 parts along the channel dimension to save computation. Each part is processed by a linear projection and $L$ Mamba layers, and then concatenated back. Both the encoder and decoder of the U-Net structure contain convolution blocks and spider mamba blocks. A subsequent Latent Feature Aligner (LFA) aligns the distribution of weather patterns between generated and real data in the latent space via KL divergence.
    - **Design Motivation**: Since the rows of the range map correspond to the LiDAR beam rings and the columns correspond to the center rays, scanning along rows and columns naturally aligns with the LiDAR imaging process. In contrast, convolutions only model locally, while self-attention disrupts the physical structure of LiDAR by connecting all points without order. Furthermore, Vision Mamba operating at the patch level loses the geometric meaning of point clouds in sparse outdoor scenes.

3. **Contrastive Learning-based Controller (CLC)**:

    - **Function**: Generates weather control signals with compact, discriminative semantic knowledge to guide the model in discriminative generation under different weather conditions.
    - **Mechanism**: Contains a weather encoder $\mathcal{W}$ and a frozen CLIP text encoder $\mathcal{C}$. The weather encoder extracts a weather embedding $\textbf{w}$ from the range map, while the CLIP encoder processes text prompts of four preset weather conditions to obtain anchor embeddings $\bm{c}_i$. Optimization is conducted using an information bottleneck contrastive learning objective: minimizing the mutual information between the weather embedding and irrelevant weather anchors $I(\textbf{w}, \textbf{c}_{i\neq j})$ while maximizing the mutual information with the matching weather anchor $I(\textbf{w}, \textbf{c}_{i=j})$. The control signal $\textbf{w}$ is concatenated with the timestep embedding and integrated into the SMG.
    - **Design Motivation**: In a unified generation framework, compact and discriminative control signals are required to distinguish different weather conditions. Directly using weather labels or simple one-hot encodings lacks semantic depth. CLIP's language supervision provides structured semantic anchors for weather control signals, and contrastive learning further compresses and focuses relevant information.

### Loss & Training

The total loss is formulated as $\mathcal{L} = \mathcal{L}_{SMG} + \mathcal{L}_{LFA} + \mathcal{L}_{CLC}$, where $\mathcal{L}_{SMG}$ is the standard diffusion denoising MSE loss, $\mathcal{L}_{LFA}$ is the KL divergence aligning the latent distributions of generated and real data, and $\mathcal{L}_{CLC}$ is the information bottleneck loss for contrastive learning. The training is split into two stages: first pre-training all modules with MDP-generated data, then fine-tuning all parameters (except MDP) with real adverse weather data from "Seeing Through Fog".

## Key Experimental Results

### Main Results

KITTI-360 unconditional generation performance:

| Method | FPD↓ | FRD↓ | MMD×10⁻⁴↓ | JSD×10⁻²↓ |
|------|------|------|-----------|-----------|
| LiDARGen | 90.29 | 579.39 | 7.39 | 7.38 |
| R2DM | 6.24 | 149.66 | 1.91 | 3.05 |
| Text2LiDAR | 4.81 | 164.16 | 0.49 | 2.01 |
| **WeatherGen** | 6.15 | **138.62** | **0.39** | **1.99** |

Seeing Through Fog weather conditional generation performance (vs. physical simulation methods):

| Method | FPD↓ | FRD×10¹↓ | MMD×10⁻⁴↓ | JSD×10⁻¹↓ |
|------|------|---------|-----------|-----------|
| LSS (Snow) | 106.37 | 142.17 | 3.59 | 2.11 |
| FSRL (Fog) | 319.32 | 210.51 | 8.56 | 3.69 |
| LISA (Rain) | 301.11 | 145.23 | 4.30 | 3.23 |
| **WG (Snow)** | **59.28** | **124.17** | **1.71** | **0.77** |
| **WG (Fog)** | 314.14 | 196.89 | 8.08 | **2.66** |
| **WG (Rain)** | **86.40** | **127.06** | 4.15 | **0.93** |

### Ablation Study

On the Seeing Through Fog "heavy snow" test set:

| Configuration | FPD↓ | Explanation |
|------|------|------|
| w/o MDP (no learnable mask) | Performance drops | Fixed-parameter simulation is not flexible enough |
| w/o SMG (replaced by U-Net)| Performance drops | Snow weather introduces much noise; lacking Spider Mamba fails to preserve LiDAR's physical structure |
| w/o CLC (weather encoder only)| Overall drop | Lack of discriminative control signals leads to inaccurate weather generation |
| w/o LFA | Performance drops | Unable to align with the real data distribution |
| Full model | **Best** | All components complement each other |

### Key Findings

- WeatherGen significantly legacy-outperforms physical simulation methods in the generation fidelity of snow and rain (FPD reduced by 44% and 71% respectively), demonstrating that learning-based methods are far superior to manual physical models in capturing complex optical propagation.
- Replacing only 7.4% of the clear weather training set with an equivalent amount of WeatherGen-generated data achieves the best performance in 18 out of 21 metrics for 3D object detection, proving the practical value of generated data.
- The physical structure preservation capability of the spider mamba scan is highly evident in visualization—generated object contours are sharper, and small objects are more complete.
- The absence of the contrastive learning controller leads to an overall performance drop, indicating that discriminative weather control signals are critical for a unified generation framework.

## Highlights & Insights

- **First unified weather LiDAR generation framework**: Shifting the paradigm from "one simulator per weather" to "one framework covering all weather," which significantly reduces data acquisition costs.
- **End-to-end design of Spider Mamba**: Scanning along beam rings and center rays perfectly suits the physical process of LiDAR imaging. This concept can be extended to other sensor data with specific imaging geometries (e.g., radar, sonar).
- **Pre-train-then-fine-tune data engineering strategy**: Pre-training with a large volume of MDP-simulated data and fine-tuning with a small amount of real data elegantly solves the scarcity of real adverse-weather data.
- **CLIP language-supervised control signals**: Anchoring weather semantics within CLIP's text space provides a structured semantic foundation for control signals. This strategy can be generalized to other scenarios requiring attribute-controlled generation.

## Limitations & Future Work

- Currently, only three weather conditions (snow, fog, rain) are supported, while extreme cases like sandstorms and hail are not considered.
- The physical model of MDP is relatively simplified; more sophisticated optical propagation modeling might further enhance data quality.
- Annotation of generated data relies on manual labeling via LabelCloud, which limits the efficiency of building large-scale datasets.
- Temporal consistency is not explored—currently, frames are generated independently, which may lead to inconsistent weather features between consecutive frames.
- The scale of the Seeing Through Fog dataset is relatively small (200-3000 frames per weather), which bounds the upper limit of fine-tuning effectiveness.

## Related Work & Insights

- **vs FSRL/LSS/LISA**: Research in these works designs independent simulators for each weather condition, but their fidelity is limited by incomplete physical models. WeatherGen learns real data distributions to generate higher-fidelity multi-weather data in a unified framework.
- **vs R2DM/LiDM/Text2LiDAR**: These diffusion-based LiDAR generation methods focus solely on clear-weather scenarios and destroy the physical structure of LiDAR using standard convolutions or self-attention. WeatherGen's spider mamba design is better suited for LiDAR data.
- **vs Vision Mamba**: Vision Mamba operates at the patch level for images, but patch-level operations in sparse LiDAR data lose geometric meaning. Spider Mamba's point-level scan along rows/columns aligns better with LiDAR characteristics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First unified weather LiDAR generation framework, elegant design of Spider Mamba.
- Experimental Thoroughness: ⭐⭐⭐⭐ Unconditional + conditional generation + downstream detection validation + comprehensive ablation, though the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive diagrams, but some technical details are relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the real pain point of data scarcity in adverse weather for autonomous driving, holding highly valuable engineering applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)
- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[CVPR 2026\] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](../../CVPR2026/autonomous_driving/buildanypoint_3d_building_structured_abstraction_from_diverse_point_clouds.md)

</div>

<!-- RELATED:END -->
