---
title: >-
  [Paper Note] Diffusion Model is a Good Pose Estimator from 3D RF-Vision
description: >-
  [ECCV 2024][Human Understanding][Human Pose Estimation] Proposes mmDiff, a diffusion-based framework for millimeter-wave (mmWave) radar human pose estimation. By employing global-local radar context extraction and structural-temporal motion consistency constraints, it effectively addresses the challenges of sparse, noisy, and inconsistent radar point clouds, significantly outperforming existing SOTA methods.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Human Pose Estimation"
  - "Millimeter-Wave Radar"
  - "Diffusion Models"
  - "Point Clouds"
  - "RF-Vision"
date: 2026-05-08
content_hash: 26b8a63cc96dae8f
---

# Diffusion Model is a Good Pose Estimator from 3D RF-Vision

**Conference**: ECCV 2024  
**arXiv**: [2403.16198](https://arxiv.org/abs/2403.16198)  
**Code**: [https://fanjunqiao.github.io/mmDiff-site/](https://fanjunqiao.github.io/mmDiff-site/)  
**Area**: Human Understanding  
**Keywords**: Human Pose Estimation, Millimeter-Wave Radar, Diffusion Models, Point Clouds, RF-Vision

## TL;DR

Proposes mmDiff, a diffusion-based framework for millimeter-wave (mmWave) radar human pose estimation. By employing global-local radar context extraction and structural-temporal motion consistency constraints, it effectively addresses the challenges of sparse, noisy, and inconsistent radar point clouds, significantly outperforming existing SOTA methods.

## Background & Motivation

**Background**: Human Pose Estimation (HPE) is a core task in computer vision, where mainstream approaches rely on RGB(D) cameras. mmWave radar is emerging as a sensor for HPE due to its advantages in penetrating obstacles, protecting privacy, and being robust to illumination. Existing mmWave HPE methods primarily enhance point cloud density through multi-frame aggregation and directly adopt LSTM or Transformer architectures from the vision domain.

**Limitations of Prior Work**: (1) The spatial resolution of mmWave radar is limited, resulting in extremely sparse point clouds (typically <100 points) and insufficient geometric information; (2) Multi-path effects produce ghost points, and parts of the human body can be missed (miss-detection); (3) Specular reflections and environmental interference lead to temporal inconsistency in sensory data, causing pose jitter and scale variations.

**Key Challenge**: Transformer/LSTM feature encoders in the vision domain are designed for dense, coherent visual data, making them incapable of effectively handling the sparsity and inconsistency of mmWave radar. Directly using MLPs to project point cloud (PC) features to joint features allows noise from missed joints to contaminate the holistic feature representation.

**Goal**: To design an HPE method specifically tailored for noisy radar point clouds, while simultaneously resolving (1) unreliable feature extraction caused by partial human body miss-detection and (2) pose instability due to signal inconsistency.

**Key Insight**: Diffusion models are inherently proficient at denoising and can reconstruct the target distribution from noisy distributions. Missed joints can be inferred from the context of detected joints, and distorted human structures can be progressively corrected.

**Core Idea**: Utilizing a conditional diffusion model for pose generation, where four dedicated modules extract clean and consistent conditional features (global/local radar context + limb length/temporal motion consistency) from noisy radar data to guide the diffusion process toward generating accurate and stable human poses.

## Method

### Overall Architecture

mmDiff employs a two-stage training scheme for the conditional diffusion model:
- **Stage 1**: Train the point cloud encoder + GRC module + coarse pose decoder to obtain global joint features $F^j$ and the coarsely estimated pose $\tilde{H}$.
- **Stage 2**: Freeze the parameters of Stage 1, and train the three conditional modules (LRC, SLC, TMC) alongside the diffusion model. The coarse pose $\tilde{H}$ is used to initialize the starting point $\hat{H}_K$ of the reverse diffusion process.

The input is a 6D radar point cloud $R_t \in \mathbb{R}^{N \times 6}$ (comprising x, y, z coordinates, velocity, energy, and amplitude), and the output is the 3D coordinates of 17 joints $H_t \in \mathbb{R}^{17 \times 3}$.

### Key Designs

1. **Conditional Diffusion Model**:

    - **Function**: Uses radar features as conditions to progressively denoise a noisy pose into the final pose.
    - **Mechanism**: The forward process progressively adds noise $q(H_k | H_{k-1}) = \mathcal{N}(H_k | \sqrt{1-\beta_k} H_{k-1}, \beta_k I)$; the reverse process employs a conditional noise estimator $\hat{\varepsilon}_k = \hat{\varepsilon}_\theta(\hat{H}_k, C, k)$ to predict and remove noise. The skeleton is encoded by a GCN into a latent representation $Z_k \in \mathbb{R}^{17 \times 96}$, and conditional features are injected via addition before each GCN block.
    - **Design Motivation**: Compared to direct regression, diffusion models can model the probability distribution of poses. Missed joints can be inferred from detected ones, and anomalous poses can be progressively rectified.

2. **Global Radar Context (GRC)**:

    - **Function**: Extracts robust joint-level features from global point cloud features to address the miss-detection issue.
    - **Mechanism**: Drawing inspiration from the cls token design in ViT, a trainable joint feature template $\bar{F}^j \in \mathbb{R}^{17 \times 1024}$ is randomly initialized and concatenated with PC features, then fed into a Global-Transformer: $F^j, F'^r = \Phi^g(\bar{F}^j, F^r)$. Each joint independently extracts relevant information from the PC features.
    - **Design Motivation**: Traditional MLP projection cannot isolate the influence of missed joints. The attention mechanism of the Transformer allows each joint token to selectively focus on relevant point cloud regions, preventing missed parts from contaminating the features of detected joints.

3. **Local Radar Context (LRC)**:

    - **Function**: Extracts high-resolution point-level features within the neighborhood of each joint.
    - **Mechanism**: The intermediate pose $\hat{H}_k$ from the diffusion process is utilized as **dynamic anchors**. KNN is applied to select the $\bar{K}=50$ nearest neighbor points around each joint, followed by point-to-point self-attention via a Local-Transformer: $C^{loc} = \bigcup_{i} \Phi^l \circ g^l(\bar{R}_k^i)$.
    - **Design Motivation**: Global features lack sufficient resolution, whereas local point clouds can provide finer details. Utilizing dynamic anchors (instead of static coarsely estimated anchors) allows the model to reflect joint error variations across different diffusion steps.

4. **Structural Limb Length Consistency (SLC)**:

    - **Function**: Extracts the lengths of 16 human body segments $\hat{L} \in \mathbb{R}^{16}$ as structural constraints.
    - **Mechanism**: Limb lengths are decoded from joint features $F^j$ using an MLP and then projected into the conditional embedding space: $C^{lim} = g_2^{lim}(g_1^{lim}(F^j))$. A limb length loss ensures accurate estimation.
    - **Design Motivation**: Since an individual's limb lengths should remain constant, this physical constraint reduces pose scale variations and structural distortions.

5. **Temporal Motion Consistency (TMC)**:

    - **Function**: Extracts motion patterns from coarsely estimated poses of the past $\Delta t = 8$ frames.
    - **Mechanism**: A shared GCN encoder encodes the historical pose sequence into a feature sequence, and 1D convolution is then used to extract motion information along the temporal dimension: $C^{tem} = g_2^{tem}(\bigcup_{i} g_1^{tem}(\tilde{H}_{t-i}))$.
    - **Design Motivation**: Human motion is continuous and smooth. 1D convolutions extract smooth motion patterns by averaging temporal features from historical frames while capturing action trends (e.g., increasing z-value of hands during "hand raising"), thereby avoiding sudden frame transitions and pose jitter.

### Loss & Training

- **Stage 1**: $\mathcal{L}_{joint} = E_{i \sim [1,17]} \|h^i - \tilde{h}^i\|_2^2$ (Joint regression L2 loss)
- **Stage 2**: $\mathcal{L}_{diff} = \mathbb{E}_{k,\varepsilon_k} \|\varepsilon_k - \hat{\varepsilon}_\theta(H_k, k, C)\|_2^2 + \lambda \cdot \mathbb{E}_{i \sim [1,16]} |l^i - \hat{l}^i|_1$
    - Diffusion denoising loss + limb length L1 regression loss ($\lambda = 5$)
- Trained for 100 epochs, batch size 1024, Adam optimizer, learning rate $2 \times 10^{-5}$
- Diffusion steps $K = 25$, constant noise schedule $\beta = 0.001$
- Average of 5 hypotheses is taken during inference

## Key Experimental Results

### Main Results (mmBody Dataset, MPJPE/PA-MPJPE mm)

| Method | Base Scene Avg | Harsh Env Avg | Overall Avg |
|------|-------------|-------------|--------|
| P4Transformer | ~72/~59 | ~82/~62 | 78.10/60.56 |
| PoseFormer | ~68/~55 | ~76/~57 | 73.52/56.07 |
| DiffPose | ~68/~56 | ~77/~59 | 73.31/58.23 |
| **mmDiff(G,L,T,S)** | ~**65**/**49** | ~**70**/**53** | **68.08/53.71** |

Compared to P4Transformer, mmDiff reduces MPJPE by 12.8% and PA-MPJPE by 11.3%, with a more pronounced reduction in harsh environments (14.7%/12.0%).

### mm-Fi Dataset

| Method | Random MPJPE | Cross-Subject MPJPE | Cross-Env MPJPE |
|------|-------------|---------------------|-----------------|
| PointTransformer | 73.09 | 75.96 | 88.28 |
| DiffPose | 73.44 | 70.31 | 86.35 |
| **mmDiff(G,S,T)** | **65.26** | **65.62** | **82.73** |

### Ablation Study

| Configuration | Avg MPJPE | Avg PA-MPJPE | Description |
|------|----------|-------------|------|
| No Diffusion (P4Trans Baseline) | 78.10 | 60.56 | Baseline |
| + Diffusion (Unconditional) | 73.31 | 58.23 | Diffusion itself can model pose distribution |
| + GRC | 70.99 | 55.80 | Global context yields significant improvement |
| + GRC + LRC | 69.79 | 55.04 | Local details are further improved |
| + GRC + LRC + TMC | 69.16 | 53.96 | Temporal constraints reduce jitter |
| + GRC + LRC + TMC + SLC | **68.08** | **53.71** | Complete model is optimal |
| W/o SLC | 69.45 | 54.53 | Limb length constraints are important |
| W/o TMC | 69.16 | 53.96 | Temporal consistency is crucial for stability |

### Model Efficiency

| Module | Latency | Parameters | GFLOPs |
|------|------|--------|--------|
| P4Transformer (Baseline) | 40.48ms | 128M | 43.50 |
| Diffusion Model + All Conditional Modules | 11.85ms | 18.51M | 0.62 |

### Key Findings

- Each conditional module makes an independent contribution, with the removal of SLC and TMC having the most significant impact.
- mmDiff's advantages are even more pronounced in harsh environments (smoke, darkness, occlusion).
- Dynamic anchors yield better LRC performance than static anchors (MPJPE 70.43 vs. 71.05).
- The joint feature template leads to better results than direct PC feature guidance (MPJPE 70.99 vs. 72.50).
- Conditional modules incur minimal computational overhead (latency < 2ms/module), making them suitable for edge deployment.

## Highlights & Insights

- **First Application of Diffusion Models to mmWave Radar HPE**: Ingeniously analogizes radar noise removal to the diffusion denoising process, establishing an intuitively sound methodological framework.
- **Clear Division of Labor Among Four Modules**: GRC handles miss-detection (joint-isolated feature extraction), LRC enhances resolution (local point-level attention), SLC maintains structural consistency (limb length constraints), and TMC preserves smooth motion (temporal feature fusion). The design logic is clear and structured.
- **Efficient Module Design**: The four conditional modules take only 18.5M parameters and 0.62 GFLOPs in total—smaller than the baseline—offering high practical deployment value.
- **Cross-Domain Generalization**: Performance in cross-subject experiments is close to random splitting, suggesting that the learned human structure and motion patterns generalize well to unseen subjects.

## Limitations & Future Work

- Performance declines in multi-target scenarios due to mutual interference of multi-person signals in radar point clouds.
- A performance gap still exists compared to RGB-D methods in standard scenarios (though outperforming them in harsh environments).
- The LRC module cannot be utilized on the mm-Fi dataset because the point clouds are too sparse (N < 100).
- The TMC module relies on the quality of coarse estimations from historical frames; poor predictions in the initial frames may lead to error accumulation.
- Diffusion inference requires 25 iterations, which, despite being fast per step, results in higher overall latency than direct regression methods.

## Related Work & Insights

- **vs. P4Transformer**: A mmWave HPE baseline method that directly employs a PC encoder + Transformer. It cannot handle noise and miss-detections effectively, whereas mmDiff reduces MPJPE by 12.8%.
- **vs. DiffPose**: A diffusion-based method for lifting RGB 2D poses to 3D. It performs poorly when directly applied to radar data due to the lack of conditional designs tailored to radar noise characteristics.
- **vs. PoseFormer**: A spatio-temporal Transformer method doing refinement from coarse poses, but lacks specialized handling of radar signal inconsistency.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to apply diffusion models to mmWave radar HPE, featuring highly targeted designs for the four conditional modules.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two public datasets with complete ablation, efficiency analysis, stability analysis (AKV), and limb length distribution statistics—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete methodology description, and high-quality figures.
- Value: ⭐⭐⭐⭐ Provides a strong baseline for privacy-preserving, non-visual HPE. Its performance in harsh environments demonstrates the practical value of RF-vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](scape_a_simple_and_strong_category-agnostic_pose_estimator.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] 3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
