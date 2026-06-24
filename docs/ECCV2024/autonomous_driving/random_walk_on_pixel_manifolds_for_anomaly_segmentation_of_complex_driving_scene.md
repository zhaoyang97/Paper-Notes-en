---
title: >-
  [Paper Note] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes
description: >-
  [ECCV 2024][Autonomous Driving][Anomaly Segmentation] Proposes Random Walk on Pixel Manifolds (RWPM), which utilizes random walks to capture the manifold structure of pixel embeddings to correct manifold distortions caused by the diversity of driving scenes. This improves the accuracy of anomaly segmentation scoring functions and allows for plug-and-play integration into existing anomaly segmentation frameworks without requiring additional training.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Anomaly Segmentation"
  - "Random Walk"
  - "Manifold Learning"
  - "Pixel Embeddings"
date: 2026-05-08
content_hash: baac38762741a0dd
---

# Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes

**Conference**: ECCV 2024  
**arXiv**: 2404.17961 (https://arxiv.org/abs/2404.17961)  
**Code**: [https://github.com/ZelongZeng/RWPM](https://github.com/ZelongZeng/RWPM)  
**Area**: Autonomous Driving  
**Keywords**: Anomaly Segmentation, Random Walk, Manifold Learning, Pixel Embeddings, Autonomous Driving

## TL;DR

Proposes Random Walk on Pixel Manifolds (RWPM), which utilizes random walks to capture the manifold structure of pixel embeddings to correct manifold distortions caused by the diversity of driving scenes. This improves the accuracy of anomaly segmentation scoring functions and allows for plug-and-play integration into existing anomaly segmentation frameworks without requiring additional training.

## Background & Motivation

Anomaly segmentation aims to detect outlier objects unseen by semantic segmentation models in driving scenes (such as animals, fallen objects, etc.), which is crucial for autonomous driving safety. Current mainstream methods adopt the Outlier Exposure (OE) strategy, inferring anomaly scores through an anomaly scoring function based on inlier class logit predictions. However, these methods overlook a key issue: **in real driving scenes, environmental diversity (lighting, road textures, etc.) causes manifold distortions in pixel embeddings**. Specifically:

1. Some inlier pixels deviate from the high-logit regions of their respective categories $\rightarrow$ generating false positives.
2. Some outlier pixels lie close to the high-logit regions of inlier categories $\rightarrow$ generating false negatives.
3. Directly using distorted pixel embeddings to compute logits severely degrades the accuracy of anomaly scores.

This work is the **first to identify and address the pixel manifold distortion issue** in anomaly segmentation.

## Method

### Overall Architecture

RWPM is a post-processing module during the inference stage. Its workflow is as follows:
1. Extract the pixel embedding map $\boldsymbol{p} \in \mathbb{R}^{d \times H \times W}$ using an encoder-decoder.
2. Divide the embedding map into $n^2$ sub-graphs (Partial Random Walk).
3. Construct a graph on each sub-graph and perform random walks to update the pixel embeddings.
4. Concatenate the corrected sub-graphs and feed them into the subsequent classifier and scoring function.

### Key Designs

1. **Graph Construction**: Constructs an affinity matrix $\mathbf{W}$ based on cosine similarity, where $\mathbf{W}_{ij} = \langle \hat{\boldsymbol{p}}_i^r, \hat{\boldsymbol{p}}_j^r \rangle$ ($i \neq j$) and the self-loop is 0. Local constraints are achieved through softmax normalization: $\mathbf{S}_{ij} = \frac{\exp(\mathbf{W}_{ij}/\tau)}{\sum \exp(\mathbf{W}_{ij}/\tau)}$, with the temperature parameter $\tau < 1.0$ (set to 0.01) to suppress the influence of non-neighboring points, replacing traditional $k$-NN search for efficient local constraint.

2. **Random Walk Process**: Unlike traditional manifold mining methods that use random walks to propagate labels or states, this work innovatively utilizes random walks to propagate and update pixel embeddings on the manifold. The iterative formula is $\mathbf{m}^{t+1} = \alpha \mathbf{S} \mathbf{m}^t + (1-\alpha) \mathbf{m}^0$, where $\alpha \in (0,1)$ controls the probability of continuing the walk vs. restarting. The closed-form solution is $\mathbf{m}^\infty = (1-\alpha)(\mathbf{I} - \alpha\mathbf{S})^{-1} \mathbf{m}^0$. Core idea: pixel embeddings on the same manifold tend to become consistent after walking, while those on different manifolds maintain their differences.

3. **Partial Random Walk Strategy**: To tackle the computational challenges of large-sized images in driving scenes (e.g., the $\mathbf{S}$ matrix of a $512 \times 1024$ image reaches $524288^2$), two optimizations are proposed:

    - **Embedding map division**: Evenly segment $\boldsymbol{p}$ into $n \times n$ sub-graphs and independently construct sub-graphs to perform random walks.
    - **Finite iteration**: Use $T$-step iterations ($T=5\sim20$) instead of the closed-form solution, reducing the time complexity from $O((HW/n^2)^3)$ to $O(Td(HW/n^2)^2)$.
    - **Calibration mechanism**: When $n > 2$, calibrate the baseline anomaly scores near the boundaries of adjacent sub-graphs.

### Loss & Training

RWPM requires no additional training and is only used during the inference stage. It is directly applied to the pretrained weights of already-trained anomaly segmentation models without changing the network structure. Key hyperparameters:
- Transition probability $\alpha = 0.99$
- Temperature $\tau = 0.01$
- Iteration steps $T = 20$ (Road Anomaly) / $T = 5$ (other datasets)
- Division parameter $n = 4$ (pixel-based) / $n = 2$ (mask-based)

## Key Experimental Results

### Main Results

| Dataset | Metric | RbA | RbA+RWPM | Gain |
|--------|------|-----|----------|------|
| Fishyscapes L&F | AP↑ | 70.81 | 71.16 | +0.35 |
| Fishyscapes L&F | FPR95↓ | 6.30 | 6.12 | -0.18 |
| Road Anomaly | AP↑ | 85.42 | 87.34 | +1.92 |
| Road Anomaly | FPR95↓ | 6.92 | 5.27 | -1.65 |
| SMIYC Anomaly Track | AP↑ | 90.86 | 92.00 | +1.14 |
| SMIYC Obstacle Track | AP↑ | 91.85 | 93.30 | +1.45 |
| Average | AP↑ | 84.73 | **86.00** | +1.27 |
| Average | FPR95↓ | 6.33 | **5.46** | -0.87 |

Component-level metrics (SMIYC Anomaly Track): mean F1 improved from 46.80 to **58.44** (+11.64).

### Ablation Study

| Configuration | AP↑ | FPR95↓ | FPS | GPU Memory |
|------|-----|--------|-----|----------|
| Without RWPM | 85.42 | 6.92 | 11.12 | 3.49GiB |
| n=1 (without division) | 87.90 | 5.17 | 0.32 | 74.96GiB |
| n=2 | 87.34 | 5.27 | 2.04 | 7.24GiB |
| n=4† | 87.17 | 5.32 | 4.35 | 3.49GiB |
| n=8† | 86.91 | 5.41 | 6.26 | 3.55GiB |

### Key Findings

- RWPM consistently improves performance across both pixel-based and mask-based architectures.
- Partial Random Walk reduces memory from 74.96GiB to 7.24GiB, speeding up the process by more than 6 times.
- Finite iterations ($T=20$) can outperform the closed-form solution ($T=\infty$), while significantly boosting efficiency.
- RWPM even improves in-distribution segmentation performance (Cityscapes mIoU: 82.25 $\rightarrow$ 82.43).

## Highlights & Insights

1. **Novel Problem Formulation**: First to reveal the problem of pixel embedding manifold distortion caused by driving scene diversity, offering a new optimization perspective for anomaly segmentation.
2. **Elegant Methodology**: Drawing inspiration from diffusion processes in manifold learning, it uses random walks to propagate embeddings instead of labels/states, enabling training-free plug-and-play capability.
3. **Convincing t-SNE Visualization**: Clearly demonstrates the clustering separation effect of inlier/outlier pixels before and after applying RWPM.
4. **Clever Local Constraints**: Replaces $k$-NN search with a softmax temperature parameter to achieve local constraints, which is both clever and highly efficient.

## Limitations & Future Work

1. Boundary pixels between segmented sub-graphs may lack cross-region manifold information, which the calibration mechanism only solves approximately.
2. Inference time overhead: Taking RbA as an example, the FPS drops from 11.12 to 2.04 ($n=2$), limiting real-time performance.
3. Random walk step size and temperature parameters need to be tuned for different datasets.
4. Lightweight manifold correction schemes could be explored, or manifold structures could be incorporated into the training stage.

## Related Work & Insights

- Manifold learning methods (diffusion processes) have been widely used in retrieval tasks; this work extends them to pixel-level dense prediction tasks.
- Unlike the approach of diffusion models such as DiffusionDet, the diffusion here is geometric diffusion on the feature manifold.
- Similar ideas can be extended to other dense prediction tasks (e.g., depth estimation, optical flow), especially in scenarios with domain shift.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to propose the issue of pixel manifold distortion; the idea of using random walk to update embeddings is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluations across multiple datasets, architectures, and metrics, with thorough ablation studies and convincing visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — The toy example in Figure 1 and the t-SNE visualization effectively convey the core ideas.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play without requiring training, though inference overhead limits real-time applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data](../../CVPR2026/autonomous_driving/climaood_improving_anomaly_segmentation_via_physically_realistic_synthetic_data.md)
- [\[CVPR 2026\] Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation](../../CVPR2026/autonomous_driving/learning_to_identify_out-of-distribution_objects_for_3d_lidar_anomaly_segmentati.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[ECCV 2024\] TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes](tod3cap_towards_3d_dense_captioning_in_outdoor_scenes.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)

</div>

<!-- RELATED:END -->
