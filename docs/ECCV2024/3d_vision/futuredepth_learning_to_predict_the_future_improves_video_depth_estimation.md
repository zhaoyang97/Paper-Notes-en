---
title: >-
  [Paper Note] FutureDepth: Learning to Predict the Future Improves Video Depth Estimation
description: >-
  [ECCV 2024][3D Vision][Video depth estimation] This paper proposes FutureDepth, which injects implicit motion and scene features into the depth decoder via a Future Prediction Network (F-Net) to learn motion cues and a Reconstruction Network (R-Net) to learn multi-frame correspondences. It achieves state-of-the-art (SOTA) accuracy and temporal consistency on four datasets (NYUDv2, KITTI, DDAD, and Sintel), with inference efficiency significantly surpassing existing video dept…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Video depth estimation"
  - "future prediction"
  - "masked auto-encoding"
  - "temporal consistency"
  - "multi-frame feature aggregation"
date: 2026-05-08
content_hash: 3d5b42294120f67b
---

# FutureDepth: Learning to Predict the Future Improves Video Depth Estimation

**Conference**: ECCV 2024  
**arXiv**: [2403.12953](https://arxiv.org/abs/2403.12953)  
**Code**: None (Qualcomm AI Research)  
**Area**: 3D Vision  
**Keywords**: Video depth estimation, future prediction, masked auto-encoding, temporal consistency, multi-frame feature aggregation

## TL;DR

This paper proposes FutureDepth, which injects implicit motion and scene features into the depth decoder via a Future Prediction Network (F-Net) to learn motion cues and a Reconstruction Network (R-Net) to learn multi-frame correspondences. It achieves state-of-the-art (SOTA) accuracy and temporal consistency on four datasets (NYUDv2, KITTI, DDAD, and Sintel), with inference efficiency significantly surpassing existing video depth estimation methods.

## Background & Motivation

**Background**: Monocular depth estimation is highly mature, but it ignores continuous video frame information, which is almost always available in practical applications such as autonomous driving and AR/VR. Video depth estimation methods leverage multi-frame information to improve accuracy and temporal consistency.

**Limitations of Prior Work**: Cost volume-based methods incur massive computational and memory overheads. Autoregressive methods offer better efficiency but remain costly (e.g., MAMo requires optical flow estimation, attention, and online gradient computation, while NVDS requires pair-wise cross-attention for each source frame) and struggle to learn underlying dynamic motion and trajectory information.

**Key Challenge**: Existing video depth estimation methods either achieve high accuracy at the cost of heavy computation or offer high efficiency with limited accuracy. Moreover, they do not consider the joint depth prediction of consecutive frames, failing to effectively learn the underlying motion trajectories and spatial correspondences, which leads to sub-optimal temporal consistency.

**Goal**: How to enable the model to implicitly learn and utilize multi-frame motion and correspondence cues to significantly improve the accuracy and temporal consistency of video depth estimation while maintaining computational efficiency.

**Key Insight**: Employing "future prediction" as a pretext task to force the network to learn motion patterns, and using adaptive masked reconstruction to compel the network to learn cross-frame correspondences.

**Core Idea**: Training the network to predict future frame features to implicitly capture motion information is an elegant representation learning approach, requiring no explicit optical flow estimation or cost volume construction during inference.

## Method

### Overall Architecture

FutureDepth consists of four components: (1) an encoder that extracts a feature volume $V_{1,T} \in \mathbb{R}^{H' \times W' \times (T \cdot C)}$ from $T$ consecutive frames; (2) a Future Prediction Network (F-Net) to learn motion cues; (3) a Reconstruction Network (R-Net) to learn multi-frame correspondences; and (4) a depth decoder + refinement network that uses the query features from F-Net and R-Net to generate the final depth map. F-Net and R-Net generate the motion query $Q_{\text{motion}}$ and the scene query $Q_{\text{scene}}$ respectively, which are fused via cross-attention into $Q_{\text{all}}$ and injected into the decoding process.

### Key Designs

1. **Future Prediction Network (F-Net)**: F-Net iteratively predicts multi-frame features one step into the future in an autoregressive manner. Given the current feature volume $V_{1,T}$ as input, it predicts $\tilde{V}_{2,T+1}$, and then uses this prediction to predict $\tilde{V}_{3,T+2}$, repeating up to $L$ steps. A key aspect is the **omission of teacher forcing**—the prediction process does not use ground-truth future features but instead iterates with its own predicted values. This forces the network to minimize error accumulation across timestamps, thereby extracting more robust motion and correspondence cues. The loss function is $\mathcal{L}_F = \sum_{i=1}^{L} \| \tilde{V}_{1+i,T+i} - V_{1+i,T+i} \|_2$. Finally, the features from the last layer of all prediction steps are averaged to produce $Q_{\text{motion}}$. Design Motivation: Predicting future features requires understanding how pixel-level scene and object details move over time, allowing F-Net to implicitly learn motion and multi-frame correspondence.

2. **Reconstruction Network (R-Net)**: R-Net learns multi-frame correspondences through adaptive masked auto-encoding. A learnable mask generator produces an adaptive mask $M_{1,T}$ to mask the feature volume as $M_{1,T} \odot V_{1,T}$, which is then input to R-Net to reconstruct the original features. The training loss is $\mathcal{L}_R = \| (1 - M_{1,T}) \odot (\hat{V}_{1,T} - V_{1,T}) \|_2 + \mathcal{L}_D(D_{1,T}, D_{1,T}^{gt})$, representing the L2 reconstruction loss of masked regions plus the depth SILog loss. The mask generator is trained separately using the depth SILog loss $\mathcal{L}_A$, learning to mask different parts of the same object across different frames (such as different regions of a white truck across frames), which forces R-Net to search for information across frames to complete the reconstruction. Design Motivation: Unlike standard video MAEs used for pre-training, R-Net functions as an auxiliary network that operates alongside the main network during inference to generate $Q_{\text{scene}}$ containing critical scene details.

3. **Feature Fusion and Refinement Network**: $Q_{\text{motion}}$ and $Q_{\text{scene}}$ first employ cross-attention to generate $Q_{\text{all}}$, with its channel dimension expanded by $T$ times. In each Transformer layer of the decoder, $Q_{\text{all}}$ is combined with the queries output from the previous layer to participate in decoding. The refinement network consists of self-attention and cross-attention layers, taking the depth map as input and iteratively refining it using $Q_{\text{all}}$ via cross-attention (repeated $N=3$ times). This significantly enhances the fine details of the depth map (e.g., railings, traffic lights).

### Loss & Training

Training is performed in stages:
- **Pre-training Stage**: The encoder-decoder is trained first (5 epochs), followed by pre-training R-Net with random masked reconstruction using L2 loss (3 epochs).
- **Main Training Stage**: The encoder-decoder is frozen, while the adaptive mask generator, F-Net, and R-Net are trained simultaneously (corresponding to $\mathcal{L}_A$, $\mathcal{L}_F$, and $\mathcal{L}_R$). Subsequently, F-Net/R-Net/mask generator are frozen, and the encoder, decoder, and refinement network are trained.
- **Final Loss**: $\mathcal{L}_{D,\text{final}} = \frac{1}{NT} \sum_{i=0}^{N} \sum_{t=1}^{T} \mathcal{L}_D(D_t^i, D_t^{gt})$, where $\mathcal{L}_D$ is the SILog loss.
- **Key Hyperparameters**: $T=4$ frames, $L=T$ prediction steps, $N=3$ refinement iterations, masking ratio $r \in [0.6, 0.9]$, and the frame stride is sampled from {1, 2, 3, 4}.
- Total training takes approximately 2 days using 2 × A100 GPUs.

## Key Experimental Results

### Main Results: NYUDv2 & Sintel

| Method | NYUDv2 $\delta<1.25$↑ | NYUDv2 Abs Rel↓ | NYUDv2 OPW↓ | Sintel $\delta<1.25$↑ | Sintel Abs Rel↓ | Sintel OPW↓ |
|---|---|---|---|---|---|---|
| NVDS | 0.950 | 0.072 | 0.364 | 0.591 | 0.335 | 0.424 |
| MAMo | 0.942 | 0.074 | 0.388 | 0.579 | 0.358 | 0.493 |
| Baseline (ours) | 0.917 | 0.093 | 0.480 | 0.477 | 0.504 | 0.611 |
| **FutureDepth** | **0.981** | **0.063** | **0.303** | **0.623** | **0.296** | **0.392** |

### Main Results: KITTI

| Method | Encoder | Abs Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|---|---|---|---|---|
| iDisc (Single Frame) | Swin-L | 0.050 | 2.067 | 0.977 |
| GEDepth (Single Frame) | - | 0.048 | 2.050 | 0.976 |
| MAMo (Video) | Swin-L | 0.049 | 1.989 | 0.977 |
| NVDS (Video) | DPT-L | 0.052 | 2.101 | 0.976 |
| **FutureDepth** | Swin-L | **0.044** | **1.920** | **0.983** |
| **FutureDepth** | DINOv2 (ViT-L) | **0.041** | **1.856** | **0.984** |

### Temporal Consistency and Efficiency (KITTI, Swin-L Encoder)

| Method | rTC↑ | aTC↓ | OPW↓ | Runtime (ms)↓ |
|---|---|---|---|---|
| NVDS | 0.951 | 0.096 | 0.356 | 930 |
| MAMo | 0.963 | 0.088 | 0.328 | 122 |
| **FutureDepth** | **0.988** | **0.076** | **0.281** | **49** |

### Ablation Study (KITTI, Swin-L)

| Model | R-Net | AM | F-Net | Refine | Sq Rel↓ | RMSE↓ | $\delta<1.25$↑ | OPW↓ |
|---|---|---|---|---|---|---|---|---|
| SF Baseline | | | | | 0.156 | 2.098 | 0.974 | 0.544 |
| MF Baseline | | | | | 0.154 | 2.094 | 0.975 | 0.540 |
| + F-Net | | | ✓ | | 0.129 | 1.978 | 0.981 | 0.311 |
| + R-Net (RM) | ✓ | | | | 0.148 | 2.040 | 0.976 | 0.478 |
| + R-Net (AM) | ✓ | ✓ | | | 0.136 | 1.999 | 0.980 | 0.416 |
| + F-Net + R-Net(AM) | ✓ | ✓ | ✓ | | 0.122 | 1.931 | 0.983 | 0.284 |
| **FutureDepth (full)** | ✓ | ✓ | ✓ | ✓ | **0.119** | **1.920** | **0.983** | **0.281** |

### Key Findings

- Naive multi-frame extension yields almost no gain (MF vs SF drops RMSE by only 0.004), indicating that simply concatenating features is insufficient to leverage temporal information.
- F-Net is the largest contributor: Sq Rel decreases by 16% and OPW by 42%, validating that future prediction effectively learns motion cues.
- Adaptive Masking (AM) outperforms Random Masking (RM): OPW decreases from 0.478 to 0.416, as AM learns to mask different parts of the same object across frames.
- FutureDepth requires only 49ms for inference, which is 19× faster than NVDS (930ms) and 2.5× faster than MAMo (122ms).
- On DDAD, the Abs Rel drops from MAMo's 0.150 to 0.114 (a 24% reduction), demonstrating strong generalization capability.

## Highlights & Insights

- **Pretext Task-Driven Representation Learning**: Instead of directly estimating optical flow or building cost volumes, the model uses two auxiliary tasks—"future prediction" and "masked reconstruction"—to let auxiliary networks autonomously learn motion and correspondence information. This design is both elegant and highly efficient.
- **Autoregressive Prediction without Teacher Forcing**: Drawing inspiration from roll-out prediction dependency in reinforcement learning, this strategy avoids distribution shift and extracts more robust motion representations.
- **Modular Design**: F-Net and R-Net are plug-and-play modules that can be combined with any backbone network (e.g., ResNet, Swin, DINOv2).
- **A Rare Balance of Efficiency and Accuracy**: Video-based methods are typically accurate but slow; FutureDepth surpasses state-of-the-art methods in both accuracy and speed.

## Limitations & Future Work

- Occluded-reappearing scenes (where objects reappear after being occluded) are not handled specifically. Properly addressing these scenarios could further improve motion understanding.
- F-Net and R-Net share a similar architecture but have different objectives. Exploring whether they can be unified into a single multi-task network is a worthwhile direction.
- During inference, masking is omitted (which enhances efficiency), but this might also result in losing some of the information learned by R-Net during the masking process.
- The training pipeline is multi-staged and relatively complex. End-to-end training could potentially simplify the process.

## Related Work & Insights

- **MAMo**: Uses optical flow and online gradient updates, achieving high accuracy but at a high computational cost. FutureDepth replaces optical flow with future prediction, offering a more elegant approach.
- **NVDS**: Computes pair-wise cross-attention for source frames, with overhead growing linearly with the number of frames. FutureDepth compresses multi-frame information into compact queries via F-Net/R-Net.
- **Video MAE (VideoMAE)**: Used to pre-train the main encoder-decoder; in contrast, the R-Net in FutureDepth is an independent auxiliary network used alongside the main network during inference.
- **Insights**: Employing "future prediction" as a self-supervised signal shows remarkable effectiveness in depth estimation. This paradigm can be extended to other sequential dense prediction tasks such as optical flow estimation and video segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Using future prediction to learn motion cues is novel and highly effective. The autoregressive training without teacher forcing is a pioneer in video depth estimation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four datasets (indoor, driving, and open-domain), covering accuracy, temporal consistency, and efficiency, combined with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the relatively complex training pipeline is supplemented with algorithmic pseudo-code.
- Value: ⭐⭐⭐⭐⭐ Achieves SOTA performance across accuracy, temporal consistency, and efficiency, holding strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](../../CVPR2025/3d_vision/video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[CVPR 2025\] Video Depth Without Video Models](../../CVPR2025/3d_vision/video_depth_without_video_models.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)

</div>

<!-- RELATED:END -->
