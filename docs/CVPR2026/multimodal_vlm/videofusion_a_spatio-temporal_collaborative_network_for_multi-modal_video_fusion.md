---
title: >-
  [Paper Note] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion
description: >-
  [CVPR2026][Multimodal VLM][Multi-modal video fusion] This paper proposes VideoFusion, the first large-scale infrared-visible video fusion framework…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Multi-modal video fusion"
  - "infrared-visible fusion"
  - "temporal consistency"
  - "cross-modal attention"
  - "video dataset"
date: 2026-05-08
content_hash: a278884b07083dff
---

# VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion

**Conference**: CVPR2026
**arXiv**: [2503.23359](https://arxiv.org/abs/2503.23359)
**Code**: [https://github.com/Linfeng-Tang/VideoFusion](https://github.com/Linfeng-Tang/VideoFusion)
**Area**: Multi-modal VLM
**Keywords**: Multi-modal video fusion, infrared-visible fusion, temporal consistency, cross-modal attention, video dataset

## TL;DR

This paper proposes VideoFusion, the first large-scale infrared-visible video fusion framework, which jointly models cross-modal complementarity and temporal dynamics via cross-modal differential reinforcement, complete-modality guided fusion, and bidirectional temporal collaborative attention, generating spatiotemporally consistent high-quality fusion videos. The authors also construct the M3SVD dataset comprising 220 videos and 153,797 frames.

## Background & Motivation

### Existing Problems

Multi-sensor fusion—particularly infrared and visible light fusion—is an important direction in computer vision with broad applications in military detection, security surveillance, and assisted driving. However, virtually all existing research focuses on **static image fusion**, overlooking the critical fact that sensors in real-world applications **typically capture continuous video sequences** rather than independent static frames.

This problem stems from two core bottlenecks:

**Data scarcity**: Large-scale, temporally synchronized, and spatially aligned multi-modal video datasets are lacking. Existing video datasets (e.g., TNO with only 3 video clips, INO with low resolution, HDO with poor imaging quality) are small in scale and limited in scene diversity.

**Modeling difficulty**: Jointly modeling spatial and temporal dependencies within a unified framework is inherently challenging.

### Limitations of Naive Approaches

Applying image fusion methods frame-by-frame to video **ignores inter-frame complementarity**, resulting in temporal incoherence and inter-frame flickering artifacts. Contemporary video fusion works (e.g., TemCoCo, UniVF) rely on DCN or optical flow networks for inter-frame compensation, but DCN's unsupervised offset estimation is unstable, and optical flow networks—typically trained on single-modality visible data—generalize poorly to multi-modal data.

### Paper Goals

The authors address the problem from two dimensions: (1) constructing a large-scale benchmark dataset to fill the data gap; and (2) proposing an attention-based spatio-temporal collaborative fusion network that adaptively aggregates cross-modal and temporal complementary information.

## Method

### Overall Architecture

VideoFusion adopts an encoder-decoder architecture that processes infrared video $\{\tilde{V}_{ir}^i\}_{i=1}^T$ and visible video $\{\tilde{V}_{vi}^i\}_{i=1}^T$, outputting high-quality fused video $\{V_f^i\}_{i=1}^T$ along with de-degraded infrared/visible reconstructed videos.

The overall pipeline proceeds as follows:

1. **Encoding stage**: 3D convolutions (Conv3D) extract shallow temporal features, followed by downsampling + Conv3D + ResBlock + CmDRM to extract multi-scale temporal features $\mathcal{F}_x^n$ ($n \in \{1,2,3\}$).
2. **Fusion stage**: CMGF modules aggregate cross-modal context at three scales to produce fused features $\mathcal{F}_f^n$.
3. **Enhancement stage**: A Restormer-based Transformer enhancement block is introduced.
4. **Decoding stage**: BiCAM establishes dynamic temporal dependencies; the fusion decoder reconstructs the fused video, while a modality disentanglement module with independent decoders reconstructs the infrared/visible videos.

### Key Designs

#### 1. Cross-modal Differential Reinforcement Module (CmDRM)

**Core Idea**: Features from different modalities contain both complementary and redundant information. The key is to extract the "differential"—the **exclusive complementary information** present in the auxiliary modality but absent in the primary modality.

- A differential feature $\mathcal{F}_d^t = \mathcal{F}_{ir}^t - \mathcal{F}_{vi}^t$ is computed to capture exclusive information from the auxiliary modality.
- The differential feature is projected into keys and values; the primary modality feature is projected into queries; cross-attention aggregates the information.
- A **learnable contribution measurement** is designed—using convolution and average pooling to compute weights $(w, \tilde{w})$—to adaptively balance the original feature and the differentially reinforced feature.
- Channel attention (CA) and spatial attention (SA) further refine the weighted features.

Key insight: Direct cross-modal attention introduces redundancy, whereas "difference-then-attention" precisely targets complementary information.

#### 2. Complete Modality Guided Fusion Module (CMGF)

CmDRM enhances single-modality representations but is insufficient for constructing a complete scene representation. The core assumption of CMGF is that **naively summing the two modality features yields a "coarse complete feature"** that lacks modality specificity.

- The complete feature $\mathcal{F}_c^t = \hat{\mathcal{F}}_{ir}^t + \hat{\mathcal{F}}_{vi}^t$ is projected into a shared query $q_c$.
- $q_c$ queries modality-specific information from both infrared and visible features (each serving as key/value) via two separate cross-attention branches.
- The outputs are aggregated via residual connections to produce the final fused feature.

#### 3. Bidirectional Temporal Collaborative Attention Mechanism (BiCAM)

This is the paper's core temporal modeling component, as Conv3D alone is insufficient to fully exploit temporal cues.

- For the current frame feature $\mathcal{F}_f^t$, multi-head cross-attention is applied with the previous frame $\mathcal{F}_f^{t-1}$ (forward) and the next frame $\mathcal{F}_f^{t+1}$ (backward) respectively.
- Boundary frames are padded by replicating the current frame as the neighbor.
- **Collaborative attention** is introduced: $\mathcal{A}_{co} = \text{softmax}(\mathcal{A}^{t-1} * \mathcal{A}^{t+1})$, where forward and backward attention maps are multiplied and then normalized via softmax to enable bidirectional dynamic interaction.
- Stacking $N$ consecutive BiCAMs—analogous to the shifted window mechanism in Swin Transformer—allows each frame to access long-range temporal context through neighboring frames as intermediaries.

### Loss & Training

The total loss consists of five components:

| Loss Term | Role | Formulation |
|:---|:---|:---|
| $\mathcal{L}_{int}$ (Intensity Loss) | Preserve salient targets | L1 distance between Y channel of fused output and max of source images |
| $\mathcal{L}_{grad}$ (Gradient Loss) | Preserve texture details | L1 distance between Sobel gradients of output and max of source gradients |
| $\mathcal{L}_{color}$ (Color Loss) | Maintain color consistency | L1 distance between CbCr channels and visible input |
| $\mathcal{L}_{sf}$ (Scene Fidelity Loss) | Leverage modality disentanglement | Pixel + gradient L1 between reconstructed IR/Vis videos and ground truth |
| $\mathcal{L}_{var}$ (Variational Consistency Loss) | Suppress temporal flickering | Align inter-frame differences of fused/reconstructed videos with source videos |

The variational consistency loss $\mathcal{L}_{var}$ is grounded in the assumption that inter-frame variations of static backgrounds should approach zero, while those of dynamic objects should align with high-quality source videos. Loss weights: $\lambda_{int}=15$, $\lambda_{grad}=1$, $\lambda_{color}=100$, $\lambda_{sf}=10$, $\lambda_{var}=100$.

Training configuration: AdamW optimizer, initial learning rate $1 \times 10^{-4}$ with cosine annealing to $1 \times 10^{-5}$, 20 epochs, $T=7$ frames during training, $T=25$ frames during testing, multi-scale channel dimensions $[C_1,C_2,C_3]=[32,64,128]$.

### M3SVD Dataset

A large-scale benchmark comprising 220 temporally synchronized and spatially aligned infrared-visible video pairs (153,797 frames total) is constructed, covering challenging scenarios including daytime, nighttime, camouflage, occlusion, low illumination, and overexposure, across 100 diverse scenes such as parks, lakes, sports fields, and intersections. Resolution: 640×480 at 30 FPS.

## Key Experimental Results

### Main Results: Quantitative Comparison under Degraded Scenarios (M3SVD)

| Method | EN↑ | MI↑ | SD↑ | SSIM↑ | VIF↑ | flowD↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| U2Fusion | 6.904 | 2.490 | 35.731 | 0.600 | 0.439 | 6.547 |
| LRRNet | 6.889 | 3.120 | 38.251 | 0.609 | 0.452 | 6.874 |
| DDFM | 6.750 | 2.656 | 32.000 | 0.609 | 0.448 | 6.778 |
| TC-MoA | 7.095 | 2.800 | 42.412 | 0.593 | 0.516 | 5.102 |
| TIMFusion | 7.063 | 3.015 | 50.824 | 0.580 | 0.409 | 5.890 |
| TemCoCo | 7.174 | 3.548 | 50.421 | 0.597 | 0.490 | 4.378 |
| **VideoFusion** | **7.167** | **4.008** | **52.465** | **0.632** | **0.526** | **3.294** |

VideoFusion achieves the best performance on MI, SSIM, VIF, and flowD. The flowD metric is reduced by approximately 25% compared to the second-best method TemCoCo (4.378→3.294).

### Ablation Study

| Configuration | EN↑ | MI↑ | SD↑ | SSIM↑ | VIF↑ | flowD↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| w/o BiCAM | 7.250 | 3.439 | 52.194 | 0.601 | 0.472 | 4.747 |
| w/o CmDRM | 6.953 | 3.557 | 48.439 | 0.612 | 0.510 | 3.728 |
| w/o CMGF | 7.046 | 2.099 | **61.403** | 0.366 | 0.233 | 7.029 |
| w/o $\mathcal{L}_{grad}$ | 7.208 | 3.702 | 52.239 | 0.615 | 0.500 | 3.669 |
| w/o $\mathcal{L}_{int}$ | 7.106 | 2.985 | 47.630 | 0.630 | 0.468 | 3.684 |
| w/o $\mathcal{L}_{var}$ | 7.211 | 3.432 | 52.245 | 0.599 | 0.480 | 6.056 |
| w/o $\mathcal{L}_{color}$ | 6.839 | 2.102 | 39.854 | 0.457 | 0.240 | 6.031 |
| **VideoFusion** | **7.167** | **4.008** | 52.465 | **0.632** | **0.526** | **3.294** |

### Key Findings

1. **BiCAM is central to temporal consistency**: Removing BiCAM degrades flowD from 3.294 to 4.747 (+44%) and also impairs $\mathcal{L}_{var}$ convergence.
2. **CMGF is irreplaceable**: Replacing CMGF with simple summation yields the highest SD but causes SSIM to collapse to 0.366 and VIF to 0.233, producing severe artifacts and distortions.
3. **$\mathcal{L}_{var}$ is critical for temporal stability**: Its removal degrades flowD from 3.294 to 6.056, nearly doubling it.
4. **$\mathcal{L}_{color}$ is essential for color quality**: Its removal reduces SD from 52.465 to 39.854 and VIF from 0.526 to 0.240.
5. **Computational efficiency**: VideoFusion has only 6.743M parameters, 267.78G FLOPs, and an inference time of 0.067s/frame, comparable to image fusion methods.
6. **Downstream extension**: Object tracking experiments based on YOLO v11 demonstrate that VideoFusion's fused results enable detection of more targets with smoother trajectories.

## Highlights & Insights

1. **"Difference-then-attention" paradigm**: CmDRM computes cross-modal differences before applying attention rather than directly performing cross-modal attention, precisely extracting complementary rather than redundant information—a principle transferable to any multi-source information fusion scenario.
2. **Elegant collaborative attention design**: BiCAM multiplies forward and backward attention maps before normalization, coupling bidirectional temporal information with minimal computational overhead.
3. **Variational consistency loss**: The intuition that "inter-frame variation of static backgrounds should be zero, while that of dynamic objects should align with the source" is formalized as a loss function, effectively suppressing flickering.
4. **Dataset contribution**: M3SVD (220 videos / 153,797 frames) substantially surpasses existing multi-modal video datasets in scale and covers challenging scenarios such as camouflage and occlusion, positioning it as a potential standard benchmark for the field.
5. **Modality disentanglement design**: Simultaneously outputting fused video and per-modality reconstructed videos with mutual reinforcement via $\mathcal{L}_{sf}$ presents a "decompose-and-fuse" training strategy worth adopting in future work.

## Limitations & Future Work

1. **Limited temporal window**: Training uses $T=7$ frames; although stacking BiCAMs extends the receptive field, large-motion sequences may still be inadequately handled.
2. **Limited to infrared-visible modality pair**: Generalizability to other modality combinations (e.g., depth+RGB, SAR+optical) has not been validated.
3. **Coarse boundary frame handling**: The strategy of replicating the current frame as a neighbor at boundaries may introduce bias.
4. **Lack of semantic-level evaluation**: Although object tracking is validated, evaluation on additional downstream tasks such as semantic segmentation and action recognition is absent.
5. **Fixed degradation model**: Training degradations (Gaussian blur + stripe noise) are predefined; robustness to more complex real-world degradations (e.g., rain, fog, motion blur) is not fully validated.

## Related Work & Insights

- **Image fusion baselines**: U2Fusion, LRRNet, DDFM, and TIMFusion serve as the primary image-level comparison methods.
- **Video fusion predecessors**: TemCoCo is the strongest baseline, using DCN for inter-frame compensation but with limited stability; RCVS adopts a frame-by-frame strategy with limited effectiveness.
- **Video restoration inspiration**: The design of BiCAM is inspired by works on video deblurring (DSTNet) and video denoising (MDIVDNet), leveraging temporal cues to combat degradation.
- **General inspiration**: The cross-modal differential reinforcement + attention paradigm is transferable to the fusion of projected features from different modalities in multi-modal large models.

## Rating

| Dimension | Score (1–5) | Notes |
|:---|:---:|:---|
| Novelty | 4 | First systematic video fusion framework with a large-scale dataset; differential reinforcement and bidirectional collaborative attention are novel designs |
| Technical Depth | 4 | Multi-module design is well-reasoned and mutually reinforcing; loss functions are grounded in clear physical intuition |
| Experimental Thoroughness | 4.5 | Multi-dataset, multi-metric evaluation with comprehensive ablations, temporal consistency metrics, and downstream tracking validation |
| Writing Quality | 4 | Clear structure with rich figures and tables |
| Value | 4 | Dataset and code are open-sourced; inference is efficient; directly applicable to security surveillance and related scenarios |
| **Overall** | **4** | Fills the gap in multi-modal video fusion with a solid methodology, thorough experiments, and a significant dataset contribution |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)

</div>

<!-- RELATED:END -->
