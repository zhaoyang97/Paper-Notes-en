---
title: >-
  [Paper Note] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion
description: >-
  [CVPR 2026][Autonomous Driving][Collaborative Perception] CoLC proposes a communication-efficient early fusion framework for collaborative perception. It reduces transmission volume via Foreground-Aware Point Sampling (FAPS), reconstructs dense pillar representations on the ego side through VQ-based LiDAR completion (CEEF), and ensures semantic and geometric consistency via Dense-Guided Dual Alignment (DGDA). The framework achieves detection performance on par with or superior to full early fusion while significantly reducing communication bandwidth.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Collaborative Perception
  - Communication Efficiency
  - Point Cloud Completion
  - Early Fusion
  - Vector Quantization
date: 2026-05-08
content_hash: 6dc2d289926b61e0
---

# CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion

**Conference**: CVPR 2026
**arXiv**: [2603.00682](https://arxiv.org/abs/2603.00682)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: Collaborative Perception, Communication Efficiency, Point Cloud Completion, Early Fusion, Vector Quantization

## TL;DR

CoLC proposes a communication-efficient early fusion framework for collaborative perception. It reduces transmission volume via Foreground-Aware Point Sampling (FAPS), reconstructs dense pillar representations on the ego side through VQ-based LiDAR completion (CEEF), and ensures semantic and geometric consistency via Dense-Guided Dual Alignment (DGDA). The framework achieves detection performance on par with or superior to full early fusion while significantly reducing communication bandwidth.

## Background & Motivation

Collaborative perception enables multiple agents to share complementary information, overcoming perception blind spots and occlusion issues inherent to single-agent systems. Existing fusion strategies fall into three categories:

- **Early Fusion**: Transmits raw point clouds, offering the highest information fidelity and natural robustness to heterogeneous models, but at prohibitive communication cost.
- **Intermediate Fusion**: Transmits BEV features with moderate communication overhead, but requires model consistency across agents.
- **Late Fusion**: Transmits detection results with minimal communication, but suffers the greatest information loss.

The authors identify a key observation: in early fusion, transmitting only foreground points leads to a substantial performance drop—worse, in fact, than transmitting only background points. This is because foreground points are responsible for completing object shapes, while background points provide contextual anchors for spatial alignment. Both are indispensable. This insight motivates CoLC's design: sampling both foreground and background points, then recovering missing information via completion on the ego side.

## Method

### Overall Architecture

CoLC consists of three complementary modules:

1. **FAPS (Foreground-Aware Point Sampling)**: Performs spatially-aware point cloud sampling on the neighbor side.
2. **CEEF (Completion-Enhanced Early Fusion)**: Reconstructs dense pillars from sparse input on the ego side and performs adaptive fusion.
3. **DGDA (Dense-Guided Dual Alignment)**: Provides semantic–geometric dual alignment supervision during training.

### Key Designs

#### 1. Foreground-Aware Point Sampling (FAPS)

Given a neighbor agent's raw point cloud $\mathcal{X}_j \in \mathbb{R}^{M \times 4}$:

- **Foreground/Background Separation**: A pretrained lightweight MLP point selector estimates a saliency map $\mathcal{S}_j \in [0,1]^M$; points are split into foreground set $\mathcal{X}_j^{fg}$ and background set $\mathcal{X}_j^{bg}$ using threshold $\tau_s = 0.5$.
- **Foreground Farthest Point Sampling (FG-FPS)**: FPS is applied to foreground points at rate $R^{fg}$ to preserve object structural integrity. Since foreground points are relatively few, the FPS overhead is negligible.
- **Background Random Sampling (BG-RPS)**: Given the large volume of background points, random sampling at rate $R^{bg}$ efficiently yields a sparse subset.

The transmitted output is a sparse point cloud containing sampled foreground and background points. A key insight is that as little as 20% foreground points combined with a sufficient number of background points effectively supports detection; an excessive foreground ratio can actually be outperformed by richer background context.

#### 2. Completion-Enhanced Early Fusion (CEEF)

The core component is a VQ-based pillar-level LiDAR completion module that reconstructs dense pillars from sparse inputs:

**VQ-based LiDAR Completion Pipeline**:

(a) **Sparse Encoder**: A Swin Transformer (depth $L=6$, embedding dimension $D=128$) encodes sparse pillars $\mathcal{P}^s$ into a global-context BEV representation, projected into quantization space $\mathbf{z}^s \in \mathbb{R}^{P \times D_c}$.

(b) **Vector Quantization**: A learnable codebook $E = \{\mathbf{e}_k\}_{k=0}^{K-1}$ ($K=128$, $D_c=128$) maps continuous latent vectors to the nearest codebook entry:

$$\mathbf{z}_i^q = \mathbf{e}_k, \quad k = \arg\min_j \|\mathbf{z}_i^s - \mathbf{e}_j\|_2$$

(c) **Dense Decoder**: Maps quantized embeddings back to pillar space, outputting reconstructed dense pillars $\hat{\mathcal{P}}^d$ and an occupancy mask $\hat{\mathcal{O}}^d$.

**Progressive Fusion Strategy** (three stages):

1. **Initial Sparse Early Fusion**: The ego point cloud and received sparse neighbor point clouds are concatenated and pillarized into $\mathcal{P}_i^{se}$.
2. **Parallel Pillar Completion**: Each neighbor's sparse pillars are completed independently; pillars with occupancy probability above $\tau_o$ are retained, with original sparse pillar values substituted at corresponding positions to preserve fidelity.
3. **Adaptive Complementary Fusion**: A spatial correlation map $\mathcal{W}_{j \to i}$ is computed (via concatenation, 1×1 convolution, and softmax) to weight the completed pillars; only empty positions in the initial fusion are updated:

$$\hat{\mathcal{P}}_i^{de} = \mathcal{M}_i^{se} \odot \mathcal{P}_i^{se} + (1 - \mathcal{M}_i^{se}) \odot \hat{\mathcal{P}}_i^f$$

#### 3. Dense-Guided Dual Alignment (DGDA)

During training, the enhanced early fusion pillars are aligned with dense full-point-cloud pillars in two spaces:

- **Semantic Distribution Alignment**: KL divergence along the channel dimension:

$$\mathcal{L}_{sda} = D_{KL}(\sigma(\hat{\mathcal{P}}_i^{de}) \| \sigma(\mathcal{P}_i^{de}))$$

- **Geometric Direction Alignment**: Cosine similarity loss:

$$\mathcal{L}_{gda} = \mathbb{E}_i\left[1 - \frac{\hat{\mathcal{P}}_i^{de} \cdot \mathcal{P}_i^{de}}{\|\hat{\mathcal{P}}_i^{de}\| \|\mathcal{P}_i^{de}\|}\right]$$

### Loss & Training

**Two-Stage Training**:
1. The LiDAR completion module is pretrained to convergence (AdamW, lr=8e-4), with the loss:

$$\mathcal{L}_\Psi = \lambda \cdot \mathcal{L}_{rec} + \mathcal{L}_{vq}$$

where $\mathcal{L}_{rec}$ comprises occupancy BCE and occupied-region MSE, and $\mathcal{L}_{vq}$ includes codebook loss and commitment loss.

2. The completion module is frozen, and the full pipeline is trained end-to-end (Adam, lr=2e-3), with the total loss:

$$\mathcal{L}_\Phi = \mathcal{L}_{det} + \gamma_1 \cdot \mathcal{L}_{sda} + \gamma_2 \cdot \mathcal{L}_{gda}$$

Hyperparameters: $\beta=0.25$, $\lambda=10$, $\gamma_1=1000$, $\gamma_2=10$.

## Key Experimental Results

### Main Results

**Table 1: Collaborative 3D Object Detection Performance (AP@0.5/0.7)**

| Method | V2XSim | OPV2V | V2XSet | DAIR-V2X |
|------|:---:|:---:|:---:|:---:|
| No Fusion | 73.72/61.65 | 74.42/54.52 | 74.18/57.43 | 64.32/53.27 |
| Early Fusion | 94.68/83.61 | 96.13/90.69 | 94.59/88.00 | 76.51/63.83 |
| Where2comm | 88.45/80.54 | 95.10/88.48 | 90.68/80.48 | 76.70/61.96 |
| ERMVP | 94.35/84.76 | 95.99/89.14 | 93.08/81.91 | 74.73/60.75 |
| **CoLC (100%)** | **95.14/87.89** | **96.88/92.93** | **95.97/89.81** | 76.71/62.17 |
| **CoLC* (50%)** | 93.47/85.28 | 96.46/91.95 | 95.05/87.72 | 76.03/62.09 |

CoLC achieves state-of-the-art AP@0.7 across all benchmarks when transmitting the full point cloud, **marginally surpassing the early fusion baseline** — attributable to the regularization effect of completion and alignment during training. CoLC* transmits only 50% of the original communication volume while approaching or exceeding early fusion performance.

**Inference Latency**: CoLC runs at 75.86 ms, comparable to Where2comm (69.7 ms) and CoBEVT (84.5 ms), and significantly faster than ERMVP (100.5 ms) and V2X-ViT (197.7 ms).

### Ablation Study

**Component Ablation (V2XSim, $R^{fg}$=0.2)**

| FAPS | CEEF | DGDA | AP@0.7 Change |
|:---:|:---:|:---:|------|
| ✓ | ✗ | ✗ | Performance drops (information loss) |
| ✓ | ✓ | ✗ | Significant recovery (completion compensates) |
| ✓ | ✓ | ✓ | Further improvement (alignment-guided) |

**VQ vs. MAE Completion**

| Method | IoU ↑ | MSE ↓ | AP@0.5/0.7 ↑ |
|------|:---:|:---:|:---:|
| MAE-based | 0.633 | 0.057 | 88.17/77.55 |
| VQ-based | 0.626 | **0.043** | **88.89/79.28** |

Although the VQ-based approach yields a marginally lower occupancy IoU, it achieves lower MSE and higher reconstruction fidelity, translating to superior detection accuracy.

### Key Findings

1. Transmitting only foreground points yields worse performance than transmitting only background points — background context is critical for early fusion.
2. CoLC is inherently robust to heterogeneous model scenarios: intermediate fusion methods may degrade below the no-fusion baseline under model heterogeneity, whereas CoLC remains consistently effective.
3. Detection performance saturates once completion quality exceeds a threshold (IoU ≥ 0.585, MSE ≤ 0.052), indicating that "good enough" completion suffices.
4. Under low bandwidth, the PC+ACF combination outperforms SEF+ACF, as completion effectively compensates for severe sparsity; under high bandwidth, the full three-stage combination performs best.

## Highlights & Insights

1. **Precise Problem Formulation**: The paper clearly identifies the information advantage and communication bottleneck of early fusion, elegantly decoupling the two via sampling and completion.
2. **Thorough Analysis of Foreground/Background Roles**: The experimental intuition in Figure 1 is compelling — foreground information is insufficient alone; background context is indispensable — and directly informs the FAPS design.
3. **Well-Motivated Choice of VQ-based Completion**: Compared to MAE-based alternatives, VQ-based completion is better suited for detection downstream tasks, as discrete priors yield more discriminative pillar features.
4. **Heterogeneous Robustness as a Key Deployment Advantage**: When vehicles from different manufacturers use different perception models, only early fusion-based approaches are naturally compatible.

## Limitations & Future Work

1. The foreground selector in FAPS requires separate pretraining, increasing deployment complexity; unsupervised or self-supervised alternatives merit exploration.
2. The completion module is frozen during end-to-end training, precluding joint optimization and potentially limiting the performance ceiling.
3. Evaluation is limited to LiDAR 3D detection; effectiveness on downstream tasks such as semantic segmentation and tracking remains unverified.
4. ICP-based alignment introduces additional computational cost that may become a bottleneck in large-scale multi-agent scenarios.
5. Security under adversarial attacks or malicious agent scenarios is not considered.

## Related Work & Insights

- **Where2comm/CoBEVT**: Representative intermediate fusion methods focusing on "what features to transmit" and "how to compress them," but constrained by model heterogeneity.
- **STAR**: Applies MAE to reconstruct masked features in intermediate fusion, but is not directly applicable to early fusion.
- **PointPillars**: Serves as CoLC's detection backbone; the pillar representation is naturally amenable to VQ-based completion.
- The approach could be extended to visual collaborative perception (transmitting and completing image patches) and multimodal fusion.

## Rating

| Dimension | Score (1–5) |
|------|:---:|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 5 |
| Overall | 4.3 |

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures](../../AAAI2026/autonomous_driving/linext_revisiting_lidar_completion_with_efficient_non-diffusion_architectures.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](../../ICCV2025/autonomous_driving/distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)
- [\[CVPR 2026\] Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals](rascene_high-fidelity_3d_scene_imaging_with_mmwave_communication_signals.md)

<!-- RELATED:END -->
