---
title: >-
  [Paper Note] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration
description: >-
  [CVPR 2026][Autonomous Driving][Trajectory Prediction] FoSS proposes a frequency-domain–time-domain dual-branch framework that organizes Fourier spectra via progressive spiral reordering (HelixSort) before feeding them into a selective state space model (SSM), and combines a temporal dynamic SSM with cross-attention fusion to achieve state-of-the-art trajectory prediction on Argoverse 1/2 while reducing parameter count by over 40% and inference latency by 22%.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Trajectory Prediction
  - Fourier Transform
  - State Space Model
  - Dual-Branch Architecture
  - Multimodal Prediction
date: 2026-05-08
content_hash: 54989ac7972dabb3
---

# FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration

**Conference**: CVPR 2026
**arXiv**: [2603.01284](https://arxiv.org/abs/2603.01284)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Trajectory Prediction, Fourier Transform, State Space Model, Dual-Branch Architecture, Multimodal Prediction

## TL;DR
FoSS proposes a frequency-domain–time-domain dual-branch framework that organizes Fourier spectra via progressive spiral reordering (HelixSort) before feeding them into a selective state space model (SSM), and combines a temporal dynamic SSM with cross-attention fusion to achieve state-of-the-art trajectory prediction on Argoverse 1/2 while reducing parameter count by over 40% and inference latency by 22%.

## Background & Motivation
Accurate trajectory prediction is critical for safety in autonomous driving, particularly in dense multi-agent environments. Existing methods face an inherent trade-off among three demands: modeling long-range cross-agent dependencies, representing multimodal futures to capture uncertainty, and satisfying strict real-time constraints.

Transformer architectures achieve high accuracy via self-attention but incur $\mathcal{O}(N^2)$ computational complexity, limiting deployment on resource-constrained systems. Recurrent models are efficient but struggle to capture long-range dependencies and fine-grained local dynamics. Methods that operate solely in the time domain tend to conflate global motion patterns with local dynamics, while standard Fourier representations lack ordered frequency semantics, making it difficult for sequence models to process spectral information effectively.

**Core observation**: Trajectory signals exhibit complementary structures in the spectral and temporal domains — **the amplitude spectrum encodes global motion trends, while the phase spectrum captures fine-grained temporal variations**. However, DFT outputs do not preserve a continuous low-to-high frequency arrangement (since $\omega$ and $T-\omega$ correspond to the same physical frequency), and directly processing them with an SSM causes the model to alternate between global and local reasoning, disrupting state evolution.

**Core idea**: Design a progressive spiral reordering module (HelixSort) that sorts DFT coefficients by spectral radius, enabling the SSM to process spectral information in a coarse-to-fine manner. Combined with long-range dependency modeling in a temporal SSM, this achieves high-accuracy multimodal trajectory prediction at linear complexity.

## Method

### Overall Architecture
FoSS adopts a dual-branch architecture: (1) **Frequency-Domain Branch (FD-Mamba)** — applies DFT to historical trajectories to decompose them into amplitude and phase, reorders the result via HelixSort, and feeds it into two parallel SSM sub-modules (Coarse2Fine-SSM for spatial interaction and SpecEvolve-SSM for channel evolution); (2) **Time-Domain Branch (TD-Mamba)** — models long-range dependencies directly on temporal sequences via input-dependent dynamic SSMs. The two branches are fused through a cross-attention layer, and learnable query vectors decode $K$ candidate trajectories that are combined via weighted fusion to produce the final prediction.

### Key Designs
1. **Progressive Spiral Reordering (HelixSort)**:

    - Function: Reorders unstructured DFT output coefficients into a sequence with monotonically increasing frequency.
    - Mechanism: (a) Reshape 1D DFT coefficients $F^{(k)} \in \mathbb{C}^T$ into a 2D grid $\mathcal{F}^{(k)} \in \mathbb{C}^{\sqrt{T} \times \sqrt{T}}$; (b) starting from the spectral center $(u_0, v_0)$, traverse outward in a spiral direction, sorting in ascending order of spectral radius $r = \sqrt{(u-u_0)^2 + (v-v_0)^2}$; (c) generate reordering indices $\pi^{(k)}$ to produce an ordered sequence $\widehat{F}^{(k)}$ satisfying $\forall i < j, r_i \leq r_j$.
    - Design Motivation: Standard DFT interleaves low- and high-frequency coefficients, forcing the SSM to switch repeatedly between global and local reasoning and disrupting state evolution. HelixSort concentrates low-frequency components at the beginning of the sequence and high-frequency components at the end, allowing the SSM to first accumulate global trends before refining local details — consistent with a coarse-to-fine reasoning strategy. Overhead is negligible: only a 0.08% increase in FLOPs and less than 0.25 MB of memory.

2. **Frequency-Domain Dual Sub-modules (Coarse2Fine-SSM + SpecEvolve-SSM)**:

    - Function: Deeply model frequency-domain features from the spatial and channel dimensions, respectively.
    - Mechanism:
        - **Coarse2Fine-SSM**: Applies FFT to input features → HelixSort reordering of amplitude and phase → depthwise separable convolution + SiLU + selective SSM + LayerNorm → iFFT to recover time-domain representation → element-wise multiplication with original features: $F_f = \text{iFFT}(A'(F_l), P'(F_l)) \odot \text{SiLU}(F_l)$.
        - **SpecEvolve-SSM**: Global average pooling extracts channel features $F_g \in \mathbb{R}^{1 \times 1 \times C}$ → FFT along channel dimension → sorted in ascending order of spectral amplitude → iFFT → element-wise gated fusion: $F_a = \text{iFFT}(A(F_g)', P(F_g)') \odot \text{SiLU}(F_g)$, yielding enhanced representation $F_{\text{enhance}} = F_a \odot F_{in}$.
        - Outputs of the two sub-modules are concatenated along the channel dimension and projected linearly to obtain the final frequency-domain representation $F_{\text{freq}}$.
    - Design Motivation: Coarse-to-fine modeling in the spatial dimension captures spatial interactions along motion trajectories, while spectral evolution in the channel dimension captures correlations across different feature dimensions — the two are complementary.

3. **Temporal Dynamic Selective SSM (TD-Mamba)**:

    - Function: Emulates self-attention behavior in the time domain at linear complexity to capture long-range temporal dependencies.
    - Mechanism: State transition matrices $A_t, B_t, C_t, D_t$ are dynamically generated from the current input $X(t)$ and its locally convolved features $\tilde{X}(t) = \text{Conv1D}(X(t))$: $A_t = f_A(X(t), \tilde{X}(t))$, where $f_A$ and analogous functions are lightweight MLPs. State update: $h(t+1) = A_t h(t) + B_t X(t)$; output: $Y_{\text{time}}(t) = C_t h(t) + D_t X(t)$. Hidden states are stabilized via SiLU + LayerNorm.
    - Design Motivation: Input-dependent parameterization allows state updates to automatically amplify salient motion patterns and suppress noise at different timesteps. Conv1D preprocessing enhances sensitivity to local dynamic variations.

### Loss & Training
- Time-domain and frequency-domain supervision are applied jointly: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \lambda \mathcal{L}_{\text{freq}}$.
- Time-domain loss: $\mathcal{L}_{\text{time}} = \|\hat{Y}_{\text{final}} - Y\|_1$ (L1 distance).
- Frequency-domain loss: $\mathcal{L}_{\text{freq}} = \|F(\hat{Y}_{\text{final}}) - F(Y)\|_1$ (L1 distance in the Fourier domain).
- Candidate trajectories are generated via cross-attention between learnable queries $Q \in \mathbb{R}^{K \times d}$ and fused features $Z$; each candidate is mapped through an MLP.
- Training: Adam optimizer, learning rate 0.001, batch size 128, 50 epochs; learning rate reduced to 10% if no improvement on the validation set for 5 consecutive epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | FoSS | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Argoverse 2 | b-minFDE6↓ | **1.69** | 1.74 (Wayformer) | 2.9% |
| Argoverse 2 | minADE6↓ | **0.61** | 0.63 (DeMo) | 3.2% |
| Argoverse 2 | minFDE6↓ | **1.07** | 1.17 (HiVT/DeMo) | 8.5% |
| Argoverse 2 | MR6↓ | **0.11** | 0.12 (Wayformer) | 8.3% |
| Argoverse 2 | Parameters | **4.18M** | 5.92M (DeMo) | −29.4% |
| Argoverse 1 | minADE1↓ | **1.67** | 1.65 (DeMo) | Approx. on par |
| Argoverse 1 | minFDE1↓ | 2.05 | **2.02** (Wayformer) | Approx. on par |
| Argoverse 1 | MR1↓ | **0.11** | 0.11 (multiple) | On par |

### Ablation Study (Argoverse 2)

| Configuration | minADE6↓ | minFDE6↓ | MR6↓ | Notes |
|------|---------|---------|------|------|
| w/o Frequency-Domain Branch | 0.71 | 1.36 | 0.17 | Frequency-domain cues are critical for global trend modeling |
| w/o HelixSort | 0.69 | 1.32 | 0.16 | Ordered spectral traversal improves structural coherence |
| w/o Fourier SSM | 0.70 | 1.35 | 0.17 | Selective SSM is indispensable for spectral-temporal fusion |
| Concat+MLP replacing cross-attention | 0.69 | 1.33 | 0.16 | Token-level cross-interaction outperforms simple concatenation |
| **Full model** | **0.65** | **1.29** | **0.15** | All components are mutually complementary and jointly optimal |

### Key Findings
- The frequency-domain branch is plug-and-play with other temporal backbones: FD-Mamba + Transformer (b-minFDE 1.77) and FD-Mamba + LSTM (1.91), validating its generality.
- Inference latency is only 64 ms (10% faster than QCNet at 71 ms; 22% faster than HiVT at 82 ms), with FLOPs of 22.1G (51% of QCNet).
- At 4.18M parameters, FoSS is the most compact model among all compared methods, clearly demonstrating its efficiency advantage.
- Minor oscillations remain in high-frequency motion scenarios such as frequent lane changes, as low-frequency decomposition may underestimate rapid lateral maneuvers.

## Highlights & Insights
- HelixSort is an elegant and effective module design: it transplants the idea of JPEG zigzag encoding to spectral reordering, providing the SSM with ordered frequency inputs at virtually zero cost (0.08% FLOPs).
- Interpreting the spectral radius of DFT coefficients as an "artificial time axis" for SSM input cleverly transforms frequency-domain analysis into a sequence modeling problem.
- The complementary design of the dual-branch architecture is well-motivated: the frequency-domain branch captures disentangled representations of global patterns and local variations, while the time-domain branch preserves the original temporal context.
- The inclusion of a frequency-domain loss ensures that predicted trajectories are consistent with ground truth not only in positional accuracy but also in frequency structure.

## Limitations & Future Work
- Performance degrades slightly in scenarios involving abrupt high-frequency motion such as frequent lane changes, as spectral decomposition inherently favors low-frequency components.
- The advantage over DeMo/Wayformer on Argoverse 1 is less pronounced than on Argoverse 2, suggesting limited frequency-domain benefits for short-horizon prediction.
- Map encoding is not incorporated (only trajectory data is used), making comparisons with HD-map-based methods potentially unfair.
- HelixSort requires padding the sequence length to a perfect square; whether this handling is optimal for non-standard input lengths is not thoroughly discussed.

## Related Work & Insights
- Compared with spectral-domain trajectory prediction methods such as Spectral TGN (ICRA 2021), the innovation of FoSS lies in combining HelixSort with SSMs to address the spectral disorder problem.
- The demonstrated success of selective SSMs (e.g., Mamba, S4) in video and speech processing provides a foundation for introducing them into trajectory prediction.
- The cross-attention fusion resolves feature-scale mismatch between time-domain and frequency-domain representations via normalization and residual connections, serving as a general strategy for multi-domain fusion.
- The plug-and-play nature of the frequency-domain branch suggests potential applicability to other temporal forecasting tasks, such as behavior prediction and traffic flow prediction.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of HelixSort and SSM for frequency-domain modeling is original, though the overall dual-branch fusion framework is relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons and ablations on Argoverse 1/2, along with efficiency analysis and plug-and-play validation, are provided; however, additional datasets such as nuScenes are absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are complete, the intuition behind HelixSort is clearly explained, and the frequency decomposition visualization in Figure 1 aids comprehension.
- Value: ⭐⭐⭐⭐ The substantial reductions in parameter count and latency have direct practical value for deployment; the frequency-domain + SSM paradigm offers meaningful reference for the trajectory prediction community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences](u4d_uncertainty-aware_4d_world_modeling_from_lidar_sequences.md)
- [\[AAAI 2026\] Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions](../../AAAI2026/autonomous_driving/walking_further_semantic-aware_multimodal_gait_recognition_under_long-range_cond.md)
- [\[CVPR 2026\] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)

</div>

<!-- RELATED:END -->
