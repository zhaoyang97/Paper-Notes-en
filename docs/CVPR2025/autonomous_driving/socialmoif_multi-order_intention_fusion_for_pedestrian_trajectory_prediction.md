---
title: >-
  [Paper Note] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction
description: >-
  [CVPR 2025][Autonomous Driving][Pedestrian Trajectory Prediction] SocialMOIF proposes a multi-order intention fusion model that comprehensively captures social intentions through a first-order direct interaction layer and a high-order neighbor indirect interaction layer. Combined with a trajectory distribution approximator based on the squeeze theorem and a global trajectory optimizer introducing KANs for the first time, it achieves SOTA performance on multiple datasets inclu…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Pedestrian Trajectory Prediction"
  - "Multi-Order Intention Interaction"
  - "Global Trajectory Optimization"
  - "Kolmogorov-Arnold Network (KAN)"
  - "Distribution Approximation"
date: 2026-05-08
content_hash: e0bba3888c3cb589
---

# SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction

**Conference**: CVPR 2025  
**arXiv**: [2504.15616](https://arxiv.org/abs/2504.15616)  
**Code**: [https://github.com/XiaodZhao/SocialMOIF](https://github.com/XiaodZhao/SocialMOIF)  
**Area**: Autonomous Driving  
**Keywords**: Pedestrian Trajectory Prediction, Multi-Order Intention Interaction, Global Trajectory Optimization, Kolmogorov-Arnold Network (KAN), Distribution Approximation

## TL;DR

SocialMOIF proposes a multi-order intention fusion model that comprehensively captures social intentions through a first-order direct interaction layer and a high-order neighbor indirect interaction layer. Combined with a trajectory distribution approximator based on the squeeze theorem and a global trajectory optimizer introducing KANs for the first time, it achieves SOTA performance on multiple datasets including ETH/UCY, SDD, NBA, and NuScenes.

## Background & Motivation

**Background**: Pedestrian trajectory prediction is a key task in intelligent transportation and autonomous driving. Existing methods are divided into knowledge-driven approaches (e.g., social force models, game-theoretic methods) and data-driven approaches (e.g., LSTM-based social pooling, Transformer-based spatiotemporal modeling, and graph neural network-based interaction modeling).

**Limitations of Prior Work**: (1) Existing methods primarily focus on the **direct interaction** (first-order) between the target pedestrian and neighbors, neglecting the high-order impacts of interactions within the neighbor group that are **indirectly propagated** to the target pedestrian. Although Kim et al. modeled $N$-order interactions, they treated each order equally, which weakens the dominant role of low-order intentions when the number of neighbors is large. (2) The latent variable distribution in generative models lacks explicit guidance, resulting in poor interpretability. (3) Downstream trajectory generation is typically serial step-by-step, resulting in cumulative errors and low efficiency.

**Key Challenge**: How to effectively incorporate high-order indirect influences while reinforcing the dominant position of first-order direct interactions? How to make the latent variable distribution of generative models more interpretable and controllable?

**Goal**: (1) Design a multi-order intention fusion network to capture both direct and indirect interaction information simultaneously; (2) Design a trajectory distribution approximator to explicitly guide latent variables; (3) Design a global trajectory optimizer to achieve parallel prediction.

**Key Insight**: High-order intention interactions can be decomposed into first-order interactions among neighbors. By capturing the internal interaction matrix of neighbors in parallel across multiple subspaces, and fusing it with the first-order interaction matrix after weighting with learnable influence factors $\eta_m$, the dominance of low-order interactions is guaranteed while high-order interactions act as supplements.

**Core Idea**: Utilize a hierarchical attention mechanism to separately model first-order (target-neighbor) and high-order (neighbor-neighbor propagation to target) intention interactions. After fusion, a distribution approximator and a KAN optimizer are utilized to generate high-quality parallel trajectory predictions.

## Method

### Overall Architecture

The input consists of historical trajectories $g_i^{1:T_H}$ (8 frames), and the output consists of predicted trajectories $\hat{g}_i^{T_H+1:T_F}$ (8 frames). The pipeline comprises four core components: (1) The Multi-Order Intention Fusion (MOIF) model, which extracts and fuses multi-order social intentions from position, velocity, distance, angle, etc.; (2) The trajectory distribution approximator, which explicitly guides the latent variable distribution by using multi-order intentions as the lower bound and the ground-truth future trajectory as the upper bound; (3) The global trajectory optimizer, which performs parallel optimization along the temporal dimension using a KAN; (4) A distance-direction fusion loss function that comprehensively supervises dynamic changes.

### Key Designs

1. **Multi-Order Intention Fusion Model (MOIF)**:

    - **Function**: Comprehensively capture direct and indirect intention interactions between the target pedestrian and neighbors.
    - **Mechanism**: Designed in two layers. **High-order intention interaction layer**: Extracts the absolute positions and velocities of $N_n$ neighbors, computing the neighbor-to-neighbor intention coefficient matrix $W_U^m$ in parallel across $M$ subspaces using multi-head attention (total intentions $\Omega = N_n^2$). **First-order intention interaction layer**: Constructs rich features between the target pedestrian and neighbors (position, distance $d$, velocity angle $\theta$, predicted endpoint distance $e$), computing the direct interaction matrix $W_S$ via attention. During fusion, the high-order matrix is weighted by learnable factors $\eta_m$ and added to the first-order matrix: $A_i = (\sum_m \eta_m W_U^m + W_S) V_S$.
    - **Design Motivation**: The first-order input incorporates richer relative features (distance, angle, predicted endpoint distance) to reinforce its dominant position, while high-order influences are adaptively controlled via learnable weights $\eta_m$ to prevent high-order interactions from overwhelming low-order ones when the neighbor count is large.

2. **Trajectory Distribution Approximator**:

    - **Function**: Explicitly guide the latent variable distribution, improving the representation quality of generated trajectories and enhancing model interpretability.
    - **Mechanism**: Inspired by the squeeze theorem, the multi-order fused intention $I_i$ serves as the lower bound and the ground-truth future trajectory feature $B_i$ acts as the upper bound. The latent variable distribution is approximated through reparameterization as $q_\varphi(\cdot|I_i^{t-1}, B_i^t) \sim N(u_{i\varphi}^t, \sigma_{i\varphi}^t)$. During training, two sets of parameters $\varphi$ (incorporating ground-truth information) and $\vartheta$ (used only during inference) are maintained, and the KL divergence is minimized to align the two distributions. An RNN updates the intention $I_i^t$ at each time step, and the latent variable and generated trajectory are concatenated for updating.
    - **Design Motivation**: In conventional generative models (like CVAE), the latent variable distribution is learned implicitly and lacks controllability. Establishing upper and lower bounds to explicitly constrain the distribution range makes the model safer and more interpretable for safety-critical tasks.

3. **Based KAN Global Trajectory Optimizer**:

    - **Function**: Introduce Kolmogorov-Arnold Networks (KAN) to trajectory prediction for the first time to achieve parallel trajectory optimization across the entire temporal domain.
    - **Mechanism**: The decoder output $g_i^{T_H+1:T_F}$ is flattened in the temporal dimension to $\Gamma^0 \in \mathbb{R}^{2T_F}$, and then optimized globally through an $L$-layer KAN (where the activation function matrix $\Phi_\ell$ of each layer is learnable): $\hat{g}_i \leftarrow \Gamma^L = (\Phi_L \circ \cdots \circ \Phi_0) \Gamma^0$. Finally, it is mapped back to the original dimensions to obtain the final trajectory for the entire prediction horizon.
    - **Design Motivation**: Traditional methods generate trajectories serially step-by-step, leading to accumulated prediction errors and low computation efficiency. KAN's learnable activation functions are more flexible than MLPs with fixed activations, making them suitable for capturing non-linear global patterns of trajectories.

### Loss & Training

The total loss combines three parts: distance, direction, and KL divergence:

$$L = \sum_{t} \{ \mathbb{E}[L_{dis} + L_{angle}] - KL[q_\varphi \| q_\vartheta] \}$$

- **Distance loss** $L_{dis} = \|\hat{g}_i^t - g_{igt}^t\|$: Standard displacement error.
- **Direction loss** $L_{angle} = -\arccos(\text{cos\_sim}(\hat{g}_i^t - \hat{g}_i^{t+1}, g_{igt}^t - g_{igt}^{t+1}))$: Supervises consistency between the predicted direction and the ground-truth direction.
- **KL divergence**: Aligns the latent variable distributions between training mode (with ground truth) and testing mode (inference only).

Training is conducted on two RTX 4090 GPUs, using $M=6$ subspaces, an $L=3$ layer KAN, 8 frames of history to predict 8 frames of future, with the best-of-20 evaluation protocol.

## Key Experimental Results

### Main Results

ETH/UCY average ADE/FDE (meters):

| Method | ETH | Hotel | Univ | Zara1 | Zara2 | Avg |
|------|-----|-------|------|-------|-------|-----|
| EqMotion | 0.36/0.54 | 0.12/0.16 | 0.21/0.39 | 0.16/0.27 | 0.11/0.19 | 0.19/0.31 |
| E-V2-Net-SC | 0.23/0.30 | 0.10/0.13 | 0.18/0.24 | 0.13/0.16 | 0.11/0.16 | 0.15/0.20 |
| **SocialMOIF** | 0.26/0.31 | **0.10/0.12** | **0.11/0.17** | **0.10/0.17** | **0.09/0.14** | **0.13/0.18** |

Other datasets ADE/FDE:

| Dataset | Prev. SOTA | SocialMOIF | Gain |
|--------|-----------|------------|------|
| NBA-Rebound | 0.54/0.79 | **0.34/0.66** | -37%/-16% |
| NBA-Scores | 0.46/0.76 | **0.30/0.56** | -35%/-26% |
| SDD | 0.21/0.34 | **0.17/0.24** | -19%/-29% |
| NuScenes | 1.04/1.47 | **0.92/1.56** | -12%/+6% |

### Ablation Study

Contribution of each component (SDD/NuScenes ADE/FDE):

| Group | Key Component | SDD | NuScenes |
|----|----------|-----|----------|
| 1 | Position, distance, and direction loss only | 0.46/0.71 | 1.26/2.15 |
| 2 | + Velocity + angle | 0.45/0.66 | 1.24/2.09 |
| Incremental addition | + High-order intention + distribution approximation + KAN optimization + direction loss | → **0.17/0.24** | → **0.92/1.56** |

### Key Findings

- Average ADE/FDE on ETH/UCY is reduced by 13.3%/10.0% compared to the previous SOTA (E-V2-Net-SC).
- The model exhibits a major improvement on the NBA dataset (ADE reduced by 35-37%), demonstrating that multi-order fusion is highly effective under intense interaction scenarios.
- The introduction of direction loss yields universal improvements across all baseline models (verified in the supplementary material of the paper).
- Qualitative analysis indicates that the model properly handles complex scenarios such as turning, decelerating to yield, and crowded environments.
- KAN is utilized in trajectory prediction for the first time, achieving parallel optimization across the temporal dimension.

## Highlights & Insights

- The design philosophy of multi-order intention fusion is solid: first-order dominance with high-order supplement, adaptively balanced by learnable factors.
- Introducing KAN for global optimization in trajectory prediction is an interesting cross-domain innovation.
- The distance-direction fusion loss is simple yet effective, focusing on the dynamic states of pedestrian movement rather than just the endpoint position.
- The trajectory distribution approximator increases the interpretability of the CVAE framework via upper- and lower-bound constraints.

## Limitations & Future Work

- FDE on NuScenes slightly increases (1.56 vs 1.47), suggesting that global optimization might introduce deviations in certain scenarios.
- In ETH scenarios where interaction in parallel traffic flows is weak, the performance is inferior to E-V2-Net-SC, indicating that the method is better suited for high-interaction environments.
- The number of subspaces of high-order intention $M=6$ and KAN depth $L=3$ are fixed hyperparameters; adaptive settings might yield better results.
- Static environmental constraints (such as obstacles and road boundaries) are not considered, which may restrict pure social force modeling in complex urban scenarios.
- The arccos in the direction loss might suffer from numerical stability issues during gradient computation.

## Related Work & Insights

- The Social Force Model [12] pioneered pedestrian interaction modeling, and SocialMOIF can be seen as its continuation in the deep learning era.
- The V2-Net + SocialCircle [41] series represents the directly comparable SOTA; SocialMOIF surpasses them through more meticulous interaction modeling.
- The theoretical basis of KAN's [24] learnable activation functions provides a good trade-off between parameter efficiency and expressiveness for trajectory optimization.
- Insight: Interaction modeling of "who influences whom" in trajectory prediction is far from saturated; high-order propagation effects warrant further exploration.

## Rating

- **Novelty**: 7/10 — The multi-order intention fusion and the introduction of KAN are novel; however, the basic framework (attention + CVAE) is relatively conventional.
- **Experimental Thoroughness**: 9/10 — Comprehensive evaluation across four mainstream datasets, detailed ablation studies, and rich qualitative analysis.
- **Writing Quality**: 7/10 — The method is clearly described but equation-dense, and some notation is not particularly intuitive.
- **Value**: 8/10 — The substantial improvements in highly interactive scenarios like NBA are impressive, and the open-source code enhances practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](../../AAAI2026/autonomous_driving/catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)
- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](gdfusion_temporal_fusion_occupancy.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)

</div>

<!-- RELATED:END -->
