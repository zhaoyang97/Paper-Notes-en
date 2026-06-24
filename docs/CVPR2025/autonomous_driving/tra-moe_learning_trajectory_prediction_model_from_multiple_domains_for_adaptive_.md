---
title: >-
  [Paper Note] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning
description: >-
  [CVPR 2025][Autonomous Driving][Trajectory Prediction] This paper proposes Tra-MoE, which uses a sparse gated Mixture-of-Experts (MoE) architecture to train a trajectory prediction model. It effectively fuses large-scale out-of-domain action-free video data with small-scale in-domain robot demonstrations. It also designs an adaptive policy conditioning technique to explicitly align 2D trajectories with visual observations, significantly improving the success rate of robot man…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Trajectory Prediction"
  - "Mixture of Experts"
  - "Cross-Domain Learning"
  - "Policy Conditioning"
  - "Robot Manipulation"
date: 2026-05-08
content_hash: 411476e3c62ff12b
---

# Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning

**Conference**: CVPR 2025  
**arXiv**: [2411.14519](https://arxiv.org/abs/2411.14519)  
**Code**: [https://github.com/MCG-NJU/Tra-MoE](https://github.com/MCG-NJU/Tra-MoE)  
**Area**: Autonomous Driving/Robotics  
**Keywords**: Trajectory Prediction, Mixture of Experts, Cross-Domain Learning, Policy Conditioning, Robot Manipulation

## TL;DR

This paper proposes Tra-MoE, which uses a sparse gated Mixture-of-Experts (MoE) architecture to train a trajectory prediction model. It effectively fuses large-scale out-of-domain action-free video data with small-scale in-domain robot demonstrations. It also designs an adaptive policy conditioning technique to explicitly align 2D trajectories with visual observations, significantly improving the success rate of robot manipulation in both simulation and real-world scenarios.

## Background & Motivation

**Background**: In robot learning, a scalable paradigm is to first learn a trajectory prediction model from action-free video data, and then train a trajectory-guided policy model with a small amount of demonstration data containing action labels. Works like ATM have achieved preliminary success in this direction but mainly rely on in-domain data to train the trajectory model.

**Limitations of Prior Work**: How to effectively utilize large-scale out-of-domain video data to jointly train the trajectory model remains under-explored. Out-of-domain data may contain different environments, objects, skills, and embodiments; direct mixed training leads to optimization conflicts. Experiments show that naively scaling up out-of-domain training data actually drops in-domain performance by 5.6 percentage points (57.6 $\rightarrow$ 52.0). Furthermore, how to effectively use the predicted 2D trajectories to condition the policy module is also an open challenge.

**Key Challenge**: Out-of-domain data can provide complementary knowledge to boost generalization capability, but the data distribution across different domains varies massively. A unified Transformer model struggles to strike a balance between parameter collaboration and specialization—collaboration requires shared parameters to learn general patterns, while specialization requires different sub-networks to handle different data distributions.

**Goal**: (1) To design a trajectory model architecture that can efficiently utilize out-of-domain data; (2) To propose a more effective policy conditioning method that allows 2D trajectories to better guide action prediction.

**Key Insight**: The authors observe that the sparse MoE architecture naturally possesses the dual capability of parameter collaboration (shared attention layers) and specialization (different experts handling different inputs), making it well-suited for multi-domain data. Meanwhile, directly mapping 2D trajectories to the image space and encoding them as learnable embeddings allows for more flexible spatial alignment.

**Core Idea**: Replace some FFN layers in the Transformer with a sparse gated MoE (Top-1) to construct Tra-MoE, expanding model capacity to digest out-of-domain data while keeping FLOPs constant; then use adaptive trajectory masking to achieve explicit spatial alignment of 2D trajectories and images.

## Method

### Overall Architecture

The overall pipeline of Tra-MoE consists of two stages: (1) **Trajectory Model Pre-training**: Jointly train the MoE trajectory prediction model on out-of-domain action-free videos $\mathcal{D}_{ood}$ and in-domain demonstrations $\mathcal{D}_{in}$. The inputs are the image observation $o_t$, query point set $\mathbf{p}_t$, and language instruction $\ell$, and the output is the future $H$-step trajectory of arbitrary points $\mathbf{p}_{t:t+H} \in \mathbb{R}^{H \times K \times 2}$. (2) **Policy Training**: Freeze the trajectory model, and train the policy model via behavior cloning solely on $\mathcal{D}_{in}$, utilizing the adaptive conditioning technique to integrate trajectory information into visual observations to predict robot actions.

### Key Designs

1. **Sparse Gated MoE Trajectory Model**:

    - Function: To expand model capacity while keeping FLOPs constant, achieving a balance between parameter collaboration and specialization.
    - Mechanism: Replace standard FFNs in selected layers of the track transformer with MoE blocks, where each MoE block consists of $N$ expert FFNs and a gating network $\mathcal{G}$. A Top-1 gating strategy is adopted, meaning only one expert is activated per token: $\mathcal{G}(\mathbf{x}_s; \Theta)_i = \text{softmax}(\text{Top-1}(g(\mathbf{x}_s; \Theta), 1))_i$, and the final output is $\mathcal{F}_{\text{sparse}}^{\text{MoE}} = \sum_{i=1}^{K} \mathcal{G}(\mathbf{x}_s)_i f_i(\mathbf{x}_s; \mathbf{W}_i)$. By default, 4 experts are used, replacing the 1st, 5th, and 8th layers (with extra replacements at the 2nd and 7th layers when data scales up).
    - Design Motivation: Most of the parameters of the MoE (attention layers) are shared and trained across all domain data, capturing general complementary patterns for collaboration; different inputs and tokens naturally activate different experts to achieve specialization. Experiments prove that even if a dense model is scaled to the same parameter size as the MoE (width scaling 48.4/depth scaling 52.5 vs MoE 61.4), it cannot achieve equivalent performance.

2. **MoE Training Strategy (Router Z-Loss)**:

    - Function: To stabilize sparse MoE training and avoid excessively large logits from the gating network.
    - Mechanism: The authors systematically studied three commonly used MoE auxiliary techniques: (i) router z-loss $\mathcal{L}_z = \frac{1}{S}\sum_{k=1}^{S}(\log\sum_{i=1}^{N}e^{g_i^{(k)}})^2$ to penalize excessively large router logits, boosting performance by 4.5 points with a weight of $\lambda_z=10^{-4}$; (ii) load-balancing loss $\mathcal{L}_{\text{lo-ba}} = N \sum_{i=1}^{N} \mathcal{Q}_i \mathcal{P}_i$ to balance expert loading, which, however, degraded performance in experiments—since forcing balance under non-uniform data distributions disrupts the advantage of specialization; (iii) adding noise to gating logits, which also led to a drop in performance (56.9 $\rightarrow$ 55.8).
    - Design Motivation: Multi-domain data distribution is naturally non-uniform; forcing balanced expert loading or adding noise negates the specialization advantages of MoE. Using only z-loss to stabilize training yields the best results.

3. **Adaptive Policy Conditioning**:

    - Function: Explicitly align 2D trajectories into the image space to provide more flexible spatial guidance for the policy model.
    - Mechanism: Construct an auxiliary mask channel where 2D trajectory points are filled into the mask based on their spatial locations, with each trajectory point set as a learnable embedding. The mask is concatenated with the image along the channel dimension to form an $H \times W \times 4$ tensor input to the encoder. The policy model adopts a dual-fusion architecture: early fusion (trajectory tokens interact with image tokens in a fusion transformer) + late fusion (fused features are concatenated with raw trajectories along the channel dimension).
    - Design Motivation: Different positions on a trajectory have different semantics—starting points emphasize local motion, while endpoints focus on global trends. Hand-drawn masks (with fixed values of 128/255) provide spatial alignment but lack adaptivity; learnable embeddings allow the model to automatically learn optimal representations for different trajectory positions. Experiments show that hand-drawn masks lead to a significant performance drop on LIBERO-Goal (81 $\rightarrow$ 58), whereas the adaptive mask improves it to 77.

### Loss & Training

The total loss of the trajectory model is $\mathcal{L}_{\text{total}} = \lambda_{\text{tra}} \cdot \mathcal{L}_{\text{tra}} + \lambda_z \cdot \mathcal{L}_z$, where $\mathcal{L}_{\text{tra}}$ is the trajectory prediction MSE loss, and $\mathcal{L}_z$ is the router z-loss ($\lambda_z = 10^{-4}$). The policy model is trained using MSE loss for behavior cloning. Trajectory ground truth (GT) is generated by CoTracker.

## Key Experimental Results

### Main Results

Simulation Experiments (LIBERO Benchmark, average success rate %):

| Method | Spatial | Goal | Object | Long | Avg. |
|------|---------|------|--------|------|------|
| ATM (in-domain only) | 67.5 | 68.5 | 68.0 | 26.5 | 57.6 |
| ATM + OOD data | 49.5 | 67.0 | 56.5 | 35.0 | 52.0 |
| **Tra-MoE + OOD data** | **62.5** | **81.0** | **73.5** | **28.5** | **61.4** |
| + Adaptive mask | **69.5** | **77.0** | **88.0** | **30.5** | **66.3** |

Real-World Experiments (5 tasks, success rate %):

| Configuration | Pour | Push | Pick&Pass | Tissue | Fold | Avg. |
|------|------|------|-----------|--------|------|------|
| Baseline | 40.0 | 45.0 | 50.0 | 30.0 | 25.0 | 38.0 |
| + Human data | 45.0 | 35.0 | 50.0 | 25.0 | 35.0 | 38.0 |
| + Human + MoE | 40.0 | 50.0 | 60.0 | 35.0 | 45.0 | 46.0 |
| + Human + MoE + Adaptive | 60.0 | 70.0 | 65.0 | 35.0 | 50.0 | **56.0** |

### Ablation Study

| Configuration | Average Success Rate | Note |
|------|-----------|------|
| Tra-MoE (4 experts, z-loss) | 61.4 | Full model |
| w/o z-loss | 56.9 | Drop of 4.5 points without z-loss |
| + load-balancing loss (1e-3) | 52.6 | Load-balancing hurts performance |
| + noise to gating | 55.8 | Adding noise also degrades performance |
| Dense scale-up (width) | 48.4 | Dense model with same parameter size performs poorly |
| Dense scale-up (depth) | 52.5 | Depth scale-up is also inferior to MoE |
| 2 experts | 56.0 | A small number of experts is already effective |
| 3 experts | 55.1 | - |
| 4 experts | 56.9 | More experts show a positive trend |

### Key Findings

- MoE architecture is crucial for utilizing out-of-domain data: naively scaling out-of-domain data drops the dense model by 5.6 points, but Tra-MoE improves by 9.6 points instead (51.8 $\rightarrow$ 61.4).
- Scaling the dense model to the same parameter size as the MoE (via width or depth extension) completely fails to match MoE's performance, proving the structural superiority of sparse activation.
- Load-balancing loss is harmful under multi-domain, non-uniform distributions; only z-loss is helpful for stabilizing training.
- When trained across RLbench data (different physics engine), Tra-MoE still outperforms the baseline by 12.6 points.
- Adaptive masking shows a 19-point advantage over hand-drawn masking on the Goal sub-task, as its learnable embeddings can handle overlaps in the beginning and end segments of trajectories.

## Highlights & Insights

- The first systematic application of MoE in robot trajectory learning—when transferring MoE from NLP/CV, they found that load-balancing loss actually hurts on multi-domain robot data, a finding that has guiding significance for subsequent work.
- Adaptive policy conditioning "paints" 2D trajectories onto images as an extra channel, which is both simple and effective. The core insight is that semantic differences at different trajectory locations require learnable representations to capture.
- The synergistic effect of out-of-domain data + MoE is noteworthy: adding out-of-domain data alone is harmful, and adding MoE alone is also harmful, but using both together yields a huge improvement of 9.6 points.

## Limitations & Future Work

- The experimental scale is relatively limited—simulation was only validated on LIBERO, and the real-world evaluation only had 5 tasks with 50 demonstrations each.
- The ratio of out-of-domain data to in-domain data (9:2) needs to be adjusted manually, lacking an automated data mixing strategy.
- More complex MoE routing strategies (e.g., expert-choice, soft MoE) were not explored.
- Future directions: (1) Scaling up the method to larger-scale internet video data; (2) Combining with 3D trajectory prediction for more precise action mapping; (3) Exploring the impact of out-of-domain data quality on model performance.

## Related Work & Insights

- **vs ATM**: ATM is the baseline of this paper, which only trains the trajectory model using in-domain data. Tra-MoE introduces MoE and out-of-domain data on top of this, improving the average success rate from 57.6 to 66.3.
- **vs RT-Trajectory**: RT-Trajectory conditions the policy using a hand-drawn trajectory mask (fixed value). The adaptive learnable embedding method of this paper is more flexible, especially performing better in trajectory overlap scenarios.
- **vs Large-Scale Robot Foundation Models (RT-2, Octo)**: These works adopt a pre-train + fine-tune paradigm and directly combine heterogeneous action spaces, which may lead to sub-optimal solutions. Tra-MoE unifies different embodiments through the trajectory space of action-free videos.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The application of MoE to robot trajectory learning is novel, but the core individual techniques (MoE, trajectory conditioning) have precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Simulation ablations are comprehensive and in-depth, and real-world experiments provide practical validation, though on a smaller scale.
- **Writing Quality**: ⭐⭐⭐⭐ — The research problem is clear, the ablation studies are logically designed, and the findings are thoroughly analyzed.
- **Value**: ⭐⭐⭐⭐ — The insights on MoE in multi-domain robotic learning (e.g., balancing loss is harmful) are of practical guidance to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[CVPR 2026\] W2W: Language-Model-Based Trajectory Prediction with Reinforcement Learning](../../CVPR2026/autonomous_driving/w2w_language-model-based_trajectory_prediction_with_reinforcement_learning.md)
- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](../../ICCV2025/autonomous_driving/donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)

</div>

<!-- RELATED:END -->
