---
title: >-
  [Paper Note] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts
description: >-
  [AAAI 2026][Autonomous Driving][End-to-end autonomous driving] ExpertAD introduces a Mixture-of-Experts (MoE) architecture into the perception and prediction modules of end-to-end autonomous driving systems. A Perception…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "End-to-end autonomous driving"
  - "mixture of experts"
  - "perception adaptation"
  - "sparse attention"
  - "inference efficiency"
date: 2026-05-08
content_hash: 9f59a9a381585cdf
---

# ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts

**Conference**: AAAI 2026
**arXiv**: [2511.11740](https://arxiv.org/abs/2511.11740)  
**Code**: None  
**Area**: Autonomous Driving
**Keywords**: End-to-end autonomous driving, mixture of experts, perception adaptation, sparse attention, inference efficiency

## TL;DR

ExpertAD introduces a Mixture-of-Experts (MoE) architecture into the perception and prediction modules of end-to-end autonomous driving systems. A Perception Adapter dynamically re-weights BEV features to amplify task-critical semantics, while a Mixture of Sparse Experts employs a router to selectively activate relevant driving task experts and uses sparse attention to reduce computation. The framework reduces inference latency by approximately 25% while maintaining or improving planning performance.

## Background & Motivation

End-to-end autonomous driving systems (ADS) have achieved notable progress through unified perception–prediction–planning pipelines, yet two core challenges remain:

**1. Semantic ambiguity interfering with decision-making**: BEV features encode diverse semantic information (roads, vehicles, traffic signs, etc.), whereas different perception tasks (tracking vs. mapping) attend to different aspects. Passing all features indiscriminately may allow non-critical dimensions to overshadow task-relevant information.

**2. Multi-task interference and inference latency**: The prediction module encompasses diverse sub-tasks such as ego-state estimation, environment interaction modeling, and navigation execution. Activating all of them simultaneously causes inter-task interference and increases computational cost. For instance, mapping aids curve planning but contributes little to straight-road driving—different scenarios require different task combinations.

Existing MoE applications in autonomous driving are largely confined to single modules (e.g., trajectory selection in planning) and suffer from unstable expert activation under dynamic scenes. Prior efficiency-oriented methods (DriveAdapter, PlanKD) trade planning quality for speed.

## Method

### Overall Architecture

ExpertAD is a plug-and-play framework that integrates into existing Transformer-based end-to-end ADS (e.g., UniAD, VAD, VADv2), replacing the original perception and prediction modules:

1. **BEV Encoder** (retained) → generates BEV features
2. **Perception Adapter (PA)** (new) → dynamically selects and amplifies task-critical feature channels
3. **Mixture of Sparse Experts (MoSE)** (new) → routes and activates relevant experts; sparse attention reduces computation
4. **Planning Module** (retained) → generates final trajectories from motion queries output by MoSE

### Key Designs

**1. Perception Adapter (PA)**

Comprises two sub-components:

**Learned Adapter**: Learns channel selection weights for each task. BEV features are first temporally normalized and pooled, then task-specific learnable parameters $w^{(t)}$ are used to compute per-channel importance scores:

$$s = \frac{1}{H \times W}\sum_{i,j} \tilde{\text{BEV}}_{:,i,j} \odot w$$

Soft channel selection weights $\lambda^{(t)} \in [0,1]^d$ are obtained via constrained optimization, ensuring focus on $\tau$ dominant channels:

$$\max_\lambda \; s^\top \lambda + \epsilon\Omega(\lambda), \quad \text{s.t.} \; \mathbf{1}^\top\lambda = \tau, \; \lambda \in [0,1]^d$$

**Alignment Layer**: Re-scales BEV features using the selection weights:

$$F_{align} = \text{MLP}(\text{BEV} \odot \lambda) + \text{BEV}$$

The MLP introduces nonlinear transformation, while the residual connection preserves original spatial information and provides a gradient shortcut. The aligned features are fed into tracking/mapping Transformers respectively, producing agent queries and map queries, which are concatenated with learnable embeddings to form the ego query.

**2. Mixture of Sparse Experts (MoSE)**

Prediction tasks are partitioned into three groups with eight sparse experts:

| Expert Category | Expert Name | Sparse Attention Type | Function |
|----------------|-------------|----------------------|----------|
| Environmental | Tracking Expert, Mapping Expert | Block-wise (block size $m$) | Dynamic foreground / map topology |
| Ego State | Velocity, Yaw, Acceleration Expert | Sliding Window (window $w$) | Smooth vehicle dynamics modeling |
| Navigation | Reference Point, BEV, Command Expert | Global TopK | Long-range dependencies and navigation commands |

Each expert fuses the ego query with expert-specific embeddings via its sparse attention mechanism:

$$\bar{\mathcal{F}}_{expert} = \text{MHCA}(\mathcal{F}_{ego}, \mathcal{F}_{expert}, \mathcal{F}_{expert})$$

**Router**: Following the MoE gating mechanism, learnable parameters $\mathbf{W}_{gate}$ map the ego query to expert logits. Gaussian noise is added during training to promote stochasticity. The Top-K experts are selected and their outputs are aggregated via weighted summation to produce the final motion query:

$$\mathcal{F}_{Motion} = \sum_{i=1}^k \mathcal{R}(\mathcal{F}_{ego})_i \cdot \bar{\mathcal{F}}_{expert_i}$$

### Loss & Training

The total loss comprises four terms:

$$\mathcal{L}_{total} = \alpha_1\mathcal{L}_{perception} + \alpha_2\mathcal{L}_{prediction} + \alpha_3\mathcal{L}_{planning} + \alpha_4\mathcal{L}_{switch}$$

The Switch Loss encourages load balancing across experts:

$$\mathcal{L}_{switch} = N \cdot \sum_{i=1}^N f_i \cdot \mathcal{P}_i$$

This penalizes experts whose actual load $f_i$ is inconsistent with the expected routing probability $\mathcal{P}_i$. Training follows the same hyperparameters as the respective baselines, using 8× A100 GPUs.

## Key Experimental Results

### Main Results

**Table 1: Overall Performance (Open-loop + Closed-loop + Efficiency)**

| Method | Avg.Col↓ | Avg.L2↓ | DS↑ | SR↑ | RC↑ | Latency↓ |
|--------|---------|---------|-----|-----|-----|---------|
| UniAD | 0.31 | 1.03 | 44.62 | 14.09 | 68.68 | 534ms |
| **Expert-UniAD** | **0.24** | **0.89** | **55.49** | **20.63** | **81.04** | **445ms** |
| VAD | 0.43 | 1.21 | 43.31 | 17.27 | 61.60 | 225ms |
| **Expert-VAD** | **0.34** | **1.10** | **52.53** | **19.53** | **76.73** | **157ms** |
| VADv2 | 0.12 | 0.33 | 75.90 | 55.01 | 90.08 | 330ms |
| **Expert-VADv2** | **0.10** | **0.28** | **78.18** | **58.34** | 89.32 | **258ms** |

**Table 2: Multi-skill Capability in Rare Scenarios (Bench2Drive220)**

| Method | Merge↑ | Overtake↑ | EmgBrake↑ | GiveWay↑ | Tsign↑ |
|--------|--------|-----------|-----------|----------|--------|
| UniAD | 12.66 | 13.33 | 20.00 | 10.00 | 13.23 |
| Expert-UniAD | 27.38 | 23.67 | 51.67 | 20.00 | 40.93 |
| VADv2 | 36.25 | 48.33 | 74.28 | 50.00 | 60.14 |
| Expert-VADv2 | 40.44 | 48.33 | 78.42 | 40.00 | 65.78 |

Averaged across three baselines: collision rate reduced by ~20%, inference latency reduced by ~25%, DS/SR/RC improved by 16%/22%/14% respectively.

### Ablation Study

- **PA hyperparameter τ**: τ=128 yields the best performance (DS=52.53, SR=18.41, RC=76.73); larger τ (e.g., 256) introduces redundancy and degrades results.
- **MoSE Top-K**: Top-4 outperforms full activation with Top-8—selective activation effectively reduces inter-task interference.
- **PA components**: MLP + ADD yields AMOTA 0.404 > ADD-only 0.390 > baseline 0.388.
- **MoSE components**: The router lowers L2 and collision rate; sparse attention substantially reduces latency (−178ms for Expert-UniAD); the two are complementary.

### Key Findings

1. The value of MoE in ADS extends beyond efficiency—dynamic expert selection reduces multi-task interference, **improving planning performance and efficiency simultaneously**.
2. Emergency braking and traffic sign scenarios see the largest gains (due to richer perceptual information), whereas overtaking and give-way scenarios require complex reasoning where MoE contributes less.
3. Cross-city generalization experiments (Boston training → Singapore testing) show that ExpertAD reduces collision rate from 0.66 to 0.46 (Expert-UniAD), demonstrating strong generalization.
4. Statistical significance tests: all improvements achieve an average p-value of 0.026 (p<0.05), confirming result reliability.

## Highlights & Insights

- **End-to-end MoE design spanning both perception and prediction**, distinguishing ExpertAD from prior methods that apply MoE solely within the planning module.
- The channel selection in PA is formulated as a differentiable constrained optimization problem, offering a more principled approach than hard pruning or static selection.
- The three expert categories are equipped with distinct sparse attention mechanisms, reflecting a deep understanding of each task's characteristics: environment → local block-wise, vehicle state → sliding window, navigation → global TopK.
- The plug-and-play design allows direct enhancement of multiple baselines including UniAD, VAD, and VADv2, demonstrating strong generality.

## Limitations & Future Work

- The number of experts (8) and the Top-K value require manual specification; adaptive expert selection warrants further exploration.
- Parameter count increases (UniAD: 89M → 125M); despite reductions in GFLOPs and latency, deployment memory overhead increases.
- Limited improvement in overtaking and give-way scenarios suggests the need for higher-level reasoning; MoE may need to be combined with LLMs or world models.
- Validation is currently limited to vision-only settings; applicability to multimodal (LiDAR+Camera) fusion remains unexplored.

## Related Work & Insights

- **UniAD** and **VAD/VADv2** represent modular end-to-end ADS; ExpertAD seamlessly enhances these frameworks.
- The success of MoE in LLMs (GLaM, Mixtral) motivates its application in ADS in this work.
- The three sparse attention variants (block-wise, sliding window, global TopK) draw from efficient Transformer designs such as Longformer.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 5 |
| Overall | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[ICCV 2025\] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts](../../ICCV2025/autonomous_driving/gm-moe_low-light_enhancement_with_gated-mechanism_mixture-of-experts.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](../../NeurIPS2025/autonomous_driving/sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)

</div>

<!-- RELATED:END -->
