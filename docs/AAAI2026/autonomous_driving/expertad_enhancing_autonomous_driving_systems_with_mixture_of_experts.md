---
title: >-
  [Paper Note] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts
description: >-
  [AAAI 2026][Autonomous Driving][End-to-end autonomous driving] This paper proposes ExpertAD, introducing the Mixture of Experts (MoE) architecture into the perception and prediction modules of end-to-end autonomous driving systems (ADS). Specifically, the Perception Adapter dynamically reweights BEV (Bird's-Eye-View) features to amplify task-critical semantics, and the Mixture of Sparse Experts (MoSE) dynamically activates relevant driving task experts via a router while empl…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "End-to-end autonomous driving"
  - "Mixture of Experts"
  - "perception adaptation"
  - "sparse attention"
  - "inference efficiency"
date: 2026-05-08
content_hash: 7357094ce9eed246
---

# ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts

**Conference**: AAAI 2026  
**arXiv**: [2511.11740](https://arxiv.org/abs/2511.11740)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: End-to-end autonomous driving, Mixture of Experts, perception adaptation, sparse attention, inference efficiency

## TL;DR

This paper proposes ExpertAD, introducing the Mixture of Experts (MoE) architecture into the perception and prediction modules of end-to-end autonomous driving systems (ADS). Specifically, the Perception Adapter dynamically reweights BEV (Bird's-Eye-View) features to amplify task-critical semantics, and the Mixture of Sparse Experts (MoSE) dynamically activates relevant driving task experts via a router while employing sparse attention to reduce computational cost. This approach reduces inference latency by approximately 25% while maintaining or improving planning performance.

## Background & Motivation

End-to-end autonomous driving systems (ADS) have achieved significant progress by unifying the perception-prediction-planning pipeline, but still face two main challenges:

**1. Semantic ambiguity interfering with decision-making**: BEV features contain various types of semantic information (roads, vehicles, traffic signs, etc.), but different perception tasks (e.g., tracking vs. mapping) focus on different aspects. Directly propagating all features might allow non-critical dimensions to overshadow critical information.

**2. Multi-task interference and inference latency**: The prediction module involves various tasks such as ego-state estimation, environmental interaction modeling, and navigation execution. Activating all tasks simultaneously causes inter-task interference and increases computational overhead. For instance, mapping tasks are helpful for curve planning but have little effect on driving straight — different scenarios require different task combinations.

Existing applications of MoE in autonomous driving are mostly limited to a single module (such as trajectory selection in planning) and suffer from unstable expert activation in dynamic scenarios. Prior efficiency optimization methods (e.g., DriveAdapter, PlanKD) traded off planning quality for execution speed.

## Method

### Overall Architecture

ExpertAD is a plug-and-play framework that can be integrated into existing Transformer-based end-to-end ADS (such as UniAD, VAD, and VADv2). It replaces the original perception and prediction modules:

1. **BEV Encoder** (retained) → Generates BEV features
2. **Perception Adapter (PA)** (added) → Dynamically selects and amplifies task-critical feature channels
3. **Mixture of Sparse Experts (MoSE)** (added) → Dynamically activates relevant experts via routing and reduces computation using sparse attention
4. **Planning Module** (retained) → Generates final trajectories based on the motion queries output by MoSE

### Key Designs

**1. Perception Adapter (PA)**

Consists of two sub-components:

**Learned Adapter**: Learns channel selection weights for each task. It first performs temporal normalization and spatial pooling on BEV features, and then computes the importance score of each channel using a task-specific learnable parameter $w^{(t)}$:

$$s = \frac{1}{H \times W}\sum_{i,j} \tilde{\text{BEV}}_{:,i,j} \odot w$$

The soft channel selection weights $\lambda^{(t)} \in [0,1]^d$ are solved via constrained optimization to ensure focus on $\tau$ dominant channels:

$$\max_\lambda \; s^\top \lambda + \epsilon\Omega(\lambda), \quad \text{s.t.} \; \mathbf{1}^\top\lambda = \tau, \; \lambda \in [0,1]^d$$

**Alignment Layer**: Recalibrates individual BEV features using the selection weights:

$$F_{align} = \text{MLP}(\text{BEV} \odot \lambda) + \text{BEV}$$

The MLP introduces a non-linear transformation, while the residual connection preserves original spatial information and provides a gradient shortcut. The aligned features are fed into tracking/mapping Transformers to output agent queries and map queries respectively, which are then concatenated with a learnable embedding to form the ego query.

**2. Mixture of Sparse Experts (MoSE)**

Divides prediction tasks into eight sparse experts across three groups:

| Expert Category | Expert Name | Sparse Attention Type | Function |
|---------|---------|-------------|------|
| Environmental | Tracking Expert, Mapping Expert | Block-wise (block size m) | Processes dynamic foreground / map topology |
| Ego State | Velocity, Yaw, Acceleration Expert | Sliding Window (window w) | Models smooth vehicle dynamics |
| Navigation | Reference Point, BEV, Command Expert | Global TopK | Captures long-range dependencies and navigation commands |

Each expert processes the ego query fused with expert-specific embeddings using its own sparse attention mechanism:

$$\bar{\mathcal{F}}_{expert} = \text{MHCA}(\mathcal{F}_{ego}, \mathcal{F}_{expert}, \mathcal{F}_{expert})$$

**Router**: Based on the MoE gating mechanism, it maps the ego query to expert logits using a learnable parameter $\mathbf{W}_{gate}$. During training, Gaussian noise is added to inject randomness. The final motion query is generated via a weighted sum of the active Top-K experts:

$$\mathcal{F}_{Motion} = \sum_{i=1}^k \mathcal{R}(\mathcal{F}_{ego})_i \cdot \bar{\mathcal{F}}_{expert_i}$$

### Loss & Training

The total loss consists of four terms:

$$\mathcal{L}_{total} = \alpha_1\mathcal{L}_{perception} + \alpha_2\mathcal{L}_{prediction} + \alpha_3\mathcal{L}_{planning} + \alpha_4\mathcal{L}_{switch}$$

where the Switch Loss encourages expert load balancing:

$$\mathcal{L}_{switch} = N \cdot \sum_{i=1}^N f_i \cdot \mathcal{P}_i$$

It penalizes discrepancy between the actual load $f_i$ and the expected routing probability $\mathcal{P}_i$ for each expert. Training is conducted on 8× A100 GPUs, maintaining the same hyperparameters as the baselines.

## Key Experimental Results

### Main Results

**Table 1: Overall Performance (Open-loop + Closed-loop + Efficiency)**

| Method | Avg.Col↓ | Avg.L2↓ | DS↑ | SR↑ | RC↑ | Latency↓ |
|------|---------|---------|-----|-----|-----|---------|
| UniAD | 0.31 | 1.03 | 44.62 | 14.09 | 68.68 | 534ms |
| **Expert-UniAD** | **0.24** | **0.89** | **55.49** | **20.63** | **81.04** | **445ms** |
| VAD | 0.43 | 1.21 | 43.31 | 17.27 | 61.60 | 225ms |
| **Expert-VAD** | **0.34** | **1.10** | **52.53** | **19.53** | **76.73** | **157ms** |
| VADv2 | 0.12 | 0.33 | 75.90 | 55.01 | 90.08 | 330ms |
| **Expert-VADv2** | **0.10** | **0.28** | **78.18** | **58.34** | 89.32 | **258ms** |

**Table 2: Multi-ability performance in rare scenarios (Bench2Drive220)**

| Method | Merge↑ | Overtake↑ | EmgBrake↑ | GiveWay↑ | Tsign↑ |
|------|--------|-----------|-----------|----------|--------|
| UniAD | 12.66 | 13.33 | 20.00 | 10.00 | 13.23 |
| Expert-UniAD | 27.38 | 23.67 | 51.67 | 20.00 | 40.93 |
| VADv2 | 36.25 | 48.33 | 74.28 | 50.00 | 60.14 |
| Expert-VADv2 | 40.44 | 48.33 | 78.42 | 40.00 | 65.78 |

On average across the three baselines, the collision rate is reduced by approximately 20%, the inference latency is reduced by roughly 25%, and DS/SR/RC are improved by 16%/22%/14% respectively.

### Ablation Study

- **PA Hyperparameter $\tau$**: Selected optimally at $\tau=128$ (DS=52.53, SR=18.41, RC=76.73); excessively large values ($\tau=256$) introduce redundancy and degrade performance.
- **MoSE Top-K**: Top-4 outperforms Top-8 fully activated, showing that selective activation effectively mitigates task interference.
- **PA Components**: The MLP + ADD combination yields a higher AMOTA of 0.404, compared to 0.390 for ADD-only and 0.388 for the baseline.
- **MoSE Components**: The Router reduces L2 error and collision rates, while Sparse Attention significantly cuts down latency (by 178ms for Expert-UniAD). The two components are highly complementary.

### Key Findings

1. The value of MoE in ADS is not limited to efficiency — dynamic expert selection reduces multi-task interference, **improving both planning effectiveness and efficiency simultaneously**.
2. The most significant improvements are observed in emergency braking and traffic sign scenarios (since they contain rich perceptual information). In contrast, overtaking and giving way scenarios require high-level reasoning, where MoE yields limited improvement.
3. Cross-city generalization experiments (training in Boston $\to$ testing in Singapore) show that ExpertAD reduces the collision rate from 0.66 to 0.46 (Expert-UniAD), demonstrating strong generalizability.
4. Statistical significance test: All improvements yield an average p-value of 0.026 ($p < 0.05$), validating the reliability of the results.

## Highlights & Insights

- **An end-to-end MoE design spanning both perception and prediction**, standing in contrast to prior approaches that restrict MoE only to the planning module.
- The channel selection of PA is formulated as a differentiable and constrained optimization problem, which is more elegant than hard pruning or static selection.
- Equipping three categories of experts with distinct sparse attention mechanisms reflects a deep understanding of each task's properties: Environmental $\to$ local blocks, Ego State $\to$ sliding windows, Navigation $\to$ global Top-K.
- The plug-and-play design allows seamless enhancement over various baselines such as UniAD, VAD, and VADv2, showing high generalizability.

## Limitations & Future Work

- The number of experts (8) and the Top-K value must be manually set; future work could explore adaptive expert selection.
- There is an increase in model parameters (e.g., UniAD from 89M to 125M); despite lower GFLOPs and latency, deployment incurs a larger footprint in memory.
- Limited improvements in overtaking and giving-way scenarios suggest a need for higher-level reasoning capabilities; MoE could potentially be integrated with LLMs/World Models.
- Currently, validation is limited to vision-only inputs, leaving the applicability to multi-modal (LiDAR+Camera) fusion unexplored.

## Related Work & Insights

- **UniAD** and **VAD/VADv2** represent state-of-the-art modular end-to-end ADS. ExpertAD seamlessly enhances their performance.
- The success of MoE in LLMs (such as GLaM and Mixtral) inspired its application in ADS.
- The three variants of sparse attention (block-wise, sliding window, global Top-K) draw inspiration from efficient Transformer designs like Longformer.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 5 |
| Overall Rating | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/drivemoe_mixture-of-experts_for_vision-language-action_model_in_end-to-end_auton.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[ICCV 2025\] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts](../../ICCV2025/autonomous_driving/gm-moe_low-light_enhancement_with_gated-mechanism_mixture-of-experts.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)

</div>

<!-- RELATED:END -->
