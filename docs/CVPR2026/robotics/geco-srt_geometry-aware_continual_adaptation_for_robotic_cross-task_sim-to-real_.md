---
title: >-
  [Paper Note] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][Robotics] This paper proposes GeCo-SRT, a geometry-aware continual adaptation method that extracts cross-domain and cross-task invariant knowledge from local geometric features, enabling knowledge accumulation across successive sim-to-real transfers to efficiently adapt to new tasks.
tags:
  - CVPR 2026
  - Robotics
date: 2026-05-08
content_hash: cfadba58daf65009
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**Conference**: CVPR 2026
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)
**Code**: N/A
**Area**: Robotics

## TL;DR

This paper proposes GeCo-SRT, a geometry-aware continual adaptation method that extracts cross-domain and cross-task invariant knowledge from local geometric features, enabling knowledge accumulation across successive sim-to-real transfers to efficiently adapt to new tasks.

## Background & Motivation

1. **The sim-to-real gap is a core bottleneck in robot learning**: Policies trained in simulation suffer significant performance degradation when deployed in the real world due to discrepancies in visual rendering and physical dynamics (Direct Deploy achieves only 3.1% average success rate).
2. **Existing methods treat each sim-to-real transfer as an isolated process**: System Identification relies on hand-crafted modeling, and Domain Randomization requires expert-level parameter tuning; both support only single-task transfer, requiring the process to restart from scratch for every new task at high cost.
3. **Lack of cross-task knowledge accumulation**: Different manipulation tasks share many geometric primitives (e.g., edges, planes), yet existing methods do not exploit these transferable geometric priors to accelerate adaptation to subsequent tasks.
4. **Catastrophic forgetting in continual learning**: When learning multiple tasks sequentially, training on new tasks overwrites knowledge from previous ones; standard PER methods prioritize sampling based on task loss, neglecting expert-level knowledge preservation.

## Method

### 3.1 Human-Correction-Based Sim-to-Real Transfer Framework

For each task $i$, a base diffusion policy $\pi^{b_i}$ (comprising a point cloud encoder $E_p^{b_i}$ and a diffusion policy head $\pi_h^{b_i}$) is first trained in simulation via imitation learning using an L2 diffusion loss:

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{(a_t, o_t) \sim \mathcal{D}_{sim}^{e_i}, \epsilon \sim \mathcal{N}(0,I)} \left[ \| \epsilon - \epsilon_\theta(a_t^k, k, o_t) \|^2 \right]$$

The base policy is deployed in the real world, and human correction trajectories $\mathcal{D}_{real}^{h_i}$ are collected via a human-in-the-loop shared autonomy framework. The base policy parameters are frozen, and a shared perceptual residual module $E_p^s$ (i.e., Geo-MoE) is introduced and continuously updated using a mixed replay buffer $D_{buf}^{m_i}$ of simulation and real-world data.

### 3.2 Geometry-aware Mixture of Experts Module (Geo-MoE)

The core insight is that **local geometric features possess dual invariance**:

- **Domain invariance**: Local geometry (e.g., planarity, linearity) is structurally consistent between simulation and the real world, unaffected by texture or material differences.
- **Task invariance**: Geometric primitives (e.g., edges, corners) are shared across different manipulation tasks (e.g., "pick cube" and "stack cube" both involve the planar surfaces of a cube).

Implementation details:

1. Local point groups $g_i$ are sampled from the input point cloud $P$ via k-NN.
2. Geometric features (planarity, linearity, saliency) of each group are estimated via local PCA.
3. A lightweight gating network $G$ produces routing weights $w_i = \text{Softmax}(G(g_i))$.
4. $M$ parallel expert networks are fused with weighted aggregation:

$$g_i' = \sum_{j=1}^{M} w_{i,j} \cdot \text{Expert}_j(g_i)$$

5. All group features are aggregated to yield a correction residual vector $g_{res}'$, which is concatenated with the frozen encoder output: $\hat{f} = \text{Concat}(E_p^{b_i}(o), g_{res}')$.

Training loss:

$$L_{total} = \text{MSE}(\hat{a}, a) + \alpha L_{balance}$$

where $L_{balance}$ is a load-balancing loss to prevent gating collapse.

### 3.3 Geometry Expert-Guided Prioritized Experience Replay (Geo-PER)

Standard PER prioritizes sampling based on task loss, neglecting knowledge degradation in underutilized experts. Geo-PER redefines the priority metric from task loss to **expert utilization**:

For each historical sample $i \in \mathcal{R}$, its expert activation vector $W_i = \{w_{i,1}, \dots, w_{i,M}\}$ is recorded, and the average expert utilization $U^{\text{new}}$ on the current new task is computed to dynamically update sampling priorities:

$$P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$$

The intuition is: if expert $j$ has low utilization on the current task (small $u_j^{\text{new}}$), historical samples that strongly activate that expert receive higher sampling probability, ensuring that underutilized expert parameters continue to receive gradient updates to counter forgetting.

## Key Experimental Results

### Experimental Setup

Four sequential robotic manipulation tasks are used for evaluation: Pick Cube → Stack Cube → Pick Banana → Plug Insert. The real hardware consists of an Xarm5 robotic arm, a Rotiq2F140 gripper, and dual RealSense depth cameras. Sixty human correction trajectories are collected per task.

### Single-Task Sim-to-Real Transfer

| Method | Pick Cube | Stack Cube | Pick Banana | Plug Insert | Avg. SR |
|--------|-----------|------------|-------------|-------------|---------|
| Direct Deploy | 5.7% | 0% | 6.7% | 0% | 3.1% |
| Action Residual | 16.7% | 3.3% | 13.3% | 0.0% | 9.2% |
| Transic | 66.7% | 30.0% | 23.3% | 33.3% | 38.3% |
| **Geo-MoE** | **80.0%** | **43.3%** | **40.0%** | **36.7%** | **50.0%** |

Geo-MoE outperforms the strongest baseline Transic by 11.7% on average, validating the advantage of geometry-aware features in bridging the observation domain gap.

### Cross-Task Continual Learning

| Method | Pick Cube SR↑ | N-NBT↓ | Stack Cube SR↑ | N-NBT↓ | Pick Banana SR↑ | N-NBT↓ | Plug Insert SR↑ | Avg. SR↑ | Avg. N-NBT↓ |
|--------|-------------|--------|--------------|--------|---------------|--------|----------------|---------|------------|
| Naive Fine-tuning | 16.7% | 100% | 3.3% | 100% | 13.3% | 100% | 3.3% | 9.2% | 75.0% |
| Transic + PER | 76.7% | 81.2% | 30.0% | 72.3% | 20.0% | 66.5% | 33.3% | 40.0% | 55.0% |
| Geo-MoE + PER | 83.3% | 34.6% | 50.0% | 76.8% | 46.7% | 7.1% | 43.3% | 55.7% | 29.6% |
| Geo-MoE + EWC | 80.0% | 70.8% | 36.7% | 77.3% | 20.0% | 50.0% | 16.7% | 38.3% | 49.5% |
| **GeCo-SRT** | **86.7%** | **28.3%** | **53.3%** | **72.0%** | **60.0%** | **5.5%** | **53.3%** | **63.3%** | **26.5%** |

GeCo-SRT achieves an average success rate of 63.3%, surpassing baselines by 52%; its average forgetting rate N-NBT is only 26.5%, substantially outperforming all other methods.

### Data Efficiency

With only 20 correction trajectories, the continual learning method achieves 76.6% success rate on Pick Cube, nearly matching the performance of training from scratch with 60 trajectories. Overall, only **1/6 of the data** is required to match the full-data performance of baselines.

## Highlights & Insights

- **Paradigm innovation**: This is the first work to elevate sim-to-real transfer from an "isolated single-instance" to a "continual cross-task" paradigm, enabling knowledge accumulation.
- **Dual geometric invariance**: The paper cleverly leverages the domain invariance and task invariance of local geometric features as a carrier for transferable knowledge.
- **Expert-level forgetting protection**: Geo-PER redefines priority from the perspective of expert utilization, precisely protecting the specialized knowledge of underutilized experts.
- **Rigorous real-world validation**: A complete continual transfer experiment across 4 tasks is conducted on a real robot (Xarm5), rather than purely in simulation.
- **Significant data efficiency**: Only 1/6 of the data is needed to reach the full-data performance of baselines, which is highly meaningful for data-scarce real-world scenarios.

## Limitations & Future Work

- **Focus limited to the observation domain gap**: The method primarily bridges the visual observation sim-to-real gap via geometric features, with limited capacity to handle complex dynamics discrepancies (e.g., friction, contact forces).
- **Limited number of tasks**: Experiments involve only 4 sequential tasks; scalability to longer task sequences remains unvalidated.
- **Dependence on human correction**: Each task still requires a human operator to collect 60 correction trajectories, leaving the degree of full autonomy insufficient.
- **Reliance on task geometric similarity**: Cross-task transfer effectiveness is highly dependent on geometric similarity; negative transfer may occur between tasks with large geometric differences.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The first to propose a continual cross-task sim-to-real paradigm; the insight of using geometric invariance as a carrier of transferable knowledge is novel.
- ⭐⭐⭐⭐ **Value**: Significantly reduces the data cost of adapting to new tasks, offering direct value for real-robot deployment.
- ⭐⭐⭐ **Experimental Thoroughness**: Real-robot experiments are complete but limited in scale (4 tasks); more diverse scenarios are absent.
- ⭐⭐⭐ **Writing Quality**: Structure is clear and motivation is well-articulated, though some mathematical notation is inconsistent.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rcnf_robot_conditioned_normalizing_flow_anomaly.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)

<!-- RELATED:END -->
