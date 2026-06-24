---
title: >-
  [Paper Note] Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks
description: >-
  [ICML 2025 (Workshop on ML4Wireless)][Autonomous Driving][Multi-UAV Control] Proposes a hierarchical collaborative LLM-based control framework that coordinates dual-level LLMs—a meta-controller LLM deployed on the HAPS and edge-controller LLMs deployed on the UAVs—to achieve joint optimization of motion planning and communication access for multi-UAVs in 3D aerial highway scenarios.
tags:
  - "ICML 2025 (Workshop on ML4Wireless)"
  - "Autonomous Driving"
  - "Multi-UAV Control"
  - "Large Language Models"
  - "Hierarchical Collaboration"
  - "Integrated Terrestrial and Non-Terrestrial Networks"
  - "Joint Motion and Communication Optimization"
date: 2026-05-08
content_hash: 6cb4ac618f236396
---

# Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks

**Conference**: ICML 2025 (Workshop on ML4Wireless)  
**arXiv**: [2506.06532](https://arxiv.org/abs/2506.06532)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: Multi-UAV Control, Large Language Models, Hierarchical Collaboration, Integrated Terrestrial and Non-Terrestrial Networks, Joint Motion and Communication Optimization

## TL;DR

Proposes a hierarchical collaborative LLM-based control framework that coordinates dual-level LLMs—a meta-controller LLM deployed on the HAPS and edge-controller LLMs deployed on the UAVs—to achieve joint optimization of motion planning and communication access for multi-UAVs in 3D aerial highway scenarios.

## Background & Motivation

Multi-UAV systems are widely applied in fields such as logistics and communication coverage, but their collective control faces critical challenges:

**Complex motion dynamics**: Most existing studies only consider motion direction, neglecting fine-grained motion dynamics such as acceleration, deceleration, and lane changing, as well as collision avoidance among multiple UAVs.

**Coupling of communication and traffic**: Increasing UAV speed improves traffic flow but leads to frequent handovers, degrading communication quality; their objectives are naturally contradictory.

**Heterogeneous network complexity**: UAVs must make access decisions within integrated terrestrial and non-terrestrial networks consisting of terrestrial base stations (BSs) and high-altitude platform stations (HAPSs), involving both horizontal handovers (between BSs) and vertical handovers (between BS and HAPS).

**Limitations of prior work**: Traditional methods like DDQN and MORL require extensive task-specific training and struggle to simultaneously handle multi-objective optimization for motion and communication.

The authors assert that the pre-trained knowledge and in-context learning capabilities of LLMs are naturally suited for such multi-objective decision-making problems, eliminating the need to retrain for each scenario.

## Method

### Overall Architecture

This paper designs a **dual-level LLM collaborative architecture** (LLM-LLM Dual Agents):

- **HAPS Level (Meta-Controller)**: An LLM deployed on the High-Altitude Platform Station (HAPS), responsible for global network orchestration and managing the connection relationship between UAVs and the HAPS (association/offloading decisions).
- **UAV Level (Edge Controller)**: An LLM deployed on each UAV, responsible for real-time motion control (acceleration/deceleration/lane-changing) and base station selection strategies.

The two levels of controllers collaborate through state sharing: offloading commands from the HAPS meta-controller constrain the vertical handover options of the UAVs, while local state changes of the UAVs feed back to influence the global decisions of the HAPS.

### Key Designs

#### 1. HAPS Meta-Controller MDP

Modeled as an MDP $(S_{meta}, A_{meta}, P_{meta}, r_{meta})$:

- **State**: Current HAPS load, data rate of each UAV, and terrestrial BS coverage.
- **Action**: $\text{Offload}\{ID\}$ (offloading a UAV to a terrestrial BS), $\text{Recall}\{ID\}$ (reconnecting a UAV back to the HAPS), and $\text{Idle}$ (maintaining the current state).
- **Decision Logic**: When the total HAPS bandwidth usage exceeds the capacity ($B_t > C$), the UAV with the poorest link quality is selected for offloading; when there is idle capacity, offloaded UAVs are recalled.

Meta-controller reward function:

$$r_{meta} = \eta_1 \sum_{j} WR_t^{ij} - \eta_2 \cdot \text{Sat}_{HAPS} - \eta_3 \cdot \mu$$

Where the first term represents the total throughput, the second term penalizes HAPS saturation events, and the third term is the handover penalty.

#### 2. UAV Edge Controller MDP

Each UAV independently runs an MDP, with its action space constrained by the HAPS meta-controller:

- **State Space** (7 dimensions/UAV): 3D position $(x, y, z)$, forward velocity $v$, heading angle $\psi$, number of available terrestrial BSs $n_R$, and number of available HAPS channels $n_H$.
- **Motion Actions** (5 types): Left lane change, keep lane, right lane change, accelerate, decelerate.
- **Communication Actions** (3 types): $a_{tele1}$ (BS with maximum weighted data rate), $a_{tele2}$ (BS considering load balancing), $a_{tele3}$ (BS with maximum instantaneous rate).

#### 3. LLM-Based Prompt Engineering

The core innovation lies in encoding the entire optimization problem into structured natural language prompts including six modules:

1. **Task Description**: Declares the mission objectives.
2. **Task Objectives**: Specific optimization metrics (maximizing speed, collision avoidance, minimizing handovers).
3. **State Definition**: Enumeration of environmental variables.
4. **Observations**: Discretized state matrices.
5. **Experience Memory**: Top-5 most similar good/bad experiences (selected by Euclidean distance in the state space).
6. **Reply Rules**: Constrains the output format to one motion action + one communication strategy.

Deployed via the Ollama framework on edge servers to ensure real-time inference latency.

#### 4. Channel Model

- **G2A Channel**: Adopts the 3GPP antenna pattern specification, considering azimuth/elevation gain, LoS probability, and path loss.
- **UAV-HAPS Channel**: LoS link + free-space path loss + Rician fading.
- **Weighted Data Rate**: $WR_t^{ij} = \frac{R_t^{ij}}{\min(Q_i, n_i)} (1 - \mu)$, where $\mu$ is the handover penalty coefficient.

### Loss & Training

The reward of the UAV edge controller consists of two parts: transport reward and communication reward.

**Transport Reward**:
$$r_t^{j, tran} = w_1 \frac{v_t^j - v_{min}}{v_{max} - v_{min}} - w_2 \delta_c - w_3 \chi_t^j$$

- $w_1$: Normalized velocity reward.
- $w_2 \delta_c$: Collision penalty ($\delta_c \in \{0,1\}$).
- $w_3 \chi_t^j$: Lane-change frequency penalty.

**Communication Reward**:
$$r_t^{j, tele} = w_4 \cdot WR_{i^*, j, t} \cdot (1 - \min(1, \xi_t^j))$$

- $\xi_t^j$: Cumulative handover probability.
- Crucial weights are designed to prioritize safety ($w_2$) and connectivity ($w_4$).

The inference process does not involve gradient training, relying entirely on the LLM's in-context learning capability.

## Key Experimental Results

### Experimental Settings

| Parameter | Value |
|------|------|
| UAV Count | 5 (Training), 5–40 (Evaluation) |
| Lane Count | 5 |
| Flying Speed Range | 5–20 m/s |
| Carrier Frequency | 2.1 GHz (BS), 2 GHz (HAPS) |
| BS Transmit Power | 40 dBm |
| Max Users per BS | 3 |
| Max Timesteps per Episode | 30 |
| Hardware | 2× Intel E5-2650 v4 + 2× NVIDIA P100 |

### Main Results

| Method | Total Reward | Communication Reward Advantage | Collision Rate | Description |
|------|--------|-------------|--------|------|
| DDQN | ~23 | Baseline | High | Traditional DRL |
| Envelope-MORL | <20 | Moderate | Moderate | Multi-objective RL |
| Llama 3.1 8B + DDQN | Moderate | Moderate | Moderate | Hybrid LLM+RL |
| Llama 3.1 70B + DDQN | High | High | Low | Hybrid with larger model |
| **LLM-LLM Dual Agent** | **~30** | **+2~3 units** | **<0.08** | **Ours** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| UAV=5 → UAV=40 | Total reward first decreases then increases (U-shaped) | Platoon effect in low density, congestion dominates in high density |
| Collision rate vs density | <0.12 (low density), sharp rise when M>30 | Dual Agent consistently stays below 0.08 |
| Number of BSs 5/10/15/20 | Network scalability verification | More BSs improve communication performance |
| Convergence speed | Dual Agent ~1.5k episodes | Approximately one order of magnitude faster than DDQN |

### Key Findings

1. **Convergence Acceleration**: The LLM-LLM framework converges within approximately 1,500 episodes, whereas DDQN requires more time and yields a lower final reward.
2. **Pareto Improvement**: Across all evaluated UAV densities (5–40), the proposed method improves the total reward by **16.3%** on average.
3. **Collision Mitigation**: In high-density scenarios (>30 UAVs), the collision rate is **21%** lower than that of DDQN.
4. **Communication Benefits**: The LLM-driven policy consistently maintains a 2–3 unit advantage in V2I throughput, while DDQN suffers a sharp performance decline at >25 UAVs due to uncoordinated handovers.

## Highlights & Insights

1. **New Paradigm of LLM as Controller**: Directly encoding the optimization problem into prompts utilizes the LLM's in-context learning capability to replace traditional RL training, avoiding massive task-specific training overhead.
2. **Ingenious Experience Memory Design**: Selecting only the top-5 good/bad experiences closest in Euclidean distance within the state space as context significantly reduces inference costs while maintaining decision quality.
3. **Hierarchical Decoupling**: Separating global network management (slow timescale) from local motion control (fast timescale) into two independent LLMs reduces the decision complexity of individual models.
4. **Joint Communication-Traffic Modeling**: The weighted data rate formula elegantly combines load balancing and handover penalties into a single metric.

## Limitations & Future Work

1. **Questionable Scalability**: Experiments only validate scenarios with 5–40 UAVs. In large-scale scenarios (hundreds of UAVs), LLM inference latency and prompt length might become bottlenecks.
2. **LLM Inference Overhead**: Although deployed locally using Ollama, calling LLM inference for each UAV at every step requires further evaluation of computational resources and latency in practical deployments.
3. **Lack of Comparison with Advanced RL**: The method is not compared against mainstream policy gradient methods like PPO or SAC.
4. **Simplified Simulation Environment**: Using the Intelligent Driver Model (IDM) to simulate motion neglects real-world environmental factors such as wind fields and GPS accuracy.
5. **Workshop Constraints**: As an ICML Workshop paper, the experimental depth and ablation analysis are limited, with a lack of detailed computational cost comparisons.
6. **Future Directions**: The authors mention plans to introduce multimodal perceptual inputs, localization in GPS-denied environments, and online adaptive strategies.

## Related Work & Insights

- **LLM for 6G**: Complementary to the LLM power control and prompt engineering work of Zhou et al. (2024, 2025), this paper extends the concept to multi-LLM collaborative scenarios.
- **RL for UAV**: The RL trajectory planning by Cherif et al. (2022, 2023) serves as a direct benchmark; this paper replaces the RL training process with LLMs.
- **Vehicle-to-Infrastructure**: The hybrid LLM+DDQN method by Yan et al. (2025) is the precursor to this study, which upgrades the hybrid framework to a pure dual-agent LLM scheme.
- **Insights**: This "replacing RL with LLMs" approach can be transferred to other multi-agent coordination scenarios (e.g., autonomous driving platoons, robot swarms), though actual constraints on inference latency must be carefully considered.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The dual-level LLM collaborative control architecture is creative |
| Technical Depth | 3 | Detailed channel modeling, but lacks in-depth analysis on the LLM part |
| Experimental Thoroughness | 3 | Baselines are reasonable, but lacks ablation and computational overhead comparisons |
| Writing Quality | 3.5 | Clear structure, but the notation in some formulas is not entirely consistent |
| Value | 3 | Proof-of-concept is meaningful, but practical deployment requires resolving latency issues |
| **Overall** | **3.3** | An interesting direction of exploration, but limited in depth as a workshop paper |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[ICLR 2026\] Rate-Distortion Optimized Pragmatic Communication for Collaborative Perception](../../ICLR2026/autonomous_driving/rate-distortion_optimized_pragmatic_communication_for_collaborative_perception.md)
- [\[NeurIPS 2025\] Continuous Simplicial Neural Networks](../../NeurIPS2025/autonomous_driving/continuous_simplicial_neural_networks.md)
- [\[ICCV 2025\] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion](../../ICCV2025/autonomous_driving/mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)
- [\[ICCV 2025\] CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving](../../ICCV2025/autonomous_driving/colmdriver_llm-based_negotiation_benefits_cooperative_autonomous_driving.md)

</div>

<!-- RELATED:END -->
