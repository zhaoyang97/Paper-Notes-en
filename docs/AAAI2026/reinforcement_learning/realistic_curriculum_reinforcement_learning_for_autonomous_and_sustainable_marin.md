---
title: >-
  [Paper Note] Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation
description: >-
  [AAAI 2026][Reinforcement Learning][Curriculum Reinforcement Learning] This paper proposes a Curriculum Reinforcement Learning (CRL) framework for autonomous and sustainable marine vessel navigation. The framework integr…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Curriculum Reinforcement Learning"
  - "Autonomous Navigation"
  - "Marine Sustainability"
  - "Diffusion Models"
  - "Fuel Consumption Prediction"
date: 2026-05-08
content_hash: fd3a13aef2e94fd1
---

# Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation

**Conference**: AAAI 2026
**arXiv**: [2601.10911](https://arxiv.org/abs/2601.10911)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Curriculum Reinforcement Learning, Autonomous Navigation, Marine Sustainability, Diffusion Models, Fuel Consumption Prediction

## TL;DR

This paper proposes a Curriculum Reinforcement Learning (CRL) framework for autonomous and sustainable marine vessel navigation. The framework integrates a high-fidelity simulation environment built on real AIS data, a diffusion model-enhanced dynamic maritime traffic simulator, and a machine learning-based fuel consumption prediction module. A multi-objective reward function simultaneously optimizes navigation safety, emission reduction, timeliness, and goal completion.

## Background & Motivation

Maritime shipping carries approximately 90% of global trade, yet faces mounting pressure for carbon reduction. The International Maritime Organization (IMO) has set a strategic target of at least 50% emission reduction by 2050. Under a business-as-usual scenario, shipping emissions are projected to increase by 10–30% above 2008 levels by 2050.

**Safety Challenges**: Maritime safety is a fundamental prerequisite for sustainable shipping. The SANCHI tanker collision resulted in 32 crew fatalities and the release or combustion of over 100,000 tonnes of oil products—investigations attributed the incident primarily to human error.

**Limitations of Existing DRL Approaches**:
1. Most prior work focuses on a single objective (collision avoidance or emission reduction alone), lacking simultaneous multi-objective optimization.
2. Scalability and generalization in highly dynamic real-world maritime environments remain limited.
3. Simulation environments diverge significantly from real-world conditions, with a lack of high-fidelity simulators grounded in real data.
4. The absence of accurate fuel consumption feedback hinders emission-aware navigation decisions.

**Goal**: This paper proposes integrating Curriculum Learning (CL) into DRL within a real-data-driven simulation environment to achieve multi-objective autonomous navigation balancing safety, emission reduction, timeliness, and goal completion.

## Method

### Overall Architecture

The system comprises three core modules:

1. **Fuel Consumption Prediction Module**: An XGBoost-based machine learning model trained on real operational data to predict fuel consumption rate.
2. **Maritime Traffic Environment Module**: A high-fidelity simulation environment built on real AIS data and augmented with a diffusion model.
3. **CRL Policy Learning Module**: Curriculum reinforcement learning based on the PPO algorithm, incorporating an Actor-Critic network and a multi-objective reward function.

### Key Designs

#### 1. Fuel Consumption Prediction Module

An XGBoost model is trained on two years of real operational data from hundreds of international ocean-going vessels.

**Input Features** (86-dimensional in total):
- Navigation parameters: travel distance, latitude/longitude, speed over ground (SOG)
- Vessel characteristics: length overall (LOA), beam, gross tonnage (GT), vessel type
- Temporal variables: month, day, hour (as proxies for ocean conditions)
- Categorical features encoded via one-hot encoding

**Output**: Fuel consumption rate (FCR) in metric tonnes per hour, aggregating contributions from main engines, auxiliary engines, auxiliary machinery, and four fuel types (HFO, LSFO, DO, LSGO).

The model is formulated as:
$$\hat{y} = f_{\text{xgboost}}(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x}), \quad f_k \in \mathcal{F}$$

#### 2. Diffusion Model-Enhanced Maritime Traffic Environment

A diffusion model is employed to generate synthetic AIS trajectories, enriching the realism and diversity of the simulation environment. Vessel trajectories are represented as sequences of positional states:

$$\mathbf{x}_0 = ((\phi_{t,1}, \lambda_{t,1}, v_{t,1}), \cdots, (\phi_{t,T}, \lambda_{t,T}, v_{t,T}))$$

The forward diffusion process incrementally adds Gaussian noise:
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

The training objective is a denoising loss:
$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

New trajectories are sampled via the reverse process, generating diverse vessel trajectories that conform to realistic motion patterns.

#### 3. Curriculum Reinforcement Learning Framework

**State representation** consists of two components:
- **Ego-vessel state** $\mathbf{s}_1^t \in \mathbb{R}^9$: current and destination latitude/longitude, heading angle, ocean current direction, and current speed (angular features encoded via sine/cosine)
- **Environmental state** $\mathbf{s}_2^t \in \mathbb{R}^{64 \times 64 \times 3}$: an ego-centric three-channel image tensor
    - Channel 0: occupancy indicator (vessel presence)
    - Channel 1: speed over ground (SOG)
    - Channel 2: course over ground (COG)

**Action space** is a continuous two-dimensional space:
$$\mathbf{a}_t = [\Delta\psi_t, v_t], \quad -\hat{\psi} \leq \Delta\psi_t \leq \hat{\psi}, \quad v_l \leq v_t \leq v_u$$

**Curriculum Learning Mechanism**: A distance threshold parameter that decreases monotonically with training episodes is introduced:
$$\omega(e) = \omega_0 \cdot (1 - \frac{e}{N_e}) + \omega_f \cdot \min(\frac{e}{N_e}, 1)$$

The initial threshold $\omega_0 = 5$ nautical miles is gradually reduced, transitioning the agent from simpler to more challenging tasks.

**Multi-Objective Reward Function**:

$$r_t = \begin{cases} 30 + 1.5g_t - f_t - s_t, & \text{if } d_{\text{cur}} < \omega(e) \\ 1.5g_t - f_t - s_t - 1.0, & \text{if } d_{\text{cur}} > d_{\text{pre}} \\ 1.5g_t - f_t - s_t - 0.1 \cdot d_{\text{cur}}, & \text{if late and } d_{\text{cur}} \geq \omega(e) \\ 1.5g_t - f_t - s_t, & \text{otherwise} \end{cases}$$

where $g_t = d_{\text{pre}} - d_{\text{cur}}$ (goal-approach reward), $f_t = \alpha \cdot \frac{f_{\text{XGBoost}}(\mathbf{x}_t)}{GT}$ (fuel penalty), and $s_t$ denotes the safety penalty based on DCPA/TCPA.

**Safety Distance Definition**:
$$d_{\text{safe}} = \max\left(\tau \cdot \frac{L_s + B_s + L_t + B_t}{2 \times 1852}, 0.5\right)$$

with buffer multiplier $\tau = 4$ and maximum time to closest point of approach $t_{\max} = 15$ minutes.

### Loss & Training

- **Actor Network**: Dual-branch architecture—environmental images are processed through lightweight convolutions (2 depthwise separable convolutional layers) for spatial feature extraction, while the ego-vessel state is processed through fully connected layers; the two branches are merged to produce the action distribution.
- **Critic Network**: Shares the input processing structure with the Actor and outputs a scalar state value.
- **Training Algorithm**: PPO (Proximal Policy Optimization)
- **Monthly Ocean Current Modeling**: Dynamic perturbations in current direction and speed are simulated based on historical distributions.

## Key Experimental Results

### Main Results

**Fuel Consumption Prediction Model Comparison**:

| Method | MAE | RMSE | R² (%) |
|--------|-----|------|--------|
| SVR | 0.4529 | 0.7050 | 45.01 |
| MLP | 0.2440 | 0.4916 | 77.06 |
| ET | 0.2116 | 0.4033 | 84.56 |
| LightGBM | 0.2015 | 0.3895 | 85.60 |
| RF | 0.1752 | 0.3832 | 86.06 |
| **XGBoost** | **0.1802** | **0.3827** | **86.10** |

**Navigation Control Model Comparison** (Instance Case 1, 21 hours, Indian Ocean):

| Method | Cumulative Reward (AR)↑ | Cumulative Fuel (AFC)↓ | Cumulative Safety Score (ASS)↓ |
|--------|------------------------|----------------------|-------------------------------|
| CL-ABDDQN | 155.527 | 20.509 | 1.466 |
| CL-A2C | -82.034 | 13.092 | 7.501 |
| DDPG | 150.424 | 33.533 | 1.840 |
| **CRL (Ours)** | **154.018** | **18.015** | **0.888** |

**Instance Case 2** (46 hours):

| Method | AR↑ | AFC↓ | ASS↓ |
|--------|-----|------|------|
| CL-ABDDQN | **298.026** | 36.229 | 5.543 |
| CL-A2C | 287.175 | 21.487 | 3.952 |
| DDPG | 279.439 | 20.974 | 5.063 |
| **CRL** | 294.148 | **19.963** | 4.754 |

**Instance Case 3** (21 hours, different origin–destination pair):

| Method | AR↑ | AFC↓ | ASS↓ |
|--------|-----|------|------|
| CL-ABDDQN | **155.035** | 20.901 | 2.332 |
| CL-A2C | -100.699 | 35.154 | 7.258 |
| DDPG | 148.615 | **9.002** | 2.710 |
| **CRL** | 145.015 | 28.618 | **2.143** |

### Ablation Study

| Instance | Method | AR↑ | AFC↓ | ASS↓ | Note |
|----------|--------|-----|------|------|------|
| Case 1 | w/o CL | 139.629 | 21.893 | 1.069 | CRL outperforms on all metrics |
| Case 1 | **CRL** | **154.018** | **18.015** | **0.888** | — |
| Case 2 | w/o CL | -708.765 | 48.951 | 3.274 | Removing CL causes task failure |
| Case 2 | **CRL** | **294.148** | **19.963** | 4.754 | — |
| Case 3 | w/o CL | 149.950 | **19.557** | 5.901 | Slightly lower fuel but worse safety |
| Case 3 | **CRL** | 145.015 | 28.618 | **2.143** | — |

### Key Findings

1. **CRL achieves the best multi-objective balance**: Although not always optimal on individual metrics, it achieves the best overall trade-off across safety, fuel consumption, and cumulative reward.
2. **Curriculum learning is indispensable**: In Instance Case 2, removing CL leads to complete task failure (AR drops from 294 to −709).
3. **CL accelerates convergence and stabilizes training**: Training reward curves show that CRL reaches high rewards faster and with smaller variance.
4. **CL-A2C fails in most scenarios**: Negative AR in two instances indicates that A2C is unsuitable for this continuous action space.
5. **DDPG achieves high fuel efficiency but poor safety**: In Case 3, DDPG attains the lowest AFC but the highest ASS.

## Highlights & Insights

- **Data-driven full-stack solution**: Every component—from fuel prediction and environment simulation to policy learning—is grounded in real data.
- **First use of diffusion models for AIS trajectory generation**: This enriches simulation diversity beyond simple historical data replay.
- **Image-based environment encoding**: Complex maritime traffic situations are compactly represented as $64 \times 64 \times 3$ visual tensors.
- **Well-motivated curriculum design**: The gradually shrinking distance threshold follows a clear intuition—first learn to reach distant goals, then progressively refine precision.
- **Safety distance formula incorporates vessel geometry**: By accounting for the LOA and beam of both ego and target vessels, the formulation is physically grounded.

## Limitations & Future Work

1. **Limited experimental scale**: Validation is restricted to three instances in a single region of the Indian Ocean, with no testing under broader geographic conditions.
2. **Poor fuel efficiency of CRL in Case 3**: AFC of 28.618 is substantially higher than DDPG's 9.002, suggesting that multi-objective weights may require further tuning.
3. **Insufficient validation of dynamic collision avoidance**: Although the ASS metric is reported, no detailed behavioral analysis of multi-vessel avoidance scenarios is presented.
4. **Lack of explicit COLREGs integration**: The international regulations for preventing collisions at sea are not hard-coded into the policy.
5. **Fuel prediction model accuracy**: An R² of 86.1% leaves approximately 14% unexplained variance, which may degrade decision quality under extreme conditions.
6. **Data accessibility**: The AIS and fuel consumption data used are proprietary, obtained through industry partnerships, making reproducibility difficult.

## Related Work & Insights

- The proposed approach contrasts with Wang et al. (2024a)'s DDPG-based harbor tug optimization and Moradi et al. (2022)'s emission-reduction route planning.
- The CL paradigm can be extended to RL tasks in other complex environments, such as UAV formation control and autonomous driving.
- The idea of using diffusion models to augment simulation environments is transferable to scenario generation in autonomous driving research.

## Rating

- Novelty: ⭐⭐⭐ — Individual modules (CRL, diffusion models, XGBoost) are not novel in isolation, but their combination and application domain carry meaningful value.
- Experimental Thoroughness: ⭐⭐⭐ — Ablation studies are complete, but the evaluation covers only three instances within a single maritime region.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with rigorous notation and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Addresses practical maritime emission reduction and safety problems with clear application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[AAAI 2026\] Speculative Sampling with Reinforcement Learning](speculative_sampling_with_reinforcement_learning.md)
- [\[ICCV 2025\] Embodied Navigation with Auxiliary Task of Action Description Prediction](../../ICCV2025/reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [\[CVPR 2026\] RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](../../CVPR2026/reinforcement_learning/radar_closedloop_robotic_data_generation_via_seman.md)

</div>

<!-- RELATED:END -->
