---
title: >-
  [Paper Note] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model
description: >-
  [ICCV 2025][Autonomous Driving][long-tail trajectory prediction] This paper proposes GALTraj, the first method to apply generative active learning to trajectory prediction. During training, it dynamically identifies tail samples on which the model fails, and employs a controllable diffusion model to synthesize new samples that preserve tail-behavior characteristics while complying with traffic rules. This effectively alleviates long-tail data imbalance, improving both tail-case performance and overall prediction accuracy.
tags:
  - ICCV 2025
  - Autonomous Driving
  - long-tail trajectory prediction
  - generative active learning
  - controllable diffusion model
  - traffic simulator
  - data augmentation
date: 2026-05-08
content_hash: 9870537d64b7de91
---

# Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model

**Conference**: ICCV 2025
**arXiv**: [2507.22615](https://arxiv.org/abs/2507.22615)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: long-tail trajectory prediction, generative active learning, controllable diffusion model, traffic simulator, data augmentation

## TL;DR

This paper proposes GALTraj, the first method to apply generative active learning to trajectory prediction. During training, it dynamically identifies tail samples on which the model fails, and employs a controllable diffusion model to synthesize new samples that preserve tail-behavior characteristics while complying with traffic rules. This effectively alleviates long-tail data imbalance, improving both tail-case performance and overall prediction accuracy.

## Background & Motivation

Data-driven trajectory prediction has achieved remarkable progress on large-scale real-world datasets, yet the long-tail problem remains a critical bottleneck:

**Fatal shortcomings of tail samples**: Rare driving behaviors (U-turns, sudden overtaking, emergency lane changes) are severely underrepresented in data. Model representations are biased toward frequently occurring head samples, leading to prediction failures in safety-critical tail scenarios. Existing benchmarks primarily evaluate overall (head-dominated) performance, which obscures this issue.

**Limitations of prior work**:
- **Modified network architectures** (e.g., hypernetworks, mixture-of-experts): increase model complexity and hyperparameters, potentially degrading head-sample performance.
- **Clustering/Kalman filter-based tail identification**: clustering assumes small clusters equal tail samples, which does not necessarily correspond to high prediction error; Kalman filter errors are inconsistent with the actual errors of the target model.
- **Traffic simulators have been used for scene diversification**, but have never been shown to effectively improve long-tail learning.

**Special challenges for data augmentation**: Trajectory prediction is a multi-agent regression task, fundamentally different from class-conditional generation in image classification. Multiple agents interact in traffic scenes, and naively generating random scenes cannot resolve long-tail imbalance—it is necessary to preserve tail-behavior characteristics while introducing scene diversity.

The authors' core insight is that existing model capacity is sufficient to handle both head and tail samples (verified experimentally); the bottleneck lies in the training process. Rather than modifying the architecture, it suffices to modify the training data. The key is to design a "tail-aware" generation strategy that distinguishes among three agent categories—tail, related, and head—and controls generation diversity accordingly.

## Method

### Overall Architecture

GALTraj is an iterative training framework:
1. Train on original data until 2/3 of the total epochs.
2. Identify tail samples on which the current model fails.
3. Augment these samples using the tail-aware generation method.
4. Update the training dataset and continue training.
5. Repeat steps 2–4.

### Key Designs

1. **Dynamic Tail Sample Mining**:

    - **Mechanism**: Tail samples are defined directly by the minADE6 error of the current prediction model. For each agent $n$ at epoch $e$, the error $\delta^{n,(e)} = \text{error}(\psi^{(e)}(\mathbf{x}^n), \mathbf{y}^n)$ is computed. If the maximum error within a scene exceeds a threshold $\tau$, the scene is marked as a tail sample.
    - $\mathcal{D}_{tr}^{tail,(e)} = \{S_j \in \mathcal{D}_{tr} \mid \max_{n \in S_j} \delta^{n,(e)} > \tau\}$
    - **Design Motivation**: Compared to clustering or Kalman filters, this directly reflects the actual weaknesses of the target model. Errors are already computed during loss calculation, requiring no additional inference—only thresholding and ID logging, incurring virtually zero overhead.

2. **Tail-Aware Generation**:

    - **Three-category agent classification**:
        - **Tail agents**: agents whose prediction error exceeds the threshold—their motion characteristics are preserved.
        - **Head agents**: agents with low prediction error—greater motion diversity is introduced.
        - **Related agents**: head agents with high interaction strength toward tail agents (attention score $> \frac{1}{|\mathcal{N}_j|}$)—moderate variation to avoid unrealistic interactions.
    - **Real Guidance for diversity control**: Using the pretrained diffusion model LCSim, reverse sampling does not start from pure noise but from an intermediate noising step $K^*$ applied to the ground-truth trajectory. $K^* = \lambda_{type} \cdot K$, where $\lambda_{tail} = 0.25$ (low noise → high fidelity), $\lambda_{rel} = 0.6$, and $\lambda_{head} = 1.0$ (high noise → high diversity).
    - **Gradient Guidance for traffic-rule compliance**: Gradient guidance is applied to head agents to enforce two constraints: (1) no off-road driving; (2) no collision with other agents (repeller). The formulation is: $p_\theta(y_{k-1} | y_k, \mathbf{x}) \approx \mathcal{N}(y_{k-1}; \mu + \Sigma^k \nabla_\mu \mathcal{C}(\mu), \Sigma^k)$.
    - **Design Motivation**: Tail agents retain their rare behavior patterns without being "smoothed away"; the diversity of head agents enriches the representation of tail samples at the scene level; moderate variation in related agents prevents unrealistic collisions; gradient guidance ensures the physical plausibility of generated scenes.

3. **Training Loop and Overfitting Mitigation**:

    - **Random temporal window shift**: Generated samples cover only the future segment; a random temporal shift $\delta t$ is applied so that part of the generated future trajectory becomes historical context $\{p_t^n\}_{-T_h+\delta t : T_f+\delta t}^{1:N}$, diversifying input features and reducing overfitting.
    - **Sampling weight decay**: Newly generated data are assigned weight 1, while historical data weights decay by $\alpha$, with a minimum sampling weight to ensure head-sample coverage.
    - **Tail mining on original data only**: Generated samples are excluded from mining to avoid redundant detection.

### Loss & Training

- Standard loss functions of the backbone models (QCNet / MTR) are used without modification.
- A pretrained LCSim diffusion model serves as the traffic generator.
- Tail sample mining requires only thresholding, with no additional inference overhead.
- The maximum proportion of tail samples does not exceed 5% of training data.
- Additional training time per epoch does not exceed 36% and decreases as the model converges.

## Key Experimental Results

### Main Results

**QCNet backbone on WOMD and AV2: long-tail and overall metrics (Table 2):**

| Method | Top1%↓ | VaR999↓ | FPR5↓ | minFDE6↓ | Dataset |
|--------|--------|---------|-------|---------|---------|
| Vanilla | 4.81 | 8.42 | 0.42 | 0.654 | WOMD |
| Resampling | 4.30 | 8.01 | 0.38 | 0.668 | WOMD |
| Contrastive | 4.12 | 6.71 | 0.31 | 0.613 | WOMD |
| **GALTraj** | **3.43** | **6.05** | **0.22** | **0.558** | **WOMD** |
| Vanilla | 4.47 | 7.22 | 0.35 | 0.545 | AV2 |
| **GALTraj** | **3.76** | **5.66** | **0.19** | **0.524** | **AV2** |

FPR5 is reduced from 0.42 to 0.22 (halved!), while minFDE6 also improves from 0.654 to 0.558.

**MTR backbone on WOMD (Table 3):**

| Method | Top1%↓ | VaR999↓ | FPR5↓ | minFDE6↓ |
|--------|--------|---------|-------|---------|
| Vanilla | 7.71 | 15.95 | 0.99 | 0.806 |
| Contrastive | 6.75 | 12.81 | 0.74 | 0.780 |
| **GALTraj** | **5.87** | **12.03** | **0.65** | **0.773** |

### Ablation Study

**Ablation of four key components (Table 4, WOMD):**

| Experiment | Real Guide | Grad Guide | Weight Decay | Temporal Shift | FPR5↓ | VaR999↓ | minFDE6↓ |
|-----------|-----------|-----------|-------------|---------------|-------|---------|---------|
| 1 (Naive) | - | - | - | - | 0.38 | 7.91 | 0.612 |
| 2 | ✓ | ✓ | - | - | 0.28 | 6.49 | 0.604 |
| 3 (w/o Real) | - | ✓ | ✓ | ✓ | 0.34 | 7.56 | 0.586 |
| 4 (w/o Grad) | ✓ | - | ✓ | ✓ | 0.26 | 6.52 | 0.601 |
| 5 (Full) | ✓ | ✓ | ✓ | ✓ | **0.22** | **6.05** | **0.558** |

### Key Findings

- **Model capacity verification** (Table 1): Evaluated on the training set, GALTraj reduces Top1% from 7.38 to 2.29, demonstrating that the architecture itself is capable of representing tail samples and that the bottleneck lies in the training process.
- **Real guidance is essential** (exp3 vs. exp5): Without it, FPR5 degrades from 0.22 to 0.34—preserving tail behavior characteristics is critical.
- **Gradient guidance protects head performance** (exp4 vs. exp5): Without it, long-tail metrics slightly degrade and minFDE6 worsens from 0.558 to 0.601—training data that violates traffic rules contaminates head-sample learning.
- **Naive augmentation has limited effect** (exp1): Simple concatenation without any guidance yields only marginal improvement, validating the necessity of tail-aware design.
- Some baselines (e.g., Resampling) produce worse minFDE6 than Vanilla (0.668 vs. 0.654), because excessive focus on tail samples causes head-sample degradation.

## Highlights & Insights

- **"Change the data, not the architecture" philosophy**: This work is the first to demonstrate in trajectory prediction that the training process—rather than the model structure—is the performance bottleneck, which carries profound implications.
- The design of **three-category agent classification with differentiated generation diversity** elegantly balances tail-behavior fidelity and scene diversity.
- **First demonstration that traffic-simulator-driven augmentation can improve long-tail learning**—previously, simulators were used solely for scene diversification.
- **No inference-time overhead**: all additional computation occurs offline during training; at inference, the method is entirely equivalent to the original backbone.
- The random temporal window shift is a simple yet effective trick: using part of the generated future trajectory as historical input achieves two goals simultaneously.

## Limitations & Future Work

- The method relies on the quality of the pretrained diffusion model (LCSim)—if the generator itself poorly models rare behaviors, performance gains will be limited.
- The threshold $\tau$ and $\lambda$ values require empirical tuning.
- The definition of tail samples changes dynamically during training, potentially causing inconsistencies across epochs.
- Although manageable, the additional training time of approximately 36% could be mitigated by faster diffusion sampling methods.
- The causal relationships among agents and their influence on tail behaviors remain unexplored.

## Related Work & Insights

- Extending the active learning framework to regression tasks (rather than traditional classification) is an important contribution.
- Real guidance (initiating reverse sampling from a noised ground-truth trajectory rather than pure noise) is an innovative application of the SDEdit idea to traffic simulation.
- GALTraj is complementary to long-tail trajectory prediction methods such as FEND: FEND modifies the architecture while GALTraj modifies the data; the two can be combined.
- Insight: the framework of generative data augmentation combined with active learning is generalizable to long-tail problems in other autonomous driving subtasks such as planning and perception.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce generative active learning into the long-tail trajectory prediction problem; the three-category agent differentiated generation strategy is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, two backbones, multiple baselines, thorough ablation, and extensive visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, compelling motivation, and information-dense figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the long-tail problem without modifying the architecture or affecting inference efficiency; highly practical and transferable.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[AAAI 2026\] SAML: A Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Prediction](../../AAAI2026/autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)

<!-- RELATED:END -->
