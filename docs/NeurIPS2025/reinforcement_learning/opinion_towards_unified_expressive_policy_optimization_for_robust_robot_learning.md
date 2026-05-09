---
title: >-
  [Paper Note] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Offline-to-Online RL] This paper proposes UEPO, a framework comprising three core components—multi-seed dynamics-aware diffusion policies, dynamic divergence regularization, and diffusion-based data augmentation—to address insufficient multimodal behavioral coverage and distribution shift in offline-to-online reinforcement learning, surpassing Uni-O4 on the D4RL benchmark.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Offline-to-Online RL
  - Diffusion Policy
  - Policy Diversity
  - Dynamics Modeling
  - D4RL
date: 2026-05-08
content_hash: 8cd2f68f61178620
---

# Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning

**Conference**: NeurIPS 2025
**arXiv**: [2511.10087](https://arxiv.org/abs/2511.10087)
**Code**: Unavailable
**Area**: Reinforcement Learning / Robot Learning
**Keywords**: Offline-to-Online RL, Diffusion Policy, Policy Diversity, Dynamics Modeling, D4RL

## TL;DR

This paper proposes UEPO, a framework comprising three core components—multi-seed dynamics-aware diffusion policies, dynamic divergence regularization, and diffusion-based data augmentation—to address insufficient multimodal behavioral coverage and distribution shift in offline-to-online reinforcement learning, surpassing Uni-O4 on the D4RL benchmark.

## Background & Motivation

Offline-to-online reinforcement learning (O2O-RL) is a critical paradigm for safe and efficient robot deployment: a base policy is first pre-trained on offline data to capture physical dynamics, and then fine-tuned through environment interaction to adapt to dynamic scenarios.

Existing methods face two core challenges:

**Insufficient multimodal behavioral coverage**: Traditional behavioral cloning (BC) relies on large amounts of expert data and struggles to cover multimodal action distributions. Although diffusion policies excel at offline modeling, fixed noise schedules and the absence of environmental feedback lead to policy degradation and distribution shift during online fine-tuning.

**Weak interface between offline and online phases**: Existing frameworks such as Uni-O4 still exhibit limitations in offline pre-training, generative model adaptation, and scalability, including high computational cost, insufficient diversity at the physical execution level, and low data efficiency.

The design of UEPO is inspired by the "pre-training + fine-tuning" paradigm in large language models, transferring this idea to robot policy learning.

## Method

### Overall Architecture

UEPO introduces three core innovations spanning both offline and online learning phases:

1. **Offline phase**: Multi-seed diffusion sampling initialization → divergence regularization to enhance diversity → diffusion-based data augmentation for dynamics model training
2. **Online phase**: Qualified policies selected from the offline phase serve as initialization for online fine-tuning

### Key Designs

#### 1. Multi-Seed Conditioned Action Sequence Generation

A state-conditioned diffusion policy is employed to model the complete action sequence distribution $p(a_{1:T}|s_{1:T})$, capturing long-horizon dependencies and multimodal behaviors.

**Mechanism**: Rather than training multiple independent models, a single pre-trained diffusion model is used to construct an ensemble of $n$ sub-policies $\{\pi_\theta^i\}_{i=1}^n$ by varying the initial noise seeds during reverse sampling. Each sub-policy initializes the reverse process with a distinct random seed $\epsilon_i \sim \mathcal{N}(0, \mathbf{I})$ on the same state sequence, generating action sequences corresponding to different behavioral modes.

**Advantage**: This approach substantially reduces computational cost compared to traditional ensemble methods that require training multiple independent models.

#### 2. Diffusion Sampling-Guided Divergence Regularization

Multi-seed initialization alone provides only initial diversity; it is further necessary to ensure that sub-policies remain divergent during dynamic execution.

**Dynamic divergence metric**: Measures the first-order (velocity) and second-order (acceleration) kinematic differences between two action sequences $a_i$ and $a_j$:

$$\text{div}(a_i, a_j) = \frac{1}{T}\sum_{t=1}^{T}\left(\|\dot{a}_{i,t} - \dot{a}_{j,t}\|_2 + (1 - \cos(\ddot{a}_{i,t}, \ddot{a}_{j,t}))\right)$$

**Adaptive perturbation**: When divergence falls below threshold $\tau$, adaptive noise is injected into the denoising estimate:

$$a_t^i \leftarrow a_t^i + \delta, \quad \delta \sim \mathcal{N}(0, \sigma_{\text{div}}^2 \mathbf{I}), \quad \sigma_{\text{div}} = \eta \cdot \frac{\tau - \text{div}(a_i, a_j)}{\tau}$$

The smaller the divergence, the larger the perturbation, forcing sub-policies to explore distinct dynamic modes.

**Synergy with sequence-level KL regularization**: The KL divergence penalty from Uni-O4 is retained, but redefined from single-step action distributions to full action sequence distributions, naturally aligning with the sequence-level diffusion policy.

#### 3. Diffusion-Based Dynamics Model Augmentation

Physically plausible synthetic trajectories generated by the diffusion policy are used to augment the training data for the dynamics model:

- Initial states are sampled from the offline data distribution
- Multi-step action sequences are generated using the diffusion policy
- Trajectories are produced through real transition dynamics
- **Key filtering**: KL divergence between real dynamics and the initial model is computed; only trajectories with $D_{KL} < \epsilon$ are retained
- Augmented data volume is controlled to 2–3× the size of the original dataset

### Loss & Training

The sub-policy objective combines likelihood maximization with sequence-level KL regularization:

$$J(\hat{\pi}^i) = \mathbb{E}_{(s,a)\sim\mathcal{D}}[\log p_\theta(a|s)] + \alpha \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\log\frac{p_\theta(a|s)}{\max_j p_\theta(a|s)}\right]$$

The joint training objective for the dynamics model:

$$\mathcal{L}(\hat{T}) = -\mathbb{E}_{(s,a,s')\sim\mathcal{D}\cup\mathcal{D}_{\text{diff}}}[\log \hat{T}(s'|s,a)]$$

## Key Experimental Results

### Main Results: D4RL Locomotion Tasks

| Environment | CQL | TD3+BC | IQL | BPPO | Uni-O4 | **UEPO** |
|---|---|---|---|---|---|---|
| halfcheetah-medium-v2 | 44.0 | 48.3 | 47.4 | 44.0 | 52.6 | **57±0.8** |
| hopper-medium-v2 | 58.5 | 59.3 | 66.3 | 93.9 | 104.4 | **108±0.5** |
| walker2d-medium-v2 | 72.5 | 83.7 | 78.3 | 83.6 | 90.2 | **91±1.4** |
| halfcheetah-medium-replay | 45.5 | 44.6 | 44.2 | 41.0 | 44.3 | **58.2±0.7** |
| hopper-medium-replay | 95.0 | 60.9 | 97.7 | 92.5 | 103.2 | **112.0±2.3** |
| walker2d-medium-replay | 77.2 | 81.8 | 73.9 | 77.6 | 98.4 | **103.8±1.7** |
| halfcheetah-medium-expert | 91.6 | 90.7 | 89.7 | 92.6 | 93.8 | **94.3±0.6** |
| hopper-medium-expert | 105.4 | 98.0 | 91.7 | 112.8 | 111.4 | **118.6±0.2** |
| walker2d-medium-expert | 108.8 | 110.1 | 109.6 | 113.1 | 118.1 | **120.7±0.3** |
| **Locomotion Total** | 698.5 | 677.4 | 692.4 | 751.0 | 816.4 | **864.6±8.5** |

### Adroit Dexterous Manipulation + Kitchen Tasks

| Environment | CQL | IQL | BPPO | Uni-O4 | **UEPO** |
|---|---|---|---|---|---|
| pen-human | 37.5 | 71.5 | 117.8 | 116.2 | **122.8±5.8** |
| hammer-human | 4.4 | 1.4 | 14.9 | 247.1 | 30.2±3.3 |
| door-human | 9.9 | 4.3 | 25.8 | 17.3 | **29.3±0.7** |
| pen-cloned | 39.2 | 37.3 | 110.8 | 101.4 | **118.4±12.4** |
| **Adroit Total** | 93.6 | 118.1 | 291.4 | 288.6 | **324.4±26.5** |
| kitchen-complete | 43.8 | 62.5 | 91.5 | 93.6 | **102.6±3.6** |
| kitchen-partial | 49.8 | 46.3 | 57.0 | 58.3 | 57.6±2.8 |
| kitchen-mixed | 51.0 | 51.0 | 62.5 | 65.0 | **70.3±5.6** |
| **Kitchen Total** | 144.6 | 159.8 | 211.0 | 216.9 | **230.5±12.0** |
| **Overall Total** | 936.7 | 970.3 | 1253.4 | 1322.0 | **1419.5±47.0** |

### Key Findings

1. **Consistent improvement over Uni-O4**: Locomotion total score improves by +48.2 (+5.9%), Adroit by +35.8 (+12.4%), and overall total by +97.5 (+7.4%).
2. **Most pronounced gains on medium-replay tasks**: halfcheetah-medium-replay improves from 44.3 to 58.2 (+31.4%), indicating greater advantage when data quality is lower.
3. **Strong scalability on Adroit dexterous manipulation**: Substantial gains on high-dimensional tasks requiring fine-grained control.
4. **Performance below Uni-O4 on certain tasks**: hammer-human (30.2 vs. 247.1) and relocate-human (2.9 vs. 27.1) indicate remaining limitations on specific tasks.
5. **Generally small standard deviations**: Suggesting good method stability, though pen-cloned (±12.4) and hammer-human (±3.3) exhibit relatively high variance.

## Highlights & Insights

- **Single-model multi-seed ensemble strategy**: Avoids the computational overhead of training multiple models; diverse sub-policies are generated from a single diffusion model by varying noise seeds, representing an elegant and efficient design.
- **Dynamics-level divergence metric**: Velocity and acceleration differences are used rather than simple distributional distances to measure policy diversity, ensuring that differences are physically meaningful at the execution level.
- **Adaptive perturbation mechanism**: Perturbation magnitude increases automatically when divergence is insufficient, forming a negative feedback loop that prevents sub-policies from converging to similar modes.
- **KL filtering for diffusion-augmented data**: Ensures that synthetic trajectories remain consistent with real dynamics, preventing the introduction of erroneous augmentation data.

## Limitations & Future Work

- **Evaluation limited to D4RL simulation benchmarks**: Real-robot experimental validation is absent.
- **Substantial performance degradation on hammer-human and relocate-human relative to Uni-O4**: The causes of degradation on these tasks are not analyzed.
- **Computational cost not discussed in detail**: Although the approach is claimed to be more efficient than multi-model ensembles, no concrete comparisons of training time or resource usage are provided.
- **Hyperparameter sensitivity not explored**: The influence of key hyperparameters such as divergence threshold $\tau$, perturbation strength $\eta$, and augmentation data ratio is not analyzed.
- **The paper cites a large number of references with weak methodological relevance** (e.g., camera calibration, multimodal recommendation), which undermines academic rigor.

## Related Work & Insights

- **Uni-O4**: The direct baseline improved upon in this work; it achieves O2O-RL through joint optimization of offline and online objectives, but exhibits limitations in policy diversity and data efficiency.
- **Diffusion Policy**: A successful application of diffusion models to policy learning, though fixed noise schedules constrain online adaptation.
- **BPPO**: Behavior Proximal Policy Optimization, which achieves strong performance on Adroit.
- **Insights**: Transferring the "pre-training + fine-tuning" paradigm from LLMs to RL policy learning is a promising direction; multi-seed sampling offers a low-cost alternative for constructing policy ensembles.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The three technical components are well-motivated; the multi-seed policy ensemble and dynamics-level divergence metric are creative contributions.
- **Effectiveness**: ⭐⭐⭐⭐ — Comprehensive improvements on D4RL, though performance degradation on certain tasks warrants explanation.
- **Reproducibility**: ⭐⭐⭐ — No code is provided, and some implementation details are insufficiently specified.
- **Impact**: ⭐⭐⭐⭐ — Direct contributions to the O2O-RL and diffusion policy communities.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization](metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)

<!-- RELATED:END -->
