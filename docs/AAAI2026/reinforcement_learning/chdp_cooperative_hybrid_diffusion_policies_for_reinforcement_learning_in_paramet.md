---
title: >-
  [Paper Note] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments
description: >-
  [AAAI 2026][Reinforcement Learning][Hybrid Action Space] This paper models the hybrid action space problem as a fully cooperative two-agent game, employing discrete and continuous diffusion policies respectively to generate actions. Sequential updates and a Q-guided codebook are introduced to resolve policy conflicts and high-dimensional scalability issues, achieving up to a 19.3% improvement in success rate.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Hybrid Action Space
  - Diffusion Policy
  - Multi-Agent Cooperation
  - Vector-Quantized Codebook
  - Parameterized Action MDP
date: 2026-05-08
content_hash: 12831018a0575e2a
---

# CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments

**Conference**: AAAI 2026
**arXiv**: [2601.05675](https://arxiv.org/abs/2601.05675)
**Code**: N/A
**Area**: Reinforcement Learning
**Keywords**: Hybrid Action Space, Diffusion Policy, Multi-Agent Cooperation, Vector-Quantized Codebook, Parameterized Action MDP

## TL;DR

This paper models the hybrid action space problem as a fully cooperative two-agent game, employing discrete and continuous diffusion policies respectively to generate actions. Sequential updates and a Q-guided codebook are introduced to resolve policy conflicts and high-dimensional scalability issues, achieving up to a 19.3% improvement in success rate.

## Background & Motivation

Hybrid Action Spaces simultaneously contain discrete choices and continuous parameters, and are prevalent in robot control and game AI. For example, a soccer shot requires first selecting left/right foot (discrete), then determining force and angle (continuous). Existing methods face two major challenges:

**Insufficient policy expressiveness**: Most methods rely on unimodal architectures (Gaussian or deterministic policies), which fail to capture multimodal distributions. When multiple action groups are equally effective, the policy either averages them or collapses to a single mode.

**Poor scalability**: Combinatorial explosion in high-dimensional discrete spaces leads to extremely low exploration efficiency. The number of on/off combinations for $n$ actuators is $2^n$; when $n=10$, there are 1,024 discrete actions.

Although HyAR provides a scalable latent space framework, it is limited by the expressiveness of deterministic policies; HyDo uses diffusion policies but inherits the scalability bottleneck of PDQN. No existing method addresses both problems simultaneously.

## Method

### Overall Architecture

CHDP models the hybrid action space as a fully cooperative game between two agents, formalized as a PAMDP. The framework consists of three core components:

- **Discrete agent**: Diffusion policy $\pi_{\theta_d}$ generates a latent representation and quantizes it into a discrete action via a codebook.
- **Continuous agent**: Diffusion policy $\pi_{\theta_c}$ generates continuous parameters conditioned on the discrete codeword.
- **Q-guided codebook**: A learnable codebook $\mathbf{E}_\zeta \in \mathbb{R}^{K \times d_e}$ embeds the high-dimensional discrete space into a low-dimensional latent space.

The inference pipeline follows sequential cooperation: (1) the discrete policy starts from noise and generates a latent representation $e$ via reverse diffusion; (2) VQ quantization maps $e$ to the nearest neighbor in the codebook, yielding discrete action index $k$ and codeword $e_k$; (3) the continuous policy generates $a^c$ via diffusion conditioned on $(s, e_k)$.

### Key Designs

**1. Multimodal Expressiveness via Dual Diffusion Policies**

Discrete policy sampling:
$$e_{i-1} = \frac{1}{\sqrt{\alpha_i}}\left(e_i - \frac{1-\alpha_i}{\sqrt{1-\bar{\alpha}_i}}\epsilon_{\theta_d}(e_i, s, i)\right) + \sqrt{\beta_i}z$$

The continuous policy is additionally conditioned on codeword $e_k$:
$$a_{i-1}^c = \frac{1}{\sqrt{\alpha_i}}\left(a_i^c - \frac{1-\alpha_i}{\sqrt{1-\bar{\alpha}_i}}\epsilon_{\theta_c}(a_i^c, s, e_k, i)\right) + \sqrt{\beta_i}z$$

This conditional formulation explicitly models the dependency between discrete and continuous actions.

**2. Sequential Update Mechanism**

This mechanism resolves optimization conflicts that arise when both policies are updated simultaneously:

- **Step 1**: Update the discrete policy first: $\mathcal{L}(\theta_d) = \mathcal{L}_d(\theta_d) - \alpha \cdot \mathbb{E}[Q_\phi(s, e, a^c)]$, where $a^c$ is sampled from the replay buffer as a fixed target.
- **Step 2**: Freeze the discrete policy; use the updated $\pi'_{\theta_d}$ to generate new latent representations, and jointly optimize the continuous policy and codebook. A stop-gradient operator prevents gradients from propagating back to the discrete policy.

**3. Q-Guided Codebook**

Inspired by VQ-VAE but without a reconstruction objective — codeword quality is determined by downstream Q-values. The quantization operation is $k = \arg\min_k \|e - e_k\|^2$. The gradient flow is deliberately asymmetric:

- The codebook receives gradients through the Q-value improvement term of the continuous policy, directing codewords toward regions supporting high-value actions.
- The discrete policy is optimized via the same Q-function but through an independent path (continuous actions are sampled from the buffer with gradients detached).
- Both are implicitly aligned to a shared latent space through the same Q-value metric.

### Loss & Training

The DQL framework is adopted; the loss for each policy is $\mathcal{L}(\theta) = \mathcal{L}_d(\theta) + \alpha \cdot \mathcal{L}_q(\theta)$, where $\alpha = \eta / \mathbb{E}[|Q_\phi(s,a)|]$ balances the two terms.

The critic uses Double Q-learning:
$$y_t = r_t + \gamma \min_{j=1,2} Q'_{\phi'_j}(s_{t+1}, e_{t+1}, a_{t+1}^c)$$

## Key Experimental Results

### Main Results

Success rate comparison (%) across 8 PAMDP benchmarks:

| Environment | HPPO | PATD3 | PDQN-TD3 | HyAR-TD3 | **CHDP** |
|-------------|------|-------|----------|----------|----------|
| Goal | 0.0 | 0.0 | 71.4 | 77.3 | **80.9** |
| Hard Goal | 0.0 | 43.0 | 0.0 | 60.2 | **79.5** |
| Platform | 66.3 | 95.1 | 96.7 | 96.6 | **99.7** |
| Catch Point | 55.7 | 86.7 | 89.8 | 86.6 | **93.8** |
| Hard Move(n=4) | 3.3 | 63.9 | 79.7 | 91.4 | **94.2** |
| Hard Move(n=6) | 2.5 | 9.8 | 31.1 | 92.3 | **93.9** |
| Hard Move(n=8) | 2.3 | 4.6 | 6.6 | 88.3 | **90.6** |
| Hard Move(n=10) | 3.4 | 10.3 | 3.3 | 69.0 | **79.8** |

CHDP achieves state-of-the-art results on all benchmarks. Hard Goal surpasses HyAR by 19.3 percentage points; Hard Move(n=10) improves by 10.8 percentage points.

### Ablation Study

| Variant | Hard Goal | Hard Move(n=6) |
|---------|-----------|----------------|
| CHDP (Full) | **75.9±3.7** | **93.9±1.0** |
| w/o Diffusion Policy | 51.3±10.2 | 45.1±19.5 |
| w/o Codebook | 71.0±6.0 | 11.1±6.9 |
| w/o Sequential Update | 32.8±4.3 | 89.4±3.5 |
| w/o Both | 31.5±16.4 | 10.7±5.1 |

Each of the three components is indispensable and addresses a distinct challenge.

### Key Findings

- The deterministic policy of HyAR collapses to a single mode in Hard Move(n=6) (selecting the same direction 100% of the time, variance = 0).
- CHDP discovers at least three strategies (frequencies of 79%/17%/4%), including a counterintuitive direction-reversal strategy.
- Learning curves show that CHDP converges faster with higher sample efficiency.

## Highlights & Insights

1. Reformulating the hybrid action space problem as a fully cooperative game is an elegant problem abstraction.
2. Qualitative experiments intuitively demonstrate the multimodal advantage of diffusion policies — revealing counterintuitive solutions unreachable by deterministic policies.
3. The codebook is guided by downstream Q-values rather than a reconstruction objective, making discrete representations naturally task-aligned.

## Limitations & Future Work

1. Validation is limited to PAMDP benchmarks; experiments on more complex real-world scenarios (e.g., robotic manipulation) are absent.
2. The multi-step sampling latency of diffusion models may hinder deployment in scenarios with strict real-time requirements.
3. The codebook size is fixed at $K$; adaptive mechanisms have not been explored.
4. Whether gradient projection or similar methods could replace sequential updates to reduce complexity warrants further investigation.

## Related Work & Insights

- **HyAR**: Provides a scalable latent space framework but is constrained by deterministic policies; CHDP's codebook design is more elegant.
- **DQL**: A pioneer in integrating diffusion models into RL; CHDP is a natural extension to hybrid action spaces.
- **VQ-VAE**: The source of inspiration for the codebook; replacing the reconstruction objective with Q-values is a key innovation in CHDP.
- **HARL**: Provides the theoretical foundation for the sequential update mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel combination of cooperative game theory, diffusion policies, and Q-guided codebook)
- Technical Depth: ⭐⭐⭐⭐ (Gradient flow design and multi-component coordination are rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Complete ablations with convincing qualitative analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and intuitive illustrations)
- Value: ⭐⭐⭐ (Benchmarks are relatively simple; practical applicability lacks validation)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Formal Verification of Diffusion Auctions](formal_verification_of_diffusion_auctions.md)
- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](../../NeurIPS2025/reinforcement_learning/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](../../NeurIPS2025/reinforcement_learning/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)

<!-- RELATED:END -->
