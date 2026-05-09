---
title: >-
  [Paper Note] Efficient Adversarial Attacks on High-dimensional Offline Bandits
description: >-
  [ICLR 2026][Image Generation][Offline multi-armed bandits] This paper exposes a security vulnerability in offline multi-armed bandit (MAB) evaluation frameworks: an attacker can completely hijack a bandit's decision-making behavior by applying imperceptibly small perturbations to publicly available reward model weights. The required perturbation norm decreases as input dimensionality increases ($\widetilde{\mathcal{O}}(d^{-1/2})$), rendering image-based generative model evaluation pipelines particularly vulnerable.
tags:
  - ICLR 2026
  - Image Generation
  - Offline multi-armed bandits
  - adversarial attacks
  - reward models
  - high-dimensional data
  - generative model evaluation
date: 2026-05-08
content_hash: 5664231396e87fac
---

# Efficient Adversarial Attacks on High-dimensional Offline Bandits

**Conference**: ICLR 2026
**arXiv**: [2602.01658](https://arxiv.org/abs/2602.01658)
**Code**: [GitHub](https://github.com/hadi-hosseini/adversarial-attacks-offline-bandits)
**Area**: Image Generation
**Keywords**: Offline multi-armed bandits, adversarial attacks, reward models, high-dimensional data, generative model evaluation

## TL;DR

This paper exposes a security vulnerability in offline multi-armed bandit (MAB) evaluation frameworks: an attacker can completely hijack a bandit's decision-making behavior by applying imperceptibly small perturbations to publicly available reward model weights. The required perturbation norm decreases as input dimensionality increases ($\widetilde{\mathcal{O}}(d^{-1/2})$), rendering image-based generative model evaluation pipelines particularly vulnerable.

## Background & Motivation

MAB algorithms have been widely adopted to evaluate generative models (diffusion models, LLMs, etc.), using strategies such as UCB to efficiently identify the best-performing model as a cost-effective alternative to exhaustive comparisons. Such evaluations typically rely on **publicly available reward models** (e.g., CLIP, BLIP, aesthetic scorers) hosted on platforms like Hugging Face.

**Core security concern**: Offline bandit evaluation—which replaces online interaction with fixed logged data—introduces two overlooked risks:

**Susceptibility to adversarial manipulation**: An attacker can bias the model selection process.

**Susceptibility to overfitting to reward models**: Repeated adjustment of evaluation metrics on the same dataset.

**Research gap**: Prior adversarial attack research has focused on tampering with reward values during training, and has never considered the more realistic threat model of **perturbing the reward model itself prior to training**.

**Intuition**: Estimating means in high-dimensional spaces is inherently unstable (the curse of dimensionality), and bandits fundamentally perform repeated high-dimensional mean estimation across arms. When observations per arm satisfy $T/K \ll d$, estimates are naturally unreliable. In this regime, a small manipulation of the reward function—exploiting the degrees of freedom in its high-dimensional parameter space—is sufficient to systematically mislead the bandit.

## Method

### Overall Architecture

The attacker $\mathscr{A}$ has access to the offline logged datasets $\mathcal{D}_1, \ldots, \mathcal{D}_K$ and applies a small perturbation $\boldsymbol{\delta}$ to the reward model $r(\cdot)$ prior to bandit training, with the goal of preventing the bandit from identifying the optimal arm.

### Key Designs

1. **Three attack strategies**:

    - **Full Trajectory Attack**: Forces the bandit to follow a fully pre-specified target trajectory $\widetilde{A}_t$; requires $(T-K)(K-1)$ constraints.
    - **Trajectory-Free Attack**: Only prevents the optimal arm $i^*$ from being selected without specifying an alternative; reduces constraints to $T-K$.
    - **Online Score-Aware (OSA) Attack**: Adds constraints incrementally online—only when the optimal arm is about to be selected—reducing the effective constraint count to $\mathcal{O}(\log T)$ and substantially lowering computational cost.

   For a linear reward model $r(\mathbf{X}) = \mathbf{w}^\top \mathbf{X}$, all attacks reduce to a convex quadratic program (QP):
    $$\boldsymbol{\delta}^* = \arg\min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_2^2 \quad \text{s.t.} \quad \boldsymbol{\delta}^\top \mathbf{T}_{i,t} > R_{i,t}, \forall (i,t) \in \mathcal{I}$$

2. **Extension from linear models to neural networks**:

    - Leveraging Neural Tangent Kernel (NTK) theory, for sufficiently wide randomly initialized networks:
    $$\text{NN}_{\boldsymbol{\theta}+\boldsymbol{\delta}}(\mathbf{X}) \approx \text{NN}_{\boldsymbol{\theta}}(\mathbf{X}) + \nabla_{\boldsymbol{\theta}}\text{NN}_{\boldsymbol{\theta}}(\mathbf{X})^\top \boldsymbol{\delta}$$
    - That is, overparameterized networks are approximately linear in parameter space, reducing the attack to the linear case.
    - When hidden layer width exceeds 750, the attack success rate reaches 100%.

3. **Theoretical guarantees**:

    - **Feasibility (Theorem 3.3)**: When $d > (T-K)(K-1)$ and the data distribution is non-degenerate, the attack is feasible with probability 1.
    - **Perturbation norm (Theorem 3.4)**: In the high-dimensional regime $d \geq KT$, the perturbation norm required for the full trajectory attack satisfies $\|\boldsymbol{\delta}^*\|_2 \leq \mathcal{O}\left(\sqrt{\frac{T^3 \log T \cdot \log d}{Kd}}\right)$, i.e., the perturbation shrinks as dimensionality grows.

### Loss & Training

The attack optimization objective is a convex quadratic program with solution complexity $\widetilde{\mathcal{O}}(|\mathcal{I}|^3 + d|\mathcal{I}|^2)$. The OSA method achieves efficient attacks by reducing the constraint count $|\mathcal{I}|$ to $\mathcal{O}(\log T)$.

Defense mechanism: Randomly shuffling a portion of the logged data before running the bandit—shuffling $T/2$ data points—significantly reduces the attack success rate.

## Key Experimental Results

### Main Results

Attack effectiveness is validated on both synthetic and real-world data:

| Setting | Attack Method | ASR | Notes |
|---------|--------------|-----|-------|
| Linear model, synthetic data | Full Trajectory | 100% | High computational cost |
| Linear model, synthetic data | Trajectory-Free | 100% | Moderate cost |
| Linear model, synthetic data | OSA | 100% | **Cost reduced by orders of magnitude** |
| Nonlinear model, synthetic data | OSA | 100% | Width > 750 |
| Real model (HF Aesthetic Scorer) | OSA | ~80% probability ASR∈[90–100%] | 5 generative models |
| Real model (HF Image Reward) | OSA | ~80% probability ASR∈[90–100%] | 30 random prompts |

| Bandit Algorithm | UCB | ETC | ε-greedy |
|---|---|---|---|
| ASR | 100% | 100% | 100% |

### Ablation Study

| Experiment | K | T | d | ASR | Perturbation Norm Trend |
|------------|---|---|---|-----|------------------------|
| Increasing dimensionality | 3 | 100 | 100→5000 | 100% | ℓ₂ and ℓ∞ **consistently decrease** |
| Random noise baseline | 3 | 100 | 1000 | ~25% | Regardless of norm magnitude |
| Defense (shuffle T/2) | 3 | 100 | 100 | Significantly reduced | — |
| Attack Fast-Slow robust algorithm | 3, 5 | 1000 | 1000 | 100% | ~0.3 |
| Attack ε-contamination | 3, 5 | 100–1000 | 1000 | 100% | ~0.2 |

### Key Findings

1. **Random perturbations are ineffective**: Regardless of norm magnitude, random noise yields an ASR of approximately 25% (equivalent to random selection), confirming that targeted optimization is necessary.
2. **Higher dimensionality implies greater vulnerability**: Both ℓ₂ and ℓ∞ norms decrease substantially as dimensionality increases.
3. **OSA is extremely efficient**: The constraint count drops from $\mathcal{O}(TK)$ to $\mathcal{O}(\log T)$ while maintaining 100% ASR.
4. **Robust bandit algorithms are equally vulnerable**: Both Fast-Slow and ε-contamination methods fail to provide a defense.

## Highlights & Insights

- A novel and highly realistic threat model is introduced: attacking reward model weights rather than training data.
- Theory and experiment are in perfect agreement: the theoretically predicted high-dimensional effects are precisely validated empirically.
- An important practical warning is raised: publicly releasing reward model weights on Hugging Face renders evaluations susceptible to manipulation.
- The OSA heuristic achieves near-optimal attacks with only $\mathcal{O}(\log T)$ constraints, offering significant engineering value.
- The proposed defense (data shuffling) is simple yet points in a promising direction for future research.

## Limitations & Future Work

- The defense mechanism remains incomplete; achieving full protection is an open problem.
- The NTK linear approximation may be insufficiently accurate for deep, narrow networks encountered in practice.
- Real-world experiments only perturb a small subset of weights while freezing the rest; the feasibility of perturbing all weights is not explored.
- Defense strategies based on ensembles of multiple reward models are not considered.
- Attacks on contextual bandits or more complex bandit variants are not addressed.

## Related Work & Insights

- **Jun et al. (2018)**: Tampers with reward values during online training; this paper shifts to pre-training attacks on model weights.
- **Garcelon et al. (2020)**: Adversarial attacks on linear contextual bandits.
- **NTK theory (Jacot et al. 2018)**: Provides the theoretical foundation for linearizing nonlinear network attacks.
- Broader implication: Any automated evaluation system relying on publicly available weighted evaluators may harbor analogous vulnerabilities.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introduces an entirely new threat model and attack paradigm; the theoretical discovery of high-dimensional effects is surprising.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic and real data across multiple bandit algorithms, though real-world validation could be further enriched.
- Writing Quality: ⭐⭐⭐⭐ The theoretical sections are rigorous, but the heavy notation may require substantial background knowledge for readers outside the bandit community.
- Value: ⭐⭐⭐⭐ Carries significant implications for AI safety and model evaluation, though direct application scenarios are relatively narrow.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] Scaling Offline RL via Efficient and Expressive Shortcut Models](../../NeurIPS2025/image_generation/scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)
- [\[ICLR 2026\] Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](verification_of_the_implicit_world_model_in_a_generative_model_via_adversarial_s.md)

<!-- RELATED:END -->
