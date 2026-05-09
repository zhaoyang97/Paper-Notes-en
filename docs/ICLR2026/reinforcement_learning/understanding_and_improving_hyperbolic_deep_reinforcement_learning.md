---
title: >-
  [Paper Note] Understanding and Improving Hyperbolic Deep Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][hyperbolic geometry] Through closed-form gradient analysis, this paper identifies the root causes of instability in hyperbolic deep RL—namely, conformal factor explosion in the Poincaré Ball and PPO trust-region breakdown induced by large-norm embeddings. It proposes Hyper++, a four-component solution comprising RMSNorm, learnable scaling, HL-Gauss categorical value loss, and the Hyperboloid model, achieving comprehensive improvements over prior baselines on ProcGen (16 environments) and Atari-5.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - hyperbolic geometry
  - PPO
  - gradient analysis
  - RMSNorm
  - categorical value loss
date: 2026-05-08
content_hash: 17b68ad28ab69c56
---

# Understanding and Improving Hyperbolic Deep Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2512.14202](https://arxiv.org/abs/2512.14202)
**Code**: [GitHub](https://github.com/Probabilistic-and-Interactive-ML/hyper-rl)
**Area**: Reinforcement Learning
**Keywords**: hyperbolic geometry, PPO, gradient analysis, RMSNorm, categorical value loss

## TL;DR
Through closed-form gradient analysis, this paper identifies the root causes of instability in hyperbolic deep RL—namely, conformal factor explosion in the Poincaré Ball and PPO trust-region breakdown induced by large-norm embeddings. It proposes Hyper++, a four-component solution comprising RMSNorm, learnable scaling, HL-Gauss categorical value loss, and the Hyperboloid model, achieving comprehensive improvements over prior baselines on ProcGen (16 environments) and Atari-5.

## Background & Motivation

**Background**: Sequential decision-making naturally gives rise to hierarchical data—each state branches into exponentially many successors, forming a tree-like structure. Euclidean space grows only polynomially in volume ($V_d(r) \propto r^d$), creating a fundamental geometric mismatch with the exponential growth of hierarchical structures. Hyperbolic space, whose volume grows exponentially, has already demonstrated success in classification, metric learning, and image-text alignment.

**Limitations of Prior Work**: Hyperbolic deep RL suffers from severe optimization difficulties. Cetin et al. (2023) first introduced hyperbolic geometry into RL, but training remained unstable, requiring SpectralNorm combined with S-RYM ($\mathbf{x}_E \mapsto \mathbf{x}_E / \sqrt{d}$) to mitigate instability—at the cost of limiting the expressive capacity of the entire encoder.

**Key Challenge**: A fundamental conflict exists between the geometric advantages of hyperbolic space (low-distortion hierarchical embeddings) and training stability (conformal factor explosion and ill-conditioned gradients).

**Goal**: (1) Formally analyze the sources of gradient ill-conditioning in hyperbolic PPO; (2) design a hyperbolic RL agent that achieves both training stability and expressive power.

**Key Insight**: Closed-form gradient derivations for both the Poincaré Ball and Hyperboloid models are used to precisely locate the sources of instability, after which corresponding components are designed to address each issue individually.

**Core Idea**: Large-norm embeddings are the root cause of hyperbolic PPO collapse. The instability can be fundamentally resolved by constraining the norm via RMSNorm, avoiding the conformal factor via the Hyperboloid model, and aligning the loss function with the geometry via categorical losses.

## Method

### Overall Architecture
Hyper++ adopts a hybrid Euclidean-hyperbolic encoder architecture (Impala-ResNet) with a shared Euclidean encoder and separate hyperbolic actor/critic heads. The core improvements are concentrated at the interface between the final encoder layer and the hyperbolic layer: RMSNorm → TanH → learnable scaling → Hyperboloid exponential map → HL-Gauss categorical value loss.

### Key Designs

1. **RMSNorm Regularization (replacing SpectralNorm)**
   - Function: Constrains the Euclidean embedding norm at the encoder output, preventing gradient explosion through the hyperbolic exponential map.
   - Mechanism: RMSNorm with $1/\sqrt{d}$ scaling is applied only to the pre-activation output of the final linear layer. Proposition 4.2 guarantees that for 1-Lipschitz activations (ReLU/TanH), $\|\hat{\mathbf{x}}\|_2 < 1$, which in turn bounds the conformal factor as $\lambda < 2\cosh^2(\sqrt{c})$.
   - Design Motivation: Lemma 4.1 shows that SpectralNorm must be applied to all encoder layers to effectively constrain the norm, but doing so severely restricts the Lipschitz constant and expressive capacity. RMSNorm requires only the final layer, preserving the freedom of all preceding layers.

2. **Learned Euclidean Feature Scaling**
   - Function: Expands the usable volume of hyperbolic space after RMSNorm constrains the embedding norm.
   - Mechanism: A scalar $\xi_\theta$ is learned to rescale embeddings as $\hat{\mathbf{x}}_E^{\text{rescale}} = \rho_{\max} \cdot \sigma(\xi_\theta) \cdot \hat{\mathbf{x}}_E$, where $\rho_{\max} = \operatorname{atanh}(\alpha)/\sqrt{c}$ and $\alpha=0.95$.
   - Design Motivation: RMSNorm restricts the usable Poincaré Ball radius to 0.76 (at $c=1$). Since usable volume scales as $\propto r^d$, at $d=32$ the volume gain from extending the radius to 0.95 is approximately $(0.95/0.76)^{32} \approx 1200\times$.

3. **Hyperboloid Model + HL-Gauss Categorical Value Loss**
   - Function: Eliminates sources of instability at both the geometric and loss levels.
   - Mechanism: The Hyperboloid MLR contains no conformal factor (the $v^{\text{HB}}$ formula lacks the $(1-c\|\mathbf{x}\|^2)^{-1}$ term), yielding more stable gradients. HL-Gauss reformulates value function learning as a classification problem over 51 discrete bins, geometrically aligned with the hyperplane distance outputs of hyperbolic MLR.
   - Design Motivation: The Poincaré Ball MLR gradient scales as $\propto (1-c\|\mathbf{x}_H\|^2)^{-2}$, exploding near the boundary. MSE regression is geometrically mismatched with hyperbolic MLR, whereas categorical losses provide a more natural alignment.

### Loss & Training

The PPO clipped surrogate objective is unchanged. The critic uses the HL-Gauss loss (51 bins, $[-10, 10]$), and TanH replaces ReLU as the final activation. Corollary 4.3 transfers the norm bounds established by RMSNorm and learnable scaling—via the Poincaré Ball–Hyperboloid isometry—to a bound on the Hyperboloid time component $x_0^{\max}$.

## Key Experimental Results

### Main Results — ProcGen (PPO, 16 environments, 25M steps, 6 seeds)

| Metric | Hyper++ | Hyper+S-RYM | Euclidean | Hyper (no reg.) |
|--------|---------|-------------|-----------|-----------------|
| Test IQM ↑ | **0.41** | 0.27 | 0.26 | 0.19 |
| Train IQM ↑ | **0.55** | 0.46 | 0.45 | 0.37 |
| Forward time | 14.7ms | 19.3ms | 14ms | — |
| NameThisGame (full run) | 35h25m | 58h21m | 17h52m | — |

### Ablation Study (ProcGen Test IQM, 6 seeds + bootstrap CI)

| Configuration | Test IQM | Note |
|--------------|---------|------|
| Hyper++ (full) | **0.40** | Baseline |
| −RMSNorm | 0.00 | Complete learning failure; norm explosion |
| −Scaling | 0.33 | Insufficient usable volume |
| +MSE (replacing HL-Gauss) | 0.33 | Geometric mismatch |
| +C51 | 0.27 | Distributional loss inferior to HL-Gauss |
| +Poincaré (replacing Hyperboloid) | 0.34 | Mild conformal factor degradation |
| +SN Full / +SN Penult. | 0.00 / 0.00 | SpectralNorm fails in both settings |
| Euclidean + full regularization | 0.35 | Competitive but below hyperbolic |

### Key Findings
- Hyper++ remains effective under PPG (a stronger baseline): PPG IQM 0.52 vs. Hyper+S-RYM 0.34 vs. Euclidean 0.47.
- On Atari-5 (DDQN, 10M steps), Hyper++ achieves the best IQM, median, mean, and optimality gap across all five games.
- Euclidean + HL-Gauss underperforms Euclidean + MSE, indicating that categorical losses benefit from hyperbolic geometry to be effective.
- Every ablated configuration underperforms the full Hyper++, demonstrating synergistic interactions among the components.

## Highlights & Insights
- **Gradient-analysis-driven design**: Rather than empirical trial-and-error, the paper first derives closed-form expressions for $\partial v / \partial \mathbf{x}_H$ and $\partial \mathbf{x}_H / \partial \mathbf{x}_E$, identifies $(1-c\|\mathbf{x}\|^2)^{-2}$ as the culprit, and then designs RMSNorm as a targeted remedy.
- **One theoretical result addressing multiple problems**: Proposition 4.2 simultaneously guarantees bounded embedding norms, bounded conformal factors, and stable gradients—a single result with three-fold impact.
- **Clear division of labor among components**: The categorical loss stabilizes $\partial L / \partial v$, the Hyperboloid stabilizes $\partial v / \partial \mathbf{x}_H$, and RMSNorm+scaling stabilizes $\partial \mathbf{x}_H / \partial \mathbf{x}_E$—each term in the chain rule of equation (3) has a dedicated component.
- **Performance and efficiency gains simultaneously**: A 52% improvement in returns is achieved alongside an approximately 30% reduction in forward pass time by eliminating the power iteration required by SpectralNorm.

## Limitations & Future Work
- The analysis focuses on the optimization perspective and does not examine what hierarchical structures are actually learned by the hyperbolic representations.
- It remains unexplored which environments are best suited for hyperbolic representations—i.e., which MDPs have more tree-like state spaces.
- The interactions among geometric choices (curvature $c$, dimension $d$) and different RL algorithm designs are not investigated.
- Hyper++ exhibits plasticity loss on ProcGen Phoenix, performing comparably to baselines in that setting.

## Related Work & Insights
- **vs. Cetin et al. (2023) Hyper+S-RYM**: Eliminates the stability–expressivity trade-off imposed by SpectralNorm, improving test IQM from 0.27 to 0.41.
- **vs. Farebrother et al. (2024) HL-Gauss**: Their work found HL-Gauss to be inconsistently effective in Euclidean RL; this paper reveals a special synergy between HL-Gauss and hyperbolic geometry.
- **vs. Mishne et al. (2023) hyperbolic numerical stability**: Their analysis addresses numerical stability of general hyperbolic networks; this paper extends the investigation to actor-critic training in RL.

## Rating
- Novelty: ⭐⭐⭐⭐ A gradient-analysis-driven remedy for hyperbolic RL with tight coupling between theory and practice.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ ProcGen (16 environments) + Atari-5 + three algorithms (PPO/PPG/DDQN) + extensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with consistent three-way correspondence among equations, components, and experiments.
- Value: ⭐⭐⭐⭐ Provides the first reliable practical solution for hyperbolic deep RL.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](cuda-l1_improving_cuda_optimization_via_contrastive_reinforcement_learning.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](deep_spi_safe_policy_improvement_via_world_models.md)
- [\[ACL 2026\] Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](../../ACL2026/reinforcement_learning/deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)

<!-- RELATED:END -->
