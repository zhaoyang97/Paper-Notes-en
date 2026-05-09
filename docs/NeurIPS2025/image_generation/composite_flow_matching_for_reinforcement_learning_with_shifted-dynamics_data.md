---
title: >-
  [Paper Note] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data
description: >-
  [NeurIPS 2025][Image Generation][Reinforcement Learning] This paper proposes CompFlow, a composite flow matching architecture that builds an online flow on top of the offline flow's output distribution to estimate the dynamics shift (Wasserstein distance) between offline and online environments. Combined with an active exploration strategy targeting high-shift regions, CompFlow achieves an average return 14.2% above the strongest baseline across 27 shifted-dynamics RL tasks.
tags:
  - NeurIPS 2025
  - Image Generation
  - Reinforcement Learning
  - Flow Matching
  - Optimal Transport
  - Shifted Dynamics
  - Wasserstein Distance
  - Offline Data
date: 2026-05-08
content_hash: 82849957324fe281
---

# Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data

**Conference**: NeurIPS 2025
**arXiv**: [2505.23062](https://arxiv.org/abs/2505.23062)
**Code**: [GitHub](https://github.com/Haichuan23/CompositeFlow)
**Area**: Image Generation
**Keywords**: Reinforcement Learning, Flow Matching, Optimal Transport, Shifted Dynamics, Wasserstein Distance, Offline Data

## TL;DR

This paper proposes CompFlow, a composite flow matching architecture that builds an online flow on top of the offline flow's output distribution to estimate the dynamics shift (Wasserstein distance) between offline and online environments. Combined with an active exploration strategy targeting high-shift regions, CompFlow achieves an average return 14.2% above the strongest baseline across 27 shifted-dynamics RL tasks.

## Background & Motivation

Online RL typically requires a large number of environment interactions, which is prohibitively costly in real-world settings such as robotics, healthcare, and wildlife conservation. Leveraging pre-collected offline data can improve sample efficiency, but when the **transition dynamics** of the offline data differ from those of the online environment (i.e., shifted dynamics), direct reuse causes distributional mismatch and policy degradation.

Core limitations of existing methods:
- **H2O**, **BC-PAR**, and others estimate dynamics shift via KL divergence or mutual information, but these measures **may be undefined or unstable** when the supports of the two dynamics do not overlap.
- **BC-VGDF** filters transitions based on value estimates, but bootstrapping bias and non-stationary targets make the estimates unreliable.
- All existing methods perform only passive filtering (discarding high-shift data) without actively exploring high-shift regions.

The core insight of CompFlow: **there exists a theoretical connection between flow matching and optimal transport**—the transport cost of a trained flow model approximates the Wasserstein distance. This connection yields a **well-defined, support-agnostic** estimator of dynamics shift.

## Method

### Overall Architecture

CompFlow consists of three core modules:

1. **Offline flow model**: learns the offline dynamics $p_{\text{off}}(s'|s,a)$ from a Gaussian prior.
2. **Online composite flow model**: built on the output distribution of the offline flow, learning the residual transport from $p_{\text{off}}$ to $p_{\text{on}}$.
3. **Dynamics-shift-guided policy training**: selectively incorporates offline data and actively explores high-shift regions based on Wasserstein distance estimates.

### Key Designs

**Composite flow architecture**:

Conventional methods learn online dynamics directly from a Gaussian prior $x_0 \sim \mathcal{N}(0, \mathbf{I})$. CompFlow proceeds in two stages:

- **Offline flow** ($t: 0 \to 1$): $x_1 = \psi_\theta^{\text{off}}(x_0, 1 \mid s, a)$, mapping Gaussian noise to the offline transition distribution.
- **Online flow** ($t: 1 \to 2$): $x_2 = \psi_\phi^{\text{on}}(x_1, 2 \mid s, a)$, mapping the offline distribution to the online transition distribution.

Theoretical guarantee (Theorem 3.1): when $W_2(p_G, p_{\text{on}}) > W_2(\hat{p}_{\text{off}}, p_{\text{on}})$ (i.e., the offline distribution is closer to the online distribution than the Gaussian prior is), the composite flow enjoys a tighter generalization error bound.

**Wasserstein distance estimation**: When the online flow is trained with OT-FM (optimal transport flow matching), the transport cost naturally approximates the 2-Wasserstein distance. The Monte Carlo estimator is:

$$\hat{\Delta}(s,a) = \left(\frac{1}{M} \sum_{j=1}^{M} \left\|\psi_\theta^{\text{off}}(x_0^{(j)}, 1 \mid s,a) - \psi_\phi^{\text{on}}(\psi_\theta^{\text{off}}(x_0^{(j)}, 1 \mid s,a), 2 \mid s,a)\right\|_2^2\right)^{1/2}$$

**Active exploration strategy**: An exploration bonus is added to the action selection objective of the standard actor-critic:

$$a = \arg\max_{a \in \mathcal{A}} \left[Q(s,a) + \beta\, \hat{\Delta}(s,a)\right]$$

where $\beta$ controls the exploration intensity. Theorem 3.5 shows that collecting more samples in high-shift regions further reduces the performance gap relative to the optimal policy.

**Data selection**: At each iteration, only offline transitions whose dynamics shift falls below the $\xi$-quantile threshold are added to the replay buffer:

$$\mathcal{B} = \{(s,a) \in \mathcal{D}_{\text{off}}: \hat{\Delta}(s,a) \leq \hat{\Delta}_{\xi\%}\} \cup \mathcal{D}_{\text{on}}$$

### Loss & Training

**Critic loss**:

$$\mathcal{L}_Q = \mathbb{E}_{\mathcal{D}_{\text{on}}}[(Q_\varsigma(s,a) - y)^2] + \mathbb{E}_{\mathcal{D}_{\text{off}}}[\mathbf{1}(\hat{\Delta}(s,a) \leq \hat{\Delta}_{\xi\%})(Q_\varsigma(s,a) - y)^2]$$

with target $y = r + \gamma Q_\varsigma(s', a') + \beta \hat{\Delta}(s,a)$.

**Policy loss** (policy improvement + behavioral cloning regularization):

$$\mathcal{L}_\pi = \mathbb{E}_{s, a \sim \pi_\varphi}[Q_\varsigma(s,a)] - \omega\, \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{off}}, \tilde{a} \sim \pi_\varphi}[\|a - \tilde{a}\|_2^2]$$

## Key Experimental Results

### Main Results

Three types of dynamics shift (morphology / kinematics / friction) × three data qualities (MR/M/ME) on Gym-MuJoCo, yielding 27 tasks in total:

| Method | Avg. Return ↑ | # Best Tasks | Gain over SAC |
|:--|:--:|:--:|:--:|
| SAC (online only) | 878 | 0 | — |
| BC-SAC | 1920 | — | +118.7% |
| H2O | 1783 | — | +103.1% |
| BC-VGDF | 1868 | — | +112.8% |
| BC-PAR | 1803 | — | +105.4% |
| **CompFlow** | **2193** | **19/27 best, 5 tied** | **+149.8%** |

CompFlow achieves the best or tied-best performance on 24 of 27 tasks, with an average return 14.2% higher than the strongest baseline (BC-SAC).

### Ablation Study

**Composite flow vs. direct flow**: On the validation set, the transition dynamics MSE of the composite flow remains consistently and significantly lower than that of a flow trained directly from a Gaussian prior throughout training, corroborating the theoretical prediction of Theorem 3.1.

**Data selection ratio $\xi$**: For $\xi \in \{20, 30, 50, 70\}$, intermediate values (30 or 50) are generally optimal. Too large a $\xi$ introduces low-quality transitions with high dynamics shift, while too small a $\xi$ discards useful data.

**Exploration intensity $\beta$**: Setting $\beta = 0$ (no exploration) yields the worst or second-worst performance across all five test tasks, confirming the necessity of active exploration. The optimal $\beta$ varies by task (friction tasks prefer larger $\beta$; morphology tasks prefer moderate $\beta$).

### Key Findings

1. Most existing baselines (H2O, BC-VGDF, BC-PAR) perform comparably to BC-SAC, indicating that they fail to effectively exploit the structural information present in offline data.
2. KL/mutual-information-based dynamics shift estimators break down under large shift or mismatched supports, whereas the Wasserstein distance remains consistently stable.
3. In a wildlife conservation experiment, CompFlow surpasses the strongest baseline by 8.8%, validating the practical utility of the approach in a real-world setting.
4. Both offline data utilization and active exploration are individually necessary; neither alone suffices.

## Highlights & Insights

- ⭐ The composite flow design is elegant and insightful—building the online flow on the output of the offline flow simultaneously leverages shared structural knowledge and enables precise measurement of the distributional shift.
- ⭐ Connecting flow matching, optimal transport, and Wasserstein distance as a unified tool for dynamics shift estimation is theoretically rigorous and practically effective.
- The active exploration strategy for high-shift regions is backed by solid theoretical support (Theorem 3.5 provides an explicit bound on the reduction in the performance gap).
- The method is highly general and can be integrated as a plug-in with any actor-critic algorithm (e.g., SAC).

## Limitations & Future Work

- The composite flow requires training two flow models (offline + online), incurring higher computational cost.
- Theorem 3.1 assumes the offline distribution is closer to the online distribution than the Gaussian prior—this may not hold when offline data deviates drastically from the online environment.
- Experiments are conducted primarily on MuJoCo continuous control tasks; high-dimensional observation spaces (e.g., image-based RL) and discrete action spaces are not evaluated.
- Both the exploration intensity $\beta$ and the data selection ratio $\xi$ require per-task hyperparameter tuning.

## Related Work & Insights

Introducing flow matching for dynamics shift estimation in cross-domain RL represents a novel perspective. Compared to DARC (classifier-based shift estimation) and H2O (KL-based Q-value penalty), the Wasserstein distance offers superior mathematical properties (metricity, continuity, no requirement for overlapping supports). The wildlife conservation experiment demonstrates the potential applicability of the approach to socially beneficial domains.

## Rating

⭐⭐⭐⭐ (4/5)

| Dimension | Rating |
|:--|:--:|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

The theoretical contributions are substantial and the composite flow architecture is cleverly designed. The experiments span 27 tasks plus a real-world scenario, providing strong empirical evidence. The primary shortcomings are insufficient analysis of computational overhead and the absence of evaluation in more complex environments (e.g., visual RL).

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](../../ICLR2026/image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)

<!-- RELATED:END -->
