---
title: >-
  [Paper Note] Recovering Hidden Reward in Diffusion-Based Policies
description: >-
  [ICML 2026][Image Generation][diffusion policy] EnergyFlow explicitly parameterizes the score field of a diffusion policy as the negative gradient of a scalar energy function. It demonstrates that under maximum-entropy o…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "diffusion policy"
  - "IRL"
  - "energy-based model"
  - "conservative field"
  - "inverse RL shaping"
date: 2026-05-08
content_hash: 4664dfa5f03e9fd4
---

# Recovering Hidden Reward in Diffusion-Based Policies

**Conference**: ICML 2026  
**arXiv**: [2605.00623](https://arxiv.org/abs/2605.00623)  
**Code**: https://github.com/sotaagi/EnergyFlow  
**Area**: Diffusion Policy / Inverse Reinforcement Learning / Energy-Based Models / Robotic Manipulation  
**Keywords**: diffusion policy, IRL, energy-based model, conservative field, inverse RL shaping  

## TL;DR
EnergyFlow explicitly parameterizes the score field of a diffusion policy as the negative gradient of a scalar energy function. It demonstrates that under maximum-entropy optimality, the score equals the gradient of the soft Q-function. This provides a scalar signal for downstream RL shaping rewards "for free" without adversarial optimization, while conservative field constraints improve OOD generalization.

## Background & Motivation

**Background**: Diffusion policies (Chi 2023, Flow Policy) have become the mainstream for robotic manipulation due to their ability to model multimodal expert action distributions and strong BC performance on RoboMimic / Meta-World. However, they are essentially behavior cloning models that learn "what the expert did" without explicitly learning "why."

**Limitations of Prior Work**: Behavior cloning suffers from two interconnected issues. First, poor OOD robustness—once the test scenario deviates from the demo distribution, action likelihood alone cannot reliably rank actions. Second, it cannot directly generate reward signals for downstream RL refinement. Existing IRL methods (max-ent IRL, GAIL, AIRL) can recover rewards from demos but require either expensive MCMC (EBM), unstable adversarial training (GAIL/AIRL), or repeated inner-loop policy optimization.

**Key Challenge**: Diffusion policies already "know" expert preferences in the latent space (score is $\nabla \log p$), but authors typically treat them only as samplers, discarding the inherent reward signals. Conversely, IRL methods aim for rewards but often restart by training EBMs or adversarial discriminators from scratch.

**Goal**: To enable a single network to simultaneously (i) act as a generative policy for action production, (ii) expose a scalar energy usable as a reward, and (iii) maintain the BC strength of diffusion policies without additional training costs.

**Key Insight**: The score function $\nabla_a \log \pi_E(a|s)$ and the soft Q-function of a max-ent expert are linearly related in log space. By restricting the score field to the negative gradient of a scalar potential (i.e., a "conservative field"), one obtains a valid energy, eliminates cyclic preferences, and tightens Rademacher complexity to improve OOD generalization.

**Core Idea**: Parameterize a scalar $E_\phi(s, a)$ such that the score $\mathcal{S}_\phi = -\nabla_a E_\phi$ is obtained via autodiff and trained using denoising score matching (DSM). Thus, $E_\phi$ serves both as the generative potential for the diffusion policy and the reward function for max-ent IRL.

## Method

### Overall Architecture

Input: Expert demos $\mathcal{D} = \{(s_i, a_i)\}$.  
Training: A scalar network $E_\phi: \mathcal{A} \times \mathcal{S} \times [0, T] \to \mathbb{R}$ is trained by adding noise to actions using variance-exploding noise $\sigma(t) = \sigma_{\min}^{1-t/T} \sigma_{\max}^{t/T}$. The DSM objective is $\mathcal{L}(\phi) = \mathbb{E}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$.  
Dual-purpose Inference: (Generation) Start from $a_T \sim \mathcal{N}(0, \sigma^2(T) I)$ and run the probability-flow ODE $da/dt = -\frac{1}{2} \frac{d[\sigma^2(t)]}{dt} \nabla_a E_\phi$. (Reward) At time $\gamma = 10^{-3}$, take $E_\phi(a, s, \gamma)$ minus a state-dependent baseline as a shaping reward for SAC.

### Key Designs

1.  **Score = Reward Gradient Equivalence (Theorem 3.3)**:
    - **Function**: Mathematically proves that $E_\phi$ trained via score matching automatically recovers the expert's soft Q-function (up to a state-dependent constant).
    - **Mechanism**: Under the max-ent optimality assumption, $\pi_E(a|s) = \exp(Q^*(s,a)/\alpha) / Z(s)$. Taking the gradient with respect to the action eliminates the partition function $Z(s)$: $\nabla_a \log \pi_E = \nabla_a Q^* / \alpha$. Since $-\nabla_a E_\phi \approx \nabla_a \log \pi_E$, then $E_\phi(a, s) = -Q^*(s,a)/\alpha + c(s)$. Corollary 3.4 further notes that $E_\phi$ actually recovers the soft advantage $A^{\text{soft}}(s,a) = Q^*(s,a) - V^*(s)$.
    - **Design Motivation**: This is the theoretical anchor. It suggests that a reward signal can be obtained "for free" simply by training a diffusion policy, avoiding GAIL/AIRL discriminators or MCMC. Unlike standard Diffusion Policies that use the model as a sampler, this reinterprets the network as an energy function.

2.  **Conservative Field Constraint (Scalar Parameterization + Autodiff Score)**:
    - **Function**: Hard-guarantees that the score field is the gradient of a scalar potential ($\nabla \times \mathcal{S}_\phi = 0$), eliminating cyclic preferences and tightening generalization bounds.
    - **Mechanism**: Instead of directly regressing a vector score $\mathcal{S}_\phi: \mathbb{R}^{|s|+|a|} \to \mathbb{R}^{|a|}$, the network outputs a scalar $E_\phi$, and $\mathcal{S}_\phi = -\nabla_a E_\phi$ is computed via autodiff. Theorem 3.6 compares Rademacher complexity: $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{unc}}) \leq \Lambda B \sqrt{d}/\sqrt{n}$ (unconstrained) vs $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{cons}}) \leq \Lambda L/\sqrt{n}$ (conservative), where the conservative version is strictly tighter in high-dimensional action spaces. Lemma 3.8 provides an OOD bound showing the conservative version's complexity is $\mathcal{O}(M \Lambda L / \sqrt{n})$.
    - **Design Motivation**: Standard diffusion policies do not guarantee conservativeness, meaning the implicit "energy" could form cyclic preferences ($a_1 \to a_2 \to a_3 \to a_1$), violating transitivity axioms of rational decision making. Conservative constraints ensure mathematical correctness for reward extraction and serve as a useful inductive bias.

3.  **Centered Shaping Reward (Removing State-Dependent Bias)**:
    - **Function**: Using raw $E_\phi$ as a reward introduces high variance due to state-dependent bias $c(s)$. The centered reward is defined as $\tilde{r}_\phi(a, s) = -(E_\phi(a, s, \gamma) - \mathbb{E}_{a' \sim \mathcal{N}(0, I)}[E_\phi(a', s, \gamma)])$.
    - **Mechanism**: Proposition 3.9 proves that while raw $E_\phi$ maintains correct within-state action ranking, cross-state comparisons are unreliable. Subtracting a state-dependent baseline (estimated via $M=16$ Monte Carlo samples) removes the bias.
    - **Design Motivation**: Experiments show that using raw $E_\phi$ causes SAC agents to plateau early because high likelihood (low energy) does not necessarily equate to task progress. Centering ensures the reward reflects "which action to choose in the current state" rather than "which state is frequently visited."

### Loss & Training
DSM loss: $\mathcal{L}(\phi) = \mathbb{E}_{t, a_0, \varepsilon}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$, where $a_t = a_0 + \sigma(t) \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$, and $t \sim \mathcal{U}[0, T]$. Setting $\lambda(t) = \sigma^2(t)$ ensures balanced contributions across noise scales. Theorem 3.11 provides a bound for action preference error propagation: $|\Delta E_\phi(a, a') - \Delta E^*(a, a')| \leq \eta \cdot \|a - a'\|_2$.

## Key Experimental Results

### Main Results

RoboMimic (5 tasks, 3 seeds):

| Method | Lift | Square | Transport | ToolHang | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LSTM-GMM | 97.8 | 64.3 | 65.6 | 46.0 | 69.0 |
| Diffusion Policy | 100.0 | 93.5 | 85.9 | 77.2 | 91.2 |
| Flow Policy | 99.6 | 91.8 | 83.6 | 74.8 | 89.6 |
| EBT-Policy | 96.2 | 78.4 | 72.4 | 58.6 | 78.8 |
| EBIL / NEAR / IQ-Learn | 92-95 | 58-68 | 48-58 | 32-44 | 61-70 |
| Implicit BC | 70.9 | 10.2 | 0.0 | 0.0 | 22.4 |
| **EnergyFlow (Ours)** | **100.0** | **95.3** | **89.4** | **84.2** | **93.8** |

Meta-World: EnergyFlow 92.5% vs Diffusion Policy 90.7%. Improvements are concentrated on difficult tasks (ToolHang +7.0 on RoboMimic).  
Real robot (AGIBOT G1): 100% success rate on Bottle/Drawer tasks (20 rollouts across 3 initial positions).

### Ablation Study

Comparison of downstream SAC reward sources (RoboMimic Square/Transport):

| Reward Source | Convergence Success | Note |
| :--- | :--- | :--- |
| Sparse only | Slow/Noisy | Signal only on success |
| Raw $E_\phi$ | Early plateau | Likelihood $\neq$ progress; biased toward state density |
| Centered $E_\phi$ | Near oracle dense | Reflects within-state action preference after removing baseline |
| Centered + Sparse | Best | Dense shaping guides exploration; sparse anchors task |

OOD Perturbations: EnergyFlow significantly outperforms Diffusion Policy and Flow Policy under medium/large initial position perturbations, validating the geometric regularization of conservative constraints.

### Key Findings
- Conservative constraints provide the most gain on difficult tasks (ToolHang +7.0), confirming that higher-dimensional action spaces benefit more from complexity reduction (Lemma 3.8).
- Centered shaping is the key to reward extraction; raw $E_\phi$ fails in RL training, while centered $E_\phi$ matches oracle performance.
- Explicit EBMs like Implicit BC fail on complex tasks (0% on Transport/ToolHang), but EnergyFlow achieves SOTA. This suggests the issue lies in training objectives (contrastive divergence vs. score matching) rather than the EBM paradigm itself.
- Inference latency is comparable to Flow Policy, proving autodiff overhead is negligible.

## Highlights & Insights
- The reinterpretation of "**score matching = max-ent IRL**" is elegant. It bridges diffusion policies and IRL by simply noting that gradients with respect to $a$ eliminate the partition function $Z(s)$.
- **Scalar parameterization + autodiff score** incurs nearly zero extra engineering cost but provides a valid energy, a conservative field, and tighter generalization bounds simultaneously.
- **Centered shaping** solves the "likelihood $\neq$ task progress" problem. If $\log \pi_E$ is used as a reward directly, the agent is trapped in high-density regions. Centering transforms the signal into a "relative within-state preference."
- Conservative constraints provide both theoretical support (Theorem 3.6) and experimental validation for OOD robustness.

## Limitations & Future Work
- The max-ent optimality assumption requires experts to act according to a Boltzmann distribution; it may fail for noisy demos or multi-expert sets.
- The recovered signal is the soft advantage $A^{\text{soft}}$; cross-state comparisons are unreliable, and state-only bias may theoretically alter the optimal policy in sequential MDPs.
- Tightness of the conservative bound depends on Lipschitz constants $L$ maintained via spectral normalization, which was not quantitatively verified during training.
- Baseline estimation relies on 16 MC samples; the impact of estimation variance on RL stability warrants further study.

## Related Work & Insights
- **vs Diffusion Policy (Chi 2023)**: Ours is a strict superset. It shares the same backbone and training speed but adds a reward output and superior BC performance.
- **vs Implicit BC / EBT-Policy**: These are EBM-based. EnergyFlow's score matching is more stable than contrastive divergence (which causes Implicit BC to fail on RoboMimic).
- **vs EBIL / NEAR**: These use EBMs for two-stage IRL but lack conservative constraints and generative capabilities. EnergyFlow unifies both.
- **vs Adversarial IRL**: Entirely bypasses adversarial training, resulting in more stable single-objective score matching.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The reinterpretation of score/energy for IRL is excellent, though individual components like DSM are known.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 10 simulation tasks, real robots, RL reward validation, and OOD perturbations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear theorem-proof structure and insightful remarks.
- **Value**: ⭐⭐⭐⭐⭐ Practical upgrade for diffusion policies by adding reward output and OOD robustness with zero inference cost penalty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](../../ICLR2026/image_generation/improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICML 2026\] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](../../ICLR2026/image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICLR 2026\] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations](../../ICLR2026/image_generation/contractive_diffusion_policies_robust_action_diffusion_via_contractive_score-bas.md)

</div>

<!-- RELATED:END -->
