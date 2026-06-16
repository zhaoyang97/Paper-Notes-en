---
title: >-
  [Paper Note] Recovering Hidden Reward in Diffusion-Based Policies
description: >-
  [ICML 2026][Image Generation][diffusion policy] EnergyFlow explicitly parameterizes the score field of a diffusion policy as the negative gradient of a scalar energy function. It proves that under maximum-entropy optimality, the score equals the gradient of the soft Q-function, thereby providing a "free" scalar signal for downstream RL reward shaping without adversa
tags:
  - ICML 2026
  - Image Generation
  - diffusion policy
  - IRL
  - energy-based model
date: 2026-05-08
content_hash: 8191e436536051b5
---
# Recovering Hidden Reward in Diffusion-Based Policies

**Conference**: ICML 2026  
**arXiv**: [2605.00623](https://arxiv.org/abs/2605.00623)  
**Code**: https://github.com/sotaagi/EnergyFlow  
**Area**: Diffusion Policy / Inverse Reinforcement Learning / Energy-Based Models / Robot Manipulation  
**Keywords**: diffusion policy, IRL, energy-based model, conservative field, inverse RL shaping

## TL;DR
EnergyFlow explicitly parameterizes the score field of a diffusion policy as the negative gradient of a scalar energy function. It proves that under maximum-entropy optimality, the score equals the gradient of the soft Q-function, thereby providing a "free" scalar signal for downstream RL reward shaping without adversarial optimization, while the conservative field constraint improves OOD generalization.

## Background & Motivation

**Background**: Diffusion policies (Chi 2023, Flow Policy) have become the mainstream for robot manipulation due to their ability to model multimodal expert action distributions and their strong BC performance on RoboMimic/Meta-World. However, they are essentially imitation learning; the models learn "what the expert did" without explicitly learning "why they did it."

**Limitations of Prior Work**: Behavioral cloning faces two related issues. First, poor OOD robustness—once the test scenario deviates from the demo distribution, action likelihood alone cannot reliably rank actions. Second, it cannot directly generate reward signals to refine downstream RL. Existing IRL methods (max-ent IRL, GAIL, AIRL) can recover rewards from demos but require either expensive MCMC (EBM), unstable adversarial training (GAIL/AIRL), or repeated inner-loop policy optimization.

**Key Challenge**: Diffusion policies already "know" expert preferences in their latent space (the score is $\nabla \log p$), but authors typically treat them only as samplers, discarding the inherent reward signal. Conversely, IRL methods aim for rewards but often retrain an EBM or adversarial discriminator from scratch.

**Goal**: To enable a single network to simultaneously (i) act as a generative policy for action production, (ii) expose a scalar energy usable as a reward, and (iii) maintain the BC strength of diffusion policies without introducing extra training costs.

**Key Insight**: It is observed that the score function $\nabla_a \log \pi_E(a|s)$ and the soft Q-function of a max-entropy expert are linearly related in log space. By restricting the score field to be the negative gradient of a scalar potential (i.e., a "conservative field"), three things are achieved simultaneously: a valid energy, elimination of cyclic preferences, and tightened Rademacher complexity for improved OOD generalization.

**Core Idea**: Parameterize a scalar $E_\phi(s, a)$ and obtain the score $\mathcal{S}_\phi = -\nabla_a E_\phi$ via autodiff, trained using denoising score matching. Thus, $E_\phi$ serves as both the generative potential of the diffusion policy and the reward function for max-ent IRL.

## Method

### Overall Architecture

EnergyFlow addresses the issue where diffusion policies discard inherent reward signals while being used as pure samplers. The architecture is straightforward: instead of directing the network to output a vector score, it outputs a scalar energy $E_\phi(s, a)$, defining the score as its negative gradient with respect to the action: $\mathcal{S}_\phi = -\nabla_a E_\phi$. This allows the same network to generate actions via a probability-flow ODE and extract $E_\phi$ at a low-noise timestep as a reward for downstream SAC, all while training with a single denoising score matching loss.

### Key Designs

**1. Score = Reward Gradient Equivalence: Reinterpreting Diffusion Score as Soft Q Gradient**

This serves as the theoretical anchor, addressing why training a diffusion policy yields a reward signal. Under the max-entropy optimality hypothesis, the expert policy follows a Boltzmann distribution $\pi_E(a|s) = \exp(Q^*(s,a)/\alpha) / Z(s)$. The key observation is that when taking the gradient with respect to the action, the partition function $Z(s)$ is independent of $a$ and cancels out: $\nabla_a \log \pi_E = \nabla_a Q^* / \alpha$. Consequently, if score matching achieves $-\nabla_a E_\phi \approx \nabla_a \log \pi_E$, then $E_\phi(a, s) = -Q^*(s,a)/\alpha + c(s)$ (Theorem 3.3). This means $E_\phi$ automatically recovers the expert's soft Q-function up to a state-only constant. Corollary 3.4 further notes that what is effectively recovered is the soft advantage $A^{\text{soft}}(s,a) = Q^*(s,a) - V^*(s)$. This connects diffusion policies and max-ent IRL without requiring adversarial discriminators or MCMC.

**2. Conservative Field Constraint: Scalar Parameterization and Generalization Bounds**

Standard diffusion policies regress a vector score $\mathcal{S}_\phi: \mathbb{R}^{|s|+|a|} \to \mathbb{R}^{|a|}$ directly, which does not guarantee a conservative field. The learned implicit energy might form cyclic preferences (e.g., $a_1 \succ a_2 \succ a_3 \succ a_1$), violating the transitivity axiom of rational decision-making (Jiang 2011), making rewards ill-defined. EnergyFlow sets the network output to a scalar $E_\phi$ and uses autodiff to find $\mathcal{S}_\phi = -\nabla_a E_\phi$, ensuring $\nabla \times \mathcal{S}_\phi = 0$ by construction. This constraint provides a useful inductive bias in high-dimensional action spaces: Theorem 3.6 shows the Rademacher complexity $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{unc}}) \le \Lambda B \sqrt{d}/\sqrt{n}$ for the unconstrained version vs. $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{cons}}) \le \Lambda L/\sqrt{n}$ for the conservative version. Lemma 3.8 extends this to OOD bounds, reducing the complexity term from $\mathcal{O}(M \Lambda B \sqrt{d}/\sqrt{n})$ to $\mathcal{O}(M \Lambda L / \sqrt{n})$.

**3. Centered Shaping Reward: Removing the Likelihood vs. Progress Trap**

Using raw $E_\phi$ as a reward is problematic due to the state-only bias $c(s)$, where high likelihood (low energy) does not necessarily imply task progress. Proposition 3.9 proves that raw $E_\phi$ only guarantees correct within-state action ranking ($\arg\min_a E_\phi = \arg\max_a Q^*$) but is unreliable for cross-state comparisons. Remark 3.10 notes this bias does not fit the potential-based reward shaping (PBRS, Ng 1999) format and may alter the optimal policy in sequential MDPs. The solution is the centered reward $\tilde{r}_\phi(a, s) = -(E_\phi(a, s, \gamma) - \mathbb{E}_{a' \sim \mathcal{N}(0, I)}[E_\phi(a', s, \gamma)])$, where the state-dependent mean is estimated via Monte Carlo sampling. This transforms the signal from "how common is this state" to "which action is better in this state."

### Loss & Training

Training utilizes a single denoising score matching loss:
$$\mathcal{L}(\phi) = \mathbb{E}_{t, a_0, \varepsilon}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$$
where $a_t = a_0 + \sigma(t) \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$, and $t \sim \mathcal{U}[0, T]$. A variance-exploding schedule is used for noise: $\sigma(t) = \sigma_{\min}^{1-t/T} \sigma_{\max}^{t/T}$. Generation starts from $a_T \sim \mathcal{N}(0, \sigma^2(T) I)$ following the probability-flow ODE:
$$da/dt = -\frac{1}{2} \frac{d[\sigma^2(t)]}{dt} \nabla_a E_\phi$$
Reward extraction occurs at a small noise timestep $\gamma = 10^{-3}$. Downstream RL utilizes SAC with the centered shaping reward, optionally augmented with a sparse task signal.

## Key Experimental Results

### Main Results

RoboMimic (ph, 5 tasks, 3 seeds):

| Method | Lift | Square | Transport | ToolHang | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|
| LSTM-GMM | 97.8 | 64.3 | 65.6 | 46.0 | 69.0 |
| Diffusion Policy | 100.0 | 93.5 | 85.9 | 77.2 | 91.2 |
| Flow Policy | 99.6 | 91.8 | 83.6 | 74.8 | 89.6 |
| EBT-Policy | 96.2 | 78.4 | 72.4 | 58.6 | 78.8 |
| EBIL / NEAR / IQ-Learn | 92-95 | 58-68 | 48-58 | 32-44 | 61-70 |
| Implicit BC | 70.9 | 10.2 | 0.0 | 0.0 | 22.4 |
| **Ours (EnergyFlow)** | **100.0** | **95.3** | **89.4** | **84.2** | **93.8** |

EnergyFlow outperformed Diffusion Policy by 1.8% on average, with significant Gains in difficult tasks (e.g., ToolHang +7.0). Real robot deployment (AGIBOT G1) achieved 100% success rates on Bottle and Drawer tasks.

### Ablation Study

Comparison of SAC reward sources (RoboMimic Square / Transport):

| Reward Source | Convergence Success | Description |
|:---|:---:|:---|
| Sparse only | Slow/Noisy | Signal only upon task completion |
| Raw $E_\phi$ | Early plateau | Likelihood $\neq$ progress; biased toward state density |
| Centered $E_\phi$ | Near oracle dense | Reflects within-state action preference |
| Centered + Sparse | Best | Dense shaping + sparse task anchoring |

### Key Findings
- The conservative constraint provides the most Benefit in difficult tasks, supporting the theory that higher action dimensions $d$ gain more from reduced complexity.
- Centered shaping is critical; raw $E_\phi$ often causes RL training to stall, while centering allows it to match oracle dense rewards.
- Explicit EBMs (Implicit BC) failed on RoboMimic (0% on Transport), but EnergyFlow reached SOTA, suggesting that the issue lies not in the EBM paradigm but in the training objective (DSM is superior to contrastive divergence).
- Inference latency is comparable to Flow Policy, proving the autodiff overhead is negligible.
- OOD robustness is significantly higher than baseline diffusion models under large perturbations.

## Highlights & Insights
- The reinterpretation that "score matching = max-ent IRL" is elegant, connecting two previously independent research streams via the simple observation that the partition function $Z(s)$ cancels during gradient calculation.
- Scalar parameterization with autodiff provides a "free lunch": it ensures a valid potential and a tighter generalization bound with almost zero engineering cost.
- Centered shaping solves the "high likelihood $\neq$ task progress" trap; this logic is transferable to any likelihood-based reward shaping method.
- Real-robot transfer worked immediately, likely due to the geometric regularization provided by the conservative constraint.

## Limitations & Future Work
- The max-ent optimality assumption requires experts to act according to a Boltzmann distribution; it may fail with noisy demos or mixed-expert datasets.
- The recovered signal is the soft advantage $A^{\text{soft}}$, making cross-state comparisons unreliable; state-only biases might still affect optimal policies in certain sequential MDPs.
- Theoretical bounds depend on Lipschitz constants, which require spectral normalization—something not always quantified in practice.
- Real-robot experiments were limited in scale (2 tasks, 3 positions).

## Related Work & Insights
- **vs. Diffusion Policy (Chi 2023)**: Ours is a strict superset, adding a reward output and stronger BC performance with similar latency.
- **vs. Implicit BC (Florence 2021)**: While both are EBMs, Implicit BC is unstable due to contrastive divergence, whereas EnergyFlow uses stable score matching.
- **vs. Adversarial IRL**: EnergyFlow avoids unstable adversarial training entirely.
- **vs. EBIL / NEAR**: These use two-stage pipelines; EnergyFlow unifies generative policy and energy recovery into a single model.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](../../ICLR2026/image_generation/improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICML 2026\] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)
- [\[AAAI 2026\] PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork](../../AAAI2026/image_generation/padiff_predictive_and_adaptive_diffusion_policies_for_ad_hoc_teamwork.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](../../ICLR2026/image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
