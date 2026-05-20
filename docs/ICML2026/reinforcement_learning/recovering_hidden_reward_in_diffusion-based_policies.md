---
title: >-
  [Paper Note] Recovering Hidden Reward in Diffusion-Based Policies
description: >-
  [ICML 2026][Reinforcement Learning][diffusion policy] EnergyFlow explicitly parameterizes the score field of diffusion policy as the negative gradient of a scalar energy function…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "diffusion policy"
  - "IRL"
  - "energy-based model"
  - "conservative field"
  - "inverse RL shaping"
date: 2026-05-08
content_hash: 6f6d4594340b6bef
---

# Recovering Hidden Reward in Diffusion-Based Policies

**Conference**: ICML 2026  
**arXiv**: [2605.00623](https://arxiv.org/abs/2605.00623)  
**Code**: https://github.com/sotaagi/EnergyFlow  
**Area**: Diffusion Policy / Inverse Reinforcement Learning / Energy-Based Model / Robotic Manipulation  
**Keywords**: diffusion policy, IRL, energy-based model, conservative field, inverse RL shaping

## TL;DR
EnergyFlow explicitly parameterizes the score field of diffusion policy as the negative gradient of a scalar energy function, and proves that under maximum-entropy optimality, the score equals the gradient of the soft Q-function. This provides a scalar signal usable as a downstream RL shaping reward "for free" without adversarial optimization, while the conservative field constraint improves OOD generalization.

## Background & Motivation

**Background**: Diffusion policy (Chi 2023, Flow Policy) has become a mainstream approach for robotic manipulation—capable of modeling multi-modal expert action distributions and achieving strong BC performance on RoboMimic / Meta-World. However, it is essentially just behavior cloning, where the model only learns "what the expert did" without explicitly learning "why".

**Limitations of Prior Work**: Behavior cloning faces two cascading issues. First, poor OOD robustness—when test scenarios deviate from the demo distribution, action likelihood alone cannot reliably rank actions. Second, it cannot directly generate reward signals for downstream RL refinement. Existing IRL methods (max-ent IRL, GAIL, AIRL) can recover rewards from demos, but require either expensive MCMC (EBM), unstable adversarial training (GAIL/AIRL), or repeated inner-loop policy optimization.

**Key Challenge**: Diffusion policy already "knows" the expert's preferences in the latent space (the score is $\nabla \log p$), but prior work treats it as a sampler and discards the embedded reward signal; meanwhile, IRL methods know they need a reward but train a new EBM or adversarial discriminator from scratch.

**Goal**: Enable a single network to simultaneously (i) serve as a generative policy for action generation, (ii) expose a scalar energy usable as reward, and (iii) retain the strong BC performance of diffusion policy without extra training cost.

**Key Insight**: It is observed that the score function $\nabla_a \log \pi_E(a|s)$ is linearly related in log-space to the soft Q-function of a max-ent expert; if the score field is constrained to be the negative gradient of a scalar potential ("conservative field"), three benefits are obtained—valid energy, elimination of cyclic preference, and a tighter Rademacher complexity bound, thus improving OOD generalization.

**Core Idea**: Parameterize a scalar $E_\phi(s, a)$, obtain the score $\mathcal{S}_\phi = -\nabla_a E_\phi$ via autodiff, and train with denoising score matching. Thus, $E_\phi$ serves as both the generative potential for diffusion policy and the reward function for max-ent IRL.

## Method

### Overall Architecture

Input: expert demos $\mathcal{D} = \{(s_i, a_i)\}$.  
Training: Scalar network $E_\phi: \mathcal{A} \times \mathcal{S} \times [0, T] \to \mathbb{R}$, add variance-exploding noise $\sigma(t) = \sigma_{\min}^{1-t/T} \sigma_{\max}^{t/T}$ to actions, DSM objective $\mathcal{L}(\phi) = \mathbb{E}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$.  
Dual-purpose inference: (generation) start from $a_T \sim \mathcal{N}(0, \sigma^2(T) I)$ and run probability-flow ODE $da/dt = -\frac{1}{2} \frac{d[\sigma^2(t)]}{dt} \nabla_a E_\phi$; (reward) at time $\gamma = 10^{-3}$, take $E_\phi(a, s, \gamma)$ minus a state-dependent baseline as the shaping reward for SAC.

### Key Designs

1. **Score = Reward Gradient Equivalence (Theorem 3.3)**:

    - Function: Mathematically proves that score matching trained $E_\phi$ automatically recovers the expert's soft Q-function (up to a state-dependent constant).
    - Mechanism: Under max-ent optimality, $\pi_E(a|s) = \exp(Q^*(s,a)/\alpha) / Z(s)$. Taking the gradient w.r.t. action cancels the partition function $Z(s)$: $\nabla_a \log \pi_E = \nabla_a Q^* / \alpha$. Thus, if $-\nabla_a E_\phi \approx \nabla_a \log \pi_E$, then $E_\phi(a, s) = -Q^*(s,a)/\alpha + c(s)$. Corollary 3.4 further notes that $E_\phi$ actually recovers the soft advantage $A^{\text{soft}}(s,a) = Q^*(s,a) - V^*(s)$ (again, up to a state-only constant).
    - Design Motivation: This is the theoretical anchor of the framework—it shows that neither GAIL/AIRL's adversarial discriminator nor EBM's MCMC are needed; simply training a diffusion policy "for free" yields a reward signal. Unlike treating diffusion as a sampler (Diffusion Policy), here the same network is reinterpreted as energy.

2. **Conservative Field Constraint (Scalar Parameterization + Autodiff Score)**:

    - Function: Hardly ensures the score field is the gradient of some scalar potential ($\nabla \times \mathcal{S}_\phi = 0$), eliminates cyclic preference, and tightens the generalization bound.
    - Mechanism: Instead of directly regressing a vector score $\mathcal{S}_\phi: \mathbb{R}^{|s|+|a|} \to \mathbb{R}^{|a|}$, the network outputs a scalar $E_\phi$, and $\mathcal{S}_\phi = -\nabla_a E_\phi$ is obtained via autodiff. Theorem 3.6 gives the Rademacher complexity comparison: $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{unc}}) \leq \Lambda B \sqrt{d}/\sqrt{n}$ (unconstrained) vs $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{cons}}) \leq \Lambda L/\sqrt{n}$ (conservative), with the conservative version strictly tighter in high-dimensional action spaces. Lemma 3.8 further provides an OOD bound: the conservative version's complexity term is $\mathcal{O}(M \Lambda L / \sqrt{n})$ instead of $\mathcal{O}(M \Lambda B \sqrt{d}/\sqrt{n})$.
    - Design Motivation: Standard diffusion policy outputs a vector score without guaranteeing conservativeness, meaning the learned implicit "energy" may form cyclic preferences (e.g., $a_1 \to a_2 \to a_3 \to a_1$), violating the transitivity axiom of rational decision (Jiang 2011), making rewards ill-defined. The conservative constraint ensures both mathematical correctness (legitimizing reward extraction) and provides a useful inductive bias (significantly reducing complexity in high-dimensional action spaces). Remark 3.7 also notes that deep networks can satisfy the Lipschitz constraint via spectral normalization, making $L$ controllable.

3. **Centered Shaping Reward (Removing State-Dependent Bias)**:

    - Function: Using raw $E_\phi$ as reward introduces a state-dependent bias $c(s)$, leading to high variance; define $\tilde{r}_\phi(a, s) = -(E_\phi(a, s, \gamma) - \mathbb{E}_{a' \sim \mathcal{N}(0, I)}[E_\phi(a', s, \gamma)])$ to cancel the bias.
    - Mechanism: Proposition 3.9 proves that raw $E_\phi$ guarantees correct within-state action ranking ($\arg\min_a E_\phi = \arg\max_a Q^*$) but is unreliable for cross-state comparison. Remark 3.10 points out that the state-only bias does not satisfy the potential-based reward shaping (PBRS, Ng 1999) form and may alter the sequential optimal policy; however, it does not affect within-state action selection. Thus, subtracting a state-dependent baseline (Monte Carlo average over $M = 16$ samples $a' \sim \mathcal{N}(0, I)$) removes the bias.
    - Design Motivation: Figure 4 directly verifies—using raw $E_\phi$ as SAC reward causes the agent to get stuck in common states (since high likelihood = low energy = high reward, but high likelihood does not equal progress), leading to early plateau; the centered version makes the reward reflect "which action to choose in the current state" rather than "which state is frequently visited", and the training curve matches the oracle dense reward. The combination of Centered Energy + Sparse is optimal—dense shaping guides early exploration, sparse ensures final task alignment.

### Loss & Training

DSM loss $\mathcal{L}(\phi) = \mathbb{E}_{t, a_0, \varepsilon}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$, where $a_t = a_0 + \sigma(t) \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$, $t \sim \mathcal{U}[0, T]$, $\sigma_{\min} = 0.01$, $\sigma_{\max} = 10$, $T = 1$. $\lambda(t) = \sigma^2(t)$ ensures balanced contribution across noise scales. Theorem 3.11 provides a bound on the propagation of score error $\eta$ to action preference: $|\Delta E_\phi(a, a') - \Delta E^*(a, a')| \leq \eta \cdot \|a - a'\|_2$, i.e., linear graceful degradation. Downstream RL uses SAC + centered shaping reward, optionally combined with sparse task signals.

## Key Experimental Results

### Main Results

RoboMimic (ph, 5 tasks, 3 seeds):

| Method | Lift | Square | Transport | ToolHang | Avg |
|--------|------|--------|-----------|----------|-----|
| LSTM-GMM | 97.8 | 64.3 | 65.6 | 46.0 | 69.0 |
| Diffusion Policy | 100.0 | 93.5 | 85.9 | 77.2 | 91.2 |
| Flow Policy | 99.6 | 91.8 | 83.6 | 74.8 | 89.6 |
| EBT-Policy | 96.2 | 78.4 | 72.4 | 58.6 | 78.8 |
| EBIL / NEAR / IQ-Learn | 92-95 | 58-68 | 48-58 | 32-44 | 61-70 |
| Implicit BC | 70.9 | 10.2 | 0.0 | 0.0 | 22.4 |
| **EnergyFlow** | **100.0** | **95.3** | **89.4** | **84.2** | **93.8** |

Meta-World (5 tasks): EnergyFlow 92.5% vs Diffusion Policy 90.7%, with gains concentrated on harder tasks (Assembly +6.2, ToolHang +7.0 on RoboMimic).  
Real robot (AGIBOT G1): 100% success rate on Bottle / Drawer tasks (3 initial positions × 20 rollouts).

### Ablation Study

Comparison of downstream SAC reward sources (RoboMimic Square / Transport):

| Reward Source | Converged Success | Notes |
|---------------|-------------------|-------|
| Sparse only | Slow and noisy | Signal only on task success |
| Raw $E_\phi$ | Early plateau | Likelihood ≠ progress, biased by state density |
| Centered $E_\phi$ | Near oracle dense | After baseline subtraction, reflects within-state action preference |
| Centered + Sparse | Best | Dense shaping + sparse anchors task |

OOD perturbation (initial position perturbation level 0/S/M/L): EnergyFlow significantly outperforms Diffusion Policy and Flow Policy at M/L perturbation levels, validating the geometric regularization effect of the conservative constraint.  
$\gamma$ sensitivity (reward extraction time): Performance stable for $\gamma \in [10^{-4}, 10^{-2}]$ (94-95%), drops for $\gamma \geq 0.1$ (data distribution corrupted by noise, score approximation fails).  
Latency (RoboMimic Square, A100): EnergyFlow $K=20$ 11.4ms (95.3%) vs Diffusion Policy 100 DDPM 32.4ms (93.5%) vs Implicit BC 50 Langevin 52.4ms (10.2%)—the conservative EBM outperforms baselines without sacrificing speed.

### Key Findings
- Conservative constraint contributes most on hard tasks (ToolHang +7 vs Diffusion Policy), confirming Lemma 3.8—the harder the task and the more complex the action space, the greater the benefit.
- Centered shaping is key for reward extraction—training RL with raw $E_\phi$ stalls, while centered version matches oracle performance.
- Explicit EBM (Implicit BC) almost completely fails on RoboMimic (Transport / ToolHang 0%), but EnergyFlow achieves SOTA, indicating the issue is not with the EBM paradigm itself but with the training objective—score matching + conservative parameterization is much more robust than traditional contrastive divergence.
- Inference latency is on par with Flow Policy, showing the overhead of autodiff for the conservative constraint is negligible.
- Real robot transfer works directly (100% on AGIBOT G1), indicating the geometric regularization from the conservative constraint is genuinely beneficial.

## Highlights & Insights
- The reinterpretation that **"score matching = max-ent IRL, reward for free"** is exceptionally clean—previously, diffusion policy and IRL were treated as separate streams, but this work connects them with a single observation: "taking the gradient w.r.t. $a$ cancels $Z(s)$". Such reinterpretation papers are highly valuable.
- **Scalar parameterization + autodiff score** adds almost zero engineering cost (just change the network's final head to 1-d and add a $\nabla_a$ in forward), yet simultaneously yields (i) valid energy, (ii) conservative field, and (iii) tighter generalization bound.
- The **centered shaping** baseline trick solves the classic problem "high likelihood ≠ task progress"—using $\log \pi_E$ directly as reward causes the agent to be attracted to high-density regions, while the centered version turns the signal into "relative within-state preference", an idea transferable to any likelihood-based reward shaping.
- The **OOD benefits of the conservative constraint** are supported both theoretically (Lemma 3.8 + Rademacher complexity) and empirically (significant outperformance at perturbation level L), providing strong evidence.
- Inference latency matches Flow Policy, rehabilitating EBM from "too expensive to use" back to a mainstream candidate.

## Limitations & Future Work
- The max-ent optimality assumption requires the expert to truly act according to a Boltzmann distribution, which may fail for noisy demos or multi-expert datasets.
- What is recovered is the soft advantage $A^{\text{soft}}$ (up to a state-only bias), making cross-state comparison unreliable; in sequential MDPs, the state-only bias may alter the optimal policy, which the authors acknowledge but do not provide an alternative for.
- The tightness of the conservative field theoretical bound (Theorem 3.6), i.e., $L \ll B \sqrt{d}$, relies on spectral normalization in deep networks, but whether this is satisfied in practice is not quantitatively checked.
- Real robot experiments only cover 2 tasks and 3 initial positions, limiting statistical significance.
- Validation is only on manipulation; transferability to other embodied tasks such as locomotion or continuous control is unknown.
- The baseline uses 16 MC samples for estimation; the impact of estimation variance on RL training stability is not systematically ablated.

## Related Work & Insights
- **vs Diffusion Policy (Chi 2023)**: This work is a strict superset—backbone, training objective, and generation latency are similar, but it additionally outputs a reward scalar and achieves stronger BC performance; this "more compact hypothesis class + more outputs" win-win design is worth emulating.
- **vs Implicit BC (Florence 2021) / EBT-Policy (Davies 2025)**: Both are EBM-based policies, but the former uses contrastive divergence and is unstable (almost complete failure on RoboMimic), EBT uses a transformer but still treats energy only as a decision score; EnergyFlow uses score matching and treats energy as reward.
- **vs EBIL (Liu 2021) / NEAR (Diwan 2025)**: Both use EBM for IRL in a two-stage pipeline (learn energy, then RL), but EBIL/NEAR lack the conservative constraint, have poor performance, and do not use energy as a generative policy; EnergyFlow unifies both.
- **vs Adversarial IRL (GAIL, AIRL, AIRL)**: Completely avoids adversarial training, using only score matching loss for stable training.
- **vs Wang & Du 2025 / Balcerak 2025**: Also observe the connection between diffusion and EBM, but EnergyFlow further leverages this connection for IRL reward extraction.

## Rating
- Novelty: ⭐⭐⭐⭐ The reinterpretation "score = soft Q gradient + conservative constraint" is elegant, though the EBM-policy / DSM components are known.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 simulated tasks + real robot + reward quality directly validated with SAC + OOD perturbation + 5 sensitivity experiments, comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory section is clear with theorem-proof structure, three Remarks clarify assumption boundaries; Figure 1 visually distinguishes EnergyFlow from Diffusion Policy.
- Value: ⭐⭐⭐⭐⭐ Directly adds reward output and OOD robustness to diffusion policy with unchanged inference speed, a highly practical upgrade for the robotics learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](../../AAAI2026/reinforcement_learning/chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)
- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](../../NeurIPS2025/reinforcement_learning/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[ICML 2026\] CAMEL: Confidence-Gated Reflection for Reward Modeling](camel_confidence-gated_reflection_for_reward_modeling.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[AAAI 2026\] Formal Verification of Diffusion Auctions](../../AAAI2026/reinforcement_learning/formal_verification_of_diffusion_auctions.md)

</div>

<!-- RELATED:END -->
