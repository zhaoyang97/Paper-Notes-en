---
title: >-
  [Paper Note] Conditional Diffusion Sampling
description: >-
  [ICML 2026][Multimodal VLM][Parallel Tempering] This paper proposes Conditional Diffusion Sampling (CDS): by deriving a class of conditional interpolants…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Parallel Tempering"
  - "Conditional Interpolants"
  - "closed-form SDE"
  - "Multi-modal Sampling"
  - "Training-free"
date: 2026-05-08
content_hash: 0abf8c101282086a
---

# Conditional Diffusion Sampling

**Conference**: ICML 2026  
**arXiv**: [2605.04013](https://arxiv.org/abs/2605.04013)  
**Code**: https://github.com/Franblueee/conditional_diffusion_sampling  
**Area**: Sampling Algorithms / Diffusion Models; MCMC; Bayesian Inference  
**Keywords**: Parallel Tempering, Conditional Interpolants, closed-form SDE, Multi-modal Sampling, Training-free

## TL;DR
This paper proposes Conditional Diffusion Sampling (CDS): by deriving a class of conditional interpolants, an **exact closed-form SDE** (without requiring neural network fitting) is obtained for unnormalized target distributions. Parallel Tempering (PT) is then used to efficiently sample the initial distribution of this SDE—combining the global exploration capability of PT with the local refinement capability of the diffusion process. CDS outperforms traditional MCMC, training-free MCMC, and neural samplers across 8 target distributions and 4 task types with fewer density evaluations.

## Background & Motivation

**Background**: Independent sampling from unnormalized multi-modal distributions $\pi(x)\propto \tilde\pi(x)$ is a fundamental problem in ML and natural sciences. Mainstream methods fall into two categories: (i) Annealed MCMC (e.g., Parallel Tempering, AIS, SMC), which transfers information between multiple chains by constructing a sequence of intermediate distributions from a reference $\pi_{\text{ref}}$ to the target $\pi$; (ii) Diffusion/interpolation-based generative models (neural samplers, stochastic interpolants), which use neural networks to fit the score or drift.

**Limitations of Prior Work**: (i) Annealing methods like PT require a massive number of intermediate distributions to be stable when the overlap between $\pi_{\text{ref}}$ and $\pi$ is small, leading to an explosion in density evaluations (a bottleneck in scenarios like molecular dynamics); (ii) Neural samplers must use a large number of target density evaluations to train neural networks to fit the drift/score, where the training cost itself may offset the "sampling savings," and retraining is required for new target distributions; (iii) Existing "training-free diffusion sampling" methods such as DiGS and RDMC rely either on Metropolis-within-Gibbs (which degrades in high dimensions) or nested MCMC (multiple density evaluations per iteration).

**Key Challenge**: The score function of diffusion sampling is **unanalytical** for general unnormalized distributions. Therefore, one must either train a neural network for fitting (leading to the cost dilemma of neural samplers) or use nested MCMC for approximation (leading to the overhead dilemma of DiGS/RDMC).

**Goal**: (i) Design a class of interpolation processes such that both the drift and score of their SDE have closed-form expressions, thereby completely avoiding neural network training; (ii) Control the initialization cost of this SDE so that the entire method significantly outperforms SOTA under a fixed density evaluation budget.

**Key Insight**: Standard stochastic interpolants (Albergo et al. 2025) study the drift of the marginal distribution, which is unanalytical. However, if one **fixes a reference point $z\sim\pi_{\text{ref}}$ and considers the conditional distribution** $\nu_{t\mid z}$, since $\nu_{t\mid z}$ is the pushforward of $\nu$ through a diffeomorphic mapping $F_{t\mid z}$, its density can be **analytically written from the target $\pi$ using the change-of-variables formula**—and the score naturally becomes closed-form!

**Core Idea**: Decompose "sampling $\pi$" into two stages: (1) At a small time $t_0$, $\nu_{t_0\mid z}$ is highly concentrated near $z$ and has extreme overlap with $\pi_{\text{ref}}$, allowing for extremely fast sampling via PT; (2) Use the closed-form SDE to transport these samples along $t_0\to 1$ to the target $\pi$.

## Method

### Overall Architecture
Two-stage pipeline (Alg. 1):

- **Stage 1 (PT for initial distribution)**: For a selected $t_0 > 0$ near zero, starting from the reference $z\sim\pi_{\text{ref}}$, use Parallel Tempering to sample the conditional distribution $\nu_{t_0\mid z}$. Since $\nu_{t_0\mid z}\to \delta_z$ as $t_0\to 0$, it almost perfectly overlaps with $\pi_{\text{ref}}$, resulting in extremely high swap acceptance and fast mixing for PT.
- **Stage 2 (Closed-form SDE transport)**: Integrate a closed-form SDE using Euler–Maruyama to transport samples from $\nu_{t_0\mid z}$ along time $t_0\to 1$ to the target $\nu$. The drift and score of the SDE are both analytically available, and an optional MH corrector can be inserted to further reduce discretization errors.

The entire method is **completely training-free**, requiring only evaluations of the target density $\tilde\pi$ and score $\nabla\log\tilde\pi$.

### Key Designs

1. **Conditional Interpolants (Core Theoretical Contribution)**:
    - **Function**: Defines a class of conditional stochastic interpolation processes such that their transition dynamics have exact closed-form expressions.
    - **Mechanism**: Standard stochastic interpolants define $x_t = F_t(z, x)$ where $z\sim\nu_{\text{ref}}, x\sim\nu$. Instead of studying the marginal distribution of $x_t$, this paper **fixes $z$** and lets $F_{t\mid z}(\cdot) = F_t(z,\cdot)$ be a diffeomorphism. Thus, $\nu_{t\mid z}$ is the pushforward of $\nu$ via $F_{t\mid z}$. From the change-of-variables formula, it follows that $\pi_{t\mid z}(x) = |\det \mathrm{J}F_{t\mid z}(F^{-1}_{t\mid z}(x))|^{-1}\pi(F^{-1}_{t\mid z}(x))$. **As long as the target $\pi$ is evaluable, the conditional density and conditional score $\nabla\log\pi_{t\mid z}$ are immediately available**. Subsequently, the conditional velocity field $u_{t\mid z}(x) = \partial_t F_{t\mid z}(F^{-1}_{t\mid z}(x))$ is defined, and combined with Fokker-Planck pairing, an exact conditional SDE that preserves $\pi_{t\mid z}$ is derived: $dx_t = (u_{t\mid z}(x_t) + \frac{\sigma_t^2}{2}\nabla\log\pi_{t\mid z}(x_t))dt + \sigma_t dW_t$.
    - **Design Motivation**: Addresses the fundamental pain point of diffusion sampling for unnormalized distributions—traditional methods must use neural networks to fit the score, while the conditional view restores the score to an analytical transformation of the target $\pi$, thereby **replacing neural training with dimensional transformation and analytical evaluation of the original density**.

2. **"Vanishing Transport Cost" at the $t\to 0$ Limit**:
    - **Function**: Ensures the initialization cost of Stage 1 monotonically decreases to 0 as $t_0\to 0$, eliminating the overhead of "running PT from scratch."
    - **Mechanism**: As $t\to 0$, $W_1(\delta_z, \nu_{t\mid z})\to 0$ (Eq. 10), meaning the conditional distribution collapses onto $z$. The authors use Lipschitz properties to prove that for any Markov kernel $K$, when the Lipschitz constant $L_t\le 1$ of the transformed kernel, the sampling error of the target $\nu_{t\mid z}$ is strictly lower than directly sampling $\nu$; for common interpolants like linear or trigonometric, $L_t\to 0$ as $t\to 0$. This implies **the smaller $t_0$ is, the easier it is for PT to jump from $\pi_{\text{ref}}$ to $\nu_{t_0\mid z}$**.
    - **Design Motivation**: Avoids a catch-22—the closed-form SDE has a singularity at $t=0$ ($F_{t\mid z}$ is not invertible at $t=0$, causing the drift to diverge), necessitating a start at $t_0>0$. However, the initial distribution at $t_0$ must be sampled. This paper proves that this new "initial sampling" task is significantly easier than the original task due to the small $t_0$—a key argument for the free-lunch nature of CDS.

3. **Division of Labor between PT and SDE (Two-stage Assembly)**:
    - **Function**: Enables PT to handle global multi-modal exploration while the SDE handles local refinement and continuous correction.
    - **Mechanism**: Stage 1 uses PT to anneal from $\pi_{\text{ref}}$ to $\nu_{t_0\mid z}$. Because $t_0$ is small, the intermediate ladder is short, swap acceptance is high, and density evaluations are few. Stage 2 uses Euler–Maruyama to integrate the closed-form SDE, pushing these "nearly correct" samples along $t_0\to 1$ to the target. **A crucial non-trivial design** is that initialization must be sampled from $\nu_{t_0\mid z}$ (rather than simply setting $x_{t_0}=z$)—the latter is proven in Appx H to result in severe degradation because diffusion cannot spread sufficient support from a single point. Furthermore, the authors find that using the SDE path is superior to using the inverse interpolation map $F^{-1}_{t_0\mid z}$ to directly map samples to $\nu$ (Fig. 5), as the continuous score-correction of the SDE automatically corrects initialization errors during transport.
    - **Design Motivation**: Complements the strengths of both methods—PT is strong at global multi-modal exploration but sensitive to the $\pi_{\text{ref}}\leftrightarrow\nu$ distance; diffusion SDEs are strong at local refinement but require the score. CDS places PT on the "shortest distance segment" and the SDE on the "full path," effectively leveraging the strengths of both while avoiding their weaknesses.

### Loss & Training
**Zero-training**. Stage 1 utilizes a non-reversible variant of PT; the SDE uses Euler–Maruyama discretization with an optional MH corrector; hyperparameters include PT steps $K$, integration steps $N$, noise schedule $\sigma_t$, and initial time $t_0$ (optimal values in Fig. 4).

## Key Experimental Results

### Main Results

| Method | Mean HVR (Aggregate over 8 tasks, higher is better) |
|------|------------------------------------|
| **CDS (Ours)** | **0.9976 ± 0.0015** |
| NRPT (SOTA non-reversible PT) | 0.9827 ± 0.0083 |
| OASMC (Optimized Annealed SMC) | 0.9287 ± 0.0277 |
| HMC | 0.6263 ± 0.1261 |
| DiGS (Diffusive Gibbs) | 0.5464 ± 0.1550 |
| MALA | 0.5241 ± 0.1494 |

Tasks cover Gaussian Mixture (2D and 16D, including non-uniform versions), Lennard-Jones (LJ-13 and LJ-55, chemical potential), Alanine Dipeptide (66D molecular dynamics), and Bayesian Neural Network (550D posterior inference).

### Ablation Study

| Configuration | Main Phenomenon | Description |
|------|---------|------|
| $t_0=1.0\to 0.0$ (Fig. 4) | RT monotonically increases, error decreases; degrades when too small | Validates the existence of an optimal $t_0$ range |
| SDE transport vs. Inverse map $F^{-1}_{t_0\mid z}$ (Fig. 5) | SDE wins overall; inverse map slightly wins only in GM-2 low budget | SDE score correction fixes initialization errors |
| Initialization $x_{t_0}=z$ vs. Sampling $\nu_{t_0\mid z}$ (Appx H) | Single point initialization degrades severely | Noise is insufficient to diffuse into support |
| ALDP 200k budget (Fig. 2) | Only CDS and NRPT reproduce correct mode ratios | A hard metric for multi-modal fidelity |

### Key Findings
- **CDS leads by a wide margin on BNN (550D)**: High-dimensional multi-modal posteriors are a weakness for traditional PT and DiGS. CDS significantly exceeds all baselines in HVR here, demonstrating the advantage of conditional SDEs in high dimensions.
- **Local samplers (MALA/HMC) perform best on LJ tasks**: Local structure dominates and mode separation is weak in LJ potentials. CDS is on par with NRPT, illustrating the principle of "method-task fit"—CDS is not universally better.
- **An optimal $t_0$ exists**: If $t_0$ is too large, the gap between $\nu_{t_0\mid z}$ and the target $\nu$ is large, degrading PT; if too small, $\nu_{t_0\mid z}$ is overly concentrated, and insufficient replica overlap leads to PT swap failure. This trade-off is the core practical hyperparameter for CDS.
- **Linear interpolants have geometric disadvantages on LJ/ALDP**: They can push particle distances near zero, causing numerical instability in high-energy regions; this suggests that future work could design task-aware geometric interpolants.
- **DiGS is on par with CDS on GM-2 but degrades as dimension increases**: This occurs because the Metropolis-within-Gibbs in DiGS worsens in high dimensions, whereas CDS does not suffer this dimensionality penalty.

## Highlights & Insights
- **The "Conditional Perspective" is an undervalued key**: Standard stochastic interpolants are neuralized because the marginal score is unanalytical. This paper changes the perspective to the conditional score, which is immediately closed-form—this trick of "resolving unanalytical components via conditioning" can be generalized to many generative modeling problems.
- **$t\to 0$ is a gift, not a problem**: The $t=0$ singularity in conventional diffusion is seen as a nuisance. This paper leverages the property that the initial distribution collapses to a Dirac as $t_0\to 0$, making Stage 1 nearly free—a design aesthetic that turns a defect into a feature.
- **PT and Diffusion are complementary, not competitive**: Previously treated as separate paths, CDS proves they are a natural "global vs. local" pair, providing a new synthesis paradigm for the sampling field.
- **Completely training-free + excellent high-dimensional performance**: Unlike neural samplers that require retraining for every new target, CDS is truly zero-shot and directly applicable to new molecules or posteriors, which is of great engineering significance.

## Limitations & Future Work
- **Dependence on the choice of interpolation map**: The authors admit that linear interpolation in potentials with singularities (LJ, ALDP) may drive trajectories through high-energy regions, causing numerical instability. Task-aware non-linear interpolants (e.g., geometry-adaptive based on $\pi$) are needed.
- **Lack of automation in selecting $t_0$**: Although Appx C provides heuristics, grid search is still required in practice, increasing tuning costs for new tasks.
- **PT swaps may still fail at extremely small $t_0$**: After the conditional distribution becomes overly concentrated, replicas may fail to overlap, still resulting in collapse. CDS does not provide a fundamental fix and relies on engineering the $t_0$ value.
- **No comparison with large-scale neural samplers like Adjoint Sampling under equal budget**: The authors exclude neural samplers by classifying them under the "amortized regime," but for industrial users, "train once, sample infinitely cheap" might not be inferior to CDS.
- **Lack of end-to-end bounds for theoretical convergence guarantees**: While the vanishing transport cost and Lipschitz properties are proven individually, a total error bound for the combined two-stage process is not provided.

## Related Work & Insights
- **vs. Parallel Tempering (NRPT)**: NRPT is the current gold standard. CDS uses PT for the shortest distance segment and SDE for the others, essentially "using PT to solve the pain points of PT."
- **vs. Neural samplers (NETS, Adjoint Sampling)**: Neural types require training before sampling, whereas CDS is training-free. However, neural methods can amortize training costs when distributions are shared, while CDS runs from scratch every time.
- **vs. DiGS / RDMC**: Both are "non-neural diffusion sampling." However, DiGS fits the marginal score with Gibbs (degrading in high dimensions), and RDMC uses nested MCMC (multiple density evaluations per step). CDS replaces the marginal with a closed-form conditional.
- **vs. Stochastic Interpolants (Albergo 2025)**: This paper is its conditional counterpart—transforming a framework "for training" into a framework "for zero-shot sampling," representing the first systematic application of this theory to the sampling side.
- **Inspiration**: The conditional reformulation technique may also be applicable to normalizing flow training, accelerating score matching, and conditional sampling under constraints.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ "Conditional interpolation → closed-form SDE" is a genuine theoretical breakthrough, pivoting diffusion sampling from "must train" to "completely training-free," with a sophisticated overall framework design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 8 distributions across 4 task types, 5 strong baselines, and detailed ablations. However, it lacks validation on higher-dimensional scientific applications (e.g., protein conformation sampling) and omits fair comparisons with the latest neural samplers from an amortized perspective.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous, and the two-stage structure is clear. However, the high density of notation presents a steep barrier for readers without a background in interpolation theory.
- **Value**: ⭐⭐⭐⭐ Highly valuable for fields like computational chemistry and Bayesian inference where "sampling on-demand per target" is required and pre-training is impossible. It also provides a generalizable "conditional-as-closed-form" idea for the ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Text-Conditional JEPA for Learning Semantically Rich Visual Representations](text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[CVPR 2026\] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](../../CVPR2026/multimodal_vlm/thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)
- [\[ICML 2026\] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling](beyond_vlm-based_rewards_diffusion-native_latent_reward_modeling.md)
- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)

</div>

<!-- RELATED:END -->
