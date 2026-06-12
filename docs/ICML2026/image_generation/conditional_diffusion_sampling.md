---
title: >-
  [Paper Note] Conditional Diffusion Sampling
description: >-
  [ICML 2026][Image Generation][Parallel Tempering] This paper proposes Conditional Diffusion Sampling (CDS): by deriving a class of conditional stochastic interpolants…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Parallel Tempering"
  - "Conditional Interpolants"
  - "closed-form SDE"
  - "multi-modal sampling"
  - "training-free"
date: 2026-05-08
content_hash: 9ea8513df834bf52
---

# Conditional Diffusion Sampling

**Conference**: ICML 2026  
**arXiv**: [2605.04013](https://arxiv.org/abs/2605.04013)  
**Code**: https://github.com/Franblueee/conditional_diffusion_sampling  
**Area**: Sampling Algorithms / Diffusion Models; MCMC; Bayesian Inference  
**Keywords**: Parallel Tempering, Conditional Interpolants, closed-form SDE, multi-modal sampling, training-free

## TL;DR
This paper proposes Conditional Diffusion Sampling (CDS): by deriving a class of conditional stochastic interpolants, it obtains an **exact closed-form SDE** for the unnormalized target distribution (without neural network fitting), and then efficiently samples the initial distribution of this SDE using Parallel Tempering (PT)—combining PT's global exploration with the local refinement of the diffusion process. On 8 target distributions and 4 task types, CDS outperforms traditional MCMC, training-free MCMC, and neural samplers with fewer density evaluations.

## Background & Motivation

**Background**: Independent sampling from unnormalized multi-modal distributions $\pi(x)\propto \tilde\pi(x)$ is a fundamental problem in ML and the natural sciences. Mainstream methods fall into two categories: (i) annealing-based MCMC (e.g., Parallel Tempering, AIS, SMC), which construct intermediate distributions from a reference $\pi_{\text{ref}}$ to the target $\pi$ to transfer information between chains; (ii) diffusion/interpolation-based generative models (neural samplers, stochastic interpolants), which fit the score or drift using neural networks.

**Limitations of Prior Work**: (i) Annealing methods like PT require a large number of intermediate distributions for stability when $\pi_{\text{ref}}$ and $\pi$ have little overlap, causing the number of density evaluations (a bottleneck in molecular dynamics, etc.) to explode; (ii) Neural samplers require extensive target density evaluations to train neural networks to fit the drift/score, making the training cost itself offset the "sampling savings," and retraining is needed for new target distributions; (iii) Existing "training-free diffusion samplers" like DiGS and RDMC either rely on Metropolis-within-Gibbs (which degrades in high dimensions) or nested MCMC (multiple density evaluations per iteration).

**Key Challenge**: The score function for diffusion sampling is **intractable** for general unnormalized distributions, so one must either train a neural network to fit it (→ cost dilemma of neural samplers) or use nested MCMC approximations (→ overhead dilemma of DiGS/RDMC).

**Goal**: (i) Design a class of interpolation processes whose SDE drift and score have closed-form expressions, thus completely avoiding neural network training; (ii) Control the initialization cost of this SDE so that the overall method significantly outperforms SOTA under a fixed density evaluation budget.

**Key Insight**: Standard stochastic interpolants (Albergo et al. 2025) study the drift of the marginal distribution, which is intractable; but if **a reference point $z\sim\pi_{\text{ref}}$ is fixed and the conditional distribution** $\nu_{t\mid z}$ is considered, since $\nu_{t\mid z}$ is the pushforward of $\nu$ through a diffeomorphic map $F_{t\mid z}$, its density can **be analytically written from the target $\pi$ via the change of variables formula**—so the score also becomes closed-form!

**Core Idea**: Decompose "sampling $\pi$" into two stages—(1) At a small time $t_0$, $\nu_{t_0\mid z}$ is highly concentrated near $z$ and overlaps greatly with $\pi_{\text{ref}}$, so PT samples it very quickly; (2) Use the closed-form SDE to transport these samples along $t_0\to 1$ to the target $\pi$.

## Method

### Overall Architecture
A two-stage pipeline (Alg. 1):

- **Stage 1 (PT for Initial Distribution)**: Choose a small $t_0>0$, start from a reference $z\sim\pi_{\text{ref}}$, and use Parallel Tempering to sample the conditional distribution $\nu_{t_0\mid z}$. As $t_0\to 0$, $\nu_{t_0\mid z}\to \delta_z$, almost completely overlapping with $\pi_{\text{ref}}$, so PT's swap acceptance is very high and mixing is very fast.
- **Stage 2 (Closed-form SDE Transport)**: Use Euler–Maruyama to integrate a closed-form SDE, transporting samples from $\nu_{t_0\mid z}$ along $t_0\to 1$ to the target $\nu$. The SDE's drift and score are fully analytic; an optional MH corrector can further reduce discretization error.

The entire method **requires no neural network training**, relying only on evaluations of the target density $\tilde\pi$ and its score $\nabla\log\tilde\pi$.

### Key Designs

1. **Conditional Interpolants (Core Theoretical Contribution)**:

    - **Function**: Defines a class of conditional stochastic interpolation processes with exact closed-form transport dynamics.
    - **Mechanism**: Standard stochastic interpolants define $x_t = F_t(z, x)$, $z\sim\nu_{\text{ref}}, x\sim\nu$. Instead of studying the marginal distribution of $x_t$, this work **fixes $z$**, lets $F_{t\mid z}(\cdot) = F_t(z,\cdot)$ be a diffeomorphism, so $\nu_{t\mid z}$ is the pushforward of $\nu$ through $F_{t\mid z}$. By the change of variables formula, $\pi_{t\mid z}(x) = |\det \mathrm{J}F_{t\mid z}(F^{-1}_{t\mid z}(x))|^{-1}\pi(F^{-1}_{t\mid z}(x))$, so **as long as the target $\pi$ is evaluable, both the conditional density and conditional score $\nabla\log\pi_{t\mid z}$ are immediately available**. The conditional velocity field is then defined as $u_{t\mid z}(x) = \partial_t F_{t\mid z}(F^{-1}_{t\mid z}(x))$, and together with the Fokker-Planck pairing, one derives the exact conditional SDE preserving $\pi_{t\mid z}$: $dx_t = (u_{t\mid z}(x_t) + \frac{\sigma_t^2}{2}\nabla\log\pi_{t\mid z}(x_t))dt + \sigma_t dW_t$.
    - **Design Motivation**: Addresses the fundamental pain point of "diffusion + unnormalized distribution" sampling—traditional methods must fit the score with neural networks, while the conditional view reduces the score to an analytic transformation of the target $\pi$, thus **replacing neural training with dimension transformation and analytic evaluation of the original density**.

2. **"Transport Cost Vanishes" Property as $t\to 0$**:

    - **Function**: Makes the initialization cost of Stage 1 decrease monotonically to zero as $t_0\to 0$, eliminating the need to "run PT from scratch."
    - **Mechanism**: As $t\to 0$, $W_1(\delta_z, \nu_{t\mid z})\to 0$ (Eq. 10), i.e., the conditional distribution collapses to $z$. Using Lipschitz properties, the authors prove that for any Markov kernel $K$, when the transformed kernel's Lipschitz constant $L_t\le 1$, the sampling error for $\nu_{t\mid z}$ is strictly lower than for $\nu$; for linear, trigonometric, and other common interpolants, $L_t\to 0$ as $t\to 0$. This means **the smaller $t_0$, the easier for PT to jump from $\pi_{\text{ref}}$ to $\nu_{t_0\mid z}$**.
    - **Design Motivation**: Avoids a catch-22—closed-form SDE has a singularity at $t=0$ ($F_{t\mid z}$ is non-invertible at $t=0$, drift diverges), so must start at $t_0>0$, and the initial distribution at $t_0$ must be sampled. This work proves that this new "sampling the initial distribution" task is actually much easier than the original task when $t_0$ is small—this is the key to the CDS free-lunch argument.

3. **Division of Labor between PT and SDE (Two-stage Assembly)**:

    - **Function**: PT handles global multi-modal exploration, SDE handles local refinement and continuous correction.
    - **Mechanism**: Stage 1 uses PT to anneal from $\pi_{\text{ref}}$ to $\nu_{t_0\mid z}$; since $t_0$ is small, the ladder is short, swap acceptance is high, and density evaluations are few. Stage 2 uses Euler–Maruyama to integrate the closed-form SDE, pushing these "almost correct" samples along $t_0\to 1$ to the target. **A crucial nontrivial design** is that initialization must sample from $\nu_{t_0\mid z}$ (not simply set $x_{t_0}=z$)—the latter is shown in Appx H to degrade severely, as diffusion cannot expand from a single point to sufficient support. The authors also find that using the inverse interpolant $F^{-1}_{t_0\mid z}$ to map samples directly to $\nu$ is worse than the SDE path (Fig. 5), because the SDE's continuous score correction can automatically fix initialization errors during transport.
    - **Design Motivation**: Combines the strengths of both methods—PT excels at global multi-modal exploration but is sensitive to the distance between $\pi_{\text{ref}}$ and $\nu$; diffusion SDE excels at local refinement but needs the score. CDS puts PT on the "shortest distance segment" and SDE on the "whole path," complementing each other's strengths.

### Loss & Training

**No training**. Stage 1 PT uses a non-reversible variant; SDE uses Euler–Maruyama discretization, with optional MH corrector. Hyperparameters are PT steps $K$, integration steps $N$, noise schedule $\sigma_t$, and initial time $t_0$ (optimal values in Fig. 4).

## Key Experimental Results

### Main Results

| Method | Mean HVR (aggregated over 8 tasks, higher is better) |
|--------|-----------------------------------------------------|
| **CDS (Ours)** | **0.9976 ± 0.0015** |
| NRPT (SOTA non-reversible PT) | 0.9827 ± 0.0083 |
| OASMC (Optimized Annealed SMC) | 0.9287 ± 0.0277 |
| HMC | 0.6263 ± 0.1261 |
| DiGS (Diffusive Gibbs) | 0.5464 ± 0.1550 |
| MALA | 0.5241 ± 0.1494 |

Tasks cover Gaussian Mixture (2D and 16D, including non-uniform versions), Lennard-Jones (LJ-13 and LJ-55, chemical potential), Alanine Dipeptide (66D molecular dynamics), Bayesian Neural Network (550D posterior inference).

### Ablation Study

| Configuration | Main Phenomenon | Description |
|---------------|----------------|-------------|
| $t_0=1.0\to 0.0$ (Fig. 4) | RT increases monotonically, error decreases; degrades if too small | Verifies existence of optimal $t_0$ interval |
| SDE transport vs inverse interpolant $F^{-1}_{t_0\mid z}$ (Fig. 5) | SDE wins overall, inverse interpolant slightly better only for GM-2 under small budget | SDE's score correction can fix initialization errors |
| Initialization with $x_{t_0}=z$ vs sampling $\nu_{t_0\mid z}$ (Appx H) | Single-point initialization degrades severely | Noise insufficient to diffuse to full support |
| ALDP 200k budget (Fig. 2) | Only CDS and NRPT reproduce correct proportion of two modes | Hard metric for multi-modal fidelity |

### Key Findings
- **CDS leads dramatically on BNN (550D)**: High-dimensional multi-modal posteriors are a weakness for traditional PT and DiGS; CDS achieves much higher HVR than all baselines in this scenario, demonstrating the advantage of conditional SDE in high dimensions.
- **Local samplers (MALA/HMC) perform best on LJ tasks**: LJ potential is dominated by local structure with weak mode separation; CDS and NRPT are tied, illustrating the "method-task match" principle—CDS is not universally superior.
- **Optimal $t_0$ exists**: If too large, $\nu_{t_0\mid z}$ is far from the target $\nu$ and PT degrades; if too small, $\nu_{t_0\mid z}$ is overly concentrated and replicas lack overlap, causing PT swap failures. This trade-off is the core practical hyperparameter of CDS.
- **Linear interpolation has geometric disadvantages on LJ/ALDP**: It pushes particle distances near zero, causing numerical instability in high-energy regions; this suggests future work could design task-aware geometric interpolants.
- **DiGS matches CDS on GM-2 but degrades with increasing dimension**: Because DiGS's Metropolis-within-Gibbs degrades in high dimensions, while CDS does not suffer this dimensionality penalty.

## Highlights & Insights
- **"Conditional perspective" is an underrated key**: Standard stochastic interpolants are neuralized due to intractable marginal scores; this work switches to the conditional score, which is immediately closed-form—this "conditioning to make the intractable tractable" trick can be generalized to many generative modeling problems.
- **$t\to 0$ is a feature, not a bug**: The $t=0$ singularity in conventional diffusion is usually a nuisance; here, the property that the initial distribution collapses to Dirac as $t_0\to 0$ makes Stage 1 almost free—turning a defect into a design feature.
- **PT and diffusion are complementary, not competing**: Previously considered two separate paths; CDS shows they are a natural "global vs local" pairing, providing a new synthesis paradigm for the sampling field.
- **Completely training-free + strong high-dimensional performance**: Unlike neural samplers that require retraining for each new target, CDS is truly zero-shot and directly applicable to new molecules/posteriors, which is of great engineering significance.

## Limitations & Future Work
- **Dependence on choice of interpolation map**: The authors acknowledge that linear interpolation may drive trajectories through high-energy regions in potentials with singularities (LJ, ALDP), causing numerical instability; future work needs task-aware nonlinear interpolants (e.g., geometry-adaptive to $\pi$).
- **Lack of automation in choosing $t_0$**: Although Appx C provides some heuristics, in practice grid search is still needed, increasing tuning cost for new tasks.
- **PT swap may still fail at extremely small $t_0$**: When the conditional distribution is overly concentrated, replicas do not overlap and collapse; CDS does not fundamentally fix this, relying only on engineering choices of $t_0$.
- **No comparison with large-scale neural samplers like Adjoint Sampling under equal budget**: The authors classify neural samplers as "amortized regime" and exclude them, but for industrial users, "train once, sample cheaply forever" may not be worse than CDS.
- **Lack of end-to-end theoretical convergence bounds**: Only transport cost vanishing and Lipschitz properties are proven separately; no total error bound for the two-stage assembly is provided.

## Related Work & Insights
- **vs Parallel Tempering (NRPT)**: NRPT is the current gold standard; CDS uses PT on the shortest segment and SDE elsewhere, essentially "using PT to solve PT's own pain points."
- **vs Neural samplers (NETS, Adjoint Sampling)**: Neural methods require training before sampling, CDS is training-free; but neural methods can amortize training cost when sharing distributions, while CDS must run from scratch each time.
- **vs DiGS / RDMC**: Both are "non-neural diffusion samplers," but DiGS fits the marginal score via Gibbs (degrades in high dimensions), RDMC uses nested MCMC (multiple density evaluations per step); CDS uses conditional to replace marginal with closed-form.
- **vs Stochastic Interpolants (Albergo 2025)**: This work is the conditional incarnation—turning a "for training" framework into a "for zero-shot sampling" framework, marking the first systematic application of this theory on the sampling side.
- **Insights**: The conditional reformulation trick may also apply to normalizing flow training, accelerating score matching, conditional sampling under constraints, etc.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Conditional interpolants → closed-form SDE" is a true theoretical breakthrough, flipping diffusion sampling from "must train" to "completely training-free," with an elegantly designed framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 task types, 8 distributions, 5 strong baselines, and detailed ablations; but lacks validation on higher-dimensional scientific applications (e.g., protein conformation sampling) and omits fair comparison with the latest neural samplers in the amortized regime.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations, clear two-stage structure; but high symbol density makes it challenging for readers without interpolation theory background.
- Value: ⭐⭐⭐⭐ Highly valuable for scenarios in computational chemistry and Bayesian inference where "sample as needed, no pretraining" is required; also provides the ML community with a generalizable conditional-as-closed-form approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](../../ICLR2026/image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)

</div>

<!-- RELATED:END -->
