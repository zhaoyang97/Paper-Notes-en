---
title: >-
  [Paper Note] Discrete Adjoint Matching
description: >-
  [ICLR 2026][Image Generation][Adjoint Matching] This paper proposes Discrete Adjoint Matching (DAM), which derives adjoint variables for discrete state spaces from a purely statistical perspective (rather than from control theory), extending the continuous-domain Adjoint Matching framework to discrete generative models based on continuous-time Markov chains (CTMCs). The approach enables effective fine-tuning of diffusion-based LLMs (LLaDA-8B), improving accuracy on Sudoku from 11.5% to 89.2%.
tags:
  - ICLR 2026
  - Image Generation
  - Adjoint Matching
  - discrete adjoint variables
  - CTMC
  - diffusion LLM fine-tuning
  - entropy-regularized reward optimization
date: 2026-05-08
content_hash: b99e1e5e2b65077e
---

# Discrete Adjoint Matching

**Conference**: ICLR 2026
**arXiv**: [2602.07132](https://arxiv.org/abs/2602.07132)
**Code**: None
**Area**: Image Generation / Fine-tuning of Discrete Generative Models
**Keywords**: Adjoint Matching, discrete adjoint variables, CTMC, diffusion LLM fine-tuning, entropy-regularized reward optimization

## TL;DR

This paper proposes Discrete Adjoint Matching (DAM), which derives adjoint variables for discrete state spaces from a purely statistical perspective (rather than from control theory), extending the continuous-domain Adjoint Matching framework to discrete generative models based on continuous-time Markov chains (CTMCs). The approach enables effective fine-tuning of diffusion-based LLMs (LLaDA-8B), improving accuracy on Sudoku from 11.5% to 89.2%.

## Background & Motivation

**Background**: Entropy-regularized reward optimization $\min_u \mathbb{E}[g(X_1)] + D_{\text{KL}}(p^u \| p^{\text{base}})$ is the standard paradigm for fine-tuning generative models, widely adopted in RLHF and conditional generation. Its optimal solution takes the closed form $p^\star(X) \propto p^{\text{base}}(X) e^{-g(X_1)}$, which shifts the model distribution toward high-reward regions while remaining close to the reference. In continuous state spaces, Adjoint Matching (AM) reformulates the optimization as a matching problem via adjoint variables, achieving strong results in image fine-tuning and molecular generation.

**Limitations of Prior Work**: AM fundamentally relies on gradient information in continuous space—the terminal adjoint is $\tilde{a}_1(X) = \nabla g(X_1)$ and the dynamics involve the Jacobian $\nabla u^{\text{base}}$. However, discrete state spaces are nowhere differentiable: $g(x)$ has no gradient, and rate functions $u_t(y,x)$ replace the drift term of the SDE. These fundamental differences prevent continuous AM from being directly applied to the discrete domain.

**Key Challenge**: CTMC-based discrete diffusion models (e.g., MDLM, LLaDA) have recently emerged as promising architectures for text generation, yet principled reward-optimization fine-tuning for such models remains an open problem. Existing approaches such as D1 rely on policy-gradient approximations that require estimating intractable likelihoods and handling non-differentiable rewards, limiting training stability.

**Key Insight**: The authors observe that the adjoint variable in AM is not fundamentally a control-theoretic concept but rather a statistical quantity—it estimates the ratio between the optimal solution and the base model. In the discrete domain, this ratio can be estimated via Dynkin's formula (a tool that expresses function values as expectations over stochastic processes), completely bypassing any differentiability requirement.

**Core Idea**: Derive discrete adjoint variables from a purely statistical perspective using Dynkin's formula, reducing the estimation of optimal CTMC rates to a matching problem, thereby enabling principled fine-tuning of CTMC-based discrete generative models.

## Method

### Overall Architecture

DAM takes as input a pretrained CTMC-based discrete generative model (e.g., the LLaDA diffusion LLM) and a reward function $r(x)$ (setting $g(x) = -r(x)$), and outputs a fine-tuned model that generates higher-reward samples. The overall pipeline proceeds as follows: (1) analytically derive the closed form of the optimal CTMC rate $u_t^\star(y,x)$; (2) construct discrete adjoint variables $\tilde{a}_t$ via Dynkin's formula as unbiased estimators of the optimal rate; (3) train a parameterized model to approximate the optimal rate via a Bregman divergence matching framework; (4) leverage the special structure of masked diffusion and importance sampling to reduce variance.

### Key Designs

1. **Discrete Adjoint Variable**:

    - **Function**: Provides an unbiased estimator of the optimal CTMC rate $u_t^\star(y,x)$.
    - **Mechanism**: The optimal rate can be written as $u_t^\star(y,x) = u_t^{\text{base}}(y,x) \cdot e^{-V_t(y)+V_t(x)}$, where $V_t(x) = -\log \sum_z p_{1|t}^{\text{base}}(z|x) e^{-g(z)}$ is the value function. The problem reduces to estimating the exponential value difference $e^{-V_t(y)+V_t(x)}$. Applying Dynkin's formula to the CTMC process yields the discrete adjoint variable $\tilde{a}_t(y;X)$, which satisfies a linear ODE with terminal condition $\tilde{a}_1(y;X) = e^{-g(y)+g(X_1)}$ (an exponential terminal cost difference, rather than the gradient $\nabla g$ used in continuous AM). Crucially, the discrete adjoint modifies the base rate **multiplicatively** ($u^{\text{base}} \cdot \mathbb{E}[\tilde{a}]$), in contrast to the additive correction of continuous AM ($u^{\text{base}} - \mathbb{E}[\tilde{a}]$).
    - **Design Motivation**: This construction bypasses the fundamental challenge of non-differentiability in discrete spaces. Dynkin's formula holds for any Feller process (including both SDEs and CTMCs): specialized to SDEs it recovers Itô's lemma, while applied to CTMCs it yields a discrete analogue of the adjoint system.

2. **Importance-Weighted Adjoint Estimator**:

    - **Function**: Provides a low-bias, low-variance, practically computable adjoint estimate.
    - **Mechanism**: The discrete adjoint ODE admits the closed-form solution $\tilde{a}_t(y;X_1) = \sum_z p_{1|t}^{\text{base}}(z|y) e^{-g(z)+g(X_1)}$, but in theory requires sampling $X_1$ from the optimal distribution $p^\star$. DAM instead samples $X_1 \sim p^u$ from the current model and corrects for bias via self-normalized importance sampling:
    $$\hat{a}_t(y;Z,\{X_1^{(k)}\}) = \frac{p_{1|t}^{\text{base}}(Z|y)}{p_{1|t}^u(Z|y)} e^{-g(Z)} \cdot \left(\frac{1}{K}\sum_k \frac{p_{1|t}^{\text{base}}(X_1^{(k)}|x)}{p_{1|t}^u(X_1^{(k)}|x)} e^{-g(X_1^{(k)})}\right)^{-1},$$
    where the importance weights (ratios $p^{\text{base}}/p^u$) are efficiently computable for CTMC models.
    - **Design Motivation**: Although the analytic solution is theoretically correct, experiments on the synthetic Pinwheel task show it exhibits substantially higher bias and variance compared to the importance-weighted version. Importance weighting makes DAM a consistent estimator (unbiased as $K \to \infty$).

3. **Exploitation of Masked Diffusion Structure**:

    - **Function**: Reduces the modeling complexity from $O(M^N)$ to $O(MN)$, making the approach compatible with modern LLM architectures.
    - **Mechanism**: In practice, nearly all base CTMCs are masked diffusion models (progressively unmasking from a fully masked state), whose rate matrices admit a special factorization $u_t^{\text{base}}(y,x) = \lambda_t^{\text{base}}(x) Q^{\text{base}}(y|x)$, where $Q^{\text{base}}$ is restricted to transitions that unmask exactly one token. The authors prove (Proposition 2.5) that the optimal rate $u_t^\star$ preserves this masked structure; therefore, it suffices to parameterize a time-independent $Q^\theta(y|x)$ and model it with an LLM, without modifying the model architecture.
    - **Design Motivation**: Computing adjoint variables directly over the full discrete state space $|\mathcal{X}| = M^N$ is completely intractable (e.g., vocabulary size 1000 and sequence length 100 yields $10^{300}$ states). Exploiting the masked structure makes large-scale LLM fine-tuning feasible.

### Loss & Training

DAM employs the generalized KL divergence (gKL) as the matching objective:
$$D_{\text{gKL}}(u,w) = \sum_{y \neq x} \left[u(y,x) - w(y,x) + w(y,x)\log \frac{w(y,x)}{u(y,x)}\right],$$
which better preserves the probabilistic structure of the discrete domain (e.g., non-negativity) compared to a naive $\ell_2$ loss. During training, trajectories are sampled from the current model and stored in a replay buffer; intermediate states $X_t$ are sampled via reciprocal projection; transition targets $y$ are sampled from the model distribution and debiased by $p_t^u(y|x)^{-1}$. Each training iteration samples $K$ model trajectories to compute the importance-weighted adjoint estimate, then updates the model with a weighted gKL loss.

## Key Experimental Results

### Synthetic Experiments: Convergence to Optimal Distribution

DAM is compared against D1 and SVDD on Checkerboard and Pinwheel tasks defined over a $91 \times 91$ discrete grid. Because the state space is small, the optimal distribution $p^\star$ can be computed exactly.

| Method | Checkerboard Visual Match | Pinwheel $D_{\text{KL}}(p^\star \| p^u)$ Convergence | Notes |
|--------|--------------------------|------------------------------------------------------|-------|
| DAM (importance-weighted) | Closest to $p^\star$ | Stably converges to $\sim 10^{-3}$ | Both jumps stable |
| DAM (analytic adjoint ablation) | Slightly worse | Converges but with higher bias | Validates importance weighting |
| D1 | Noticeable deviation | Plateaus, does not converge | Limited by policy-gradient approximation |
| SVDD | Noticeable deviation | Plateaus, does not converge | Limitation of value-function regression |

### Math Reasoning: Fine-tuning LLaDA-8B-Instruct

| Task | Seq. Len. | LLaDA Baseline | + D1 | + DAM | DAM Gain |
|------|-----------|----------------|------|-------|---------|
| GSM8K | 128 | 68.6% | 75.6% | **75.7%** | +7.1 pp |
| GSM8K | 256 | 76.8% | 79.8% | **79.9%** | +3.1 pp |
| MATH500 | 128 | 28.8% | 31.2% | **32.6%** | +3.8 pp |
| MATH500 | 256 | 30.8% | **37.2%** | 36.4% | +5.6 pp |
| Countdown | 128 | 34.8% | 43.8% | **60.2%** | +25.4 pp |
| Countdown | 256 | 19.5% | 31.3% | **55.5%** | +36.0 pp |
| Sudoku | 128 | 11.5% | 23.8% | **89.2%** | +77.7 pp |
| Sudoku | 256 | 6.4% | 12.9% | **88.1%** | +81.7 pp |

### Test-Time Generalization (Cross Sequence Length)

| Method | Train Len. | Countdown 128/256/512 | Sudoku 128/256/512 |
|--------|------------|-----------------------|-------------------|
| DAM | 128 | 60.2 / 59.8 / 59.0 | 89.2 / 88.6 / 84.9 |
| D1 | 128 | 43.8 / 33.6 / 28.1 | 23.8 / 16.9 / 10.0 |
| DAM | 256 | 58.6 / 55.5 / 49.6 | 87.0 / 88.1 / 87.1 |
| D1 | 256 | 33.2 / 31.3 / 37.1 | 18.4 / 12.9 / 11.0 |

### Key Findings

- **DAM substantially outperforms D1 on Countdown and Sudoku**: The gap on Sudoku exceeds 65 percentage points, demonstrating that DAM's principled optimization is far superior to policy-gradient approximations on tasks requiring exact constraint satisfaction.
- **Smaller gap on GSM8K and MATH500**: On these tasks, the approximation assumptions of D1 may already be sufficiently accurate, leaving little room for DAM to demonstrate advantage.
- **Importance-weighted adjoint estimator clearly outperforms the analytic version**: In the Pinwheel synthetic experiment, the importance-weighted variant exhibits significantly lower bias and variance.
- **Strong generalization**: DAM fine-tuned models maintain stable performance across different test sequence lengths (Sudoku: 89.2→88.6→84.9), while D1 degrades substantially (23.8→16.9→10.0).

## Highlights & Insights

- **Statistical perspective bypasses non-differentiability**: The entire derivation requires no gradients whatsoever, instead relying on Dynkin's formula—a fundamental tool from probability theory. This reveals that adjoint variables are intrinsically statistical quantities rather than differential ones; the two coincide in the continuous domain only by coincidence.
- **Deep implications of multiplicative vs. additive correction**: Continuous AM applies an additive correction $u^{\text{base}} - \mathbb{E}[\tilde{a}]$, whereas discrete DAM applies a multiplicative correction $u^{\text{base}} \cdot \mathbb{E}[\tilde{a}]$. This reflects the natural geometric difference between the two domains: a "shift" relative to the base model in continuous space versus a "scaling factor" in discrete space.
- **Masked structure preservation theorem**: Proposition 2.5 proves that the optimal rate automatically preserves the structural constraints of masked diffusion, meaning DAM can seamlessly employ existing LLM architectures for parameterization and inference, with minimal deployment overhead.

## Limitations & Future Work

- **Validated only on masked CTMC models**: All experiments are based on masked diffusion models (LLaDA); the approach has not been validated on non-masked CTMCs (e.g., uniform transition) or other discrete generative models. The paper itself identifies extending DAM to non-masked CTMCs as future work.
- **Narrow experimental scope**: Only synthetic tasks and mathematical reasoning are evaluated; broader applications such as code generation, protein design, and natural language dialogue remain unexplored.
- **Incompatible with autoregressive LLMs**: DAM is designed specifically for CTMC-based discrete diffusion models and cannot be directly applied to RLHF for autoregressive models such as GPT; its applicability is therefore tied to the emerging ecosystem of diffusion-based LLMs.
- **Computational efficiency concerns**: Each training step requires sampling $K$ model trajectories for importance weighting plus replay buffer management, making the actual training cost substantially higher than simple policy-gradient methods.
- **Limited advantage on GSM8K/MATH500**: On these mainstream math reasoning benchmarks, DAM offers only marginal improvements over D1, lacking compelling evidence that theoretical elegance translates into significant practical gains.

## Related Work & Insights

- **vs. Adjoint Matching (AM)**: The continuous-domain counterpart, which derives adjoint variables via control theory and requires gradient information. DAM demonstrates that a statistical perspective can fully replace the control-theoretic one and is strictly more general.
- **vs. D1 (Zhao et al., 2025)**: The current state-of-the-art method for masked CTMC fine-tuning, based on policy gradients. DAM substantially outperforms D1 on constraint-satisfaction tasks but offers limited advantage on general mathematical reasoning.
- **vs. SVDD (Li et al., 2024)**: A value-function regression approach that estimates the value function via reward regression. DAM directly estimates the exponential value difference rather than the value function itself, yielding better theoretical properties.
- **vs. DiffuCoder / DRAKES**: Other discrete diffusion fine-tuning methods that typically require approximate handling of non-differentiable rewards or intractable likelihoods. DAM's matching framework naturally accommodates non-differentiable rewards.
- **Flow Matching / Score Matching**: DAM continues the elegant "matching" training paradigm and extends it to the discrete domain.
- **Insight**: When fine-tuning discrete generative models, matching-based methods may be more suitable than policy-gradient methods, as the former exploit the structured prior knowledge encoded in the generative model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Deriving discrete adjoint variables from a statistical perspective to bypass non-differentiability is an elegant and genuinely theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic experiments thoroughly validate the theory, but the application scope is narrow and advantages on GSM8K/MATH500 are marginal.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear; the parallel presentation of statistical and control-theoretic perspectives is well-structured.
- Value: ⭐⭐⭐⭐ — Provides a principled theoretical framework for fine-tuning discrete diffusion models, though practical impact depends on the broader adoption of diffusion-based LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](loopholing_discrete_diffusion_deterministic_bypass_of_the_sampling_wall.md)
- [\[CVPR 2026\] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](../../CVPR2026/image_generation/diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)
- [\[ICLR 2026\] Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning](embracing_discrete_search_a_reasonable_approach_to_causal_structure_learning.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)

</div>

<!-- RELATED:END -->
