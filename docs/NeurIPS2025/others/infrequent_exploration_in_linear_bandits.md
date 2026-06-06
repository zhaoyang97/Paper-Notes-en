---
title: >-
  [Paper Note] Infrequent Exploration in Linear Bandits
description: >-
  [NeurIPS 2025][linear bandits] This paper proposes the INFEX framework, which executes a baseline algorithm (e.g., LinUCB/LinTS) at designated exploration steps according to a given schedule and selects arms greedily at…
tags:
  - "NeurIPS 2025"
  - "linear bandits"
  - "infrequent exploration"
  - "greedy policy"
  - "regret bound"
  - "exploration-exploitation"
date: 2026-05-08
content_hash: 4080149117547e3d
---

# Infrequent Exploration in Linear Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2510.26000](https://arxiv.org/abs/2510.26000)  
**Code**: None  
**Area**: Other
**Keywords**: linear bandits, infrequent exploration, greedy policy, regret bound, exploration-exploitation

## TL;DR
This paper proposes the INFEX framework, which executes a baseline algorithm (e.g., LinUCB/LinTS) at designated exploration steps according to a given schedule and selects arms greedily at all other time steps. It is proven that as long as the number of exploration steps exceeds $\omega(\log T)$, INFEX achieves the same poly-logarithmic regret as full-time exploration while substantially reducing computational overhead (80%–99% of time steps are greedy).

## Background & Motivation

**Background**: In linear bandit problems, UCB and Thompson Sampling perform exploration at every step, yielding optimal theoretical guarantees with logarithmic or sublinear regret. At the other extreme, purely greedy policies require strong distributional assumptions (e.g., sufficient context diversity) to succeed, and typically incur linear regret in the fixed arm-set setting.

**Limitations of Prior Work**: Continuous exploration can be harmful or unethical in healthcare and safety-critical systems, while purely greedy strategies necessarily fail in the absence of context diversity. The intermediate regime—whether infrequent exploration suffices to guarantee near-optimal regret—has received little systematic study.

**Key Challenge**: The literature lacks a rigorous analytical framework characterizing how exploration frequency affects regret. $\epsilon$-greedy explores occasionally but achieves suboptimal regret of $O(T^{2/3})$; Explore-Then-Commit (ETC) separates phases but is likewise suboptimal.

**Core Problem**: (a) Is per-step exploration necessary to achieve logarithmic regret? (b) What is the threshold lower bound on exploration frequency? (c) Can infrequent exploration simultaneously yield computational benefits?

**Core Idea**: Design a general-purpose framework, INFEX, which takes an arbitrary baseline exploration algorithm and an exploration schedule as inputs, explores at scheduled time steps, and acts greedily otherwise. The framework theoretically guarantees that asymptotic regret is unaffected by exploration frequency, provided it exceeds $\omega(\log T)$.

## Method

### Overall Architecture
INFEX takes two inputs: (1) a baseline exploration algorithm $\mathsf{Alg}$ (e.g., LinUCB or LinTS); and (2) an exploration schedule $\mathcal{T}_e \subset \mathbb{N}$ specifying which time steps are designated for exploration. At each step $t$: if $t \in \mathcal{T}_e$, the arm is selected according to $\mathsf{Alg}$ and the reward is observed; otherwise, the agent greedily selects $X_t = \arg\max_{x} x^\top \hat{\theta}_{t-1}$ using the ridge regression estimate $\hat{\theta}_{t-1}$.

### Key Designs

1. **Modular Exploration Framework**:

    - Function: Explicitly decouples exploration from exploitation, allowing any baseline algorithm to serve as the exploration component.
    - Mechanism: $\mathsf{Alg}$ is invoked at time steps $t \in \mathcal{T}_e$; greedy selection is used at all other steps. The exploration frequency function $f(t) = |\mathcal{T}_e \cap \{1,\ldots,t\}|$ characterizes the cumulative number of exploration steps up to $t$.
    - Design Motivation: The framework is highly general (compatible with LinUCB, LinTS, or any algorithm with logarithmic regret guarantees) and confines computationally intensive exploration operations to sparse time steps, reducing overall computational cost.

2. **Regret Decomposition and Analysis**:

    - Function: Establishes a regret upper bound under infrequent exploration.
    - Mechanism: The total regret is decomposed into three terms: (a) the regret of the baseline algorithm over $f(T)$ exploration steps, $\mathcal{R}_{\mathsf{Alg}}(f(T))$; (b) a constant term $G_{\text{const}}$ independent of $T$ that depends on the density of the exploration schedule; and (c) the regret incurred during greedy steps, $G(T) = O\!\left(\frac{(\log T + d\log\log T)^2}{\Delta}\right)$.
    - Key Lemma: The regret of greedy steps is related to the number of times the optimal arm is selected, $N_{\text{opt}}(t)$—each selection of the optimal arm reduces the estimation error of $\hat{\theta}$ with respect to $x^*$, thereby controlling subsequent greedy-step regret.

3. **Necessity of the Exploration Threshold (Theorem 3)**:

    - Function: Proves that $\omega(\log T)$ exploration steps are necessary.
    - Mechanism: A counterexample is constructed showing that if $f(t) \neq \omega(\log t)$, there exists a problem instance for which the expected regret of INFEX is at least $\Omega(T^{1-\epsilon})$.
    - Design Motivation: This establishes a non-improvable theoretical lower bound and identifies a phase transition threshold for exploration frequency.

### Theoretical Results

**Theorem 1 (INFEX Regret)**: If the baseline $\mathsf{Alg}$ achieves logarithmic regret and $f(t) = \omega(\log t)$, then

$$\mathcal{R}_{\text{INFEX}}(T) \leq \mathcal{R}_{\mathsf{Alg}}(f(T)) + G_{\text{const}}(\tau_{\mathsf{Alg}}, f) + O\left(\frac{(\log T + d\log\log T)^2}{\Delta}\right)$$

where only the last term depends on $T$, and it is consistent with the instance-dependent bound of LinUCB.

**Theorem 2 (New Bound for LinTS)**: The first instance-dependent logarithmic regret upper bound for LinTS is established.

## Key Experimental Results

### Main Results — Regret Comparison ($d=10$, $T=10000$)

| Method | K=10 Regret | K=100 Regret | K=1000 Regret |
|--------|-------------|--------------|---------------|
| LinUCB (full exploration) | Baseline | Baseline | Baseline |
| INFEX(LinUCB, m=5) | ≈Baseline or better | ≈Baseline or better | ≈Baseline or better |
| INFEX(LinUCB, m=20) | ≈Baseline | ≈Baseline | Slightly above baseline |
| INFEX(LinUCB, m=100) | Slightly above baseline | Slightly above baseline | Above baseline |
| LinTS (full exploration) | Baseline | Baseline | Baseline |
| INFEX(LinTS, m=5/20/100) | **All outperform baseline** | **All outperform baseline** | **All outperform baseline** |
| Pure Greedy | Linear regret | Linear regret | Linear regret |
| $\epsilon$-greedy | Suboptimal | Suboptimal | Suboptimal |

### Computational Efficiency Comparison

| Method | K=10 | K=100 | K=1000 |
|--------|------|-------|--------|
| LinUCB | 1× | 1× | 1× |
| INFEX(LinUCB, m=5) | ~5× speedup | ~5× speedup | ~3× speedup |
| INFEX(LinUCB, m=100) | ~20× speedup | ~15× speedup | ~10× speedup |
| LinTS | 1× | 1× | 1× |
| INFEX(LinTS, m=5) | ~3× speedup | ~3× speedup | ~2× speedup |

### Key Findings
- **INFEX(LinTS) achieves lower regret than pure LinTS**: Occasional greedy steps prevent over-exploration by Thompson Sampling—an unexpected yet meaningful finding.
- Fixed periods ranging from $m=5$ to $m=100$ (80%–99% greedy) consistently maintain near-optimal regret.
- OLSBandit is computationally inefficient due to the requirement for $\Omega(d^2 \log T)$ forced sampling steps.
- $\epsilon$-greedy has an $\Omega(T^{2/3})$ regret lower bound and cannot achieve logarithmic regret.

## Highlights & Insights
- **Bridging a Theoretical Gap**: A complete theoretical framework is established between the two extremes of per-step exploration and pure greedy, proving that $\omega(\log T)$ is the exact threshold for asymptotically optimal regret.
- **Modular Design**: INFEX is a meta-algorithm that can wrap any linear bandit algorithm, offering flexibility and ease of integration into existing systems.
- **Practical Value**: In safety-critical applications (e.g., healthcare, finance), restricting exploration to 1%–20% of time steps while retaining theoretical guarantees is of significant practical relevance.
- **New Regret Bound for LinTS**: Theorem 2 establishes the first instance-dependent bound for LinTS, which is of independent value beyond the INFEX framework.

## Limitations & Future Work
- The analysis requires a positive minimum gap $\Delta > 0$ and a unique optimal arm, and does not directly extend to contextual linear bandits with time-varying arm sets.
- The $\omega(\log T)$ threshold applies to deterministic schedules; adaptive schedules may be superior but are analytically more challenging.
- Only instance-dependent regret is analyzed; the optimal exploration strategy under minimax regret ($O(\sqrt{T})$) remains an open problem.
- Empirical validation is limited to synthetic data, with no experiments on real-world applications such as recommendation systems or clinical trials.

## Related Work & Insights
- **vs. LinUCB/LinTS**: INFEX replaces most time steps with greedy selection, substantially reducing computation while maintaining or improving regret.
- **vs. $\epsilon$-greedy**: $\epsilon$-greedy has an insurmountable $\Omega(T^{2/3})$ regret lower bound; INFEX achieves logarithmic regret.
- **vs. ETC (Explore-Then-Commit)**: ETC explores exclusively in the early phase and exploits exclusively thereafter; INFEX alternates throughout, avoiding ETC's suboptimal regret.
- **vs. Pure Greedy**: Pure greedy requires strong distributional assumptions, whereas INFEX requires no context diversity assumption.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fills an important theoretical gap in the study of exploration frequency in linear bandits.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments are thorough, but real-world validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical development is rigorous, theorem statements are clear, and discussions are thorough.
- Value: ⭐⭐⭐⭐ Significant theoretical contributions with direct practical relevance for safety-critical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](../../ICML2026/others/a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[NeurIPS 2025\] Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits](gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] Alias-Free ViT: Fractional Shift Invariance via Linear Attention](alias-free_vit_fractional_shift_invariance_via_linear_attention.md)

</div>

<!-- RELATED:END -->
