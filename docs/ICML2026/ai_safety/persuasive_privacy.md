---
title: >-
  [Paper Note] Persuasive Privacy
description: >-
  [ICML 2026][AI Safety][Differential Privacy] This paper reformulates "privacy" as the relative scoring rule loss of a Receiver under the worst-case data-prior using a Sender-Receiver Stackelberg game and Bayesian Persuas…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Bayesian Persuasion"
  - "Scoring Rules"
  - "Stackelberg Game"
  - "Deterministic Mechanisms"
date: 2026-05-08
content_hash: 28acbd6fbd68539b
---

# Persuasive Privacy

**Conference**: ICML 2026  
**arXiv**: [2601.22945](https://arxiv.org/abs/2601.22945)  
**Code**: None  
**Area**: AI Safety / Privacy Theory  
**Keywords**: Differential Privacy, Bayesian Persuasion, Scoring Rules, Stackelberg Game, Deterministic Mechanisms

## TL;DR
This paper reformulates "privacy" as the relative scoring rule loss of a Receiver under the worst-case data-prior using a Sender-Receiver Stackelberg game and Bayesian Persuasion framework. It provides a unified definition $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP, subsumes pure and probabilistic DP as special cases, and provides non-trivial formal privacy guarantees for **deterministic algorithms** (such as noiseless empirical means) for the first time.

## Background & Motivation

**Background**: Over the past two decades, differential privacy (DP) and its variants (Rényi DP, $f$-divergence privacy, pufferfish privacy, QIF, etc.) have become the de facto standards for data privacy. These are based on algebraic criteria regarding the near-indistinguishability of output distributions on adjacent datasets, providing clean worst-case guarantees and favorable composition/post-processing properties.

**Limitations of Prior Work**: DP faces three long-standing criticisms in engineering implementation: the parameter $\varepsilon$ is difficult to explain to regulators and the public; semantic interpretations (e.g., Kasiviswanathan–Smith, Wasserman–Zhou) are post-hoc patches that struggle to map directly to realistic concerns of "what I am afraid of leaking"; and actual deployments often result in $\varepsilon$ values so large they become meaningless. Crucially, DP and almost all its variants **cannot** provide non-trivial guarantees for deterministic mechanisms (any deterministic function with different outputs for adjacent datasets is judged non-private), yet scenarios like the US Decennial Census rely heavily on deterministic "invariant statistics."

**Key Challenge**: DP models "privacy" as the indistinguishability of adjacent output distributions. This purely algorithmic definition cannot incorporate a Sender's preferences regarding "what specifically is feared to be leaked." Simultaneously, the implicit assumption of a worst-case "universal prior" automatically excludes deterministic mappings. To achieve a framework that is (i) semantically tailorable, (ii) interpretable, and (iii) applicable to deterministic algorithms, the Sender, Receiver, and utility functions in the game must be explicitly modeled.

**Goal**: Construct a game-theoretic meta-framework that can (a) generate new privacy definitions as needed while maintaining rigorous proofs, (b) conversely evaluate existing DP guarantees, and (c) provide non-trivial guarantees for deterministic mechanisms.

**Key Insight**: Data release is viewed as a variant of Bayesian Persuasion—the Sender holds the ground truth $x$ and commits to a Markov kernel $M$; the Receiver holds a prior $Q$, observes $T\sim M(x,\cdot)$, and makes a Bayesian decision. The negation of the Sender's "privacy function" $\rho(d,x)$ is the Receiver's loss (asymmetric information + shared utility + robust Sender). Combined with classic results from Grünwald–Dawid, the loss induced by the Receiver's optimal decision automatically constitutes a **proper scoring rule**, placing "privacy assessment" and "probabilistic prediction" within the same mathematical language.

**Core Idea**: Define the "relative privacy score" $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ as the improvement in the Receiver's ability to predict the ground truth before and after release. A unified PP definition is given as a $\kappa$-tail probability condition across the triple worst case of $x$, Receiver prior $Q\in\mathcal{Q}_x$, and randomness $T$.

## Method

### Overall Architecture
The framework consists of a three-layer structure:

**Layer 1 (Game Semantics)**: A Sender→Receiver Stackelberg game. The Sender holds the true value $x\in\mathsf{X}$ and publicly commits to a mechanism $M:(\mathsf{X},\mathcal{T})\to[0,1]$ (Markov kernel) belonging to a privacy class $\mathfrak{C}$ (Assumption 1: transparency, $M$ and $\mathfrak{C}$ are data-independent). The Receiver holds a prior $Q\in\mathcal{P}$ and makes a Bayes-rational decision after observing $T\sim M(x,\cdot)$ (Assumption 2).

**Layer 2 (Scoring Rule Formulation)**: The Sender uses a "privacy function" $\rho:(\mathcal{D},\mathsf{X})\to\mathbb{R}$ to represent preferences over different Receiver decisions. Proposition 1 proves that if the Receiver's loss function $\ell=\rho$ (Assumption 3), the Sender simultaneously obtains the worst-case data-averaged loss, representing the most robust adversary model. Under this assumption, the Receiver's optimal decision is $d^P\in\arg\inf_d \mathbb{E}_{X\sim P}[\rho(d,X)]$, and the induced $S(P,x)=\rho(d^P,x)$ is a negatively-oriented **proper scoring rule** (Proposition 2, essentially the conclusion of Grünwald–Dawid 2004), termed the privacy score.

**Layer 3 (PP Definition)**: The relative score $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ measures the "improvement in the Receiver's ability to predict $x$ after observing $T$." A mechanism $M$ is $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-Persuasive Private if and only if:

$$\inf_{S\in\mathcal{S}}\inf_{x\in\mathsf{X}}\inf_{Q\in\mathcal{Q}_x}\mathbb{P}_x[\Delta_S(Q,T,x)\le\kappa]\ge 1-\delta$$

The quadruple $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$ corresponds to: the set of semantics to be protected, the considered class of adversary priors, the maximum allowable privacy loss, and the failure probability.

### Key Designs

1. **Privacy = Relative Score + Worst-case Prior Class (Definition 3)**:
    - **Function**: Unifies all "privacy" definitions as a worst-case upper bound on how much the Receiver's posterior prediction improved.
    - **Mechanism**: The relative value $\Delta_S$ is used instead of the absolute $S(Q_T,x)$. If an absolute value were used, and a Receiver already had a prior $Q=\delta_x$ (fully informed), all mechanisms would appear "equally private," causing the definition to degenerate. By using the difference, the mechanism only needs to "prevent the Receiver's belief from shifting significantly toward the truth." The worst case spans three dimensions: $x$ across the data space (Assumption 5), $Q$ across the Sender-restricted adversary class $\mathcal{Q}_x$, and $T$ across mechanism randomness (Assumption 4, $\kappa$-$\delta$ tail condition).
    - **Design Motivation**: The triple worst case retains DP-style algebraic robustness, while the explicit $\mathcal{Q}_x$ turns "how strong the Sender assumes the adversary to be" into a tunable knob, preventing deterministic algorithms from being dismissed by the implicit DP assumption that the "adversary knows everything."

2. **DP as a Special Case of PP (Proposition 6)**:
    - **Function**: Uses PP as a "semantic interpreter" for DP to explain what DP is actually protecting.
    - **Mechanism**: By setting $L$ as the discrete negative log-score and the adversary class as $\mathcal{H}=\{Q\in\mathcal{P}_2:\exists(x,x')\in\mathfrak{N},Q(\{x,x'\})=1\}$ (two-point alternative-hypothesis priors, corresponding to DP's adjacent pairs), $M$ is $(\varepsilon,\delta)$-PDP if and only if $M$ is $(L,\mathcal{H},\varepsilon,\delta)$-PP; $\delta=0$ corresponds to pure $\varepsilon$-DP. The proof reveals that the minimum is reached where the Receiver's prior probability of the truth $Q(\{x\})\to 0$—providing a new interpretation: "$(\varepsilon,\delta)$-DP protects against information gain even when the Receiver initially has almost no belief in the ground truth." Replacing the tail probability with expectation recovers Rényi DP and broader $f$-divergence privacy (Appendix C).
    - **Design Motivation**: Previous DP semantics were constructed post-hoc (e.g., hypothesis testing). PP places DP directly into a first-principles game-theoretic derivation, allowed for each assumption to be tested and relaxed against real-world scenarios.

3. **Receiver vs. Sender Post-processing Distinction (Definition 5–6, Prop 4–5)**:
    - **Function**: Clarifies whether the fact that PDP "does not satisfy the post-processing inequality" is actually a problem.
    - **Mechanism**: The traditional post-processing inequality ("$M\in\mathfrak{C}\Rightarrow MK\in\mathfrak{C}$") confuses two things: arbitrary processing by the Receiver after obtaining the output (receiver post-processing: $M\otimes K$) vs. the Sender applying a transformation before release (sender post-processing: $MK$ where only the marginal is released). All PP guarantees satisfy the former (Proposition 4, true "adversary robustness"); however, some PP guarantees do not satisfy the latter (Proposition 5). The remedy is simple: if the Sender releases both the raw output of $M$ and the result of $K$ (i.e., $M\otimes K$), it reduces to the receiver case.
    - **Design Motivation**: This clarification refines the long-criticized lack of a post-processing inequality in PDP—it only loses an algebraic tool for "using complex mechanisms to prove simple ones," not actual adversary robustness. It also lays the foundation for composition rules ($\kappa_1+\kappa_2$, $\delta_1+\delta_2$, Proposition 3) under conjugate prior families.

### Loss & Training
The framework is **definitional rather than training-based**: there are no parameters to learn. "Training" corresponds to the Sender choosing a mechanism $M$ that satisfies the PP inequality. Theoretical analysis utilizes proper scoring rules (Dawid–Sebastiani score, negative log-score, interval score, etc.) to recover different DP variants. Composition rules require $\mathcal{Q}_x$ to be conjugate to the mechanisms considered (Definition 4), which holds naturally for Gaussian prior families.

## Key Experimental Results

As a purely theoretical work, it does not involve large-scale numerical experiments, but it provides two illustrative cases to verify that the framework "covers scenarios DP cannot."

### Main Results Table (PP Relationship with Existing Definitions)

| Existing Definition | Corresponding PP Instantiation | Scoring Rule $S$ | Adversary Prior Class $\mathcal{Q}_x$ | Remarks |
|---------|---------------|--------------|----------------------------|------|
| Pure $\varepsilon$-DP | $(L,\mathcal{H},\varepsilon,0)$-PP | Neg. Log-score $L$ | Adjacent two-point priors $\mathcal{H}$ | Prop 6 where $\delta=0$ |
| $(\varepsilon,\delta)$-PDP | $(L,\mathcal{H},\varepsilon,\delta)$-PP | Neg. Log-score $L$ | $\mathcal{H}$ | Proposition 6 |
| Rényi DP | Expectation-based PP (App. C) | Neg. Log-score | $\mathcal{H}$ | Assump. 4 changed to expectation |
| $f$-divergence privacy | Exp. PP + $f$ score | $f$-divergence score | $\mathcal{H}$ | Appendix C |
| **Noiseless Mean** | $(\mathcal{I},\mathcal{G}_x^r,\kappa,\delta)$-PP | Marginal DSS | Gaussian + Prior quality constraint | Proved by PP, impossible for DP |

### Key Findings
- **Uncovering DP's Implicit Assumptions**: The proof of Prop 6 locates DP's extreme point at $Q(\{x\})\to 0$. This implies DP protects an extreme adversary who "almost doesn't believe the truth," explaining why $\varepsilon$ is difficult to tune or explain in practice.
- **Deterministic Mechanisms are not Inherently Unprivate**: By tailoring the adversary strength $\mathcal{Q}_x$ (e.g., Gaussian priors with condition number constraints) and the scoring rule (Marginal DSS) to the scenario, noiseless empirical means can be proven PP for sufficiently large $n$, aligning with long-standing intuition in Statistical Disclosure Control (SDC).
- **Insights from Bayesian Persuasion**: The framework is essentially a Sender-commit, Receiver-best-respond Stackelberg game. It incorporates three key differences from classic BP: asymmetric information (Sender knows $x$), shared utility ($\ell=\rho$), and a robust Sender (worst-case).

## Highlights & Insights
- **First-principles vs. Post-hoc Semantics**: Unlike previous DP interpretations derived after the fact, PP derives DP from game theory and scoring rules. This provides native answers to questions like "why the $\exp(\varepsilon)$ factor."
- **Tunable Adversary Class $\mathcal{Q}_x$**: Transforming "how strong the adversary is" from an implicit assumption (DP's default "knows all adjacent info") to an explicit parameter is valuable for engineering; regulators can discuss $\mathcal{Q}_x$ instead of an abstract $\varepsilon$.
- **Proper Scoring Rules as a Privacy Language**: The Grünwald–Dawid equivalence ensures $S$ and $(\rho, \mathcal{D})$ are interchangeable. This means any statistical prediction tool (CRPS, Brier score, etc.) can be used to define new privacy semantics.
- **Engineering Implications of Post-processing Split**: By distinguishing sender vs. receiver post-processing, "imperfect" definitions like PDP are no longer dismissed; they instead provide a trade-off curve—sacrificing sender post-processing for tighter privacy parameters.

## Limitations & Future Work
- **Theoretical Framework, Engineering Gap**: The paper does not provide best practices for choosing $\mathcal{Q}_x$ in industrial pipelines; the calibratability and auditability of prior classes require further work.
- **Strong Conjugacy Conditions**: Composition rules rely on $\mathcal{Q}_x$ being conjugate to the mechanism. How composition works outside conjugate families (e.g., mixture priors, neural network posterior approximations) remains unsolved.
- **Limited Deterministic Examples**: Section 5 only covers empirical means and linear constraints; PP guarantees for complex deterministic releases (e.g., full invariant statistics for censuses) must be constructed per scenario.
- **Lack of Integration with DP Libraries**: There is no translation of existing real-world deployments (OpenDP, Google, Apple) into PP format to determine which $\mathcal{Q}_x$ they actually correspond to.

## Related Work & Insights
- **vs. Differential Privacy (Dwork et al. 2006)**: DP is a special case of $(L,\mathcal{H},\varepsilon,\delta)$-PP. PP provides first-principles semantics and extends to deterministic mechanisms, though it sacrifices the sender post-processing closure of DP (remediable via $M\otimes K$).
- **vs. Pufferfish (Kifer & Machanavajjhala 2014)**: Both allow users to define "secrets" and "adversary priors." PP additionally treats the Sender as the first player in a game and requires semantics to come from proper SRs, resulting in a more symmetric theoretical structure.
- **vs. Bayesian Persuasion (Kamenica & Gentzkow 2011)**: DP assumes the Sender does not know the truth and has information symmetry with the Receiver. PP modifies this specifically for privacy contexts with Sender knowledge, utility correlation, and robustness.
- **vs. Quantitative Information Flow / $f$-divergence privacy**: PP subsumes these through expectation-based variants, unifying "information flow" and "scoring rule" perspectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewriting DP from an algebraic definition to a game-theoretic first-principles derivation using proper SRs is a major breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient theoretical proof covering DP/PDP/Rényi DP and mean-release cases, though lacking a real deployment case study.
- Writing Quality: ⭐⭐⭐⭐⭐ The conceptual progression is excellent; assumptions are clearly numbered and discussed.
- Value: ⭐⭐⭐⭐⭐ Provides a reconfigurable, tailorable "meta-framework" for privacy, offering immediately useful theoretical tools for SDC and official statistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)

</div>

<!-- RELATED:END -->
