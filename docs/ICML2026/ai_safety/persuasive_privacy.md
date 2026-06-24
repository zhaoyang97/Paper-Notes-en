---
title: >-
  [Paper Note] Persuasive Privacy
description: >-
  [ICML 2026][AI Safety][Differential Privacy] This paper reformulates "privacy" as the relative scoring rule loss of a Receiver under the worst-case data-prior using a Sender–Receiver two-party Stackelberg game and Bayesian Persuasion. It provides a unified definition $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP, which subsumes pure DP and probabilistic DP as special cases, while providing non-trivial formal privacy guarantees for **deterministic algorithms** (e.g.…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Bayesian Persuasion"
  - "Scoring Rules"
  - "Stackelberg Game"
  - "Deterministic Mechanisms"
date: 2026-05-08
content_hash: f7a6112615ae05ce
---

# Persuasive Privacy

**Conference**: ICML 2026  
**arXiv**: [2601.22945](https://arxiv.org/abs/2601.22945)  
**Code**: None  
**Area**: AI Safety / Privacy Theory  
**Keywords**: Differential Privacy, Bayesian Persuasion, Scoring Rules, Stackelberg Game, Deterministic Mechanisms

## TL;DR
This paper reformulates "privacy" as the relative scoring rule loss of a Receiver under the worst-case data-prior using a Sender–Receiver two-party Stackelberg game and Bayesian Persuasion. It provides a unified definition $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP, which subsumes pure DP and probabilistic DP as special cases, while providing non-trivial formal privacy guarantees for **deterministic algorithms** (e.g., noiseless empirical mean) for the first time.

## Background & Motivation

**Background**: Over the past two decades, differential privacy (DP) and its variants (Rényi DP, $f$-divergence privacy, pufferfish privacy, QIF...) have become the de facto standards for data privacy. They are based on algebraic criteria where "output distributions on adjacent datasets are nearly indistinguishable," providing clean worst-case guarantees and favorable composition/post-processing properties.

**Limitations of Prior Work**: DP faces three long-standing issues in engineering deployment: the parameter $\varepsilon$ is difficult to explain to regulators and the public; semantic interpretations (e.g., Kasiviswanathan–Smith, Wasserman–Zhou) are post-hoc patches that struggle to map directly to real-world concerns of "what I am afraid of leaking"; and actual deployments often result in excessively large or meaningless privacy budgets. More critically, DP and almost all its variants **cannot** provide non-trivial guarantees for deterministic mechanisms (any deterministic function that differs on adjacent datasets is judged non-private), yet scenarios like the US Decennial Census rely heavily on deterministic "invariant statistics."

**Key Challenge**: DP models "privacy" as the indistinguishability of adjacent output distributions. This purely algorithmic definition cannot incorporate a Sender's preferences regarding specific leakage concerns. Simultaneously, the implicit assumption of a worst-case "universal prior" automatically disqualifies deterministic mappings. To simultaneously achieve (i) semantic tailorability, (ii) explainability, and (iii) coverage of deterministic algorithms, the Sender, Receiver, and utility functions in the game must be explicitly modeled.

**Goal**: Construct a game-theoretic meta-framework that can (a) generate new privacy definitions as needed while retaining rigorous proofs, (b) conversely evaluate existing DP-family guarantees, and (c) provide non-trivial guarantees for deterministic mechanisms.

**Key Insight**: Data publishing is viewed as a variant of Bayesian Persuasion—the Sender holds the truth $x$ and commits to a Markov kernel $M$; the Receiver holds a prior $Q$, observes $T\sim M(x,\cdot)$, and makes a Bayesian decision. The negative of the Sender's "privacy function" $\rho(d,x)$ is the Receiver's loss (information asymmetry + shared utility + Sender robustness). Combined with the classic results of Grünwald–Dawid, the loss induced by the Receiver's optimal decision automatically constitutes a **proper scoring rule**, placing "privacy evaluation" and "probabilistic prediction" within the same mathematical language.

**Core Idea**: Define "relative privacy score" $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ as the improvement in the Receiver's ability to predict the truth before and after the release. A unified PP definition is established using a $\kappa$-tail probability condition over the triple worst-case of $x$, Receiver prior $Q\in\mathcal{Q}_x$, and randomness $T$.

## Method

### Overall Architecture
This paper addresses the lack of an interpretable, tailorable, and deterministic-compatible privacy definition by rewriting data publishing as a Sender–Receiver Stackelberg game. The Sender holds the truth $x$ and publicly commits to a mechanism $M$ (Markov kernel) and its privacy class; the Receiver holds a prior $Q$, observes the output $T\sim M(x,\cdot)$, and makes a Bayes-rational decision. Via the Grünwald–Dawid equivalence, the loss from the Receiver's optimal decision is a proper scoring rule, allowing "privacy" to be expressed as the upper bound of the "relative improvement in the Receiver's prediction capability." The framework follows three steps: establishing game semantics and transparency assumptions, translating Sender preferences into scoring rules, and defining unified PP inequalities. The framework's consistency is then verified by subsuming pure/probabilistic DP and clarifying the distinction between "receiver post-processing" and "sender post-processing."

### Key Designs

**1. Game Semantics and Scoring Rules: Translating "Sender Preferences" into proper scoring rules**

To make privacy tailorable to specific leakage concerns, the Sender's preferences are explicitly modeled. The Sender holds $x\in\mathsf{X}$ and commits to mechanism $M:(\mathsf{X},\mathcal{T})\to[0,1]$ and its privacy class $\mathfrak{C}$ (Assumption 1: transparency, $M$ and $\mathfrak{C}$ are data-independent). The Receiver holds prior $Q\in\mathcal{P}$, observes $T\sim M(x,\cdot)$, and makes a Bayes-rational decision (Assumption 2). The Sender uses a "privacy function" $\rho:(\mathcal{D},\mathsf{X})\to\mathbb{R}$ to represent preferences over Receiver decisions. Proposition 1 shows that if the Receiver's loss $\ell=\rho$ (Assumption 3), the Sender adopts the most robust worst-case data-averaged loss adversary model. The Receiver's optimal decision $d^P\in\arg\inf_d \mathbb{E}_{X\sim P}[\rho(d,X)]$ induces $S(P,x)=\rho(d^P,x)$, which is precisely a negatively-oriented proper scoring rule (Proposition 2, essentially Grünwald–Dawid 2004), termed the privacy score. This aligns "privacy evaluation" with "probabilistic prediction."

**2. Privacy = Relative Score + Worst-case Prior Class: Tuning Adversary Strength (Definition 3)**

Addressing the issue where DP's "all-knowing adversary" assumption invalidates deterministic algorithms, privacy is defined via relative rather than absolute scores. $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ measures the gain in prediction ability. $M$ is $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-Persuasive Private if and only if:

$$\inf_{S\in\mathcal{S}}\inf_{x\in\mathsf{X}}\inf_{Q\in\mathcal{Q}_x}\mathbb{P}_x[\Delta_S(Q,T,x)\le\kappa]\ge 1-\delta$$

The quadruple represents the set of protected semantics $\mathcal{S}$, the adversary prior class $\mathcal{Q}_x$, the maximum allowed privacy loss $\kappa$, and failure probability $\delta$. Using relative $\Delta_S$ is critical: with absolute values, if a Receiver has a perfectly informed prior $Q=\delta_x$, all mechanisms appear equally private, and the definition trivializes. Relative scoring ensures the mechanism limits the Receiver's belief shift toward the truth. The worst-case spans the data space (Assumption 5), the Sender-specified adversary class $\mathcal{Q}_x$, and mechanism randomness (Assumption 4). This maintains DP-style algebraic robustness while making adversary strength a tunable parameter.

**3. DP as a Special Case of PP: Game-theoretic Interpretation of DP (Proposition 6)**

The framework acts as a "semantic interpreter" for DP. By setting $L$ as the discrete log-loss score and taking the adversary class $\mathcal{H}=\{Q\in\mathcal{P}_2:\exists(x,x')\in\mathfrak{N},Q(\{x,x'\})=1\}$ (two-point alternative-hypothesis priors corresponding to DP's adjacent pairs), $M$ is $(\varepsilon,\delta)$-PDP if and only if $M$ is $(L,\mathcal{H},\varepsilon,\delta)$-PP. Setting $\delta=0$ yields pure $\varepsilon$-DP. The proof reveals that the minimum is reached as the Receiver's prior probability $Q(\{x\})\to 0$. This gives a new interpretation: $(\varepsilon,\delta)$-DP protects against information gain when the Receiver initially places almost no belief in the truth, explaining why $\varepsilon$ is difficult to tune. Replacing tail probabilities with expectations recovers Rényi DP and $f$-divergence privacy (Appendix C).

**4. Receiver vs Sender Post-processing: Clarifying PDP Post-processing "Deficiencies" (Def 5–6, Prop 4–5)**

PDP is often criticized for not satisfying the "post-processing inequality." This paper argues this confuses two concepts. Traditional $M\in\mathfrak{C}\Rightarrow MK\in\mathfrak{C}$ conflates arbitrary post-processing by the Receiver (receiver post-processing: $M\otimes K$) with additional transformations by the Sender before release (sender post-processing: $MK$). All PP guarantees satisfy the former (Proposition 4: true adversary robustness), even if some do not satisfy the latter (Proposition 5). The remedy is simple: the Sender releases the original output of $M$ alongside the transform $K$, which reduces to the receiver case. This recalibration suggests that "missing" property in PDP is merely an algebraic convenience for proving privacy of complex mechanisms, not a loss of robustness.

### Loss & Training
The framework is **definitional rather than training-based**: there are no parameters to learn. "Training" corresponds to the Sender selecting a mechanism $M$ to satisfy the PP inequality. Theoretical analysis utilizes proper scoring rules (Dawid–Sebastiani score, log-loss, interval score, etc.) to recover DP variants. Composition rules require $\mathcal{Q}_x$ to be conjugate to the mechanisms (Definition 4), which holds for Gaussian prior families.

## Key Experimental Results

As a theoretical work, there are no large-scale numerical experiments. Instead, two illustrative cases demonstrate coverage of scenarios DP cannot handle.

### Main Results (Relation between PP and existing definitions)

| Existing Definition | Corresponding PP Instance | Scoring Rule $S$ | Adversary Prior Class $\mathcal{Q}_x$ | Remarks |
|---------|---------------|--------------|----------------------------|------|
| Pure $\varepsilon$-DP | $(L,\mathcal{H},\varepsilon,0)$-PP | Log-loss $L$ | Adjacent two-point class $\mathcal{H}$ | Prop 6 where $\delta=0$ |
| $(\varepsilon,\delta)$-PDP | $(L,\mathcal{H},\varepsilon,\delta)$-PP | Log-loss $L$ | $\mathcal{H}$ | Proposition 6 |
| Rényi DP | Expected PP (Appendix C) | Log-loss | $\mathcal{H}$ | Assump 4 changed to expectation |
| $f$-divergence privacy | Expected PP + $f$-score | $f$-divergence score | $\mathcal{H}$ | Appendix C |
| **Noiseless empirical mean** | $(\mathcal{I},\mathcal{G}_x^r,\kappa,\delta)$-PP | Marginal Dawid–Sebastiani | Gaussian prior + condition constraints | PP provable, DP fails |

### Key Findings
- **DP's Implicit Assumptions Exposed**: The proof of Prop 6 locates DP's extreme point at $Q(\{x\})\to 0$, suggesting DP protects against a semantically counter-intuitive adversary who initially disbelieves the truth.
- **Deterministic Mechanisms are not Inherently Unprivate**: By tailoring adversary strength $\mathcal{Q}_x$ and scoring rules (Marginal DSS), the noiseless empirical mean can be proven PP for sufficiently large $n$, aligning with long-standing intuitions in Statistical Disclosure Control (SDC).
- **Bayesian Persuasion Insights**: The framework is essentially a Sender-commit, Receiver-best-respond Stackelberg game, but introduces three key differences from classic BP: information asymmetry, shared utility, and Sender robustness.

## Highlights & Insights
- **First-principles over post-hoc semantics**: Instead of deriving semantics after the DP definition, PP derives DP from first principles (games and scoring rules), providing a native answer to "why $\exp(\varepsilon)$."
- **Tailorable Adversary Class $\mathcal{Q}_x$**: Makes the "assumed adversary strength" an explicit parameter. Regulators can discuss $\mathcal{Q}_x$ rather than the obscure $\varepsilon$.
- **Proper Scoring Rule as Privacy Language**: The Grünwald–Dawid equivalence ensures any statistical prediction tool (CRPS, Brier score, etc.) can define new privacy semantics.
- **Post-processing Bifurcation**: Distinguishing sender vs receiver post-processing suggests that definitions like PDP are not "broken" but offer a trade-off: sacrificing sender post-processing closure for tighter privacy parameters.

## Limitations & Future Work
- **Theoretical framework, engineering gap**: Lacks best practices for choosing $\mathcal{Q}_x$ in industrial pipelines.
- **Strong Conjugacy Requirements**: Composition rules depend on $\mathcal{Q}_x$ being conjugate to the mechanism. Composition for non-conjugate families (e.g., neural posteriors) remains unsolved.
- **Limited Deterministic Examples**: Only covers empirical means and linear constraints; PP guarantees for complex deterministic releases (e.g., full census statistics) require per-scenario construction.
- **Integration with DP Libraries**: Does not yet map real $(\varepsilon,\delta)$ deployments to PP forms to identify their implied $\mathcal{Q}_x$.

## Related Work & Insights
- **vs Differential Privacy (Dwork et al. 2006)**: DP is a special case of $(L,\mathcal{H},\varepsilon,\delta)$-PP. PP provides first-principles semantics and expansion to deterministic mechanisms.
- **vs Pufferfish (Kifer & Machanavajjhala 2014)**: Both allow custom "secrets" and "prior classes." PP treats the Sender as the first player and requires semantics to stem from proper scoring rules.
- **vs Bayesian Persuasion (Kamenica & Gentzkow 2011)**: BP assumes Sender ignorance and information symmetry. PP modifies this for privacy contexts (Sender knows truth, utility correlation, robustness).
- **vs Quantitative Information Flow / $f$-divergence privacy**: PP subsumes these through expectation-based variants (Appendix C–D), unifying "information flow" and "scoring rule" perspectives.

## Rating
- Novelness: ⭐⭐⭐⭐⭐ Rewriting DP via Stackelberg + proper SR allows the first non-trivial privacy proofs for deterministic mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ Pure theory. Covers DP/PDP/Rényi DP and empirical mean cases; lacks real-world deployment case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from privacy functions to scoring rules to PP definitions is highly readable.
- Value: Provides a "meta-framework" for reconfigurable privacy definitions, immediately useful for SDC and official statistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICLR 2026\] Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding](../../ICLR2026/ai_safety/privacy_beyond_pixels_latent_anonymization_for_privacy-preserving_video_understa.md)

</div>

<!-- RELATED:END -->
