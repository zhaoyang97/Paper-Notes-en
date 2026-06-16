---
title: >-
  [Paper Note] Persuasive Privacy
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper reformulates "privacy" as the relative scoring rule loss of a Receiver under the worst-case data-prior using a Sender–Receiver two-party Stackelberg game and Bayesian Persuasion. It provides a unified definition $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP, subsumes pure DP and probabilistic DP as special
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: 696e190018febe07
---
# Persuasive Privacy

**Conference**: ICML 2026  
**arXiv**: [2601.22945](https://arxiv.org/abs/2601.22945)  
**Code**: None  
**Area**: AI Safety / Privacy Theory  
**Keywords**: Differential Privacy, Bayesian Persuasion, Scoring Rules, Stackelberg Games, Deterministic Mechanisms

## TL;DR
This paper reformulates "privacy" as the relative scoring rule loss of a Receiver under the worst-case data-prior using a Sender–Receiver two-party Stackelberg game and Bayesian Persuasion. It provides a unified definition $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP, subsumes pure DP and probabilistic DP as special cases, and provides non-trivial formal privacy guarantees for **deterministic algorithms** (e.g., noiseless empirical means) for the first time.

## Background & Motivation

**Background**: Over the past two decades, differential privacy (DP) and its variants (Rényi DP, $f$-divergence privacy, pufferfish privacy, QIF, etc.) have become the de facto standards for data privacy. Based on algebraic criteria where "output distributions on adjacent datasets are nearly indistinguishable," they provide clean worst-case guarantees and favorable composition/post-processing properties.

**Limitations of Prior Work**: DP faces three long-criticized issues in engineering deployment: the parameter $\varepsilon$ is difficult to explain to regulators and the public; semantic interpretations (such as Kasiviswanathan–Smith or Wasserman–Zhou) are post-hoc patches that struggle to map directly to realistic concerns about "what I am afraid of leaking"; and actual deployments often exhibit excessively large or meaningless privacy budgets. More critically, DP and nearly all its variants **cannot** provide non-trivial guarantees for deterministic mechanisms (any deterministic function that outputs differently for adjacent datasets is judged non-private), yet scenarios like the US Decennial Census rely heavily on deterministic "invariant statistics."

**Key Challenge**: DP models "privacy" as the indistinguishability of adjacent output distributions. This purely algorithmic definition cannot incorporate the Sender's preferences regarding "what specifically is feared to be leaked." Meanwhile, the implicit assumption of a worst-case "universal prior" automatically disqualifies deterministic mappings. To simultaneously achieve (i) semantic tailorability, (ii) interpretability, and (iii) coverage of deterministic algorithms, one must explicitly model the Sender, Receiver, and utility functions in a game.

**Goal**: Construct a game-theoretic meta-framework that can (a) generate new privacy definitions as needed while retaining rigorous proofs, (b) conversely evaluate existing DP-family guarantees, and (c) provide non-trivial guarantees for deterministic algorithms.

**Key Insight**: Data publishing is viewed as a variant of Bayesian Persuasion—the Sender holds the ground truth $x$ and commits to a Markov kernel $M$; the Receiver holds a prior $Q$, observes $T\sim M(x,\cdot)$, and makes a Bayesian decision. Defining the negative of the Sender's "privacy function" $\rho(d,x)$ as the Receiver's loss (information asymmetry + shared utility + Sender robustness), and combined with classic results from Grünwald–Dawid, the loss induced by the Receiver's optimal decision automatically constitutes a **proper scoring rule**. This places "privacy assessment" and "probabilistic prediction" within the same mathematical language.

**Core Idea**: Define the "relative privacy score" $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ as the improvement in the Receiver's ability to predict the ground truth before and after publication. A unified PP definition is established using a $\kappa$-tail probability condition over the triple worst-case of $x$, Receiver prior $Q\in\mathcal{Q}_x$, and randomness $T$.

## Method

### Overall Architecture
This paper addresses the lack of an interpretable, tailorable unified privacy definition that covers deterministic algorithms by rewriting data publishing as a Sender–Receiver Stackelberg game. The Sender holds ground truth $x$ and publicly commits to a mechanism $M$ (Markov kernel) and its privacy class; the Receiver holds prior $Q$ and makes a Bayes-rational decision after observing $T\sim M(x,\cdot)$. Using the Grünwald–Dawid equivalence, the loss induced by the Receiver's optimal decision is automatically a proper scoring rule. Consequently, "privacy" is expressed as the upper bound of the worst-case "relative improvement in the Receiver's predictive ability" regarding the ground truth. The framework follows three steps: establishing game semantics and transparency assumptions, translating Sender preferences into scoring rules, and formulating a unified PP inequality. The paper then verifies self-consistency by subsuming pure/probabilistic DP as special cases and clarifying the distinction between "receiver post-processing" and "sender post-processing."

### Key Designs

**1. Game Semantics and Scoring Rule Formulation: Translating "Sender Preferences" into Proper Scoring Rules**

To make privacy tailorable to "specific leakage concerns," the first step is to explicitly model Sender preferences. The Sender holds $x\in\mathsf{X}$ and publicly commits to mechanism $M:(\mathsf{X},\mathcal{T})\to[0,1]$ and its privacy class $\mathfrak{C}$ (Assumption 1: transparency, $M$ and $\mathfrak{C}$ are data-independent). The Receiver holds prior $Q\in\mathcal{P}$ and makes a Bayes-rational decision upon observing $T\sim M(x,\cdot)$ (Assumption 2). The Sender uses a "privacy function" $\rho:(\mathcal{D},\mathsf{X})\to\mathbb{R}$ to represent preferences over Receiver decisions. A critical step is Proposition 1: if the Receiver's loss function $\ell = \rho$ (Assumption 3), the Sender gains a robust worst-case data-averaged loss adversary model. The Receiver's optimal decision is $d^P\in\arg\inf_d \mathbb{E}_{X\sim P}[\rho(d,X)]$, and the induced $S(P,x)=\rho(d^P,x)$ is exactly a negatively-oriented proper scoring rule (Proposition 2, essentially the conclusion of Grünwald–Dawid 2004), termed the privacy score. This effectively puts "privacy assessment" and "probabilistic prediction" into the same mathematical language—every proper SR corresponds to a Bayesian decision problem and vice versa.

**2. Privacy = Relative Score + Worst-case Prior Class: Tuning Adversary Strength (Definition 3)**

Addressing the issue where DP's implicit "all-knowing adversary" assumption invalidates deterministic algorithms, this paper defines privacy via relative rather than absolute scores. $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ measures the "improvement in the Receiver's prediction of $x$ after seeing $T$." Mechanism $M$ is $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-Persuasive Private if and only if:

$$\inf_{S\in\mathcal{S}}\inf_{x\in\mathsf{X}}\inf_{Q\in\mathcal{Q}_x}\mathbb{P}_x[\Delta_S(Q,T,x)\le\kappa]\ge 1-\delta$$

The quadruple corresponds to the set of protected semantics $\mathcal{S}$, the adversary prior class $\mathcal{Q}_x$, the maximum allowed privacy loss $\kappa$, and the failure probability $\delta$. Using the relative $\Delta_S$ is crucial: if an absolute score were used, all mechanisms would be "equally private" when the Receiver already knows everything ($Q=\delta_x$), causing the definition to degenerate. The relative difference ensures the mechanism "does not significantly shift the Receiver's belief toward the truth." The worst case spans three dimensions: $x$ across the data space (Assumption 5), $Q$ across the Sender-specified adversary class $\mathcal{Q}_x$, and $T$ across mechanism randomness (Assumption 4). This retains DP-style algebraic robustness while making adversary strength a tunable knob via $\mathcal{Q}_x$.

**3. DP as a Special Case of PP: Explaining What DP Protects (Proposition 6)**

The self-consistency of the framework is demonstrated by its role as a "semantic interpreter for DP." By choosing $L$ as the discrete negative log-score and the adversary class $\mathcal{H}=\{Q\in\mathcal{P}_2:\exists(x,x')\in\mathfrak{N},Q(\{x,x'\})=1\}$ (priors over adjacent pairs), $M$ is $(\varepsilon,\delta)$-PDP if and only if $M$ is $(L,\mathcal{H},\varepsilon,\delta)$-PP (where $\delta=0$ yields pure $\varepsilon$-DP). The proof reveals that the minimum is reached as the Receiver's prior $Q(\{x\})\to 0$. This provides a new interpretation of DP: "$(\varepsilon,\delta)$-DP protects against information gain even when the Receiver initially places almost no belief in the truth," explaining why $\varepsilon$ is difficult to tune or explain in practice. By replacing the tail probability with expectation, one can similarly recover Rényi DP and broader $f$-divergence privacy (Appendix C).

**4. Decoupling Receiver vs. Sender Post-processing (Def 5–6, Prop 4–5)**

PDP has long been criticized for "not satisfying the post-processing inequality." This paper argues that this criticism conflates two concepts. The traditional $M\in\mathfrak{C}\Rightarrow MK\in\mathfrak{C}$ conflates arbitrary post-processing by the Receiver after receiving the output (receiver post-processing: $M\otimes K$) with additional transformations by the Sender before publication (sender post-processing: $MK$ publishing only the marginal). All PP guarantees satisfy the former (Proposition 4, the true adversary robustness), even if some do not satisfy the latter (Proposition 5). The remedy is simple: the Sender publishes both the original output and the transformed result to reduce it to the receiver case. This decoupling recalibrates the qualitative assessment of PDP as losing an algebraic tool rather than adversary robustness.

### Loss & Training
The framework is **definitional rather than training-oriented**: there are no parameters to learn; "training" corresponds to the Sender selecting a mechanism $M$ to satisfy the PP inequality. Theoretical analysis utilizes various proper scoring rules (Dawid–Sebastiani score, negative log-score, interval score, etc.) to recover different DP variants. Composition rules require $\mathcal{Q}_x$ to be conjugate to the considered mechanisms (Definition 4), which holds naturally for Gaussian prior families.

## Key Experimental Results

As a theoretical work, this paper does not involve large-scale numerical experiments but provides two illustrative cases to verify that the framework covers scenarios where DP fails.

### Main Results (Relation between PP and existing definitions)

| Existing Definition | Corresponding PP Instantiation | Scoring Rule $S$ | Adversary Prior Class $\mathcal{Q}_x$ | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Pure $\varepsilon$-DP | $(L,\mathcal{H},\varepsilon,0)$-PP | Negative Log-score $L$ | Adjacent Pair Class $\mathcal{H}$ | Degeneration of Prop 6 at $\delta=0$ |
| $(\varepsilon,\delta)$-PDP | $(L,\mathcal{H},\varepsilon,\delta)$-PP | Negative Log-score $L$ | $\mathcal{H}$ | Proposition 6 |
| Rényi DP | Expectation-based PP | Negative Log-score | $\mathcal{H}$ | Assumption 4 as Expectation |
| $f$-divergence privacy | Expectation PP + $f$-score | $f$-divergence score | $\mathcal{H}$ | Appendix C |
| **Noiseless Mean** | $(\mathcal{I},\mathcal{G}_x^r,\kappa,\delta)$-PP | Marginal Dawid–Sebastiani | Gaussian + Quality Constraints | Not provable by DP; provable by PP |

### Key Findings
- **DP's Implicit Assumption Unmasked**: The proof of Prop 6 identifies the extremum at $Q(\{x\})\to 0$, implying DP protects against an extreme adversary who initially disbelieves the ground truth, explaining why $\varepsilon$ is counter-intuitive.
- **Deterministic Mechanisms are not Naturally Unprivate**: By tailoring adversary strength $\mathcal{Q}_x$ (e.g., Gaussian priors with bounded condition numbers) and scoring rules (Marginal DSS), noiseless empirical means can be proven PP for sufficiently large $n$, aligning with long-standing intuition in Statistical Disclosure Control (SDC).
- **Bayesian Persuasion Insights**: The framework is essentially a Sender-commit, Receiver-best-respond Stackelberg game, but introduces three key differences from classic BP: information asymmetry (Sender knows $x$), shared utility ($\ell=\rho$), and Sender robustness (worst-case).

## Highlights & Insights
- **First-principles instead of post-hoc semantics**: Unlike previous DP semantic interpretations which were derived after the fact, PP derives DP from game theory and scoring rules, providing native answers to why parameters like $\exp(\varepsilon)$ appear.
- **Tailorable Adversary Class $\mathcal{Q}_x$**: Explicitly parameterizing "how strong the adversary is assumed to be" (which is implicit in DP) is highly valuable for engineering; regulators can discuss $\mathcal{Q}_x$ instead of an abstract $\varepsilon$.
- **Proper Scoring Rule as Privacy Language**: The Grünwald–Dawid equivalence ensures $S$ and $(\rho,\mathcal{D})$ are interchangeable, meaning any statistical prediction tool (CRPS, Brier score, etc.) can define new privacy semantics.
- **Post-processing Decoupling**: Distinguishing between sender and receiver post-processing suggests that definitions like PDP should not be dismissed due to lack of algebraic closure, but rather viewed as a trade-off for tighter privacy parameters.

## Limitations & Future Work
- **Theoretical Framework, Deployment Gap**: The paper does not provide best practices for selecting $\mathcal{Q}_x$ in industrial pipelines; calibrating and auditing adversary prior classes remains future work.
- **Strong Conjugacy Conditions**: Composition rules rely on $\mathcal{Q}_x$ being conjugate to mechanisms; composition outside Gaussian or simple conjugate families (e.g., neural network posterior approximations) is unresolved.
- **Scalar/Simple Deterministic Examples**: Section 5 only covers empirical means and linear constraints; PP guarantees for complex deterministic releases (like full Census invariant statistics) require case-by-case construction.
- **No Integration with DP Libraries**: Existing $(\varepsilon, \delta)$ deployments have not been translated into PP form to identify their corresponding $\mathcal{Q}_x$.

## Related Work & Insights
- **vs Differential Privacy (Dwork et al. 2006)**: DP is a special case of $(L,\mathcal{H},\varepsilon,\delta)$-PP. PP provides first-principles semantics and extends to deterministic mechanisms at the cost of sender post-processing closure.
- **vs Pufferfish (Kifer & Machanavajjhala 2014)**: Both allow custom "secrets" and "adversary priors." PP adds the Sender as the first player in a game and enforces semantics via proper SR.
- **vs Bayesian Persuasion (Kamenica & Gentzkow 2011)**: BP assumes the Sender does not know the truth and utility is symmetric; PP modifies this for privacy by assuming the Sender knows the truth and requires robustness.
- **vs Quantitative Information Flow / $f$-divergence privacy**: PP subsumes these through expectation-based variants (Appendix C–D), unifying the "information flow" and "scoring rule" perspectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Sender–Receiver Stackelberg + proper SR reformulates DP into first-principles derivation, enabling the first non-trivial privacy proofs for deterministic mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical; covers DP/PDP/Rényi DP and empirical mean cases with rigorous proofs, though lacks real-world deployment case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The conceptual stack is logically progressive; assumptions are clearly numbered and discussed, making it highly accessible for a theoretical privacy paper.
- Value: ⭐⭐⭐⭐⭐ Provides a "meta-framework" for reconfigurable, tailorable, and auditable privacy definitions, offering immediate theoretical tools for SDC and official statistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)
- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](../../ICLR2026/ai_safety/unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)

</div>

<!-- RELATED:END -->
