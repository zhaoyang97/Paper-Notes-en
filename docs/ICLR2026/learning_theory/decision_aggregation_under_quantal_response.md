---
title: >-
  [Paper Note] Decision Aggregation under Quantal Response
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper investigates the aggregation of binary decisions from $n$ experts under bounded rationality characterized by quantal response. It proves that when collective rationality is below a threshold $g(n)$ dependent on the group size, naive majority voting is the minimax optimal robust aggregator. Furthermore, group
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: e41c0b8eed18ae9a
---
# Decision Aggregation under Quantal Response

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=VtN1z92lvu](https://openreview.net/forum?id=VtN1z92lvu)  
**Code**: To be confirmed  
**Area**: Learning Theory / Information Aggregation / Bounded Rationality  
**Keywords**: Decision Aggregation, Quantal Response, Bounded Rationality, Majority Voting, Minimax Regret

## TL;DR
This paper investigates the aggregation of binary decisions from $n$ experts under bounded rationality characterized by quantal response. It proves that when collective rationality is below a threshold $g(n)$ dependent on the group size, naive majority voting is the minimax optimal robust aggregator. Furthermore, groups with bounded rationality can unexpectedly outperform fully rational groups because randomness in decision-making encodes weak signals that are lost in deterministic behavior; this phenomenon is empirically validated using the temperature parameter of LLMs as a natural "rationality knob."

## Background & Motivation
**Background**: Information aggregation—combining judgments from multiple experts into a collective decision—is a core problem in collective intelligence. Classic approaches (e.g., robust predictive aggregation, Condorcet’s Jury Theorem) mostly assume experts are **fully rational**, meaning they perform precise Bayesian inference upon receiving signals and choose actions maximizing expected utility. In the absence of knowledge about the joint distribution of signals ("signal structure"), researchers use "worst-case regret" to evaluate aggregators: the maximum performance gap compared to an omniscient aggregator that knows everything.

**Limitations of Prior Work**: Real-world experts are rarely fully rational. Humans exhibit cognitive biases and noisy judgments. Moreover, when applying this theory to AI, LLMs naturally perform **stochastic selection** via temperature parameters—higher temperatures lead to more random outputs. The assumption of full rationality fits neither humans nor machines. Existing works focus either on full rationality or adversarial experts, lacking an analysis that systematically incorporates a "bounded rationality" framework into robust aggregation.

**Key Challenge**: Intuitively, being "less rational" should be detrimental—the average utility (0.43) of an individual "John" (who follows intuition and sometimes bets against it) is lower than that of a fully rational "Mia" (0.5). However, the paper provides a counter-intuitive observation: a group of Johns can achieve a utility of 0.51 after aggregation, surpassing a group of Mias who are always at 0.5. **Individual "errors" actually inject information at the collective level**. The problem is: under realistic constraints where the signal structure is unknown, can this advantage be sustained? What aggregator should be used to realize it?

**Goal**: Under the setting of bounded rationality modeled by quantal response and conditionally independent and identically distributed (c.i.i.d.) signals, answer two sub-questions: (1) Without knowing the signal structure, what aggregator is minimax optimal? (2) Can bounded rationality actually outperform full rationality at the group level, and if so, under what conditions?

**Key Insight**: By using the McKelvey-Palfrey quantal response function $\psi_\lambda$ to compress the "degree of rationality" into a scalar parameter $\lambda$, the aggregation problem can be framed as a minimax regret optimization over signal structures. The key insight is that bounded rationality **regularizes** the space of possible reporting structures, smoothing out extreme structures that favor complex or non-monotonic aggregators, thereby allowing simple majority voting to emerge as the optimal choice.

**Core Idea**: Parameterize bounded rationality as $\psi_\lambda$ and prove, within the minimax regret framework, that "sufficiently low rationality leads to majority voting optimality and the bounded rationality advantage." The authors further point out that the LLM temperature $t$ is effectively $1/\lambda$, making the theory empirically testable.

## Method

### Overall Architecture
Consider a state $\omega\in\{0,1\}$ with a known prior $\mu=\Pr[\omega=1]$. A decision-maker (DM) faces $n$ anonymous experts. Expert $i$ observes a private signal $S_i$, where signals are conditionally independent and identically distributed (c.i.i.d.) given $\omega$. Experts do not report posteriors; instead, they provide a binary decision $X_i\in\{0,1\}$ generated via a quantal response function. The DM only observes the count of positive reports $X=\sum_i X_i$ and uses an aggregator $f:\{0,\dots,n\}\to[0,1]$ to output a prediction (where $f(x)$ is the probability of guessing $\omega=1$ given $x$ ones).

Utility is defined as "$1$ for a correct guess, $-1$ for an incorrect one." Since the DM and experts share the same objective, there is no incentive for strategic lying. The evaluation metric is **minimax regret**: the maximum gap between the DM and an **omniscient aggregator** $\mathrm{opt}_{\hat\theta}$ that knows the true report structure $\hat\theta$. The objective is to solve:

$$\mathrm{opt}_{\hat\Theta}\in\arg\min_f\ \max_{\hat\theta\in\hat\Theta}\ R(f,\hat\theta),\qquad R(f,\hat\theta)=U(\mathrm{opt}_{\hat\theta},\hat\theta)-U(f,\hat\theta).$$

The difficulty lies in the infinite-dimensional space of signal structures. The main strategy is to **use a geometric dimensionality reduction to compress the infinite-dimensional problem into a three-parameter problem**, then prove the optimality threshold of majority voting and the advantage of bounded rationality within this three-parameter space.

### Key Designs

**1. Quantal Response: Modeling "Rationality" as Temperature-Adjustable Logistic Choice**

In classic aggregation, experts are binary Bayesian optimizers, which is unrealistic and prevents quantification of rationality levels. This paper adopts the McKelvey-Palfrey quantal response function to randomize decisions. Let $v$ be the expected utility difference between buying ($X=1$) and selling ($X=0$) given signal $S$, where $v=E[u(1,\omega)\mid S]\in[-1,1]$. The probability of reporting $1$ is:

$$\varphi_\lambda(v)=\frac{e^{\lambda v}}{e^{-\lambda v}+e^{\lambda v}}=\frac{1}{1+e^{-2\lambda v}}.$$

Substituting the posterior $p=\Pr[\omega=1\mid S_i]$ (where $v=2p-1$) yields the response as a function of the posterior: $\psi_\lambda(p)=\dfrac{1}{1+e^{2\lambda(1-2p)}}$. The parameter $\lambda$ serves as the "rationality knob": $\lambda=0$ corresponds to pure randomness (coin flip), while $\lambda\to\infty$ recovers a deterministic threshold rule. This form is **mathematically identical** to a softmax with temperature in LLM output layers—if internal logits are viewed as expected utilities, rationality $\lambda$ is equivalent to the inverse temperature $1/t$.

**2. Minimax Regret Framework: Omniscient Benchmarking and Robust Aggregation**

Since the DM does not know the signal structure, direct utility maximization is impossible. The **omniscient aggregator** $\mathrm{opt}_{\hat\theta}$ serves as an unattainable upper bound, deciding based on whether $\Pr_{\hat\theta}[\omega=1\mid X=x]$ exceeds 0.5. The **robust aggregator** minimizes the worst-case gap $R(f,\hat\theta)$. This design transforms the realistic constraint of "unknown signal structure" into an analyzable mathematical object.

**3. Three-Signal Dimensionality Reduction: A Geometric Lemma for Tractability**

Minimax optimization over all c.i.i.d. signal structures is infinite-dimensional. The key technical contribution is a geometric reduction: since expert reports depend only on their posteriors $s\in[0,1]$, each posterior can be encoded as a point on a curve in $\mathbb{R}^3$. All valid report structures constitute the **convex hull** of this curve. The paper proves a critical lemma: **any four points on the curve are non-coplanar**. Combined with Carathéodory's theorem, any report structure can be equivalently represented by a signal structure with **at most three posteriors** $\{0,p,1\}$. This collapses the optimization into a manageable three-parameter space.

**4. Main Theorem: Majority Voting Threshold $g(n)$ and Bounded Rationality Advantage**

In the three-parameter space, the paper proves two conclusions. First, **Optimality of Majority Voting**: when $\lambda\le g(n)$, majority voting $f^{\mathrm{maj}}$ is the minimax optimal robust aggregator. Intuitively, lower $\lambda$ (more randomness) regularizes the report structure space, making simple majority voting more robust than complex non-monotonic aggregators. Particularly for $n\le 2$, $g(n)=\infty$.

Second, **Bounded Rationality Advantage**: While full rationality is best for a single expert ($n=1$), for any $n\ge 2$, there exists a signal structure $\theta^*$ and a finite $\lambda^*$ such that the optimal utility under bounded rationality **strictly exceeds** that of full rationality. For $n>2$, this advantage is achievable even with simple majority voting. The proof is constructive: a structure $\theta^*$ is designed where fully rational experts consistently report 0 (utility 0.5), but finite $\lambda^*$ introduces "noise" that encodes hidden information, allowing the aggregator to achieve utility strictly greater than 0.5.

## Key Experimental Results

### Main Results
Two empirical studies using `gpt-4o-mini` were conducted, where temperature $t$ mapped to rationality $\lambda$, to verify the theory's predictions.

| Study | Task | Setting | Key Finding |
|------|------|------|----------|
| Bayesian Decision | Ball-and-urn posterior inference | 400 scenarios × 20 repetitions, $t\in\{0,0.5,1\}$ | As temperature increases, $\lambda$ decreases: $t=0$ gives $\lambda\to\infty$, $t=0.5$ gives $\lambda=13.25$, $t=1.0$ gives $\lambda=8.93$. Confirmed LLMs follow quantal response. |
| Multiple-choice QA | MathQA | 500 questions × 20 responses, plurality voting $f^{\mathrm{plu}}$, $n\in\{1,3,5\}$ | For $n\ge 3$, high-temperature aggregation is more accurate, replicating the "bounded rationality advantage." |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| $n=1$, Low $t=0$ | Highest utility/accuracy | Randomness is purely harmful for a single expert. |
| $n=1$, High $t\ge0.5$ | Slight decrease | A single random expert is worse than a deterministic one. |
| $n=3,5$, High $t=0.5/1$ | Counter-intuitive increase | Under multiple experts, randomness provides informational diversity that improves aggregate performance. |

### Key Findings
- **Group Scale as a Switch**: Randomness (high temperature/low $\lambda$) is a burden for individuals but an asset for groups. When $n$ increases from 1 to 3, the effect of temperature flips from "harmful" to "beneficial," aligning with the predicted $g(n)$ threshold.
- **Temperature = Inverse Rationality**: The $\lambda$ fitted via logistic regression decreases monotonically with temperature, quantitatively establishing the $\lambda\leftrightarrow 1/t$ mapping.
- **Moderation over Extremes**: The U-shaped curve of worst-case regret suggests an optimal moderate level of rationality; pursuing perfect determinism ($t=0$) is suboptimal in group settings.

## Highlights & Insights
- **The Paradox of Informative Errors**: The study formalizes the folk wisdom that "group randomness can uncover deep truths." Fully rational experts may collectively discard weak signals, while the "jitter" of bounded rationality leaks these signals to the aggregator.
- **Hardcore Geometric Reduction**: Reducing the infinite-dimensional signal space to three posteriors $\{0,p,1\}$ via the "no-four-points-coplanar" property is a elegant and powerful proof technique that is transferable to other robust aggregation problems.
- **Quantal Response ≡ Temperature Softmax**: This structural equivalence bridges decades of behavioral economics with modern LLM sampling, providing a theoretical explanation for optimal temperature settings in ensemble methods and self-consistency.
- **Vindication of Majority Voting**: Within the bounded rationality regime, the simplest majority vote is minimax optimal, providing clear guidance against unnecessarily complex weighting schemes in engineering.

## Limitations & Future Work
- **Strong c.i.i.d. Assumption**: Real-world experts often have correlated or even adversarial signals, which the current model ignores.
- **Homogeneous Rationality**: The model assumes all experts share the same $\lambda$, without considering heterogeneity in rationality levels within a group.
- **Binary States + Shared Utility**: The restriction to binary states and shared objectives (no strategic lying) limits the scope. Extending this to multi-state or adversarial settings remains a challenge.
- **Experimental Scale**: The study used `gpt-4o-mini` with small $n$ (up to 5). Robustness in larger groups or more complex tasks (e.g., medical diagnosis) requires further verification.

## Related Work & Insights
- **vs. Classic Robust Aggregation**: While previous works sought robust aggregation under full rationality, this paper is the first to integrate quantal response into a minimax framework and use geometric reduction to reveal the paradoxical benefits of bounded rationality.
- **vs. Quantal Response Equilibrium (QRE)**: Unlike the QRE literature focused on strategic manipulation with heterogeneous utilities, this work focuses on honest aggregation with a common goal.
- **vs. LLM Temperature Studies**: Previous research often viewed temperature through the lens of individual accuracy or creativity. This work formally links temperature to collective decision theory, providing a theoretical foundation for self-consistency and ensemble accuracy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies bounded rationality, minimax robust aggregation, and LLM temperature with a provable "bounded rationality advantage."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solidly validates theoretical predictions, though the tasks and group sizes are relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear narrative, naturally bridging introductory stories with complex theory and empirical testing.
- **Value**: ⭐⭐⭐⭐ Significant conceptual and practical implications for collective intelligence, AI ensembles, and voting mechanisms.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Minimax-Optimal Aggregation for Density Ratio Estimation](minimax-optimal_aggregation_for_density_ratio_estimation.md)
- [\[ICLR 2026\] Online Decision-Focused Learning](online_decision-focused_learning.md)
- [\[ICLR 2026\] Conformalized Decision Risk Assessment](conformalized_decision_risk_assessment.md)
- [\[ICLR 2026\] Dimension-Free Decision Calibration for Nonlinear Loss Functions](dimension-free_decision_calibration_for_nonlinear_loss_functions.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)

</div>

<!-- RELATED:END -->
