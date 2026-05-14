---
title: >-
  [Paper Note] Procurement Auctions with Predictions: Improved Frugality for Facility Location
description: >-
  [NeurIPS 2025][LLM Safety][procurement auctions] This paper studies procurement auction design for the strategic uncapacitated facility location problem. It proves that the frugality ratio of the classical VCG auction is…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "procurement auctions"
  - "facility location"
  - "frugality"
  - "learning-augmented mechanisms"
  - "VCG auctions"
date: 2026-05-08
content_hash: 970e85b5210f7593
---

# Procurement Auctions with Predictions: Improved Frugality for Facility Location

**Conference**: NeurIPS 2025
**arXiv**: [2512.09367](https://arxiv.org/abs/2512.09367)
**Code**: None
**Area**: LLM Safety
**Keywords**: procurement auctions, facility location, frugality, learning-augmented mechanisms, VCG auctions

## TL;DR

This paper studies procurement auction design for the strategic uncapacitated facility location problem. It proves that the frugality ratio of the classical VCG auction is exactly 3 (improving the previously known upper bound of 4), and designs learning-augmented auction mechanisms that exploit prediction information to achieve near-optimal frugality when predictions are accurate, while maintaining constant-factor robustness when predictions are arbitrarily inaccurate.

## Background & Motivation

When government agencies, retail chains, or banks need to open new facilities to serve customers, they face the classical **Uncapacitated Facility Location (UFL)** problem: balancing customer connection costs (distance to the nearest open facility) against facility opening costs.

However, most existing work assumes that the designer has full knowledge of opening costs, which is often unrealistic in practice. Each location may be owned by a distinct strategic agent whose true opening cost is **private information**, giving agents an incentive to misreport costs in order to receive higher compensation. To prevent manipulation, one must design **truthful auctions** that incentivize agents to report costs honestly.

A key challenge in evaluating procurement auction performance is the lack of an appropriate benchmark. The **frugality** literature proposes using the cost of the "second-best solution" as a benchmark:
- If the second-best solution is much costlier than the optimal, the instance resembles a monopoly and the institution must pay more.
- If the second-best solution is close to the optimal, a good auction can exploit competition to reduce costs.
- The **frugality ratio** quantifies the ratio of the auction's total payment to the cost of the second-best solution.

Previously, nearly all work on frugal mechanism design operated within an adversarial framework (worst-case analysis), assuming no prior information about private costs. In practice, however, institutions can often obtain **predicted values** of costs through historical data, expert estimates, or data-driven models. This motivates the central question: **Can prediction information be exploited to improve frugality while maintaining robustness when predictions are inaccurate?**

## Method

### Overall Architecture

**Problem Formulation**: Given a set of users $U$ and a set of facility locations $L$, each facility $\ell$ has a private opening cost $o_\ell$, and connection costs $d(u, \ell)$ form a metric space. An auction mechanism $\mathcal{M}$ receives bids $b_\ell$ from facility owners, and outputs a subset $S$ of facilities to open together with payments $p_\ell$ for each facility.

**Key Definitions**:
- **Truthfulness**: Reporting the true cost is a dominant strategy for every facility.
- **Frugal solution**: A second-best solution $F$ that contains none of the facilities in the optimal solution.
- **Frugality ratio**: $\text{frugality}(\mathcal{M}) = \max_{U, \mathbf{o}, d} \frac{p(\mathcal{M})}{c(F)}$

The paper presents three main results: a tight analysis of the VCG auction, the learning-augmented auction PredictedLimits, and the error-tolerant auction ErrorTolerant.

### Key Designs

**Result 1: The Frugality Ratio of VCG is Exactly 3**

The previously known upper bound for the VCG frugality ratio was 4. The authors prove the exact value of 3 via a tighter analysis and a matching lower bound.

The core of the upper bound proof is a **rerouting argument**: for each facility in the winning set, its threshold payment is upper-bounded by reassigning its users to other facilities (either within the winning set or in the frugal solution). This is formalized through a Unified Payment Upper Bound Lemma (Lemma 3.1), which serves as the foundation for all subsequent results.

The lower bound is established via a star-metric construction: $k$ users, $k+1$ facilities (one central, $k$ peripheral). The VCG total cost is $3k$, the frugal solution cost is $k+2$, and the ratio approaches 3.

**Result 2: Learning-Augmented Auction PredictedLimits**

The core innovation is to leverage predictions $\hat{o}_\ell$ of facility opening costs to improve the frugality ratio. The mechanism accepts a parameter $\epsilon \in (0, 2]$:

1. Compute the predicted optimal solution $\widehat{\text{OPT}}$ based on predicted costs.
2. Define a modified cost function: if facility $\ell \in \widehat{\text{OPT}}$ and its reported cost exceeds the predicted value, scale it up by $\frac{2}{\epsilon}$.
3. Select the solution minimizing the modified cost.

$$o'_\ell(S) = \begin{cases} \frac{2}{\epsilon} \cdot o_\ell, & \text{if } S = \widehat{\text{OPT}} \text{ and } o_\ell > \hat{o}_\ell \\ o_\ell, & \text{otherwise} \end{cases}$$

**Intuition**: Unlike typical learning-augmented mechanisms that shrink predicted costs, this mechanism **scales costs upward**. This stems from the fundamental difference in objective — minimizing payments rather than social cost. By amplifying potentially inflated bids, the auction can more tightly control facility payments.

A key property is that the scaling is **solution-dependent** rather than facility-dependent: the scaling is triggered only when evaluating $\widehat{\text{OPT}}$ as a complete set, and alternative solutions are unaffected. This ensures that the negative impact of inaccurate predictions is confined to the evaluation of the single predicted optimal solution.

**Guarantees**:
- Consistency (accurate predictions): $(1 + \epsilon)$
- Robustness (arbitrarily inaccurate predictions): $\max\{5, 3 + \frac{2}{\epsilon}\}$

**Result 3: Error-Tolerant Auction ErrorTolerant**

Define the prediction error as $\eta = \max_\ell \max\{\hat{o}_\ell / o_\ell, o_\ell / \hat{o}_\ell\}$ and introduce a tolerance parameter $\lambda > 1$.

Two-phase verification mechanism:
- If all facilities in $\widehat{\text{OPT}}$ satisfy $o_\ell \leq \lambda \hat{o}_\ell$: scale down the total cost of $\widehat{\text{OPT}}$ by $1/\lambda^2$ (aggressive scaling ensures $\widehat{\text{OPT}}$ is selected even under errors of magnitude $\eta \leq \lambda$).
- Otherwise: fall back to the per-facility penalty logic of PredictedLimits.

### Loss & Training

This is a theoretical work and involves no training. The core analytical tool is the Unified Payment Upper Bound Lemma (Lemma 3.1), which bounds threshold payments via a carefully constructed rerouting map $\pi_f$ that reassigns users of a removed facility in ranked order.

## Key Experimental Results

### Main Results: Frugality Ratio Guarantees of Different Mechanisms

| Mechanism | Consistency (Accurate Predictions) | Robustness (Worst Case) | Notes |
|:---|:---:|:---:|:---|
| VCG (no predictions) | — | 3 (tight) | Improves prior upper bound of 4 |
| PredictedLimits ($\epsilon = 0.1$) | 1.1 | 23 | Near-optimal consistency |
| PredictedLimits ($\epsilon = 1$) | 2 | 5 | Balanced setting |
| PredictedLimits ($\epsilon = 2$) | 3 | 4 | Optimal robustness |

### Error-Tolerant Mechanism Analysis: Frugality Ratio of ErrorTolerant

| Condition | Frugality Ratio Upper Bound | Notes |
|:---|:---:|:---|
| $\eta = 1$ (perfect predictions) | $1 + \lambda + 2\epsilon$ | Approaches optimal as $\lambda, \epsilon \to 0$ |
| $\eta \leq \lambda$ (error within tolerance) | $\eta(1 + \lambda) + 2\epsilon$ | Grows linearly with error |
| $\eta > \lambda$ (error exceeds tolerance) | $\max\{2\lambda^4 + 3\lambda^2, 3 + \frac{2}{\epsilon}\}$ | Falls back to constant-factor guarantee |

### Key Findings

1. **The tight frugality ratio of VCG is 3**: This is a substantive improvement over the upper bound of 4 due to Talwar (2003), with tightness established via a constructive lower bound. The proof relies on a finer rerouting analysis than previously known.

2. **Consistency–robustness tradeoff**: The parameter $\epsilon$ governs the tradeoff between performance under accurate predictions and worst-case guarantees. Smaller $\epsilon$ yields near-optimal frugality (approaching 1) when predictions are accurate, at the cost of degraded robustness in the worst case.

3. **Counterintuitive scaling direction**: Unlike existing learning-augmented mechanisms that typically shrink predicted costs, the proposed mechanism scales costs upward. This reflects the fundamental difference in objective — minimizing payments rather than social cost.

4. **Solution-set dependence is critical**: Cost scaling acts only on the complete predicted optimal solution, localizing the negative impact of inaccurate predictions and serving as the key design principle enabling good robustness.

5. **Layered guarantees of the error-tolerant mechanism**: ErrorTolerant provides three tiers of guarantees — perfect predictions, approximately accurate predictions, and arbitrary predictions — making it more flexible for practical deployment.

## Highlights & Insights

- The **Unified Payment Upper Bound Lemma** (Lemma 3.1) is the technical centerpiece of the paper, unifying the analyses of VCG, PredictedLimits, and ErrorTolerant through a parameterized rerouting argument.
- This is the first application of the learning-augmented framework to frugal mechanism design for facility location, opening a new research direction.
- The counterintuitive upward-scaling design reveals a fundamental distinction between efficiency (minimizing social cost) and frugality (minimizing payments).
- The two-phase verification in ErrorTolerant (global scaling vs. per-facility penalty) demonstrates an elegant engineering-theoretic design.

## Limitations & Future Work

- All results concern deterministic auctions; randomized auctions may yield further improvements in frugality ratio.
- The robustness guarantee of ErrorTolerant involves terms of order $\lambda^4$, requiring careful selection of the error-tolerance parameter.
- Whether the Pareto-optimal frontier of the consistency–robustness tradeoff has been achieved remains an open question.
- Extensions to capacitated facility location or budget-constrained settings are natural directions for future work.
- The practical acquisition and evaluation of prediction quality is not addressed.

## Related Work & Insights

- The work most closely related to this paper is that of XL22 on learning-augmented frugal mechanism design for path auctions; the present paper extends similar ideas to the more complex facility location setting.
- Learning-augmented mechanism design is an active recent research direction, with applications spanning facility location, scheduling, auction design, and social welfare.
- The rerouting argument technique may be applicable to mechanism design for other combinatorial optimization problems.

## Rating

- **Novelty**: 4/5 — First application of the learning-augmented framework to frugal mechanism design for facility location; the counterintuitive upward-scaling direction is a notable highlight.
- **Technical Depth**: 4.5/5 — The unified payment lemma and careful case analysis demonstrate strong mathematical sophistication.
- **Experimental Thoroughness**: 3/5 — Pure theoretical work with no empirical experiments, though the theoretical results are complete (tight bounds with matching upper and lower bounds).
- **Value**: 3/5 — Primarily a theoretical contribution; practical deployment would require additional engineering effort.
- **Writing Quality**: 4/5 — Clear structure, well-organized proofs, and consistent notation throughout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[NeurIPS 2025\] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)

</div>

<!-- RELATED:END -->
