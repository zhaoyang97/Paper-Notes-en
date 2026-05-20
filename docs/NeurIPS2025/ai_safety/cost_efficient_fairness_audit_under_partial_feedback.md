---
title: >-
  [Paper Note] Cost Efficient Fairness Audit Under Partial Feedback
description: >-
  [NeurIPS 2025][AI Safety][fairness audit] Under the partial feedback setting, this paper proposes a fairness auditing framework with a novel cost model…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "fairness audit"
  - "partial feedback"
  - "equalized odds"
  - "rejection sampling"
  - "exponential family mixture"
date: 2026-05-08
content_hash: a987fc6ae1b67a17
---

# Cost Efficient Fairness Audit Under Partial Feedback

**Conference**: NeurIPS 2025
**arXiv**: [2510.03734](https://arxiv.org/abs/2510.03734)  
**Code**: Provided with the paper (supplementary)  
**Area**: AI Safety
**Keywords**: fairness audit, partial feedback, equalized odds, rejection sampling, exponential family mixture

## TL;DR

Under the partial feedback setting, this paper proposes a fairness auditing framework with a novel cost model, delivering near-optimal audit algorithms for both black-box and mixture model scenarios, reducing audit cost by approximately 50% compared to natural baselines.

## Background & Motivation

Machine learning classifiers are widely deployed in high-stakes decisions such as loan approval, criminal risk assessment, and university admissions, yet they may amplify biases present in training data. **Fairness auditing** aims to verify whether a classifier satisfies specific fairness criteria (e.g., equalized odds).

A critical practical challenge is **partial feedback**—true labels are observable only for positively classified individuals. In bank lending, for instance, only approved applicants generate repayment records, while the repayment capacity of rejected applicants remains entirely unknown.

Prior work either studies learning fair classifiers under partial feedback or conducts fairness audits under complete labels, but **no existing work addresses cost-sensitive fairness auditing under partial feedback**. Moreover, acquiring missing labels is itself costly—forcing loans on rejected applicants incurs real losses upon default—yet this cost asymmetry has not been modeled previously.

## Core Problem

Given a classifier $f: \mathcal{X} \times \mathcal{A} \to \{0,1\}$, the goal is to complete a $(\gamma, \varepsilon, \delta)$-fairness audit under partial feedback at minimum audit cost:

- **Fair hypothesis** $\mathsf{FAIR}$: equalized odds difference $\Delta \leq \gamma$
- **Unfair hypothesis** $\mathsf{UNFAIR}$: $\Delta > \gamma + \varepsilon$

**Audit cost model**: The auditor observes online samples and, for individuals classified negatively ($f=0$):

- Requesting features incurs cost $c_{feat}$
- Requesting a label that turns out negative ($Y=0$) incurs an additional cost $c_{lab}$ (e.g., loan default loss)

In practice $c_{lab} \gg c_{feat}$, and this cost model accurately captures the high cost of obtaining negative-outcome labels in real-world settings.

## Method

### 1. Black-Box Model — Baseline

Labels are requested for all individuals with $f=0$, and stopping times are used to guarantee estimation accuracy. Audit cost upper bound:

$$\text{cost} \leq \widetilde{O}\left(\sum_{a \in \mathcal{A}} \sum_{y \in \{0,1\}} \frac{c_{feat} \cdot \mathbb{P}[f=0] + c_{lab} \cdot \mathbb{P}[f=0, Y=0]}{q_{y,a} \cdot \varepsilon^2}\right)$$

**Limitation**: Labels are requested indiscriminately for all individuals, wasting query cost on groups $A \neq a$.

### 2. Black-Box Model — RS-Audit (Rejection Sampling Audit)

**Core Insight**: Estimating $\mathbb{P}[f=1|Y=y, A=a]$ requires only the conditional probability $\mathbb{P}[Y=y|A=a]$, not the joint probability $\mathbb{P}[Y=y, A=a]$. Therefore, for group $a$, samples with $A \neq a$ can be **discarded via rejection sampling**, and labels are requested only for the target group.

**Algorithm**:

1. For each $(y, a)$, estimate $p_{y|a} = \mathbb{P}[f=1, Y=y | A=a]$ from a historical positive-class database $D$ using the Past subroutine.
2. In the online stream, via the Online subroutine, observe only individuals with $A=a$ and collect labels until $\tau$ samples with $Y=y$ are accumulated.
3. Exploit the concentration properties of the Negative Binomial distribution to guarantee estimation accuracy.
4. Compute $\hat{\Delta}$ and compare against threshold $\gamma + \varepsilon/2$ to determine fairness.

**Audit cost** (Theorem 2):

$$\text{cost} \leq \widetilde{O}\left(\sum_{a \in \mathcal{A}} \sum_{y \in \{0,1\}} \frac{c_{feat} \cdot \mathbb{P}[f=0|A=a] + c_{lab} \cdot \mathbb{P}[f=0, Y=0|A=a]}{q_{y|a} \cdot \varepsilon^2}\right)$$

Compared to the Baseline, the numerator is reduced from $\mathbb{P}[f=0]$ to $\mathbb{P}[f=0|A=a]$ and the denominator is increased from $q_{y,a}$ to $q_{y|a}$, yielding a dual improvement.

### 3. Lower Bound (Theorem 3)

By constructing two families of instances with similar KL divergence but different fairness properties, it is shown that the expected cost of any algorithm is at least $\widetilde{\Omega}$, matching the upper bound of RS-Audit up to logarithmic factors. **The black-box setting is thus essentially completely characterized.**

### 4. Mixture Model — Exp-Audit

When the feature distribution follows an exponential family mixture model $\mathbb{P}[X|Y=y, A=a] = \mathcal{E}_{\theta^*_{y,a}}(X)$, incorporating structural assumptions can substantially reduce cost.

**Three-step algorithm**:

1. **Learning from truncated samples**: Treat historical positive-class data as samples truncated by $f$, and recover distribution parameters $\hat{\theta}_{y,a}$ using the TruncEst subroutine.
2. **Constructing a MAP oracle**: Use the estimated parameters and coarse mixture weights $\tilde{q}_{y|a}$ to build a maximum a posteriori discriminator $M_a(x) = \arg\max_y \tilde{q}_{y|a} \mathcal{E}_{\hat{\theta}_{y,a}}(x)$.
3. **Estimating mixture weights via proxy labels**: Request only features (cost $c_{feat}$) and assign proxy labels using the MAP oracle, thereby avoiding requests for true labels (cost $c_{lab}$).

**Audit cost** (Theorem 4):

$$\text{cost} \leq \widetilde{O}\left(\sum_{a \in \mathcal{A}} \left(\frac{c_{feat} \cdot \mathbb{P}[f=0|A=a]}{q_{m,a} \cdot \varepsilon^2} + \sum_{y \in \{0,1\}} \frac{c_{lab} \cdot \mathbb{P}[f=0, Y=0|A=a]}{q_{y|a}}\right)\right)$$

**Key improvement**: The $c_{lab}$ term is independent of $\varepsilon$ (it is a problem-dependent constant); only the $c_{feat}$ term depends on $\varepsilon^{-2}$. Since $c_{lab} \gg c_{feat}$, the total cost is substantially reduced.

### Assumptions

- **Structural assumption** (Assumption 1): A lower bound $\alpha > 0$ on the positive-class probability, along with log-concavity and smoothness of the exponential family.
- **Separation assumption** (Assumption 2): $\|\theta^*_{1,a} - \theta^*_{0,a}\| \geq \Omega(\sqrt{\log(1/\varepsilon)})$, which is significantly weaker than the $\Omega(1/\varepsilon)$ separation required by clustering methods.

## Key Experimental Results

| Dataset | Comparison | Cost Reduction |
|--------|------------|----------------|
| Adult Income | RS-Audit vs. Baseline | ~50% cost savings |
| Law School | RS-Audit vs. Baseline | Similar improvement |
| Adult Income | Exp-Audit vs. RS-Audit | Significantly lower cost, independent of $\varepsilon$ |
| Synthetic | Exp-Audit vs. RS-Audit | Exp-Audit cost curve is flat; RS-Audit grows with $\varepsilon$ |

- Black-box experiments use a Logistic Regression classifier.
- Mixture model experiments use penultimate-layer embeddings from a neural network, assuming a multivariate Gaussian distribution.
- On Adult Income, the separation condition is not enforced, yet Exp-Audit still demonstrates a marked advantage, reflecting robustness.

## Highlights & Insights

1. **First work to study cost-sensitive fairness auditing under partial feedback**, addressing an important gap.
2. **Carefully designed cost model**: Distinguishing $c_{feat}$ from $c_{lab}$ accurately captures the asymmetric cost of label acquisition.
3. **Complete characterization of the black-box setting**: The RS-Audit upper bound matches the lower bound up to logarithmic factors.
4. **Truncated sample perspective**: Interpreting partial feedback data as truncated samples is a conceptual contribution that bridges two distinct research areas.
5. **Generalization of the exponential family MAP oracle**: Extending MAP oracle results from spherical Gaussians to general exponential families has independent theoretical value.
6. **Classifier-agnostic algorithms**: The framework applies to auditing arbitrary classifiers.

## Limitations & Future Work

1. **Strong mixture model assumptions**: The exponential family type (i.e., $h, T, W$) is assumed known, as are the structural and separation conditions.
2. **Binary groups and binary classification only**: Extensions to multi-class settings and continuous sensitive attributes are not discussed.
3. **Uniform cost model**: In practice, default costs may vary across individuals (non-uniform costs).
4. **Unknown classifier scenario not covered**: Audit complexity when the classifier itself is unknown (e.g., drawn from a hypothesis class) remains an open problem.
5. **Limited experimental scale**: Validation is conducted on only two standard datasets; large-scale, high-dimensional settings are not evaluated.
6. **Practical verifiability of the separation condition**: Although the algorithm can detect online whether the separation condition holds and switch strategies accordingly, the additional cost of such switching is not analyzed.

## Related Work & Insights

| Aspect | Ours | Prior Work |
|--------|------|------------|
| Setting | Partial feedback + cost model | Full-label audit / partial feedback classification |
| Objective | Minimize audit cost | Hypothesis testing / fair classifier learning |
| vs. Bechavod et al. 2019 | Audit (testing) problem | Online fair classification (learning) problem |
| vs. Keswani et al. 2024 | Cost-sensitive auditing | Data collection + fair classification |
| vs. Chugg et al. 2023 | Partial feedback | Sequential testing under full labels |
| vs. Si et al. 2021 | Partial feedback + optimal cost | Optimal transport framework, full labels |
| Truncated sample technique | First use in fairness auditing | Applied only to distribution parameter estimation |

The conceptual connection between **truncated samples and partial feedback** is an inspiring contribution, potentially generalizable to other partial observation settings (e.g., selection bias auditing in recommender systems). The cost model design is transferable to **active learning**, enabling query strategies under asymmetric label acquisition costs. The exponential family generalization of the MAP oracle may be applicable to **semi-supervised fair classification**, leveraging feature distributional structure when labels are scarce. The fairness–accuracy tradeoff analysis (Remark 4) suggests that auditing and retraining could be unified within a single framework.

## Rating

- Novelty: ⭐⭐⭐⭐ (First cost-sensitive fairness audit under partial feedback; novel truncated sample perspective)
- Experimental Thoroughness: ⭐⭐⭐ (Limited dataset scale, but theoretical validation is thorough)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; problem motivation is well articulated)
- Value: ⭐⭐⭐⭐ (Fills an important gap; theoretical contributions are solid)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[NeurIPS 2025\] Efficient Fairness-Performance Pareto Front Computation](efficient_fairness-performance_pareto_front_computation.md)
- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)
- [\[NeurIPS 2025\] Matchings Under Biased and Correlated Evaluations](matchings_under_biased_and_correlated_evaluations.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)

</div>

<!-- RELATED:END -->
