---
title: >-
  [Paper Note] PAC-Bayes Bounds for Cumulative Loss in Continual Learning
description: >-
  [ICLR 2026][Learning Theory][Cumulative Loss] This paper generalizes existing online and time-uniform offline PAC-Bayes bounds to the continual learning setting, providing the **first cumulative loss (learning plasticity) bounds applicable to arbitrary task distributions and learning algorithms**, and verifies non-vacuous risk certificates on visual continual learning tasks.
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Continual Learning"
  - "PAC-Bayes"
  - "Cumulative Loss"
  - "Learning Plasticity"
  - "Risk Certificate"
date: 2026-05-08
content_hash: 7ee7c6c1d06b37ef
---

# PAC-Bayes Bounds for Cumulative Loss in Continual Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=hWw269fPov](https://openreview.net/forum?id=hWw269fPov)  
**Area**: Learning Theory / Continual Learning / PAC-Bayes  
**Keywords**: Continual Learning, PAC-Bayes, Cumulative Loss, Learning Plasticity, Risk Certificate

## TL;DR
This paper generalizes existing online and time-uniform offline PAC-Bayes bounds to the continual learning setting, providing the **first cumulative loss (learning plasticity) bounds applicable to arbitrary task distributions and learning algorithms**, and verifies non-vacuous risk certificates on visual continual learning tasks.

## Background & Motivation
**Background**: Continual learning requires a trade-off between "remembering old tasks" and "learning new tasks," corresponding to the contradiction between memory stability and learning plasticity. Numerous empirical algorithms (EWC regularization, replay, variational inference, etc.) exist to mitigate catastrophic forgetting, and there is an increasing demand for **reliable risk certificates** for these algorithms.

**Limitations of Prior Work**: Theoretical progress lags behind empirical developments. Existing theoretical results either only characterize **forgetting** (memory stability) or are restricted to narrow settings—such as continuous linear regression or the NTK (infinitely wide networks) regime—and are often tied to specific optimization algorithms (SGD, OGD). There is almost a total vacuum regarding **learning plasticity**, especially for universal upper bounds.

**Key Challenge**: Continual learning is measured by two types of quantities. **Average loss** (corresponding to average accuracy) uses the final posterior $Q_T$ to evaluate all tasks retrospectively; this requires recomputation as new tasks arrive and storage of data from every task. **Cumulative loss** (CuL) accumulates the current test error of each task during the learning process, corresponding to forward transfer, intransigence, and learning plasticity. Existing PAC-Bayes work (meta-learning, or the forgetting/average error bounds of Friedman & Meir 2025) does not cover CuL.

**Goal**: To derive **algorithm-agnostic, architecture-agnostic, and task-structure-agnostic** high-probability upper bounds for cumulative loss in continual learning, and further analyze the tightness of these bounds under different task similarities and orders.

**Key Insight**: The authors observe that the PAC-Bayes framework for online learning (time-uniform bounds from Haddouche & Guedj 2022/2023 or Chugg et al. 2023) naturally possesses a structure where the "posterior evolves with the data sequence." By expanding each "task" from a single sample to $m$ i.i.d. samples, this machinery can be adapted for continual learning.

**Core Idea**: Formalize the continual learning process using **online predictive sequences + stochastic kernels**, then combine the PAC-Bayes "triad" (Markov inequality / change-of-measure / Hoeffding) with "bad-event analysis" and "Laplace's method" to obtain cumulative loss bounds that hold for general hypothesis classes and degenerate into oracle bounds under the Gibbs posterior.

## Method

### Overall Architecture
The paper is a purely theoretical work. The main logic follows: "Formalize continual learning into a probabilistic framework suitable for PAC-Bayes → Derive general upper bounds within this framework → Derive more interpretable oracle bounds for the Gibbs posterior → Use oracle bounds to compare different task scenarios."

The formal setting is as follows: Data space $Z = X \times Y$, fixed sample size $m$ per task, and arbitrary task sequence length $T$. The $t$-th task corresponds to an unknown distribution $D_t$, given an i.i.d. sample $S_t \sim D_t^m$. The learner maintains a sequence of priors $(P_t)$ and posteriors $(Q_t)$: the priors start from a data-free $P_1=P$, with the **key setting being that the posterior of the previous task serves as the prior for the next task** ($P_t = Q_{t-1}$). Consequently, the KL term represents the "change in posterior between adjacent tasks." The entire process evolves along an adapted filtration $(F_t)$, using stochastic kernels (probability measures mapping dataset $S$ to $H$) to allow for data-dependent priors. The target quantity to be upper-bounded is the cumulative loss:

$$\mathrm{CuL}((Q_t)_{t=1}^{T}) = \sum_{t=1}^{T} \mathbb{E}_{h_t \sim Q_{t,S_{1:t}}}\big[\mathbb{E}_{z_t \sim D_t}[\ell(h_t, z_t)\mid F_{t-1}]\big],$$

which is the sum of expected test errors for each task at the moment it is "just learned." The framework covers both offline continual learning (samples given in batches per task) and online continual learning (samples arriving one by one and not reusable).

### Key Designs

**1. PAC-Bayes Formalization for Continual Learning: Using online predictive sequences to create a boundable probabilistic structure**

To derive bounds for CuL, a language describing "posterior evolution along a task sequence" is required. The authors perform a key modification of the online framework by Haddouche & Guedj: whereas the original framework has one sample per time step, this work replaces each step with a task of $m$ samples. The core object is the **online predictive sequence**—a sequence of stochastic kernels $(P_t)$ where $P_t(S,\cdot)$ must be $F_{t-1}$-measurable (depending only on information from previous tasks) and absolutely continuous with respect to $P_{t-1}$. This ensures a causal structure where the "prior only sees the past and the posterior can use current data," sufficient to describe various continual learning algorithms. Based on this, the authors strictly distinguish between CuL, average loss (AL), and meta-learning loss: AL evaluates all tasks using $Q_T$ and requires additional memory proportional to the number of tasks; MetaL measures future unseen tasks; only CuL is sequentially additive, requiring only a single term update for new tasks, making it the correct object for characterizing plasticity. This formalization serves as the foundation for all subsequent bounds and the interface for connecting online PAC-Bayes machinery to continual learning.

**2. General Bounds for Cumulative Loss: From non-converging naive bounds to a $\sqrt{T}$ law converging with the number of tasks**

The classic Catoni bound is first generalized to $m>1$ and sub-Gaussian losses, yielding Corollary 3.1:

$$\tfrac{1}{T}\mathrm{CuL} \le \tfrac{1}{T}\sum_{t=1}^{T}\hat{L}(Q_t, S_t) + \tfrac{1}{\lambda T}\sum_{t=1}^{T}\mathrm{KL}(Q_t\|P_t) + \tfrac{\lambda K^2}{m} + \tfrac{\log(1/\delta)}{\lambda T}.$$

While clean in form, the right-hand side contains a $\lambda K^2/m$ term that does not decay with the number of tasks $T$. In continual learning, bounds that "converge as tasks increase even if samples per task are fixed" are preferred. To achieve this, the authors assume bounded loss $\ell\in[0,K]$ and $m \gg \sqrt{T}$. Through detailed analysis of task/posterior dependencies and **bad-event analysis** (a less common technique in PAC-Bayes), they derive Theorem 3.2. Setting $\lambda = T\sqrt{T}/K$ and $\delta_2 = e^{-T\sqrt{T}}$, the simplified readable version is Eq. (3):

$$\tfrac{1}{T}\mathrm{CuL} \le \tfrac{1}{T}\sum_{t=1}^{T}\hat{L}(Q_t,S_t) + \tfrac{K}{T\sqrt{T}}\sum_{t=1}^{T}\mathrm{KL}(Q_t\|Q_{t-1}) + \tfrac{K}{4\sqrt{T}}\sqrt{2m} + \tfrac{K(1+\log(1/\delta))}{T\sqrt{T}}.$$

As long as $m > \sqrt{T}/2$, the right side converges to the left as $m, T \to \infty$. This not only provides a risk certificate but also yields a practical rule of thumb: **the number of samples per task should exceed the square root of the total number of tasks.** For example, if a model is retrained daily for a year, only a few dozen samples are needed to provide a valid high-probability certificate for cumulative loss.

**3. Oracle Bounds under Gibbs Posterior: Eliminating the KL term to bound cumulative loss by the loss of "predictors trained on past tasks"**

Since the KL complexity term in general bounds is difficult to estimate for large models, the authors turn to the **Gibbs posterior** $\hat{Q}_t^\lambda(h) \propto e^{-\lambda\hat{L}(h,S_t)}\hat{Q}_{t-1}^\lambda(h)$ (and its expected version). The advantage of the Gibbs posterior is that it can eliminate the entire KL term on the right side of Eq. (1), allowing for an oracle upper bound on cumulative loss. Under assumptions such as $H$ being a compact bounded subset and the total expected loss having a strict global minimum that is twice differentiable, Theorem 4.1 gives:

$$\lim_{m,T\to\infty}\tfrac{1}{T}\mathrm{CuL}((Q_t^\lambda)) \le \lim_{T\to\infty}\tfrac{1}{T}\sum_{t=2}^{T} L(h^*_{1:t-1}, D_t),$$

where $h^*_{1:t-1}$ is the optimal solution for the total loss of the first $t-1$ tasks. Intuitively: the cumulative loss of the current task is upper-bounded by the loss of the "optimal predictor trained on all previous tasks" evaluated on the current task. Variants can be obtained under different minimality assumptions (Corollary 4.2). This proof largely relies on **Laplace's method**, which is also rare in PAC-Bayes. The value of the oracle bound lies in translating abstract cumulative loss into "how well the previous optimal predictor transfers," enabling quantitative comparisons of task similarity.

**4. Scenario Analysis of Task Similarity and Order: Using oracle bounds to explain the impact of "task scheduling" on cumulative error**

With oracle bounds, the authors provide explicit bounds for four typical scenarios under assumptions of Lipschitz loss (where distribution similarity is reflected in the loss via Wasserstein distance) and over-parameterization: ① When all tasks are i.i.d., $\frac{1}{T}\mathrm{CuL}\le L_1^* + O(1/T)$; ② In alternating distributions, the optimum $\frac{L_1^*+L_2^*}{2}$ is reachable if dimension $d\ge d_1+d_2$, otherwise an extra term proportional to task distance $G_H d(D_1,D_2)$ appears; ③ A single switch between distributions; ④ Gradual task drift (adjacent distance $\le r$, total radius $\le \phi$) adds an $rG_H$ term. The conclusion is that **task order significantly affects cumulative error when capacity is limited** (single switch is better than alternating), while **sufficiently over-parameterized models can essentially eliminate the impact of task order.** This section connects the abstract bounds to comparable concrete situations, offering insights into the relationship between forward transfer, model complexity, and task similarity.

## Key Experimental Results

### Main Results (Vision Continual Learning, Verifying General Bounds)
$T=120$ tasks, CNN model, 5 random seeds. Reported is the average cumulative error $\frac{1}{T}\mathrm{CuL}$ vs. Eq. (3) upper bound (error percentage, lower is better):

| Dataset | Method | CuL | Upper Bound (Eq. 3) | Error@t=120 | Bound@t=120 |
|--------|------|-----|-----------|-------------|-----------|
| Perm.-MNIST | EWC | 1.0 | 10.6 | 1.0 | 5.1 |
| Perm.-MNIST | VI | 15.5 | 17.9 | 4.7 | 6.8 |
| Split-MNIST | EWC | 0.9 | 4.2 | 0.9 | 2.5 |
| Split-MNIST | VI | 17.6 | 19.4 | 5.1 | 7.5 |
| Split-CIFAR10 | EWC | 34.4 | 47.9 | 34.7 | 39.7 |
| Split-CIFAR10 | VI | 49.8 | 52.2 | 49.5 | 52.5 |

Split-ImageNet (Random model approx. 98% cumulative error):

| Method | CuL | Upper Bound (Eq. 3) | Forgetting |
|------|-----|-----------|-----------|
| EWC | 33.5 | 40.1 | 5.7 |
| SGD | 34.8 | 41.4 | 7.8 |
| Replay | 55.7 | 62.5 | 2.8 |

### Key Findings
- **Bounds are tight for VI but loose for EWC**: VI directly optimizes the right-hand side of Eq. (3), whereas EWC's training process is not directly linked to the bound.
- **Bounds tighten with more tasks**: Multiple terms in Eq. (3) decay with $T$, and the KL term decreases over time—using the previous posterior as the current prior provides an increasingly "informative" prior. Except for VI on Split-CIFAR10, all empirical bounds are non-vacuous, providing useful risk certificates especially for later tasks.
- **Bounds are looser for early tasks**: This likely stems from stochasticity in the early training stages; however, this aligns with the intuition that "continual retraining of complex models favors generalization on new tasks."
- **Oracle bounds verify task order effects** (Linear regression + SGLD approximating Gibbs sampling): Cumulative loss for a single task switch is significantly better than for alternating tasks, consistent with theory; in the over-parameterized regime, cumulative error is nearly constant, and the bound is almost exact. In this setting, the bound is several orders of magnitude tighter than NTK-based SGD generalization bounds.

## Highlights & Insights
- **First universal upper bound for learning plasticity**: While previous theory focused on forgetting/average error in specific settings, this work addresses the gap in cumulative loss to characterize plasticity with an algorithm/architecture/task-agnostic bound.
- **"Previous posterior as next prior" is the stroke of genius**: This makes the KL term represent the change between adjacent posteriors, which both respects the continual learning constraint of "no looking back at old data" and explains why the bound naturally tightens as tasks progress.
- **The $m>\sqrt{T}$ rule is directly actionable**: The theoretical derivation yields an operational rule of thumb—samples per task should exceed the square root of the number of tasks—offering guidance for real-world continual retraining systems.
- **Novel methodology**: Integration of bad-event analysis and Laplace's method into PAC-Bayes provides a reusable technical template for deriving convergent and oracle-type bounds.

## Limitations & Future Work
- Assumes bounded or sub-Gaussian loss; heavy-tailed losses require further extension (referencing Haddouche & Guedj 2023).
- The strict global minimum assumption could be relaxed to a finite number of global minima; current oracle bounds are for the expected Gibbs posterior, while the empirical version would require modified minimality assumptions.
- In offline continual learning, cumulative error is often less relevant than average error (where infinite samples per task allow learning from scratch), and in online scenarios with fuzzy task boundaries, $m$ and $T$ must be estimated to use the bounds.
- General bounds contain KL complexity terms difficult to scale for large models (oracle bounds avoid this); subsequent work could consider other divergence measures or model compression bounds to alleviate this.

## Related Work & Insights
- **vs. Friedman & Meir (2025)**: They provide PAC-Bayes bounds for **average error/forgetting**; this paper turns to **cumulative loss** to characterize plasticity.
- **vs. NTK/SGD theoretical bounds (Bennani & Sugiyama 2020, etc.)**: Those bounds are tied to the NTK regime and specific optimization algorithms. This work's bound is agnostic and empirically orders of magnitude tighter than NTK-based bounds.
- **vs. Meta-learning PAC-Bayes (Pentina & Lampert 2015, etc.)**: Meta-learning allows retrospective access to old data and assumes tasks come from a single distribution. Follow-the-leader methods require access to historical data. This work covers both offline and online continual learning without rereading old data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First universal PAC-Bayes bound for cumulative loss/plasticity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Non-vacuous bounds verified on vision tasks + oracle bounds on linear regression; sufficient coverage but scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Framework and theorems progress logically; symbols are rigorous; details are delegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides computable risk certificates for CL algorithms and yields an actionable $\sqrt{T}$ rule of thumb.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] On the Bayes Inconsistency of Disagreement Discrepancy Surrogates](on_the_bayes_inconsistency_of_disagreement_discrepancy_surrogates.md)
- [\[ICLR 2026\] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity](barriers_for_learning_in_an_evolving_world_mathematical_understanding_of_loss_of.md)
- [\[ICLR 2026\] Multi-Synaptic Cooperation: A Bio-Inspired Framework for Robust and Scalable Continual Learning](multi-synaptic_cooperation_a_bio-inspired_framework_for_robust_and_scalable_cont.md)

</div>

<!-- RELATED:END -->
