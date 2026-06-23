---
title: >-
  [Paper Note] f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness
description: >-
  [ICLR 2026][Interpretability][Paper Note] The study redefines "how important a sample is" as "whether the change in loss after its removal is statistically significantly different from training randomness." By leveraging the hypothesis testing framework of f-differential privacy, the authors propose f-influence and the f-INE algorithm, which enables estimation
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: 0942993ece46fb64
---
# f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TwkcMNACXo](https://openreview.net/forum?id=TwkcMNACXo)  
**Code**: TBD  
**Area**: Interpretability / Data Attribution (Influence Estimation)  
**Keywords**: Data Attribution, Influence Estimation, Hypothesis Testing, f-Differential Privacy, Training Randomness, LLM Data Poisoning Detection  

## TL;DR
The study redefines "how important a sample is" as "whether the change in loss after its removal is statistically significantly different from training randomness." By leveraging the hypothesis testing framework of f-differential privacy, the authors propose f-influence and the f-INE algorithm, which enables estimation via a single training run. This ensures that influence scores remain consistent across different random seeds and allows for reliable detection of poisoned samples in Llama-3.1-8B.

## Background & Motivation

**Background**: Influence Estimation (Data Attribution) seeks to answer a simple question: how much does a single training sample contribute to a model's final prediction? The ideal metric is Leave-One-Out (LOO) training, which involves re-training the entire model after removing a sample to observe the resulting change. Since LOO is prohibitively expensive, mainstream methods (Influence Functions, TraceIn, TRAK, LESS) essentially use various inexpensive approximations to estimate LOO.

**Limitations of Prior Work**: Existing methods collapse under training randomness. Minor variations in random seeds, weight initialization, batch size, and data shuffling order can cause the same sample to be identified as a "critical sample" in one run but "irrelevant" in the next. The paper quantifies this using the Jaccard consistency score: the consistency of Influence Functions and TraceIn is only ~0.56, and TRAK is ~0.74, all far from the ideal value of 1.0. This instability renders data cleaning and filtering meaningless, as one cannot guarantee whether the removed or retained samples are truly the intended ones.

**Key Challenge**: Under training randomness, influence lacks a well-defined total order. The paper provides a sharp example: removing $d_1$ consistently increases accuracy by 0.1%, while removing $d_2$ has a 0.1 probability of increasing accuracy by 1% (and 0% otherwise), yielding the same expectation. However, $d_1$ is more "useful" during a single re-training, whereas $d_2$ is more "useful" when picking the best out of multiple runs. A scalar value (such as the mean) cannot characterize this criterion-dependent ranking. Current methods are unstable precisely because they do not incorporate randomness into the definition of influence.

**Goal**: Provide a definition of influence that remains stable under randomness, can be efficiently estimated in a single training run, and scales to Large Language Models (LLMs).

**Core Idea**: **[Equating Influence Estimation to Hypothesis Testing/Privacy Auditing]** If a suspect sample is removed and the model is re-trained, and the subsequent loss reduction significantly exceeds random noise, then the sample is worth removing. This is naturally a hypothesis testing problem of $H_0$ (training on $D$) vs. $H_1$ (training on $D\setminus S$). The ease of distinguishing between these two distributions is precisely what the trade-off curve in f-differential privacy (f-DP) characterizes.

## Method

### Overall Architecture

The method consists of two layers: the **definition layer** redefines influence as the "difficulty" of hypothesis testing (f-influence / $G_\mu$-influence). Utilizing the compositionality and asymptotic normality of f-DP, the authors prove that under highly iterative SGD training, this converges to a Gaussian influence characterized by a single parameter $\mu$ with a total order. The **algorithm layer** (f-INE) leverages compositionality to reduce the "influence of the entire training trajectory" to "step-wise influence $\times$ number of steps," allowing for the collection of gradient similarity signals along the trajectory in a **single training run** to estimate $\mu$.

```mermaid
flowchart TD
    A[Suspect subset S ⊆ D] --> B[Hypothesis Testing Definition<br/>H0: Train on D<br/>H1: Train on D\S]
    B --> C[f-influence:<br/>Trade-off curve T_P,Q]
    C -->|Composition + Asymptotic Normality| D[Gμ-influence:<br/>Single parameter μ, Total Order]
    D --> E[f-INE Stage 1:<br/>Single training run collecting<br/>Gradient similarity (with/without S)]
    E --> F[Obtain distribution samples<br/>P ~ with S, Q ~ without S]
    F --> G[f-INE Stage 2:<br/>Scan thresholds for type-I/II errors<br/>μ = Φ⁻¹1-α − Φ⁻¹β]
    G --> H[Take μ with largest magnitude as final influence]
```

### Key Designs

**1. f-influence: Defining "Influence" as "Distributional Separability" via Trade-off Curves.** Let $P$ and $Q$ be the distributions of the test statistic $\ell$ under $H_0$ (training on $D$) and $H_1$ (training on $D\setminus S$), respectively. Following f-DP, for a rejection rule $0\le\phi\le1$, the type-I error is $\alpha_\phi=\mathbb{E}_P[\phi]$ and the type-II error is $\beta_\phi=1-\mathbb{E}_Q[\phi]$. The trade-off function $T(P,Q)(\alpha)=\inf_\phi\{\beta_\phi:\alpha_\phi\le\alpha\}$ is the minimal sufficient statistic for distinguishing $P$ and $Q$. It inherently characterizes randomness because influence is measured at the distribution level rather than on a single realization. Two key differences from Gaussian Differential Privacy (GDP) are: influence estimation is data-dependent ($S$ is sampled from a given $D$, rather than worst-case adjacent datasets), and influence can be positive or negative (GDP privacy is always non-negative).

**2. $G_\mu$-influence: Compressing the Entire Curve into an Interpretable, Signed Scalar.** Trade-off curves only provide a partial order (e.g., if the curves for $d_1$ and $d_2$ do not dominate each other), which is impractical for ranking. The paper uses the Gaussian special case $f=T(\mathcal{N}(0,1),\mathcal{N}(\mu,1))$, where influence is fully characterized by a single parameter $\mu$: $\mu=\Phi^{-1}(1-\alpha)-\Phi^{-1}\big(T(P,Q)(\alpha)\big)$, which holds for all $\alpha$. The semantics of $\mu$ are straightforward—the change in the test statistic after removing $S$ is at least as large as the difference between $\mathcal{N}(0,1)$ and $\mathcal{N}(\mu,1)$, and the sign of $\mu$ directly indicates the direction of influence (positive/negative). This "rescues" the definition back to a total order.

**3. Composition + Asymptotic Normality: Theoretical Guarantees for SGD.** This acts as the bridge for implementing general f-influence into a calculable $\mu$. Compositionality (Theorem 2.6) states that if $S$ has $f_i$-influence on each step of algorithm $A_i$, the $k$-step composite algorithm has at most $f_1\otimes\cdots\otimes f_k$-influence. Corollary 2.7 further links total trajectory influence to step-wise influence for Gaussian cases: $|\tilde\mu|\le|\mu\sqrt{k}|$. Asymptotic normality (Theorem 2.8), similar to the Central Limit Theorem, indicates that if an algorithm can be decomposed into a large number of nearly identically distributed update steps, the composite trade-off curve asymptotically becomes a Gaussian influence $G_\mu$. Together, these imply that for highly iterative algorithms like SGD, we can reliably work within the $G_\mu$ class.

**4. f-INE Algorithm: Single Training + Gradient Similarity + Decorrelation.** Naively estimating $G_\mu$ would requires hundreds of re-trainings to plot histograms of $\ell_D$ and $\ell_{D\setminus S}$, which is infeasible. f-INE achieves this in one run using three techniques: (i) **Step-wise substitution**: utilizing compositionality to estimate and scale step-wise influence; (ii) **Gradient similarity in place of loss**: using a first-order Taylor approximation of the loss difference between adjacent steps $\ell(\theta_t,z_{\text{test}})-\ell(\theta_{t+1},z_{\text{test}})\approx\eta\nabla\ell(\theta_t,z_{\text{test}})^\top\nabla\ell(\theta_t,z')$, which improves scalability and removes trends; (iii) **Sample decorrelation**: handled by taking first-order differences to eliminate linear trends and using "difference-in-differences" (training an auxiliary model and subtracting its signal $\tilde O=O-\hat O$) to further reduce correlation. Stage 1 collects gradient similarity signals $\tilde O$ and $\tilde O'$ (with and without $S$, respectively); Stage 2 scans thresholds $\tau$ to calculate $\alpha_\tau, \beta_\tau$ and uses the closed-form $\mu_\tau=\Phi^{-1}(1-\alpha_\tau)-\Phi^{-1}(\beta_\tau)$, ultimately taking the $\mu$ with the largest magnitude. The overall complexity is $O(Tnd)$, comparable to TraceIn/LESS.

## Key Experimental Results

### Main Results

**Consistency (Data Shuffling / Random Seed)**: Measured by the average Jaccard consistency score ($[0,1]$, where 1 is perfect consistency).

| Method | Consistency Score ↑ |
|------|--------------|
| Influence Functions | 0.567 |
| TraceIn | 0.564 |
| TRAK | 0.736 |
| **f-INE (Ours)** | **0.938** |

**Computational Complexity / Scalability** ($n$: training samples, $d$: model dimensions, $T$: iterations, $k\ll d$: projection dimensions, $M$: ensemble models):

| Method | Complexity | Scalability |
|------|--------|----------|
| Influence Functions | $O(nd^2+d^3)$ | Low |
| TRAK | $O(M(nk^2+k^3))$ | Medium |
| TraceIn / LESS | $O(Tnd)$ | High |
| **f-INE (Ours)** | $O(Tnd)$ | High |

**Identifying Mislabeled Samples (MNIST, 20% Label Noise, MLP)**: Samples are ranked by descending self-influence; mislabeled samples should appear first. The recall curve of f-INE is on par with TraceIn (only 0.05% higher) but is 13.85% and 3.83% higher than TRAK and Influence Functions, respectively, with a smoother curve and lower variance.

### LLM Attribution (Llama-3.1-8B, Poisoned LIMA Instruction Tuning)

Biased instructions regarding "Joe Biden" and "Abortion" were injected into LIMA to induce negative sentiment (increasing negative responses by 40%/60% compared to a clean model). Among methods scalable to LLMs, only f-INE and LESS (an optimized TraceIn for LLMs using cosine similarity + LoRA checkpoints) were evaluated.

- **Utility (Recall of Poisoned Instructions)**: f-INE recovers **>60%** of Joe Biden poisoning instructions in the top 20% most influential samples, compared to only **44%** for LESS. f-INE outperforms LESS and the random baseline across both entities and various $p$ values.
- **Stability**: Across three training runs, the average coefficient of variation for f-INE influence scores is significantly lower than that of LESS.

### Key Findings
- The root cause of instability in existing influence estimation methods is the failure to incorporate randomness into the definition; moving to a distributional/hypothesis testing level achieves consistency across random sources.
- The difference from LESS demonstrates that while LESS only compares means, f-INE compares the entire distribution (accounting for variance), leading to a win-win in utility and stability.
- The hypothesis testing framework is also applicable to membership inference and privacy auditing in natural language, explicitly bridging influence estimation and DP auditing.

## Highlights & Insights
- **Elegant Conceptual Bridge**: Connecting influence estimation $\leftrightarrow$ hypothesis testing $\leftrightarrow$ f-differential privacy is the paper's greatest conceptual contribution. The compositionality and asymptotic normality of GDP are leveraged to prove that influence under SGD converges to single-parameter Gaussian influence.
- **Thorough Diagnostics**: The paper uses Jaccard consistency and the $d_1/d_2$ counter-example to explain why influence lacks a total order under randomness before "rescuing" it via the Gaussian case.
- **Practical Application**: With $O(Tnd)$ complexity and compatibility with LESS-style LoRA+cosine optimizations, the method was successfully scaled to Llama-3.1-8B, demonstrating real-world utility in poisoning detection.

## Limitations & Future Work
- **White-box Assumption**: The algorithm requires observing model parameters and gradients at each update step, making it inapplicable to black-box or API-only scenarios.
- **Imperfect Consistency**: While f-INE's consistency (0.938) far exceeds baselines, it is not 1.0; Gaussian approximations may be less accurate when training steps are limited or update steps vary significantly.
- **Test Task Scope**: LLM experiments were focused on sentiment steering/poisoning detection; generalized capability attribution and cross-task influence remain to be verified.
- **Auxiliary Model Overhead**: The "difference-in-differences" approach requires training an additional auxiliary model to decorrelate signals, which increases training costs by a constant factor.
- Scanning thresholds for the "largest magnitude $\mu$" is a best-case estimate sensitive to noise, which might lead to overestimation.

## Related Work & Insights
- **Influence Estimation Lineage**: From Influence Functions (Koh & Liang 2017) to TraceIn, TRAK, and LESS, this paper points out a shared blind spot: treating influence as a deterministic quantity. f-INE elevates "randomness" to a first-class citizen.
- **f-Differential Privacy / GDP** (Dong et al. 2022): Trade-off functions, compositionality, and asymptotic normality serve as the theoretical foundation for this method, with the distinction of being data-dependent and allowing signed influence.
- **Privacy Auditing / Membership Inference** (Shokri et al. 2017; Nasr et al. 2023; Steinke et al. 2023): Techniques for "single-training auditing" are transferred here to enable single-training influence estimation.
- **Insight**: When an estimator is "unstable," rather than pursuing more precise point estimates, one should ask where its randomness originates and whether it belongs in the definition. Upgrading a metric from a scalar to a distribution often resolves both stability and interpretability issues.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Re-founding influence estimation on hypothesis testing/f-DP is an original contribution at the definitional level, not just another LOO approximation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage includes consistency, mislabel detection, and LLM poisoning attribution across scales up to 8B; however, baselines are limited (only LESS for LLMs), and stress tests on more diverse tasks are needed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from motivation to counter-example to definition to algorithm is very clear.
- **Value**: ⭐⭐⭐⭐⭐ Scenarios like data cleaning, poisoning detection, and behavior attribution depend heavily on reliable influence scores; achieving "consistency across random sources" has direct practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Deleuzian Representation Hypothesis](the_deleuzian_representation_hypothesis.md)
- [\[ICLR 2026\] Estimating Dimensionality of Neural Representations from Finite Samples](estimating_dimensionality_of_neural_representations_from_finite_samples.md)
- [\[ICLR 2026\] Influence Dynamics and Stagewise Data Attribution](influence_dynamics_and_stagewise_data_attribution.md)
- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICLR 2026\] The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology](the_shape_of_adversarial_influence_characterizing_llm_latent_spaces_with_persist.md)

</div>

<!-- RELATED:END -->
