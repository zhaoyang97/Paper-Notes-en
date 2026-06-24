---
title: >-
  [Paper Note] The Coverage Principle: How Pre-Training Enables Post-Training
description: >-
  [ICLR2026][Learning Theory][coverage profile] This paper theoretically answers "what pre-training actually leaves for post-training (RL / test-time scaling)"—the answer is not cross-entropy, but a quantity called **coverage profile**. The authors prove that next-token prediction **implicitly** optimizes coverage, and that coverage generalizes faster than cross-entropy without being hindered by sequence length, thereby explaining the anomaly of "why models with lower cross-ent…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Pre-training"
  - "Test-time Scaling"
  - "coverage profile"
  - "next-token prediction"
  - "maximum likelihood"
  - "Best-of-N"
  - "generalization analysis"
date: 2026-05-08
content_hash: cbf4f0b9640bb9a5
---

# The Coverage Principle: How Pre-Training Enables Post-Training

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=AUXvYQlQLZ](https://openreview.net/forum?id=AUXvYQlQLZ)  
**Code**: Provided in supplementary material (including reproduction scripts for Figure 1/2)  
**Area**: Learning Theory / Pre-training / Test-time Scaling  
**Keywords**: coverage profile, next-token prediction, maximum likelihood, Best-of-N, generalization analysis

## TL;DR
This paper theoretically answers "what pre-training actually leaves for post-training (RL / test-time scaling)"—the answer is not cross-entropy, but a quantity called **coverage profile**. The authors prove that next-token prediction **implicitly** optimizes coverage, and that coverage generalizes faster than cross-entropy without being hindered by sequence length, thereby explaining the anomaly of "why models with lower cross-entropy can have worse Best-of-N performance."

## Background & Motivation

**Background**: Modern language models follow a two-stage pipeline: "large-scale pre-training (next-token prediction + cross-entropy) $\rightarrow$ targeted post-training (usually RL with verifiable rewards or test-time scaling like Best-of-N)." The industry defaults to the assumption that "greater pre-training investment and lower cross-entropy lead to stronger post-trained models," thus using cross-entropy / perplexity as the core metrics for pre-training quality.

**Limitations of Prior Work**: This default assumption **fails empirically**. Multiple studies have observed that starting post-training from a next-token predictor with lower cross-entropy does **not necessarily** result in better downstream performance, sometimes even worse (Figure 1 in the paper plots curves showing negative correlation between cross-entropy/KL and Pass@N). In other words, cross-entropy—the quantity we spend massive compute to minimize—**cannot reliably predict downstream success**.

**Key Challenge**: Cross-entropy (equivalently, sequence-level KL divergence) measures the average log-likelihood over the **entire distribution**. It suffers huge, even infinite, costs from "missing mass" (i.e., the model assigning too low or zero probability to rare high-quality answers), and this cost grows linearly with sequence length $H$. However, what downstream Best-of-N / RL truly needs is just for the "model to reserve **enough probability mass to be sampled** for high-quality answers"—this is a tail/threshold property, which is fundamentally different from average likelihood.

**Goal**: To precisely characterize the relationship between next-token prediction loss and downstream performance, identify a metric "more predictive of downstream success than cross-entropy," and explain the mechanism and conditions under which next-token prediction produces a model suitable for downstream use.

**Key Insight**: The authors propose using **coverage** as a lens to re-examine pre-training—it directly quantifies "how much probability mass the model places on high-quality answers," which is the necessary and sufficient condition for Best-of-N success.

**Core Idea**: Replace cross-entropy with the **coverage profile** to connect pre-training and post-training, and prove a phenomenon called the **coverage principle**: maximum likelihood / next-token prediction implicitly pushes the model toward high coverage, and coverage generalizes faster than cross-entropy, bypassing issues like sequence length dependency.

## Method

### Overall Architecture

This is a **theoretical** paper; the "method" is a chain of arguments aimed at grounding "pre-training $\rightarrow$ downstream success" in coverage rather than cross-entropy. The chain can be viewed as follows:

1. **Establish Metric**: Define coverage profile $\text{Cov}_N$ and prove it is **necessary and sufficient** for Best-of-N (Section 2).
2. **Negate Old Metric**: Prove that while cross-entropy / KL provides a scaling law to coverage (Proposition 3.1), this conversion degrades into a vacuous prediction under finite samples due to sequence length $H$ (Proposition 3.2); thus, cross-entropy is the wrong metric (Section 3).
3. **Main Positive Result**: Prove the coverage principle—next-token prediction (Maximum Likelihood) implicitly optimizes coverage, which **generalizes faster** and does not depend on $H$ (Theorems 4.1 / 4.2, Section 4).
4. **Address Optimizer**: Shift analysis from "ideal MLE solution" to more realistic one-pass SGD, noting that naive SGD coverage is hindered by $H$, whereas **gradient normalization** eliminates this dependence (Section 5).
5. **Propose Interventions**: Propose three categories of algorithmic interventions with provable benefits—test-time decoding and checkpoint selection tournaments (Section 6).

The causal relationship of the entire chain is clear: Metric (Coverage) $\rightarrow$ Why old metrics fail ($H$-dependency of cross-entropy) $\rightarrow$ Why the new metric is naturally optimized by pre-training (Coverage Principle) $\rightarrow$ How to remedy under real optimizers (Gradient Normalization) $\rightarrow$ Practical enhancements (Interventions). The following key designs expand on these individual components.

### Key Designs

**1. Coverage Profile: Quantifying the Sampleability of Correct Answers as a Tail CDF**

The pain point is that cross-entropy / KL takes the **mean** log-density ratio, which can explode due to missing mass, failing as a predictor for downstream success. The authors adopt a tail quantity. Define the coverage profile for model $\hat\pi$ relative to reference distribution $\pi$ (taken as data distribution $\pi_D$):

$$\text{Cov}_N(\pi \,\|\, \hat\pi) := \Pr_{x\sim\mu,\,y\sim\pi(\cdot|x)}\!\left[\frac{\pi(y\mid x)}{\hat\pi(y\mid x)} \ge N\right],$$

where $N$ is the number of samples in Best-of-N. Intuitively, small $\text{Cov}_N$ means few bad samples have density ratios $\pi/\hat\pi$ exceeding $N$, meaning $\hat\pi$ does not suppress answers favored by $\pi$. Thus, taking $\tilde\Theta(N)$ samples will likely hit a high-quality answer. The authors prove (Propositions F.6 / F.7) that for any downstream policy $\pi_T$, Best-of-N suboptimality $\asymp \text{Cov}_N(\pi_T \,\|\, \hat\pi)$. Thus, **good coverage is necessary and sufficient for Best-of-N success**. By leveraging transitivity, the problem reduces to studying when next-token prediction makes $\text{Cov}_N(\pi_D\,\|\,\hat\pi)$ small. Formally, the coverage profile is the **entire CDF** of the log-density ratio $\log\frac{\pi(y|x)}{\hat\pi(y|x)}$, whereas KL is merely its **mean**.

**2. Coverage Principle: Next-token prediction implicitly optimizes coverage and generalizes faster than cross-entropy**

This is the central theorem (Theorem 4.1). First, why cross-entropy fails: the authors prove sequence-level KL grows linearly with $H$ even in simple autoregressive linear models (Proposition 3.2, $D_{KL}\gtrsim H/n$). Applying the KL-to-coverage scaling law $\text{Cov}_N \le \frac{D_{KL}}{\log(N/e)}$ (Proposition 3.1) leads to the vacuous conclusion that test-time compute must scale exponentially with sequence length. However, experiments (Figure 2) show coverage remains nearly constant across different $H$, and Best-of-N remains successful.

The authors explain this using Mendelson’s "small-ball" anti-concentration techniques: exploiting the unique structure of log-loss, they prove the MLE satisfies:

$$\text{Cov}_N(\hat\pi) \lesssim \underbrace{\frac{1}{\log N}\inf_{\varepsilon>0}\!\left(\frac{\log\mathcal{N}_\infty(\Pi,\varepsilon)}{n}+\varepsilon\right)}_{\text{fine-grained}} + \underbrace{\frac{\log\mathcal{N}_\infty(\Pi, c\log N)+\log(1/\delta)}{n}}_{\text{coarse-grained}}.$$

Key takeaways: ① **The fine-grained term is independent of $H$ and the density ratio $\log W_{\max}$**, depending only on the covering number at scale $\varepsilon$. ② This term is scaled by $1/\log N$, meaning **convergence is faster in the deeper tails (larger $N$)**—a novel implicit bias of log-loss. For overparameterized autoregressive linear models, the authors replace $H$ with "inherent variance" $\sigma_\star^2$ (effective sequence length), yielding $\mathbb{E}[\text{Cov}_N(\hat\pi)]\lesssim\sqrt{\sigma_\star^2/(n\log N)}+B^2/n$ (Theorem 4.2). This is the coverage principle: **Pre-training implicitly optimizes coverage, and does so faster than cross-entropy.**

**3. Gradient Normalization: Enabling length-independent coverage for real one-pass SGD**

The main results apply to the "ideal MLE solution," but real pre-training uses one-pass SGD. The authors prove naive sequence-level SGD is hindered by $H$: $\mathbb{E}[\frac1T\sum_t\text{Cov}_N(\pi_{\theta_t})]\lesssim\frac{1}{\log N}(\sqrt{\sigma_\star^2/T}+B^2H/T)$, where the $H$-dependence is proved to be tight (Proposition 5.1). The cause is prompt heterogeneity: some gradients grow with $H$, forcing small learning rates, while others require large rates.

The solution is a simple intervention—**Gradient Normalization**: use mini-batch gradients $\hat g$ for normalized updates $\theta_{t+1}\leftarrow\text{Proj}_\Theta(\theta_t+\eta\cdot\frac{\hat g}{\lambda+\|\hat g\|})$. Theorem 5.1 proves this achieves a **sequence-length independent** coverage bound $\sqrt{\sigma_\star^2/(T\log N)}+B^2/T+B/(K\log N)$. This behavior aligns with Adam/SignSGD, suggesting these optimizers work well because they implicitly optimize coverage.

**4. Test-time Decoding and Checkpoint Tournaments: Plug-and-play coverage interventions**

The authors provide two interventions with provable benefits. **Test-time training-style decoding** (Theorem 6.1): sampling one token, performing a gradient step on its log-likelihood, and resetting parameters after the sequence. This "improper" sampling bypasses the $H$-lower bound of Proposition 5.1. **Coverage tournaments for checkpoint selection** (Theorem 6.2): select $\hat\pi=\arg\min_\pi\max_{\pi'}\widehat{\text{Cov}}_N(\pi'\|\pi)$ using empirical coverage $\widehat{\text{Cov}}_N(\pi'\|\pi)=\frac1n|\{i:\pi'(y_i|x_i)/\pi(y_i|x_i)\ge N\}|$. This method picks the model most difficult to be outperformed by any competitor and removes the realizability assumption $\pi_D\in\Pi$. Using this instead of cross-entropy for checkpoint selection provides a stronger starting point for RL (the $\diamond$ markers in Figure 1).

### Loss & Training
The paper analyzes the standard Maximum Likelihood target $\hat L_n(\pi)=\sum_{i=1}^n\log\pi(y_i\mid x_i)$. Contributions at the "Training Strategy" level include: ① Gradient normalization updates $\theta_{t+1}\leftarrow\text{Proj}_\Theta(\theta_t+\eta\,\hat g/(\lambda+\|\hat g\|))$; ② Token-level SGD combined with test-time training decoding. Analysis assumes realizability (Assumption 2.1, $\pi_D\in\Pi$) and bounded parameters/features (Assumption 2.2).

## Key Experimental Results

> As a theoretical paper, these are **verification experiments** on a graph reasoning task to corroborate that "coverage predicts Pass@N better than KL" and "coverage is independent of sequence length."

### Main Results: Coverage vs. KL as Predictors of Pass@N (Figure 1)

| Observed Attribute | KL / Cross-Entropy Performance | Coverage Profile Performance |
| :--- | :--- | :--- |
| Trend during training | **Monotonically decreases** | May **degrade during training** (synced with downstream) |
| Correlation with Pass@N (Small $N$) | Comparable to coverage | Comparable to KL |
| Correlation with Pass@N (Large $N$) | Significantly worse, even **negatively correlated** | Remains a **better predictor** |
| Checkpoint selection | Selected models have weak Pass@N (red dots) | Tournament selection yields superior Pass@N ($\diamond$) |

Core conclusion: Cross-entropy improves monotonically while Pass@N does not; coverage tracks with Pass@N fluctuations, especially in the tail regions.

### Analysis Experiments: Coverage Dependency on Sequence Length (Figure 2)

| Quantity | Behavior with Sequence Length $H\in\{8,16,24\}$ | Implication |
| :--- | :--- | :--- |
| Sequence-level KL (Converged) | **Grows linearly** with $H$ | Confirms $H/n$ lower bound in Prop 3.2 |
| Coverage $\text{Cov}_{N=16}$ (Converged) | Nearly **independent of $H$** | Confirms Theorem 4.1/4.2 |
| KL / Cov Ratio | Increases significantly with $H$ | Shows Prop 3.1 scaling is too conservative |

### Key Findings
- **Cross-entropy as a "reverse indicator"**: On graph reasoning tasks, cross-entropy/KL can be negatively correlated with Best-of-N performance, falsifying the default assumption that lower loss always implies stronger downstream performance.
- **Missing mass is the culprit**: KL is penalized by $\log W_{\max}$ (up to $H$) for rare answers even good learners cannot cover, while coverage as a tail CDF is immune. In Bernoulli models, KL can be $+\infty$ while coverage remains $\lesssim\log(1/\delta)/n$.
- **Faster tail convergence**: The fine-grained term of coverage scales with $1/\log N$; generalization is faster as $N$ increases—an implicit bias unique to log-loss.
- **Optimizer-level remedies**: Naive SGD coverage is tight on $H$, but gradient normalization (similar to Adam/SignSGD) eliminates this dependence.

## Highlights & Insights
- **Change the ruler, not the model**: The "AHA" moment is realizing we use the wrong metric to measure pre-training—it's not that cross-entropy is not low enough, but that it cannot measure what downstream tasks need (tail coverage).
- **Coverage = CDF of log-density ratio; KL = its mean**: This characterization is highly portable—any scenario where the mean is skewed by tails should consider threshold/CDF measures.
- **"Inherent variance $\sigma_\star^2$ = Effective sequence length"**: Replacing nominal length $H$ with the number of truly uncertain tokens aligns with the empirical observation that most tokens in LMs are near-deterministic.
- **Theoretical explanation for Adam**: Gradient normalization improves coverage and shares roots with Adam/SignSGD, providing a coverage-based explanation for why adaptive optimizers are more stable in LLM training.
- **Deployable interventions**: Using coverage tournaments instead of cross-entropy to select checkpoints for RL is a low-cost improvement that only changes the selection criteria.

## Limitations & Future Work
- **Strong realizability assumptions**: Core theorems rely on $\pi_D\in\Pi$ (Assumption 2.1), which real Transformers do not satisfy. Even with mis-specification discussions in the appendix, direct applicability is limited.
- **Limited model classes**: Detailed analysis of SGD/Normalization/TTT-decoding is confined to **autoregressive linear models**, which are far from non-linear Transformers.
- **Observability of coverage**: Coverage, like KL, is not directly measurable (cross-entropy is just an upper bound). Scaling laws are theoretical predictions rather than engineering metrics that can be deployed immediately.
- **Synthetic tasks**: Verification is on graph reasoning, lacking end-to-end validation on full-scale LLM pre-training $\rightarrow$ RL pipelines.
- **Future Directions**: Extending coverage analysis to non-linear models and modern generalization theories (e.g., benign overfitting); designing cheap proxy indicators for coverage to make tournament selection practical.

## Related Work & Insights
- **vs. Scaling Law Paradigm (Kaplan, Hoffmann, etc.)**: They focus on cross-entropy/perplexity. This paper proves these metrics degrade with sequence length and argues for coverage as a more relevant metric for Best-of-N / RL.
- **vs. "Lower loss $\neq$ stronger downstream" (Liu 2022, Zeng 2025, Chen 2025, etc.)**: While prior work observed this empirically, this paper provides the **theoretical mechanism** (missing mass + faster coverage generalization).
- **vs. Best-of-N / Test-time Scaling analysis (Yue 2025, Wu 2025)**: They showed BoN predicts RL performance; this paper proves BoN success is equivalent to coverage, linking test-time scaling to pre-training metrics.
- **vs. Test-time Training / Dynamic Evaluation (Krause 2019, Sun 2024, Akyürek 2025)**: This paper uses the idea of updating parameters during decoding but provides **provable coverage gains** that bypass lower bounds of "proper" sampling methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefines pre-training outcomes via coverage and proves the Coverage Principle, offering a new fine-grained understanding of Maximum Likelihood generalization.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; verification experiments used controlled tasks rather than end-to-end LLM validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical chain is very clear, progressing from metrics to main results, optimizers, and interventions.
- Value: ⭐⭐⭐⭐⭐ Provides a provable explanation for a major empirical anomaly and offers actionable interventions for checkpoint selection and optimizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Predictors to Samplers via the Training Trajectory](from_predictors_to_samplers_via_the_training_trajectory.md)
- [\[ICLR 2026\] Strong Correlations Induce Cause Only Predictions in Transformer Training](strong_correlations_induce_cause_only_predictions_in_transformer_training.md)
- [\[ICLR 2026\] Training-Free Determination of Network Width via Neural Tangent Kernel](training-free_determination_of_network_width_via_neural_tangent_kernel.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks](tractability_via_low_dimensionality_the_parameterized_complexity_of_training_qua.md)

</div>

<!-- RELATED:END -->
