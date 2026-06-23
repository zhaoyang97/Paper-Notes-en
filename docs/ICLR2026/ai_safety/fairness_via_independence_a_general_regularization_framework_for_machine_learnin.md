---
title: >-
  [Paper Note] Fairness via Independence: A General Regularization Framework for Machine Learning
description: >-
  [ICLR 2026][AI Safety][utility-fairness trade-off] This paper proposes using Cauchy-Schwarz (CS) divergence as a fairness regularization term to minimize the statistical dependence between "model predictions" and "sensitive attributes." Using a unified framework that is model-agnostic and independent of specific fairness definitions, it simultaneously improves $\Delta$
tags:
  - ICLR 2026
  - AI Safety
  - utility-fairness trade-off
date: 2026-05-08
content_hash: 530e7f1a192a5d0c
---
# Fairness via Independence: A General Regularization Framework for Machine Learning

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=sbEb0Ld6MK](https://openreview.net/forum?id=sbEb0Ld6MK)  
**Code**: TBD  
**Area**: AI Safety / Fairness / Group Fairness  
**Keywords**: Group Fairness, Cauchy-Schwarz Divergence, Independence Regularization, Debiasing, Utility-Fairness Trade-off

## TL;DR
This paper proposes using Cauchy-Schwarz (CS) divergence as a fairness regularization term to minimize the statistical dependence between "model predictions" and "sensitive attributes." Using a unified framework that is model-agnostic and independent of specific fairness definitions, it simultaneously improves $\Delta$DP and $\Delta$EO while maintaining accuracy and demonstrating greater robustness to hyperparameter variations.

## Background & Motivation

**Background**: In high-stakes decision-making such as credit, hiring, healthcare, and education, machine learning models often inherit or even amplify biases present in training data. This manifests as undesirable correlations between "prediction outcomes" and "sensitive attributes" (e.g., gender, race, age). Group fairness is the most studied fairness definition, with common metrics being Demographic Parity ($\Delta$DP) and Equal Opportunity ($\Delta$EO).

**Limitations of Prior Work**: Existing debiasing methods follow two main streams, each with significant drawbacks. The first category **directly embeds specific fairness definitions into the training objective** (e.g., specialized DP or EO regularizers). While these target specific metrics effectively, they suffer from **poor generalization**: models trained for DP often perform poorly on EO (as shown in Figure 1 and Table 2, where DP regularization yielded the worst $\Delta$EO in 7/10 cases across 5 datasets). The second category **minimizes statistical dependence between predictions and sensitive attributes** (using information theory or adversarial learning). This approach is more flexible across different fairness definitions, but **performance depends heavily on the choice of distance/dependence measure**, shifting the burden of fairness performance onto the metric selection.

**Key Challenge**: While the second category is theoretically more general, commonly used metrics (gap parity, MMD, KL divergence, HSIC) are **highly sensitive to minor perturbations in model parameters**, resulting in poor robustness. As seen in the loss landscape of Figure 2, the "inner circles" for KL and HSIC span from $-2$ to $2$, implying fairness collapses with slight parameter changes. The problem thus becomes: **Can a principled, cross-task consistent, and perturbation-robust dependence measure be found?**

**Goal**: To identify such a measure that allows the regularizer to generalize across multiple fairness definitions while providing a stable and consistent utility-fairness trade-off under different hyperparameters.

**Key Insight**: Theoretically, CS divergence is known to have **tighter upper bounds** compared to KL divergence and gap parity. Given that the core of fairness is "ensuring predictions are independent of sensitive attributes" ($\hat{Y} \perp S$ is essentially DP), using a dependency measure with a tighter bound to approximate this independence should yield more generalized and robust fairness.

**Core Idea**: Replace KL/MMD/HSIC with Cauchy-Schwarz divergence as the fairness regularization term, unifying "fairness" into a single principle: "minimizing the CS divergence between the prediction distribution and sensitive attributes."

## Method

### Overall Architecture

The entire methodology can be condensed into a single optimization objective: adding a **CS divergence penalty term that measures the difference in prediction distributions between two sensitive groups** to the standard classification loss.

Let the dataset be $\mathcal{D}=\{(x_i,y_i,s_i)\}_{i=1}^M$, where $x_i$ denotes features excluding sensitive attributes, $y_i\in\{0,1\}$ is the task label, and $s_i\in\{0,1\}$ is the sensitive attribute. The model outputs prediction probabilities $z_i=f(x_i,s_i)\in[0,1]$. Splitting the sample batch by sensitive attributes yields two prediction distributions:

$$P = P(\hat{Y}\mid S=0), \qquad Q = P(\hat{Y}\mid S=1).$$

Ideal group fairness (DP) requires $\hat{Y}\perp S$, equivalent to $P=Q$. Instead of directly constraining $\Delta$DP/$\Delta$EO values, this paper uses CS divergence to measure the "distributional distance" between $P$ and $Q$ and minimizes it. The final training objective is:

$$\min_{\theta}\; \mathcal{L}_{\text{BCE}} + \alpha\,\tilde{D}_{\text{CS}}(P,Q) + \frac{\beta}{2}\lVert\theta\rVert_2^2,$$

where $\mathcal{L}_{\text{BCE}}$ is the binary cross-entropy (for accuracy), $\tilde{D}_{\text{CS}}$ is the empirical estimation of CS divergence (for fairness), $\lVert\theta\rVert_2^2$ is L2 regularization, and $\alpha$ controls the fairness-accuracy trade-off. The framework is **model-agnostic** (works with MLPs for tabular data or ResNets for images), **independent of fairness definitions** (not specifically designed for DP or EO), and **distribution-free** (the kernel estimation does not assume a specific parametric form for predictions).

### Key Designs

**1. CS Divergence Regularization Based on the "Prediction $\perp$ Sensitive Attribute" Principle**

This step addresses the "poor generalization" of the first category of methods. Rather than writing dedicated regularizers for DP and EO, the paper returns to the fundamental definition of group fairness—$\hat{Y}\perp S$ (where DP is exactly the independence of predictions and sensitive attributes). By using a **measure of distributional dependence** to approximate this independence, multiple fairness definitions are addressed simultaneously. CS divergence is derived from the Cauchy-Schwarz inequality (equality holds if and only if $p,q$ are linearly dependent). For two probability densities $p,q$:

$$D_{\text{CS}}(p;q) = -\log\!\left(\frac{\left(\int p(x)q(x)\,dx\right)^2}{\int p(x)^2 dx \,\int q(x)^2 dx}\right).$$

It is symmetric, non-negative, and $D_{\text{CS}}=0$ if and only if $p(x)=q(x)$. Substituting $p=P$ and $q=Q$, "the closer the prediction distributions of the two sensitive groups, the smaller the divergence," which directly corresponds to fairness. Crucially, while DP/EO regularizers anchor to specific metric differences, CS divergence anchors to the deeper goal of "distributional consistency"—making it **simultaneously** effective for $\Delta$DP and $\Delta$EO.

**2. Empirical Mini-batch CS Divergence via Kernel Density Estimation**

Since the integral in the theoretical definition cannot be computed directly, this step uses Kernel Density Estimation (KDE) to transform CS divergence into a differentiable, mini-batch computable regularization term. Given two groups of samples, the empirical estimate of CS divergence is:

$$\tilde{D}_{\text{CS}}(p;q) = \log\!\left(\frac{1}{N_1^2}\sum_{i,j}\kappa(x_i^p,x_j^p)\right) + \log\!\left(\frac{1}{N_2^2}\sum_{i,j}\kappa(x_i^q,x_j^q)\right) - 2\log\!\left(\frac{1}{N_1 N_2}\sum_{i,j}\kappa(x_i^p,x_j^q)\right),$$

where $\kappa$ is the Gaussian (RBF) kernel $\kappa_\sigma(x,x')=\exp(-\lVert x-x'\rVert_2^2/2\sigma^2)$, and the bandwidth $\sigma$ is selected using the median heuristic. Computing on all $n$ samples requires $O(n^2)$ complexity; this paper follows MMD/HSIC practices by **estimating on mini-batches**. The extra overhead per step is only $O(B^2)$ ($B\ll n$), utilizing vectorized matrix operations on the same sample batch used for prediction loss. This makes it highly efficient and applicable to tabular and image models. The "distribution-free" nature comes from the sampling-based estimation without parametric assumptions.

**3. Tighter Theoretical Bounds: Why CS is More Robust than KL/MMD/DP**

This provides the theoretical justification for why CS is superior. Prior work has shown that "test-time fairness loss is upper-bounded by training loss"; therefore, **a tighter generalization error bound for the training distance function ensures better test-time fairness**. Proposition 4.2 states: for any Gaussian distributions $p,q$ with positive definite covariance, $D_{\text{CS}}(p;q)\le D_{\text{KL}}(p;q)$ and $D_{\text{CS}}(p;q)\le D_{\text{KL}}(q;p)$. CS is tighter than KL (the Gaussian assumption is used for closed-form comparison and is not required for training). Geometrically, CS divergence uses **cosine distance** (measuring angular/directional differences in feature space), whereas MMD uses Euclidean and DP mean disparity uses Manhattan distance. When group distributions have large variance/scale differences, MMD and DP **overestimate** the difference due to lack of normalization, while CS normalization focuses on "direction," providing a tighter bound—this explains why CS has the smallest inner circle in the loss landscape (Figure 2).

**4. Unifying Existing Regularizers as a "Metric Selection" Problem**

The paper classifies existing debiasing methods into three categories: (i) specialized fairness embedding (DP/EO), (ii) latent representation alignment (MMD), and (iii) prediction-sensitive attribute dependency minimization (HSIC, PR). All share the structure $\mathcal{L}_{\text{fairness}}=D(\cdot,\cdot)$, differing only in the metric $D$. The paper argues these are all instances of the same principle (preventing predictions from carrying sensitive information), and CS divergence is the superior choice due to its tighter bound and more reasonable geometry. This perspective also allows for natural extension to multiple sensitive attributes by either penalizing $\tilde{D}_{\text{CS}}(P_{\hat{Y}|S}, P_{\hat{Y}}P_S)$ or summing across attributes $\sum_k \tilde{D}_{\text{CS}}(P_{\hat{Y}|S_k}, P_{\hat{Y}}P_{S_k})$.

### Loss & Training

The training objective is $\min_\theta \mathcal{L}_{\text{BCE}} + \alpha\tilde{D}_{\text{CS}}(P,Q) + \frac{\beta}{2}\lVert\theta\rVert_2^2$. Hyperparameters $\alpha$ (fairness weight) was searched in $(1\mathrm{e}{-6},150)$ and $\beta$ (L2 weight) in $(1\mathrm{e}{-3},10)$ via grid search. MLPs were used for tabular data and ResNets for images. The kernel was fixed to RBF with median heuristic bandwidth to ensure fair comparison.

## Key Experimental Results

Datasets: 4 tabular (Adult, COMPAS, ACS-I, ACS-T) + 1 image (CelebA-A); sensitive attributes include gender and race. Results are averaged over 10 splits. Utility is measured by ACC/AUC (higher is better), and fairness by $\Delta$DP/$\Delta$EO (lower is better). Baselines: Vanilla MLP, DP regularization, MMD, HSIC, PR.

### Main Results (Partial tabular data, improvements relative to MLP in parentheses)

| Dataset/Attr | Method | ACC(%)↑ | $\Delta$DP(%)↓ | $\Delta$EO(%)↓ |
|------|------|------|------|------|
| Adult / Gender | MLP | 85.63 | 16.52 | 8.43 |
| Adult / Gender | DP | 82.42 | 1.29 (92%) | 20.15 (**Worsened 139%**) |
| Adult / Gender | PR | 81.81 | 0.71 (96%) | 12.45 (Worsened 48%) |
| Adult / Gender | **Ours** | 83.31 | 2.42 (85%) | **2.27 (73%)** |
| COMPAS / Race | MLP | 66.99 | 17.24 | 19.44 |
| COMPAS / Race | HSIC | 64.52 | 2.21 (87%) | 2.72 (86%) |
| COMPAS / Race | **Ours** | 65.62 | **1.79 (90%)** | **1.48 (92%)** |
| ACS-I / Gender | MLP | 82.04 | 10.26 | 2.13 |
| ACS-I / Gender | DP | 81.32 | 0.96 (91%) | 5.37 (**Worsened 152%**) |
| ACS-I / Gender | **Ours** | 81.86 | 0.77 (93%) | **0.90 (58%)** |

**Image Data CelebA-A (Young / Non-Young)**: CS reduced $\Delta$DP by **97.36%** and $\Delta$EO by **98.58%**, the most notable result.

### Key Findings

- **CS suppresses both DP and EO, while specialized regularizers sacrifice one for the other**: DP regularization minimized $\Delta$DP but made $\Delta$EO worse than vanilla MLP ($-139\%, -152\%$); CS kept $\Delta$DP competitive while significantly reducing $\Delta$EO (Adult gender 73%, ACS-I gender 58%).
- **Minimal accuracy cost**: ACC drop was generally $< 3.1\%$, and AUC drop $< 2.2\%$. In some cases, AUC even improved (Adult gender $+0.02\%$, race $+0.58\%$; COMPAS race $+0.35\%$).
- **Superior Pareto Front & Late Collapse**: Trade-off curves (Figure 3) show CS has lowest $\Delta$DP for the same accuracy. PR/DP perform well at low accuracy but fairness collapses as accuracy increases; CS maintains fairness into higher accuracy ranges.
- **Hyperparameter Sensitivity**: Fairness is much more sensitive to $\alpha$ than $\beta$. Changing $\beta$ ten-thousandfold (1e-3 to 10) only slightly reduced $\Delta$DP, whereas a 5x increase in $\alpha$ (1e-2 to 5e-2) slashed $\Delta$DP.

## Highlights & Insights

- **Reframing "Which Fairness Definition" as "Which Dependence Metric"**: The paper intelligently simplifies the problem. By identifying that $\hat{Y}\perp S$ is the core, it transforms the task into finding the best measure of independence, justified by theory (tighter bounds) and geometry (cosine vs. others).
- **Clear Causal Chain from Theory to Robustness**: The logical flow from "tighter bounds $\rightarrow$ better test-time generalization $\rightarrow$ perturbation robustness" is well-supported by loss landscape visualizations.
- **Zero-cost Integration**: The regularizer works on mini-batches with $O(B^2)$ complexity, is vectorized, and reuses the same samples as the prediction loss, making it easy to plug into any training pipeline.

## Limitations & Future Work

- **Scope**: Validation was limited to general ML tasks (tabular + one image dataset); extension to structured tasks like graph learning is pending.
- **Gaussian Assumption**: Proposition 4.2's "CS $\le$ KL" is theoretically derived under Gaussian assumptions. Since real distributions aren't Gaussian, the "tighter bound" benefit is empirically supported rather than theoretically guaranteed for all cases.
- **Debiasing Type**: Focuses on in-process debiasing for binary classification/sensitive attributes. Multi-class/multi-value extensions were noted but not extensively tested.
- **Kernel/Bandwidth**: Used fixed RBF/median heuristic. The impact of hyperparameter $\alpha$ is sensitive, and systematic costs of tuning were not fully explored.

## Related Work & Insights

- **vs. DP/EO Regularizers**: These target specific metrics and lack generalization (optimizing DP hurts EO); CS constrains distribution consistency, improving both.
- **vs. MMD**: Both are kernel methods, but MMD uses unnormalized Euclidean distance, leading to looser bounds when scales differ; CS uses normalized cosine distance for better robustness.
- **vs. HSIC / PR**: These also minimize dependency; this paper provides a unified metric selection framework and shows CS is superior due to tighter bounds.
- **vs. Adversarial Debiasing**: Adversarial methods are notoriously unstable and hard to tune; CS is a single differentiable regularizer that is more stable.

## Rating
- Novelty: ⭐⭐⭐⭐ First use of CS divergence for fairness with theoretical bounds, though it is a "metric migration."
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets with 10 splits each; comprehensive trade-off/T-SNE/sensitivity analysis, though limited image datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-theory-experiment chain.
- Value: ⭐⭐⭐⭐ Model/distribution-agnostic, easy to implement, improves both DP/EO, high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Convergent Differential Privacy Analysis for General Federated Learning](convergent_differential_privacy_analysis_for_general_federated_learning.md)
- [\[ICLR 2026\] A General Framework for Black-Box Attacks Under Cost Asymmetry](a_general_framework_for_black-box_attacks_under_cost_asymmetry.md)
- [\[ICLR 2026\] RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility](resfl_an_uncertainty-aware_framework_for_responsible_federated_learning_by_balan.md)
- [\[ICLR 2026\] Fair Graph Machine Learning under Adversarial Missingness Processes](fair_graph_machine_learning_under_adversarial_missingness_processes.md)
- [\[ICLR 2026\] ReTrace: Reinforcement Learning-Guided Reconstruction Attacks on Machine Unlearning](retrace_reinforcement_learning-guided_reconstruction_attacks_on_machine_unlearni.md)

</div>

<!-- RELATED:END -->
