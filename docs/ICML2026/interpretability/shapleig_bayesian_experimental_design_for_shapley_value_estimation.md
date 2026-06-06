---
title: >-
  [Paper Note] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation
description: >-
  [ICML 2026][Interpretability][Shapley Values] For expensive games where evaluation budgets are extremely limited (e.g., requiring model retraining)…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Shapley Values"
  - "Bayesian Experimental Design"
  - "EIG"
  - "Gaussian Processes"
  - "Hamming Kernel"
date: 2026-05-08
content_hash: 7118ff505a73e055
---

# ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation

**Conference**: ICML 2026  
**arXiv**: [2606.02247](https://arxiv.org/abs/2606.02247)  
**Code**: Yes (Open repository in Appendix D.1.5)  
**Area**: Explainable Machine Learning / Shapley Value Estimation / Bayesian Experimental Design  
**Keywords**: Shapley Values, Bayesian Experimental Design, EIG, Gaussian Processes, Hamming Kernel

## TL;DR
For expensive games where evaluation budgets are extremely limited (e.g., requiring model retraining), this work utilizes a Gaussian Process (GP) with a Hamming kernel as a surrogate for the value function. It adaptively selects the next coalition based on the "Expected Information Gain (EIG) for Shapley Values" and compresses the EIG computation complexity from $O(4^p t)$ to $O(p^4 + t^3)$.

## Background & Motivation

**Background**: Shapley Value (SV) is the most widely used axiomatic attribution metric in explainable ML. However, exact calculation requires enumerating $2^p$ coalitions and calling the value function $\nu(S)$ for each. Existing methods generally follow two paradigms: (1) Monte Carlo methods (permutation sampling, MSR, SVARM) which sample coalitions from a **fixed predefined distribution**; (2) Surrogate regression methods (Kernel SHAP, Leverage SHAP, Regression MSR) which fit a surrogate model and then extract SVs, typically using fixed sampling distributions as well.

**Limitations of Prior Work**: When the value function itself is expensive—such as TabPFN feature importance (requiring in-context inference), Ghorbani-Zou style data valuation (retraining a RF/GB), HyperSHAP hyperparameter importance (running a round of HPO), or local explanations for large vision models (costly API calls)—the budget may be limited to a few hundred evaluations. Fixed distribution sampling wastes precious query budget on samples that provide redundant information relative to previous coalitions.

**Key Challenge**: The natural inclination is to perform "adaptive coalition selection." However, within the BED framework, EIG usually lacks a closed-form solution. Standard EIG on GPs typically targets the uncertainty of the **surrogate itself** (e.g., Uncertainty Sampling, US), which does not directly align with the downstream SV quantity. Furthermore, even with a selection criterion, brute-force traversal of $2^p$ coalitions is exponential.

**Goal**: (i) Derive a closed-form expression for "EIG for SV"; (ii) reduce EIG computation from $O(4^p t)$ to a polynomial in $p$; (iii) outperform SOTA sampling/surrogate methods in low-budget regimes.

**Key Insight**: SV is a **linear transformation** of the value function $\phi=A\nu$ (based on the Linearity axiom of Shapley). By reframing "selecting coalitions for SV" as a **Bayesian Linear Inverse Problem + Goal-Oriented OED (GOODE)**, the EIG depends only on the GP posterior covariance rather than specific observations, allowing for a closed-form: $-\tfrac12 \log\det(A\Sigma_{\theta\mid y}A^\top)+C$.

**Core Idea**: Use a Hamming kernel GP as the surrogate + derive a closed-form EIG with SV as the linear end-goal + expand the multiplicative structure of the Hamming kernel using Elementary Symmetric Polynomials (ESP) to make EIG calculable in polynomial time.

## Method

### Overall Architecture
ShaplEIG is a greedy Bayesian Adaptive Design (BAD) loop. **Input**: Player set $P=\{1,\dots,p\}$, value function $\nu:2^P\to\mathbb{R}$ to be estimated, initial coalition subset $\mathcal{C}_0$ (sampled using leverage scores, $T_0=p+1$), candidate pool $\mathcal{C}$, and budget $T$ (typically $\le 512$ in the paper). **Output**: Consistent estimates $\hat\phi$ for all $p$ SVs. In each round $t$, four steps are performed: (1) Select the coalition that maximizes $\mathrm{EIG}^{(t)}_\phi(z^{(i)})$ by **exhaustive search** in the pool $\mathcal{C}$ or over up to 1024 candidates; (2) Evaluate the chosen coalition via $\nu$; (3) Add the new $(z,\nu(z))$ to dataset $\mathcal{D}_{t+1}$; (4) Refit GP hyperparameters $\xi$ via MLE/MAP (this step is the primary mechanism for "true adaptivity," allowing historical observations $y$ to influence subsequent choices). Upon termination, SVs are read directly from the GP posterior mean via $\hat\phi = A\mu_{\nu\mid\mathcal{D}_{T+1}}$.

### Key Designs

1.  **Hamming Kernel GP as Value Function Surrogate**:
    - **Function**: Provides a fully probabilistic surrogate for $\nu$ on the binary coalition space $\{0,1\}^p$, offering well-calibrated uncertainty in low-data regimes.
    - **Mechanism**: Represents coalitions as indicator vectors $z\in\{0,1\}^p$ and uses a weighted Hamming distance kernel $k_\xi(z,z')=\prod_{j=1}^p \xi_j^{\mathbb{1}[z_j\ne z'_j]}$ (with one learnable weight $\xi_j$ per player). For a fixed $\xi$, $\nu(Z)\mid\mathcal{D}_t,\xi$ is a $2^p$-dimensional MVN with **closed-form posterior mean and covariance**.
    - **Design Motivation**: Traditional Kernel SHAP uses a linear surrogate with fixed weights, where posterior uncertainty depends only on the selected coalitions and is **independent of observations**, making the design inherently non-adaptive. A GP with learnable $\xi$ injects information from observations into subsequent EIG computations through kernel hyperparameters, enabling outcome-adaptive design. The multiplicative structure of the Hamming kernel also facilitates ESP expansion, which is essential for $O(p^4)$ computation.

2.  **SVs as the Linear End-Goal for GOODE with Closed-form EIG**:
    - **Function**: Converts the difficult "EIG for the $p$-dimensional SV vector $\phi$" (usually requiring nested MC) into a single log-det expression.
    - **Mechanism**: The Shapley axioms guarantee $\phi = A\nu$, where $A \in \mathbb{R}^{p \times 2^p}$ has elements $\frac{\mathbb{1}_S}{p}\binom{p-1}{|S|-1}^{-1} - \frac{1-\mathbb{1}_S}{p}\binom{p-1}{|S|}^{-1}$. Under a GP prior, this is a linear projection from "parameters → end-goal" in a Bayesian linear inverse problem. EIG can be written as $\mathrm{EIG}_\phi(z^{(i)}) \propto C' + \log[e_i^\top(\Sigma_{\nu\mid\mathcal{D}_t}+\sigma_\epsilon^2 I)e_i] - \log[e_i^\top(\Sigma_{\nu\mid\mathcal{D}_t}+\sigma_\epsilon^2 I - Q)e_i]$, where $Q_{i,i}=(A\Sigma_{\nu\mid\mathcal{D}_t}e_i)^\top (A\Sigma_{\nu\mid\mathcal{D}_t}A^\top)^{-1}(A\Sigma_{\nu\mid\mathcal{D}_t}e_i)$. This instantiates Attia (2018) GOODE results for the SV setting.
    - **Design Motivation**: Applying EIG directly to the untransformed $\nu$ (known as ITL in information theory) on a GP reduces to pure US, selecting coalitions where the surrogate itself is most uncertain while ignoring the impact on the downstream uncertainty of $\phi$. The paper also demonstrates that EPIG is computationally infeasible over the $2^p$ target distribution. Embedding $A$ into the log-det directly optimizes for the downstream quantity of interest.

3.  **Compressing EIG from $O(4^p t)$ to $O(p^4 + t^3)$ using ESP**:
    - **Function**: Makes closed-form EIG calculable for games up to $p \approx 100$. Vectorization over candidate batches yields $O(p^4+t^3+|W|t^2)$.
    - **Mechanism**: A naive algorithm would construct the $2^p \times 2^p$ matrix $\Sigma_{\nu\mid\mathcal{D}_t}$ and then perform projections, with a dominant term of $O(4^p t)$. Theorems B.1/B.2 in the paper rewrite the linear term $AK_\xi(Z,z^{(i)}) \in \mathbb{R}^p$ and the quadratic term $AK_\xi(Z,Z)A^\top \in \mathbb{R}^{p \times p}$ as "weighted kernel sums across coalitions." By identifying shared weights and correlating sums with **univariate/bivariate ESPs**, the complexity drops to $O(p^2)$ and $O(p^4)$ respectively. This is enabled by the multiplicative structure of the Hamming kernel.
    - **Design Motivation**: Without this optimization, closed-form EIG is only "theoretically elegant"—a $p=10$ game would already imply $4^{10} \approx 10^6$ dimensional matrices. This allows the method to scale to real-world tasks like the Crime dataset ($p=101$).

### Loss & Training
The GP hyperparameters $\xi$ are retrained via MAP from the posterior $p(\xi \mid \mathcal{D}_{t+1})$ at each round (or according to a refit schedule for large $p$). Value function evaluations are assumed to have Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)$, though the paper notes consistency for noiseless GPs: when all $2^p$ coalitions are evaluated, GP interpolativity $\mu_{\nu(z)\mid \mathcal{D}_{2^p+1}} = \nu(z)$ ensures $\mu_\phi = \phi(\nu)$. This **structural consistency** is an advantage over Regression MSR (where tree surrogates are inconsistent and require residual correction).

## Key Experimental Results

Experiments used 15 games across 4 task categories with $p \in [8, 101]$, each run with 30 or 100 seeds. All baselines were evaluated with an equal budget per round for fair comparison.

### Main Results
| Task Category | Representative Game | $p$ | ShaplEIG vs SOTA (Low-Budget MSE) |
| :--- | :--- | :--- | :--- |
| FI (TabPFN) | Diabetes Reg. | 10 | Strictly superior to Kernel/Leverage SHAP, Perm. Sampling, and Reg. MSR throughout. |
| DV (RF on Bike Sharing) | Bike Sharing | 10 | Leads all baselines by multiple orders of magnitude. |
| HPI (XGBoost on Chess) | Chess | 16 | Competitive with Reg. MSR initially, then leads in the low MSE regime. |
| LE (ViT 16-patch) | ImageNet | 16 | Outperforms all competitors throughout. |
| LE (RF on Crime, Large $p$) | Crime | 101 | Despite scheduled hyperparameter retraining, it remains feasible and eventually leads. |

The paper describes ShaplEIG as strictly dominant across all budgets on most games, with Regression MSR occasionally matching it over narrow intervals. Other baselines are significantly outperformed in low-budget regions.

### Ablation Study
| Configuration | Key Finding |
| :--- | :--- |
| Full ShaplEIG | Best overall performance. |
| GP + Random sampling | Significantly outperformed by ShaplEIG in most games. |
| GP + Leverage Score Sampling | Occasionally approaches ShaplEIG but is consistently surpassed. |
| GP + Uncertainty Sampling (US) | **Worse than GP + Random**, suggesting classic BED US criteria are unsuitable for SV. |
| ShaplEIG (Large $p \ge 60$, periodic refit) | Slightly outperformed by weak baselines in the first 100 rounds, then overtakes them. |

### Key Findings
- **Performance Source**: High performance is not merely from the GP surrogate itself—ShaplEIG beats GP+Random and GP+US, indicating that EIG-based coalition selection is the primary contribution.
- **US vs. Random**: Counter-intuitively, US performs worse than random for SV estimation because it targets coalitions where the surrogate is most uncertain, ignoring the downstream impact on SV. This justifies the necessity of EIG formulated directly for SV.
- **Computational Overhead**: For $p \le 16$, hyperparameter retraining takes $\le 2$ min/round and EIG takes $< 1$ sec. For $p \le 100$, retraining takes $\le 25$ min/round and EIG takes $\le 30$ sec. **Hyperparameter retraining is the bottleneck**, making GP-based ablations (even without EIG) equally expensive, which supports the efficiency of the EIG derivation itself.
- **Sweet Spot**: ShaplEIG is ideal when $\nu$ is very expensive (heavy retraining/API calls) and $p$ is moderate (optimal for $\le 16$, acceptable for $\sim 100$).

## Highlights & Insights
- **Goal-Oriented Targeting**: Directly rejects the naive view that "an accurate surrogate naturally leads to accurate SVs." It proves that while EIG for $\nu$ reduces to US, it is a highly effective criterion for the linear projection $A\nu$.
- **Hamming Kernel + ESP Synergy**: The multiplicative structure $\prod_j \xi_j^{\mathbb{1}[\dots]}$ allows the kernel sums over player subsets to map perfectly to Elementary Symmetric Polynomials. This custom-fit synergy provides the polynomial-time efficiency that other kernel classes (like categorical RBF) cannot achieve.
- **Consistency and Closed-form**: Offers engineering advantages over Regression MSR. While MSR's tree surrogate is biased and requires residual MSR correction, ShaplEIG is automatically consistent under noiseless GP settings.
- **Transferable Design**: The GOODE + Hamming-GP + ESP framework can be reused for any linear functional of $\nu$ in expensive games (e.g., Sobol indices, fANOVA decomposition) simply by changing the matrix $A$.

## Limitations & Future Work
- **Bottleneck**: The authors acknowledge that GP hyperparameter retraining is the primary cost. For $p > 100$, the multi-minute per-round cost makes the method viable only if $\nu$ is even more expensive. It is inefficient for cheap $\nu$ (e.g., standard SHAP).
- **Evaluation Scenarios**: Evaluations relied on **precomputed games** (cached value tables) for ground-truth comparison. The framework has not been stress-tested in end-to-end "in-the-loop" scenarios like retraining LLMs or diffusion models where evaluations take hours.
- **Greedy Selection**: The suboptimality of greedy BAD and consistency under noisy GPs are discussed primarily in the Appendix. The poor performance of pure exploration (US) suggests sensitivity to game symmetry, though this is not yet quantified.
- **Future Directions**: (i) Amortized/lazy hyperparameter retraining; (ii) Batch BED for picking multiple coalitions; (iii) More expressive kernels that preserve ESP compatibility; (iv) Extension to high-order Shapley interaction indices.

## Related Work & Insights
- **vs. Kernel SHAP / Leverage SHAP / Regression MSR**: These use fixed sampling distributions and observation-agnostic surrogates. ShaplEIG uses an outcome-adaptive design targeting SV directly and is inherently consistent.
- **vs. BayesSHAP (Slack 2021)**: BayesSHAP uses Bayesian linear models with US. ShaplEIG improves this by (a) using GPs for better outcome-adaptivity and (b) upgrading the criterion from US to EIG for SV.
- **vs. Mitchell 2022 / Nguyen 2025 (GP+BQ for SV)**: Their kernels are defined on permutations or data distributions, usually reducing uncertainty for one player at a time with fixed kernels (non-adaptive). ShaplEIG's kernel is in the coalition space and targets the joint uncertainty of all players' SVs with adaptive retraining.
- **vs. Active Learning / ITL / EPIG**: ITL collapses to US here, and EPIG is computationally intractable over the $2^p$ space. ShaplEIG leverages the linearity of SV to achieve a closed-form solution, bypassing standard nested MC issues in BED.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces the GOODE perspective to Shapley estimation with original theoretical and algorithmic results (closed-form EIG + ESP $O(p^4)$ computation).
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 15 games across 4 categories with $p \in [8, 101]$, including strong baselines and ablations, though lacks end-to-end online "expensive $\nu$" verification.
- Writing Quality: ⭐⭐⭐⭐ Clearly links BED terminology, Shapley axioms, and GP derivations, though the reliance on the appendix for core proofs makes the main text feel slightly abstract.
- Value: ⭐⭐⭐⭐ Provides a SOTA estimator for the specific "expensive $\nu$ + low budget" regime with a clear interface for related problems like fANOVA or Sobol indices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)
- [\[ICML 2026\] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere](neural_collapse_by_design_learning_class_prototypes_on_the_hypersphere.md)

</div>

<!-- RELATED:END -->
