---
title: >-
  [Paper Note] Combinatorial Bandit Bayesian Optimization for Tensor Outputs
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] Addressing expensive black-box systems where the output is a multi-modal tensor, this paper proposes a Tensor Output Bayesian Optimization framework (TOBO). It utilizes a Tensor Gaussian Process (TOGP) as a surrogate model to capture cross-modal dependencies and employs Upper Confidence Bound (UCB) sampling. The framew
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 7228cbf822c5af78
---
# Combinatorial Bandit Bayesian Optimization for Tensor Outputs

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=BgX4s1d2d0](https://openreview.net/forum?id=BgX4s1d2d0)  
**Code**: To be confirmed  
**Area**: Bayesian Optimization / Combinatorial Multi-Armed Bandit / Gaussian Process / Tensor Learning / Learning Theory  
**Keywords**: Tensor outputs, Bayesian optimization, Combinatorial bandits, Tensor Gaussian Process, regret bounds

## TL;DR
Addressing expensive black-box systems where the output is a multi-modal tensor, this paper proposes a Tensor Output Bayesian Optimization framework (TOBO). It utilizes a Tensor Gaussian Process (TOGP) as a surrogate model to capture cross-modal dependencies and employs Upper Confidence Bound (UCB) sampling. The framework is further extended to a Combinatorial Bandit Bayesian Optimization (CBBO) setting where only a subset of tensor elements contributes to the objective. A CMAB-UCB2 criterion is designed to jointly select input points and subsets, both of which are proven to achieve sublinear regret of $\tilde{O}(\sqrt{T})$.

## Background & Motivation

**Background**: Bayesian Optimization (BO) is a primary method for optimizing expensive, black-box objective functions where each evaluation is costly. It is widely applied in hyperparameter search, experimental design, and robotics. The standard approach involves fitting a Gaussian Process (GP) surrogate model to observed data and optimizing an acquisition function (e.g., UCB) to balance exploration and exploitation. While most BO handles scalar outputs, multi-output BO (MOBO) has recently emerged.

**Limitations of Prior Work**: To the authors' knowledge, no existing work handles **tensor-valued outputs**, where the system response to an input $x$ is an $m$-mode tensor $f(x)\in\mathbb{R}^{t_1\times\cdots\times t_m}$. A direct approach would be vectorizing the tensor for MOBO, but this **destroys the inherent cross-modal structural dependencies** (mode-wise dependency), which are critical in complex systems like spatio-temporal processes.

**Key Challenge**: Existing tensor-output GPs (e.g., HOGP) generally adopt **separable covariance** structures for computational efficiency, where the joint covariance is expressed as the Kronecker product of an input kernel and individual mode kernels. The cost of separability is the assumption that "cross-modal correlation structures remain constant across inputs," which is frequently violated in complex systems, leading to poor prediction, unstable posterior inference, and degraded BO performance.

**Goal**: (1) Develop a flexible and scalable GP surrogate capable of capturing "input-dependent intra-tensor correlations"; (2) Design acquisition functions and sequential query strategies for tensor-output BO with regret guarantees; (3) Address more practical settings where only $k<T$ elements in the tensor contribute to the objective, requiring joint selection of "query points + element subsets."

**Key Insight**: Generalize the "Linear Model of Coregionalization" (LMC) from vector-valued outputs to **tensor-valued outputs**. Mixing a set of base kernels allows cross-element correlations to vary with inputs, thereby bypassing the separability assumption. Selecting a subset of elements is naturally modeled as a Combinatorial Multi-Armed Bandit (CMAB), where each tensor element is a base arm and a super-arm of size $k$ is selected in each round.

**Core Idea**: Implement TOBO using "LMC Tensor Kernels + UCB," and solve subset selection (TOCBBO) by coupling a "GP surrogate + Combinatorial Bandit UCB" into the CMAB-UCB2 criterion, providing sublinear regret proofs for both.

## Method

### Overall Architecture

The problem is divided into two layers. The first layer is **TOBO (Tensor Output BO)**: the goal is $f: \mathcal{X} \to \mathcal{Y}$ where input $x \in \mathcal{X} \subset \mathbb{R}^d$ and the output is an $m$-mode tensor. Since a whole tensor cannot be maximized directly, a bounded linear operator $\mathcal{L}_f$ is used to scalarize the tensor to find:

$$x^\star = \arg\max_{x \in \mathcal{X}} \mathcal{L}_f f(x).$$

The TOBO cycle follows three classic BO steps: fit $f$ using a Tensor Output GP (TOGP) $\to$ compute posterior mean/variance $\to$ select $x_{n+1}$ using a UCB acquisition function $\to$ observe noisy output $y_i = f(x_i) + \varepsilon_i$.

The second layer is **TOCBBO (Combinatorial Bandit BO)**: only $k$ elements in the tensor count toward the objective. A binary indicator vector $\lambda \in \{0,1\}^T, \mathbf{1}^\top \lambda = k$ represents the selection (equivalent to choosing a super-arm $S$ of size $k$). The goal becomes joint optimization:

$$(x^\star, \lambda^\star) = \arg\max_{x \in \mathcal{X}, \lambda \in \Lambda} \mathcal{H}_f \tilde f(x, \lambda), \quad \tilde f(x, \lambda) = e(\lambda) \mathrm{vec}(f(x)).$$

Joint search over $x$ and $\lambda$ is infeasible ($\binom{T}{k}$ combinations). CMAB-UCB2 **decouples this into two steps**: fixing the current best super-arm to select $x$, then fixing $x$ to select $\lambda$ via combinatorial bandit UCB.

### Key Designs

**1. LMC Tensor Output Kernel: Enabling Input-Dependent Cross-Element Correlation**

This is the foundation of the work, addressing the limitation where separable covariances assume static mode-wise correlations. The authors generalize vector-valued LMC to tensors: let $\mathrm{vec}(f(x)) = \sum_{\ell=1}^m \sum_{j=1}^{r_\ell} a_{\ell j} u_{\ell j}(x)$, where loading vectors $a_{\ell j} = \mathrm{vec}(A_{\ell j})$ are parameterized by core tensors $A_{\ell j} \in \mathbb{R}^{t_1 \times \cdots \times t_m}$, and $\{u_{\ell j}\}$ are independent scalar GPs with base kernels $k_{\ell j}(x, x')$. This yields a **non-separable kernel** (Definition 1):

$$K(x,x')=\sum_{l=1}^{m}\sum_{j=1}^{t_l}\mathrm{vec}(A_l)\,\mathrm{vec}(A_l)^\top k_{lj}(x,x').$$

Crucially, cross-element correlation is governed by a **mixture** of base kernels $\{k_{lj}\}$, allowing the correlation structure to vary with input $x$. When all components share the same base kernel, it collapses back to a **separable kernel** (Definition 2). The authors prove (Proposition 1) that both classes of kernels are symmetric positive semi-definite (SPSD). For efficiency, core tensors can use CP/Tensor-Train decompositions, and Nyström approximations reduce the complexity of $(K_n + \eta I_{nT})^{-1}$ to $O(nTn_\ell^2 + n_\ell^3)$.

**2. TOGP Surrogate + UCB Acquisition: Integrating Tensor Uncertainty**

With the defined kernel, the TOGP posterior of $\mathrm{vec}(f(x))$ given $n$ observations remains Gaussian, with mean:

$$\hat\mu_n(x)=\mu+K_n^\top(x)(K_n+\eta I_{nT})^{-1}(Y_n-\mathbf{1}_n\otimes\mu),$$

and covariance $\hat K_n(x,x) = \sigma^2 [K(x,x) - K_n^\top(x)(K_n + \eta I_{nT})^{-1} K_n(x)]$. Hyperparameters are estimated via MLE. The acquisition function applies $\mathcal{L}_f$ to the posterior:

$$\alpha_{\text{UCB}}(x\mid\mathcal{D}_n)=\mathcal{L}_f\hat\mu_n(x)+\beta_n\,\hat K_n(x,x)^{1/2},$$

selecting $x_{n+1} = \arg\max_x \alpha_{\text{UCB}}$. This approach ensures UCB explores directions where the **holistic tensor posterior uncertainty** is highest.

**3. Partial Observation TOGP (PTOGP): Learning from Subsets**

In CBBO, only $k$ elements corresponding to super-arm $S_i$ are observed. The authors use a binary selection matrix $e(\lambda) \in \{0,1\}^{k \times T}$ to project TOGP onto the selected subset, resulting in PTOGP: prior $\tilde f(x,\lambda) \sim \mathcal{PTOGP}(e(\lambda)\mu, \tau^2 e(\lambda)K(x,x')e(\lambda')^\top)$. The posterior for a new pair $(x, \lambda)$ is Gaussian on $\mathbb{R}^k$. Importantly, because the covariance kernel encodes cross-element correlations, **unobserved elements can be inferred from the observed $k$ elements**, enabling the evaluation of arbitrary candidate super-arms in the CMAB step.

**4. CMAB-UCB2: Decoupled Joint Selection**

To avoid combinatorial explosion, CMAB-UCB2 splits the $n+1$ round into two: first, **select input fixing the super-arm** to the current best $\lambda_n^\star$:

$$x_{n+1}=\arg\max_{x\in\mathcal{X}}\mathcal{H}_f\tilde\mu_n(x,\lambda_n^\star)+\tilde\beta_n\|\tilde K_n(x,x;\lambda_n^\star,\lambda_n^\star)\|^{1/2};$$

second, **select the super-arm fixing $x_{n+1}$** by maximizing a UCB over candidate super-arms:

$$\lambda_{n+1}=\arg\max_{\lambda\in\Lambda}\mathcal{H}_f\tilde\mu_n(x_{n+1},\lambda)+\tilde\rho_n\|\tilde K_n(x_{n+1},x_{n+1};\lambda,\lambda)\|^{1/2}.$$

### Loss & Training

There is no neural network training; "training" refers to MLE for hyperparameters $\Theta = \{\theta, a, \sigma^2, \tau^2\}$. Core tensor ranks are selected via cross-validation using MAE ($r^\star = \arg\min_r \mathrm{MAE}(r)$). Theoretical guarantees provide that for **TOBO** (Theorem 1), regret $R_N$ satisfies with high probability:

$$R_N\le L\sqrt{C_1 N\gamma_N(K,\eta)}\,\beta_N+\tfrac{\pi^2}{6},$$

where $\gamma_N$ is the maximum information gain. **TOCBBO** (Theorem 2) exhibits an additional term for super-arm selection. Under a separable kernel with a Gaussian base kernel, both achieve sublinear regret of $O(\sqrt{T \log T})$, marking the first such analysis for tensor-valued outputs.

## Key Experimental Results

Experiments used synthetic data and real-world cases, comparing against baselines that vectorize tensors for MOGPs: sMTGP, MLGP, MVGP, paired with either UCB or random sampling.

### Main Results: Synthetic Data

For prediction accuracy (NLL/MAE, lower is better):

| Setting | Metric | TOGP (Ours) | sMTGP | MLGP | MVGP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (1) | NLL | **503.0** | 749.4 | 707937.1 | 11152.2 |
| (1) | MAE | **0.1571** | 0.1684 | 0.9428 | 0.6746 |
| (2) | NLL | **-18.1** | -5.0 | 7066.9 | 46.54 |
| (3) | MAE | **0.1372** | 0.1501 | 1.1670 | 1.0000 |

TOGP achieved the lowest NLL/MAE across all settings. Optimization performance (lower $\text{MSE}_x$, higher Accuracy for CBBO):

| Task | Method | Set (1) MSE$_x$ | Set (1) Acc | Set (3) MSE$_x$ | Set (3) Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BO | TOBO | **0.0000** | – | **0.0001** | – |
| BO | sMTGP-UCB | 0.0001 | – | 0.0048 | – |
| CBBO | TOCBBO | **0.0023** | **1.00** | **0.0021** | **1.00** |
| CBBO | sMTGP-UCB | 0.1832 | 0.67 | 0.0075 | 0.86 |
| CBBO | MVGP-UCB | 0.0032 | 1.00 | 0.0151 | 0.71 |

### Ablation Study

| Configuration | Description | Observation |
| :--- | :--- | :--- |
| TOBO / TOCBBO (Full) | TOGP + UCB | Optimal across all settings. |
| TOGP-RS | TOGP + Random Sampling | Significant degradation in $\text{MSE}_x$. |
| sMTGP-UCB / MVGP-UCB | Alternative surrogates | Prediction and optimization both degrade. |

### Key Findings
- **TOGP Surrogate is the Driver**: Replacing it with vectorized MOGPs degrades both prediction and optimization.
- **Kernel Selection**: Non-separable kernels are more powerful for complex dependencies but carry higher computational costs.
- **Real-world Success**: On the REEN dataset, TOGP achieved the best NLL (15.67) and MAE (0.0883). Optimal optimization curves were observed across CHEM, MAT, PRINT, and REEN datasets.

## Highlights & Insights
- **Generalizing LMC to Tensors**: Using core tensors with mixed base kernels allows cross-element correlations to be input-dependent, unifying existing separable kernels as special cases.
- **PTOGP Inference**: The encoding of structural correlation allows the model to infer unobserved parts of the tensor, making subset evaluation possible during the CMAB step.
- **Pragmatic Decoupling**: CMAB-UCB2 bypasses the combinatorial explosion of the joint search space while maintaining theoretical regret bounds.
- **Theoretical Pioneering**: This work provides the first sublinear regret analysis for Bayesian Optimization with tensor outputs.

## Limitations & Future Work
- **Manual Kernel Type Selection**: Choosing between separable and non-separable kernels is currently manual; an automated selection mechanism is lacking.
- **Combinatorial Scale**: The CMAB-UCB2 step requires evaluating candidate super-arms; while efficient for $k=T/6$, very large $T$ and $k$ could still pose computational bottlenecks.
- **Experimental Scale**: Experiments utilized relatively low-dimensional tensors; scalability to massive multi-modal tensors requires further testing.

## Related Work & Insights
- **vs. High-order GP (HOGP)**: HOGP uses separable Kronecker structures (input-independent). This work uses LMC structures for input-dependent, non-separable correlations.
- **vs. Multi-output BO (MOBO)**: MOBO is designed for vectors; information-theoretic methods in MOBO become computationally infeasible as the number of objectives increases. This work is the first to handle tensor outputs with regret analysis.
- **Inspiration**: When outputs have strong inherent structures (tensors, graphs, sequences), encoding this structure directly into the kernel is superior to vectorization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] BoGrape: Bayesian optimization over graphs with shortest-path encoded](bogrape_bayesian_optimization_over_graphs_with_shortest-path_encoded.md)
- [\[ICLR 2026\] Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models](adaptive_acquisition_selection_for_bayesian_optimization_with_large_language_mod.md)
- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](symmetry-aware_bayesian_optimization_via_max_kernels.md)
- [\[ICLR 2026\] Contextual Causal Bayesian Optimisation](contextual_causal_bayesian_optimisation.md)
- [\[ICLR 2026\] NExCO: Native Solution Expansion for Diffusion-based Combinatorial Optimization](nexco_native_solution_expansion_for_diffusion-based_combinatorial_optimization.md)

</div>

<!-- RELATED:END -->
