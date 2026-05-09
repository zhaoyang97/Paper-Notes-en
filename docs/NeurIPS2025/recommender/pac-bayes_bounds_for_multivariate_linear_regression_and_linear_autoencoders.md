---
title: >-
  [Paper Note] PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders
description: >-
  [NeurIPS 2025][Recommender Systems][PAC-Bayes] This paper extends PAC-Bayes generalization bounds from single-output linear regression to **multivariate linear regression**, and further adapts them to **linear autoencoders (LAEs)** in recommender systems. Through theoretical development, the computational complexity is reduced from O(n⁴) to O(n³), and experiments demonstrate that the bounds are tight and highly correlated with practical metrics such as Recall@K and NDCG@K.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - PAC-Bayes
  - generalization bounds
  - linear autoencoder
  - multivariate linear regression
date: 2026-05-08
content_hash: 09284f8c926bb805
---

# PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders

**Conference**: NeurIPS 2025
**arXiv**: [2512.12905](https://arxiv.org/abs/2512.12905)
**Code**: None
**Area**: Statistical Learning Theory / Recommender Systems
**Keywords**: PAC-Bayes, generalization bounds, linear autoencoder, multivariate linear regression, recommender systems

## TL;DR

This paper extends PAC-Bayes generalization bounds from single-output linear regression to **multivariate linear regression**, and further adapts them to **linear autoencoders (LAEs)** in recommender systems. Through theoretical development, the computational complexity is reduced from O(n⁴) to O(n³), and experiments demonstrate that the bounds are tight and highly correlated with practical metrics such as Recall@K and NDCG@K.

## Background & Motivation

**Background**: Linear autoencoders (LAEs) such as EASE and EDLAE achieve strong performance in recommender systems, often matching or surpassing deep learning models. However, this empirical success lacks theoretical explanation. Recommender systems research has long relied excessively on empirical comparisons, and biased evaluation persists due to weak baselines and unreliable sampling metrics.

**Limitations of Prior Work**:
   - **Lack of theoretical understanding of LAEs**: Despite their widespread use, LAEs lack theoretical guarantees on generalization performance.
   - **Limitations of existing PAC-Bayes bounds**: Existing PAC-Bayes bounds for linear regression are restricted to the single-output case ($p=1$) and cannot directly handle multi-output settings.
   - **Computational efficiency**: Optimizing PAC-Bayes bounds in high-dimensional spaces is typically expensive, and LAE models can have hundreds of millions of parameters.

**Key Challenge**: LAEs demonstrate strong practical performance (surpassing classical methods such as ALS), yet statistical learning theory has not provided rigorous generalization guarantees for them. Classical VC-dimension PAC bounds are typically vacuous for large models, while PAC-Bayes theory offers the potential for tighter, non-vacuous bounds—but requires extending from single-output to multi-output settings and from Gaussian data assumptions to bounded data assumptions.

**Goal**: To establish rigorous PAC-Bayes generalization bounds for LAE models, with both theoretical significance (explaining why LAEs generalize well) and practical value (connecting bounds to real ranking metrics).

**Key Insight**: Starting from the single-output linear regression PAC-Bayes bounds of Shalaeva et al., the paper proceeds incrementally: (1) single-output → multi-output; (2) Gaussian assumption → bounded data assumption; (3) incorporating LAE-specific constraints (zero diagonal, hold-out), and developing efficient computation methods.

**Core Idea**: By interpreting LAEs as constrained multivariate linear regression models over bounded data, tight and computable PAC-Bayes generalization bounds can be established.

## Method

### Overall Architecture

The methodology proceeds in four progressive steps:

```
Single-output PAC-Bayes bound (Shalaeva)
    → Multivariate linear regression PAC-Bayes bound (Section 3)
        → LAE bound adapted to bounded data (Section 4)
            → Efficient computation methods (Section 5)
```

The core mechanism is that LAEs under relaxed MSE can be viewed as constrained multivariate linear regression models, so PAC-Bayes bounds for multivariate linear regression transfer naturally.

### Key Designs

#### Module 1: PAC-Bayes Bound for Multivariate Linear Regression (Theorem 3.2)

**Function**: Extends Shalaeva's single-output bound to the multi-output case ($p > 1$).

**Mechanism**: In the multivariate setting, different outputs may exhibit statistical dependence. A generalized Gaussian data assumption (Assumption 3.1) is introduced, allowing the output noise covariance matrix $\Sigma_e$ to be a general positive definite matrix (rather than a scalar), and the input covariance $\Sigma_x$ to be positive semidefinite (permitting degeneracy).

**Key Result**: The bound has a form analogous to Shalaeva's, but the $\Psi$ term involves the eigendecomposition of $\Sigma_W$:

- The exact form depends on eigenvalues $\eta_i$ and non-centrality parameters $b_i$ of $\Sigma_W$.
- The upper bound simplifies to $\ln \mathbb{E}_\pi[\exp(2\lambda^2\|\Sigma_W\|_F^2 / m)]$, depending on sample size $m$, guaranteeing convergence.

**Design Motivation**: The single-output bound is a special case with $p=1$; the multi-output extension is necessary for recommender systems where user vectors are multi-dimensional.

#### Module 2: Convergence Analysis (Theorem 3.3)

**Function**: Establishes rigorous sufficient conditions for convergence of the PAC-Bayes bound.

**Mechanism**: The convergence analysis in Shalaeva et al. lacks rigorous justification for exchanging limits and expectations. This paper provides sufficient conditions via the **Dominated Convergence Theorem**:

$$\mathbb{E}_{W\sim\pi}\left[\exp\left(\lambda\|(\Sigma_x+\mu_x\mu_x^T)^{1/2}(W^*-W)\|_F^2\right)\right] < \infty$$

**Key Instances**:
- If prior $\pi$ has bounded support → holds for any $\lambda > 0$.
- If prior $\pi$ is Gaussian → holds when $\lambda < 1/(2\nu_1\sigma^2)$ (where $\nu_1$ is the largest eigenvalue).

#### Module 3: Adapting to LAEs under Bounded Data Assumption (Section 4)

**Function**: Adapts the multivariate linear regression bound to LAE models in recommender systems.

**Mechanism**:

1. **Data assumption replacement**: Recommendation data consists of binary matrices $H \in \{0,1\}^{n \times m}$, which do not satisfy the Gaussian assumption. A bounded data assumption (Assumption 4.1) is introduced, requiring only that three finite cross-correlation matrices $\Sigma_{xx}$, $\Sigma_{xy}$, $\Sigma_{yy}$ exist.

2. **Relaxed MSE**: The classical MSE evaluates only held-out interactions. This paper removes the mask on predictions, allowing all prediction terms to participate in evaluation. This reduces the LAE evaluation to the standard empirical risk form of linear regression.

3. **LAE-specific constraints**:

    - **Zero diagonal constraint**: $\text{diag}(W) = 0$, preventing the model from degenerating to the identity mapping.
    - **Hold-out constraint**: Input $X$ and target $Y$ cannot both be 1 at the same position.

4. **Explicit expression for the true risk (Lemma 4.2)**:
$$R^{true}(W) = \|W\Sigma_{xx}^{1/2} - \Sigma_{xy}^T\Sigma_{xx}^{-1/2}\|_F^2 - \|\Sigma_{xy}^T\Sigma_{xx}^{-1/2}\|_F^2 + \text{tr}(\Sigma_{yy})$$

#### Module 4: Efficient Computation Methods (Section 5)

**Function**: Makes the bounds practically computable on large-scale datasets.

**Mechanism**:

1. **Closed-form optimal posterior under Gaussian constraints (Theorem 5.2)**: Assuming both prior $\pi$ and posterior $\rho$ are element-wise independent Gaussian distributions, the optimal posterior $\rho = (U, S)$ admits a **closed-form solution** by exploiting the linear structure of LAEs—a significant advantage over Dziugaite & Roy (2017), which requires SGD and Monte Carlo sampling for neural networks.

2. **Complexity reduction under zero diagonal constraint (Theorem 5.4)**:

    - Direct computation requires eigendecomposition of each $A^{(i)}$ → $O(n^4)$.
    - It is shown that $\mathbb{E}_{\pi'}[e^{\lambda R^{true}}] \leq \mathbb{E}_\pi[e^{\lambda R^{true}}]$ → the unconstrained version serves as an upper bound → $O(n^3)$.

3. **Grid search over $\lambda$**: The optimal $\lambda$ is selected from a finite grid $\Lambda = \{1, 2, 4, \ldots, 512\}$, choosing the tightest bound.

### Loss & Training

This paper does not involve model training—the LAE weight matrix $W$ is obtained by methods such as EASE on the training set. The paper **focuses exclusively on evaluating generalization bounds at test time**, entirely independently of the training procedure.

Evaluation uses the relaxed MSE (Equation 8), which removes the mask on predictions relative to the classical MSE, allowing direct correspondence with the empirical risk of linear regression.

## Key Experimental Results

### Main Results

PAC-Bayes bounds are computed on three real-world datasets, alongside reported practical ranking metrics:

| Dataset | # Users | # Items | # Interactions |
|---------|---------|---------|----------------|
| ML 20M | ~138K | ~27K | ~20M |
| Netflix | ~480K | ~18K | ~100M |
| MSD | ~571K | ~42K | ~34M |

**PAC-Bayes Bound vs. Practical Ranking Metrics** (EASE model, varying regularization parameter $\gamma$):

| $\gamma$ | ML20M LH | ML20M RH | ML20M Recall@50 | Netflix LH | Netflix RH | MSD LH | MSD RH |
|---|---------|---------|----------------|-----------|-----------|--------|--------|
| 50 | 61.66 | 128.66 | 0.3434 | 87.22 | 178.11 | 15.96 | 32.60 |
| 200 | 60.06 | 123.67 | 0.3471 | 85.96 | 174.55 | 15.76 | 31.94 |
| 500 | 59.46 | 121.41 | 0.3489 | 85.35 | 172.64 | 15.66 | 31.62 |
| 1000 | 59.19 | 120.17 | 0.3502 | 85.00 | 171.44 | 15.64 | 31.50 |
| 2000 | 59.09 | 119.34 | 0.3510 | 84.72 | 170.45 | 15.68 | 31.52 |
| 5000 | 59.19 | 118.91 | 0.3506 | 84.48 | 169.47 | 15.83 | 31.77 |

(LH = left-hand side / true risk; RH = right-hand side / bound; both in relaxed MSE units)

### Ablation Study

**Bound Tightness Analysis**:

| Metric | This Work | Dziugaite & Roy (2017) Neural Networks |
|--------|----------|----------------------------------------|
| RH/LH ratio | **< 3×** (all cases) | Up to 10× |
| Non-vacuous | ✓ Always non-vacuous | ✓ But looser |

**Correlation Between Bounds and Ranking Metrics**:
- Models with smaller LH/RH generally correspond to higher Recall@50 and NDCG@100.
- The negative correlation is as expected: LH/RH is a loss metric (lower is better), while Recall/NDCG are ranking metrics (higher is better).
- Minor discrepancies appear on some datasets (e.g., on MSD, the optimal LH/RH occurs at $\gamma=1000$, while the optimal Recall occurs at $\gamma=500$).

### Key Findings

1. **The bounds are tight**: RH is consistently within 3× of LH, substantially better than existing PAC-Bayes bounds for neural networks.
2. **The bounds reflect actual performance**: LH/RH is strongly negatively correlated with Recall@K and NDCG@K, indicating that the theoretical bounds genuinely predict practical recommendation quality.
3. **Computational feasibility**: The $O(n^3)$ complexity enables practical computation on large-scale real datasets such as Netflix (18K items).
4. **Dual effect of regularization**: Increasing $\gamma$ makes the model more conservative and the bound tighter, but excessively large $\gamma$ degrades ranking performance.

## Highlights & Insights

1. **Solid theoretical contributions**: A complete extension of PAC-Bayes bounds from single-output to multi-output linear regression is achieved, and the convergence argument in Shalaeva's bounds is rigorously completed.
2. **First theoretical treatment of LAEs**: To the authors' knowledge, PAC-Bayes theory is applied to linear autoencoder recommender models for the first time.
3. **Elegance of the closed-form solution**: The linear structure of LAEs yields a closed-form optimal posterior, avoiding the iterative SGD and Monte Carlo sampling required in the neural network setting.
4. **Clever computational optimization**: Theorem 5.4 reduces the $O(n^4)$ complexity introduced by the zero diagonal constraint to $O(n^3)$ via a concise inequality.
5. **Bridging theory and practice**: The bounds are not only theoretically non-vacuous but also quantitatively correlated with practical ranking metrics—a property that is uncommon in the generalization theory literature.

## Limitations & Future Work

1. **Gap between relaxed MSE and real metrics**: The bounds are based on relaxed MSE rather than Recall@K/NDCG@K as used in practice; the theoretical connection between these quantities remains to be established.
2. **Requirement of known $\Sigma_{hh}$**: The algorithm requires the user-item cross-correlation matrix as input, which in practice can only be approximated using the full dataset.
3. **Restricted to linear models**: The bounds are designed for the linear structure of LAEs and cannot be directly extended to nonlinear autoencoders (e.g., VAEs).
4. **Prior not optimized**: The current approach uses $\pi = \bar{\mathcal{N}}(W, \sigma^2 I)$ as the prior; more sophisticated priors may yield tighter bounds.
5. **Heuristic selection of $\sigma$ and $\Lambda$**: The hyperparameter $\sigma = 0.001$ and the search grid for $\lambda$ are set heuristically without theoretical justification.
6. **Validation only on EASE**: Although the theory is independent of the training method, experiments are conducted only on EASE; models such as EDLAE and ELSA are not evaluated.

## Related Work & Insights

- **Shalaeva et al. (2020)**: This paper directly extends their single-output linear regression PAC-Bayes bounds and corrects gaps in their convergence argument.
- **Dziugaite & Roy (2017)**: First demonstrated non-vacuous PAC-Bayes bounds for neural networks; this paper adopts their Gaussian prior/posterior framework but achieves closed-form solutions.
- **Germain et al. (2016)**: Their PAC-Bayes bounds for linear regression do not converge (independent of $m$); the bounds in this paper address this deficiency.
- **EASE (Steck 2019)**: The PAC-Bayes bounds provide a theoretical explanation for the empirical success of EASE.
- **Generalization theory for recommender systems**: Matrix completion generalization analyses such as Srebro et al. (2004) represent rare prior examples.

**Implications for Research**:
- PAC-Bayes theory is particularly well-suited to "simple yet effective" linear models; future work may explore applications to other linear recommender models.
- The existence of closed-form solutions highlights unique theoretical advantages of linear models that merit further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The multivariate PAC-Bayes bound and LAE adaptation constitute meaningful theoretical extensions, though the methodological framework largely follows prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Complete experiments across three large-scale real-world datasets and multiple regularization parameters, though comparisons with other LAE variants are absent.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear, and the structure is progressively organized; however, the high density of notation presents a barrier for non-theoretically-oriented readers.
- Value: ⭐⭐⭐⭐ — Provides the first rigorous theoretical generalization bounds for LAE recommender models and connects them to practical metrics, offering value to both theory and application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information](../../AAAI2026/recommender/generalization_bounds_for_semi-supervised_matrix_completion_with_distributional_.md)
- [\[NeurIPS 2025\] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[NeurIPS 2025\] Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)

</div>

<!-- RELATED:END -->
