---
title: >-
  [Paper Note] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods
description: >-
  [ICML 2026][AI Safety][Continuous Fairness] This paper extends the "Iterative Null-Space Projection (INLP)" fairness method, originally designed for linear models by Ravfogel et al.…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Continuous Fairness"
  - "Null-Space Projection"
  - "Kernel Methods"
  - "Empirical Feature Space"
  - "SVR"
date: 2026-05-08
content_hash: 2d445fa9298b454b
---

# Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods

**Conference**: ICML 2026  
**arXiv**: [2511.03304](https://arxiv.org/abs/2511.03304)  
**Code**: https://github.com/Felix-St/FairKernelDecomposition (Available)  
**Area**: AI Safety / Algorithmic Fairness / Kernel Methods  
**Keywords**: Continuous Fairness, Null-Space Projection, Kernel Methods, Empirical Feature Space, SVR

## TL;DR
This paper extends the "Iterative Null-Space Projection (INLP)" fairness method, originally designed for linear models by Ravfogel et al., to kernel methods. By deriving a closed-form transformation $\mathbf{T}$ acting directly on the kernel matrix $\mathbf{K}$ within the empirical feature space, the resulting $\mathbf{K}_{(m)}$ remains a positive semi-definite kernel but is stripped of predictive information regarding continuous protected attributes. This allows any kernel-based algorithm (e.g., KRR, SVR) to be converted into a "continuously fair" version, achieving competitive or superior fairness–accuracy Pareto fronts on Crimes, ACSIncome, and ACSTravelTime datasets.

## Background & Motivation
**Background**: Most mainstream fair machine learning research assumes that both protected attributes and targets are discrete—such as "race" bins or binary "gender"—and defines metrics like Demographic Parity or Equalized Odds accordingly. However, anti-discrimination laws (e.g., in the EU) explicitly mention attributes like "age," which is inherently continuous. Furthermore, "race" in social science surveys often appears as continuous values like "percentage of Black population." Forcing these into bins is unnatural and leads to information loss. "Continuous fairness" (where both the target and protected attributes are continuous) is thus a neglected but practically necessary setting.

**Limitations of Prior Work**: Current methods for continuous fairness typically embed a fairness metric (HGR, GDP, PF) as a regularization term or adversarial constraint into the optimization objective. This approach binds the fairness score to a specific model and optimizer, requiring a loss function redesign for every new score. Another approach, INLP (Ravfogel et al., 2020), offers a more elegant conceptual framework: iteratively finding directions that predict the protected attribute and projecting the data into their null space. While INLP is a "model-agnostic + metric-agnostic" preprocessing method, it has only been verified for linear models or neural network embeddings and **cannot be directly applied to kernel methods**. The feature spaces induced by kernels are often infinite-dimensional (e.g., RBF kernels), making it impossible to store feature vectors for projection naively.

**Key Challenge**: To bring the decoupling advantage of INLP—"stripping information first, then feeding it to any downstream model"—to kernel methods (especially Kernel Ridge Regression and Support Vector Regression for regression tasks), a method must be found to **perform null-space projection using only the $n\times n$ kernel matrix $\mathbf{K}$**. This requires avoiding the infinite-dimensional feature space while ensuring the projected matrix remains a valid (positive semi-definite) kernel to maintain the convexity of downstream optimization.

**Goal**: (1) Derive a closed-form transformation acting directly on $\mathbf{K}$ that is equivalent to null-space projection in the feature space; (2) Prove that the transformation preserves PSD properties, is extensible to test points, and supports multiple iterations; (3) Validate its effectiveness as a general preprocessing tool on real-world "continuous fairness" datasets.

**Key Insight**: The authors utilize the "empirical feature space" as a classic tool: the kernel matrix $\mathbf{K}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top = \mathbf{G}\mathbf{G}^\top$, where $\mathbf{G}\coloneq \mathbf{Q}\boldsymbol{\Lambda}^{1/2}$ is an $n$-dimensional explicit representation isometric to the subspace spanned by the training set. Performing INLP on $\mathbf{G}$ is a finite-dimensional operation, which can then be rewritten back to $\mathbf{K}$ using the kernel trick.

**Core Idea**: "Perform INLP in the empirical feature space, then algebraically transform all projections on $\mathbf{G}$ into a single right-multiplication transformation $\mathbf{T}^{\mathbf{K}}=\mathrm{Id}-\mathbf{M}\mathbf{K}$ on the kernel matrix." This method is termed Fair Kernel Decomposition (FKD).

## Method

### Overall Architecture
Input: Raw kernel matrix $\mathbf{K}\in\mathbb{R}^{n\times n}$ (induced by any kernel, RBF used here), continuous protected attribute vector $\mathbf{p}\in\mathbb{R}^n$, iteration count $m$, and ridge regression regularization $\tilde{\alpha}$.
Output: A "fair kernel matrix" $\mathbf{K}_{(m)}$ stripped of information related to $\mathbf{p}$, and a corresponding cumulative transformation matrix $\mathbf{T}_{(m)}$, such that $\mathbf{K}_{(m)} = \mathbf{K}_{(0)}\mathbf{T}_{(m)}$. Any downstream kernel method (KRR and SVR in experiments) is trained normally using $\mathbf{K}_{(m)}$. For new test points, the test kernel $\mathbf{K}_{\text{test}}\in\mathbb{R}^{k\times n}$ is transformed using $\mathbf{T}_{(m)}$ for out-of-sample extension.

The overall pipeline consists of an outer iteration and inner closed-form updates: In each round $(i)$, (i) fit a ridge regression direction $\mathbf{w}$ on the current $\mathbf{K}_{(i-1)}$ to predict $\mathbf{p}$; (ii) construct its null-space projection; (iii) compress this projection into a right-multiplication $\mathbf{T}^{\mathbf{K}_{(i)}}$ on $\mathbf{K}_{(i-1)}$ via Theorem 3.2 to obtain $\mathbf{K}_{(i)}$; (iv) accumulate the multiplication into the total transformation $\mathbf{T}_{(i)}$. Multiple iterations are used because a single projection only eliminates the "most significant predictive direction," while residual non-linear dependencies are stripped away round by round.

### Key Designs

1.  **Null-Space Projection in Empirical Feature Space (Core Theoretical Bridge)**:
    *   **Function**: Replaces the "desirable but impossible" INLP in the infinite-dimensional kernel feature space with an equivalent finite-dimensional operation on $\mathbf{G}$.
    *   **Mechanism**: Decomposes the PSD kernel matrix as $\mathbf{K}=\mathbf{G}\mathbf{G}^\top$ where $\mathbf{G}=\mathbf{Q}\boldsymbol{\Lambda}^{1/2}$. Each row $\mathbf{g}_i$ of $\mathbf{G}$ represents the sample $\mathbf{x}_i$ in an $n$-dimensional representation isometric to the training subspace, where $k(\mathbf{x}_i,\mathbf{x}_j)=\langle \mathbf{g}_i,\mathbf{g}_j\rangle$. Running INLP on $\mathbf{G}$ involves using ridge regression to find $\mathbf{w} = \mathbf{G}^\top(\mathbf{G}\mathbf{G}^\top+\tilde{\alpha}\mathrm{Id})^{-1}\mathbf{p}$ (this formulation is crucial as it expresses $\mathbf{w}$ purely in terms of $\mathbf{K}$), then performing the projection $\mathbf{P}^{\mathbf{G}}=\mathrm{Id}-\mathbf{w}(\mathbf{w}^\top\mathbf{w})^{-1}\mathbf{w}^\top$. Lemma 3.1 proves that the product of consecutive projections is still a projection.
    *   **Design Motivation**: Direct projection in feature space is infeasible for infinite-dimensional kernels like RBF. In INLP, $\mathbf{w}$ typically exists as a linear combination of samples, allowing the entire process to close on $\mathbf{K}$, avoiding the explicit construction of $\mathbf{G}$ while remaining $\mathcal{O}(n^2)$ in memory.

2.  **Closed-Form Kernel Matrix Transformation $\mathbf{T}^{\mathbf{K}}=\mathrm{Id}-\mathbf{M}\mathbf{K}$ (Theorem 3.2 + Cor. 3.3)**:
    *   **Function**: Compresses the two-step process of "projecting on $\mathbf{G}$ and calculating $\mathbf{G}'{\mathbf{G}'}^\top$" into a single right-multiplication on $\mathbf{K}_{(m-1)}$, proving the result remains PSD.
    *   **Mechanism**: Define $\tau_{\text{norm}}\coloneq(\mathbf{w}^\top\mathbf{w})^{-1}$ and $\mathbf{M}\coloneq(\mathbf{K}_{(m)}+\tilde{\alpha}\mathrm{Id})^{-1}\mathbf{p}\,\tau_{\text{norm}}\,\mathbf{p}^\top(\mathbf{K}_{(m)}+\tilde{\alpha}\mathrm{Id})^{-1}$. Then $\mathbf{K}_{(m)}=\mathbf{K}_{(m-1)}(\mathrm{Id}-\mathbf{M}\mathbf{K}_{(m-1)})$. The cumulative form $\mathbf{T}_m=\prod_{i=0}^{m-1}\mathbf{T}^{\mathbf{K}_{(i)}}$ ensures $\mathbf{K}_{(m)}=\mathbf{K}_{(0)}\mathbf{T}_m$, which naturally applies to test kernels for out-of-sample extension. Corollary 3.3 ensures valid kernels through PSD preservation. The authors also state extension to multiple protected attributes only requires replacing $\mathbf{p}\in\mathbb{R}^{n}$ with a matrix $\mathbf{p}\in\mathbb{R}^{n\times l}$.
    *   **Design Motivation**: Unlike Pérez-Suay's HSIC-based approach (KRR-FKL), FKD does not modify the downstream objective or depend on a specific fairness score. It encapsulates fairness logic in kernel preprocessing, achieving "model-agnostic + metric-agnostic" properties. Using $\mathrm{Id}-\mathbf{M}\mathbf{K}$ rather than a general similarity transform preserves PSD.

3.  **Nystroem Approximation + Algorithmic Implementation (Algorithm 1)**:
    *   **Function**: Reduces the $\mathcal{O}(n^3)$ matrix inversion bottleneck in each iteration to an acceptable complexity for medium-scale data.
    *   **Mechanism**: Algorithm 1 maintains $\mathbf{B}=(\mathbf{K}_{(i-1)}+\tilde{\alpha}\mathrm{Id})^{-1}$, $\tau_{\text{norm}}=(\mathbf{p}^\top \mathbf{B}\mathbf{K}_{(i-1)}\mathbf{B}\mathbf{p})^{-1}$, $\mathbf{M}=\mathbf{B}\mathbf{p}\tau_{\text{norm}}\mathbf{p}^\top \mathbf{B}$, and $\mathbf{T}^{\mathbf{K}_i}=\mathrm{Id}-\mathbf{M}\mathbf{K}_{(i-1)}$ per round. The inversion of $\mathbf{B}$ is replaced with Nystroem approximation. Matrix multiplication orders are optimized to save memory (Appendix C).
    *   **Design Motivation**: The exact version's complexity of $\mathcal{O}(m\cdot n^3)$ is impractical for tens of thousands of samples. Nystroem makes the inverse term manageable. Experimental results (§4.5) show the fairness-accuracy Pareto front of the approximate version nearly overlaps with the exact version, proving no loss of critical properties. However, $\mathcal{O}(n^2)$ storage of the kernel matrix still limits large-scale expansion.

### Loss & Training
The method itself introduces no new loss functions. Inner ridge regression uses standard closed-form solutions. Downstream models are trained with their respective standard objectives (KRR: closed-form; SVR: standard dual QP). Fairness is achieved via kernel matrix preprocessing. Hyperparameters include the number of iterations $m$ (controlling stripping strength) and ridge regularization $\tilde{\alpha}$ (controlling the "granularity" of information removal). RBF bandwidth and downstream hyperparameters are fixed via grid-search on non-fair baselines.

## Key Experimental Results

### Main Results
Evaluation is performed on three typical fair regression benchmarks: Communities & Crimes (predicting crime rate, protected attribute = \% Black population), ACSIncome (Montana, protected attribute = Age), and ACSTravelTime (Montana, protected attribute = Age). Prediction accuracy is measured by MAE, and fairness is reported across three metrics: HGR [DP], GDP [DP], and PF [EO]. Results are shown as Pareto fronts of "MAE vs. fairness" (Figure 1) via 5-fold cross-validation. Baselines include KRR-FKL, NN-HGR, and a dummy regressor.

| Dataset | Fairness Metric | Best Performing Method | Notes |
| :--- | :--- | :--- | :--- |
| Crimes | HGR | NN-HGR / KRR-FKL (Strong reg.); SVR-FKD (Slightly better in weak reg.) | Gap closes on GDP/PF |
| Crimes | GDP / PF | SVR-FKD (Clear lead in weak reg.) | Pareto front further towards bottom-left |
| ACSIncome | GDP (Overall) | **SVR-FKD** (Significantly best) | KRR-FKD outperformed by KRR-FKL, but both better than NN-HGR |
| ACSTravelTime | All 3 metrics | **Only SVR-FKD** | Only one to improve fairness while maintaining low MAE; others collapse to dummy level |

### Ablation Study
| Configuration | Key Finding | Description |
| :--- | :--- | :--- |
| SVR-FKD vs SVR-INPL (Fig. 3) | SVR-INPL improves fairness much slower; MAE rises while fairness plateaus at large $m$ | Proves linear null-space projection fails to capture non-linear dependencies (especially on HGR/PF) |
| Multiple Attributes (Fig. 2a) | "Multi" better handles multiple attributes without performance collapse | Theorem 3.2 naturally supports $\mathbf{p}\in\mathbb{R}^{n\times l}$ |
| Ridge Regularization $\tilde{\alpha}$ (Fig. 2b) | Larger $\tilde{\alpha}$ increases MAE faster; small $\tilde{\alpha}$ allows "finer" information stripping | Recommended: small $\tilde{\alpha}$ + adjust strength via $m$ |
| Nystroem Approx. (Fig. 4) | Curves qualitatively identical to exact version | Engineering approximation does not compromise fairness performance |

### Key Findings
*   SVR + FKD is the strongest combination: It achieves the Pareto front across nearly all datasets. The authors hypothesize that the $\epsilon$-insensitive loss of SVR is more robust to the preprocessed kernel structure, whereas KRR is sometimes outperformed by the HSIC approach (KRR-FKL).
*   The necessity of "non-linear projection" is most evident in ACSTravelTime—all baselines collapse to dummy performance, whereas FKD suppresses fairness while maintaining MAE, indicating non-linear coupling between attributes and targets in reality.
*   The role of $\tilde{\alpha}$ is not classic "overfitting prevention" but "granularity control of information stripping," providing a new intuition for tuning: use $m$ as the primary knob.
*   Extension to multiple protected attributes is virtually free: By increasing the dimension of $\mathbf{p}$, the theory and algorithm remain unchanged—a benefit of the metric-agnostic preprocessing paradigm.

## Highlights & Insights
*   The combination of "Empirical Feature Space + INLP" is a clean paradigm synthesis: Empirical feature spaces are a classic tool in kernel learning, and INLP is a known NLP debiasing trick. Combining them for the "continuous fairness + regression + arbitrary kernel" gap requires Theorem 3.2 to close the projection into a PSD-preserving kernel transformation, which is a non-trivial contribution.
*   The "preprocessing instead of constraint" decoupling is transferable: As long as a downstream model is based on a kernel matrix, $\mathbf{K}_{(m)}$ can replace the original kernel without changing training code—offering a different engineering approach compared to fairness loss terms.
*   The form $\mathbf{T}^{\mathbf{K}}=\mathrm{Id}-\mathbf{M}\mathbf{K}$ is noteworthy: It is PSD-preserving, supports closed-form iteration, extends to multiple attributes, and can be accelerated by Nystroem. This form likely has utility in other "attribute stripping" tasks in kernel methods beyond fairness.
*   The reinterpretation of $\tilde{\alpha}$ as a knob for "information stripping vs. signal preservation" is insightful, suggesting that classic tools should not always be used with classic intuitions.

## Limitations & Future Work
*   Storage bottleneck remains: $\mathcal{O}(n^2)$ storage for the kernel matrix is an inherent ceiling for kernel methods. Nystroem only addresses the $\mathcal{O}(n^3)$ inversion. Scaling to hundreds of thousands of samples may require Random Fourier Features or performing projections directly in Nystroem's low-rank representation.
*   Conservative experimental setup: Evaluation uses classic fairness benchmarks whose validity has been questioned (Bao et al., 2021). Sample sizes are small (e.g., Montana), so scalability to large-scale real-world tasks is unverified.
*   Fairness evaluation is highly sensitive to hyperparameters: Different $m / \tilde{\alpha}$ choices yield different Pareto points without an automated selection principle.
*   Scope limited to regression + continuous attributes: "Classification + continuous attributes," "Regression + discrete attributes," Gaussian Processes, and feature-space projections are left for future work. Privacy-preserving scenarios (stripping identity features) are potential research directions.
*   Lack of integration with modern deep models: Most current ML systems are end-to-end neural networks. This method acts on kernel matrices; integration into deep models would require acting on intermediate representations with external kernels.

## Related Work & Insights
*   **vs Ravfogel 2020 (INLP)**: Both share the "iterative direction finding + projection" logic. This paper extends the scope from linear models/embeddings to kernel feature spaces (including infinite-dimensional ones). The key addition is Theorem 3.2 compressing projection into a PSD-preserving kernel transformation for regression.
*   **vs Pérez-Suay 2017 (KRR-FKL)**: Their work embeds HSIC independence into KRR optimization (the "loss modification" school). This paper belongs to the "preprocessing" school—metric-agnostic and compatible with SVR. While KRR-FKD is weaker than KRR-FKL, SVR-FKD outperforms most, showing the flexibility of the preprocessing paradigm.
*   **vs Mary 2019 (NN-HGR)**: NN + HGR bound regularization is a strong baseline. Ours performs better on GDP/PF metrics and medium datasets, is training-gradient free, and does not require neural architecture selection.
*   **vs Tan 2020 (GP fair subspace)**: Both involve subspace methods, but Tan finds alignment subspaces in the hypothesis space, whereas Ours projects directly on the data feature space, offering a more intuitive geometric interpretation.

## Rating
*   Novelty: ⭐⭐⭐⭐ Solid "filling the gap" work extending INLP to kernels via closed-form PSD transformations.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive datasets, metrics, baselines, and ablations, though constrained by small dataset scales.
*   Writing Quality: ⭐⭐⭐⭐ Rigorous derivation and clear motivation regarding the real-world need for continuous fairness.
*   Value: ⭐⭐⭐⭐ Provides a pluggable, PSD-preserving, extensible tool for kernel-based continuous fairness, specifically valuable for SVR in anti-discrimination regression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](prism_gauge-invariant_tangent-space_differentially_private_lora.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[ICLR 2026\] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](../../ICLR2026/ai_safety/adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)

</div>

<!-- RELATED:END -->
