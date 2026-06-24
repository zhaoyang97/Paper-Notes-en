---
title: >-
  [Paper Note] On the Variability of Concept Activation Vectors
description: >-
  [AAAI 2026][Concept Activation Vectors] This paper presents the first theoretical analysis of the variability of Concept Activation Vectors (CAVs) in the TCAV framework. It proves that the variance of CAVs decays at a rate of $O(1/N)$ (where $N$ is the number of random samples), while the variance of TCAV scores remains $O(1)$ due to "boundary points," and can only be reduced to $O(1/s)$ by averaging over multiple runs.
tags:
  - "AAAI 2026"
  - "Concept Activation Vectors"
  - "TCAV"
  - "variance analysis"
  - "asymptotic normality"
  - "interpretable AI stability"
date: 2026-05-08
content_hash: 55fe219752da8b8a
---

# On the Variability of Concept Activation Vectors

**Conference**: AAAI 2026
**arXiv**: [2509.24058](https://arxiv.org/abs/2509.24058)  
**Code**: To be released  
**Area**: Other
**Keywords**: Concept Activation Vectors, TCAV, variance analysis, asymptotic normality, interpretable AI stability

## TL;DR

This paper presents the first theoretical analysis of the variability of Concept Activation Vectors (CAVs) in the TCAV framework. It proves that the variance of CAVs decays at a rate of $O(1/N)$ (where $N$ is the number of random samples), while the variance of TCAV scores remains $O(1)$ due to "boundary points," and can only be reduced to $O(1/s)$ by averaging over multiple runs.

## Background & Motivation

### State of the Field

**Background**: TCAV (Testing with Concept Activation Vectors) is one of the core methods in concept-based interpretability. It obtains a concept direction vector (CAV) by training a linear classifier to separate concept embeddings from random embeddings, and then measures the sensitivity of model predictions to that direction.

**Limitations of Prior Work**: TCAV relies on random sampling to construct reference sets, causing results to vary across runs. Kim et al. recommend averaging over multiple runs, but provide no quantitative guidance on how many runs or how many samples are needed to obtain stable results.

**Core Problem**: Given a fixed computational budget, is it better to perform one large-sample run or to average over multiple small-sample runs? Theoretical guidance has been absent.

**Key Insight**: Leveraging asymptotic theory for imbalanced logistic regression, this paper analyzes the convergence behavior of the CAV estimator as the number of random samples tends to infinity.

## Method

### Theoretical Framework

The computation of CAVs is formalized as an imbalanced logistic regression problem: the number of concept samples is fixed while the number of random samples $N \to \infty$. The asymptotic properties of the CAV estimator $\hat{\beta}_N$ are analyzed under this limit.

### Key Theoretical Results

1. **Theorem 1: Asymptotic Normality of CAVs**:
    - Statement: Under the "mean-enclosing" assumption, $\sqrt{N}(\hat{\beta}_N - \beta_0) \Rightarrow \mathcal{N}(0, \Sigma)$
    - Corollary: The variance of CAVs satisfies $\text{tr}(\text{Cov}(\hat{\beta}_N)) = O(1/N)$
    - Significance: Increasing the number of random samples effectively stabilizes the CAV direction estimate
    - Proof sketch: A Taylor expansion of the loss gradient at the optimum is combined with the law of large numbers (for Hessian convergence) and the central limit theorem (for score convergence), via Slutsky's theorem

2. **Corollary 1: Variance of Sensitivity Scores**:
    - Statement: $\sqrt{N}(S(\mathbf{x}, \beta_N) - S(\mathbf{x}, \beta_0)) \xrightarrow{D} \mathcal{N}(0, V(\mathbf{x}))$
    - Significance: The variance of sensitivity scores also decays at $O(1/N)$

3. **Unexpected Finding on TCAV Score Variance**:
    - Statement: The variance of TCAV scores does **not** decay with $N$, remaining $O(1)$
    - Cause: TCAV scores are computed by thresholding sensitivity scores; "boundary points" (samples with sensitivity near 0) are highly sensitive to small changes in the CAV and contribute a constant variance term
    - Remedy: Averaging over multiple runs reduces variance as $\text{Var}(T_{\text{multi}}) = O(1/s)$

### Practical Recommendations

- To stabilize **TCAV scores**: use a large number of independent runs (large $s$), each with a relatively small sample size
- To stabilize **CAV directions** (e.g., for downstream applications such as bias mitigation): increase the per-run sample size $N$
- There is no universal setting: the optimal allocation depends on the specific method and implementation

## Key Experimental Results

### Cross-Modal Validation


### Main Results

| Data Type | Dataset | Model | CAV variance $\propto 1/N$? | TCAV variance stable? |
|-----------|---------|-------|----------------------------|----------------------|
| Image | ImageNet + Broden | ResNet | ✅ | ✅ |
| Tabular | UCI Adult | 2-layer MLP | ✅ | ✅ |
| Text | IMDB | CNN classifier | ✅ | ✅ |

### CAV Variance Decay Validation


### Ablation Study

| $N$ (number of random samples) | CAV variance (trace) approximate magnitude |
|-------------------------------|-------------------------------------------|
| 10 | ~$10^{-1}$ |
| 50 | ~$10^{-2}$ |
| 100 | ~$10^{-2.5}$ |
| 200 | ~$10^{-3}$ |

Across all three domains, the empirical variance is consistent with the theoretically predicted $1/N$ decay rate.

### Multi-Run Averaging

| $s$ (number of runs) | TCAV variance |
|---------------------|---------------|
| 2 | ~0.01 |
| 5 | ~0.004 |
| 10 | ~0.002 |
| 20 | ~0.001 |

Variance decreases at a rate of $1/s$, consistent with Conjecture 1.

### Key Findings
- CAV variance decay is independent of classifier type — logistic regression, SVM, and mean-difference methods all exhibit the same $O(1/N)$ behavior
- The non-decaying TCAV variance is attributable to the "boundary point" effect — samples with sensitivity near zero become overly sensitive to small CAV perturbations after thresholding
- Computational budget allocation tradeoff: multiple runs are more efficient than a single large-sample run for stabilizing TCAV scores, whereas the opposite holds for stabilizing CAV directions

## Highlights & Insights
- **"Boundary points" prevent TCAV variance from converging**: This finding is counterintuitive — even as CAVs become highly precise, TCAV scores can remain unstable. The root cause lies in the discontinuity of the thresholding operation (indicator function), a general issue affecting all threshold-based statistics
- **Theoretical analysis paradigm from LIME to TCAV**: This work draws on the stability analysis of LIME by Garreau & Mardaoui (2021) and establishes a theoretical framework for analyzing the reliability of XAI methods
- **High practical value**: Concrete computational budget allocation recommendations are provided, offering direct guidance for practitioners using TCAV

## Limitations & Future Work
- The theoretical analysis assumes perfect optimizer convergence; incomplete solver convergence in practice may introduce additional variance
- The "mean-enclosing" assumption (Assumption 1) is generally satisfied but no verifiable sufficient conditions are provided
- The variance behavior of non-linear concept boundary methods (e.g., CAR, Concept Gradient) is not analyzed
- Conjecture 1 lacks a formal proof and relies on an independence assumption

## Related Work & Insights
- **vs. LIME stability analysis (Garreau & Mardaoui 2021)**: The instability of LIME stems from insufficient sampling, with variance also decaying as $1/N$; the instability of TCAV is more fundamental — TCAV scores remain unstable even after CAVs have converged
- **vs. Adversarial CAV (Soni et al. 2020)**: That work improves CAV robustness via adversarial perturbations; this paper provides a theoretical analysis and proposes the simpler multi-run averaging scheme

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical analysis of TCAV variance; the "boundary point" finding is insightful
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical predictions validated across three data modalities
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and practical recommendations are explicit
- Value: ⭐⭐⭐⭐ Makes an important contribution to the theoretical understanding of XAI method reliability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Autonomous Concept Drift Threshold Determination](autonomous_concept_drift_threshold_determination.md)
- [\[ACL 2025\] Partial Colexifications Improve Concept Embeddings](../../ACL2025/others/partial_colexifications_improve_concept_embeddings.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](../../ICML2026/others/polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)
- [\[ICML 2025\] Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability](../../ICML2025/others/synthesizing_images_on_perceptual_boundaries_of_anns_for_uncovering_and_manipula.md)

</div>

<!-- RELATED:END -->
