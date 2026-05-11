---
title: >-
  [Paper Note] SHAP Values via Sparse Fourier Representation
description: >-
  [NeurIPS 2025][Interpretability][SHAP values] This paper proposes FourierShap, an algorithm that first approximates a black-box predictor as a sparse Fourier representation and then leverages closed-form SHAP value formu…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "SHAP values"
  - "Fourier representation"
  - "sparse Walsh-Hadamard transform"
  - "feature attribution"
  - "accelerated computation"
date: 2026-05-08
content_hash: 7cf2e4fe2c82cbda
---

# SHAP Values via Sparse Fourier Representation

**Conference**: NeurIPS 2025
**arXiv**: [2410.06300](https://arxiv.org/abs/2410.06300)
**Code**: Not available
**Area**: Explainable AI / Shapley Value Computation
**Keywords**: SHAP values, Fourier representation, sparse Walsh-Hadamard transform, feature attribution, accelerated computation

## TL;DR

This paper proposes FourierShap, an algorithm that first approximates a black-box predictor as a sparse Fourier representation and then leverages closed-form SHAP value formulas for Fourier basis functions to efficiently compute feature attributions, achieving 10–10,000× speedups over KernelShap while supporting a tunable accuracy–efficiency trade-off.

## Background & Motivation

SHAP (SHapley Additive exPlanations) is the most widely used local feature attribution method in explainable AI. Its core is the Shapley value formula:

$$\phi_i(v) = \frac{1}{n} \sum_{S \subseteq [n] \setminus \{i\}} \frac{v(S \cup \{i\}) - v(S)}{\binom{n-1}{|S|}}$$

This formula involves an exponential summation over all possible subsets ($2^n$ terms), making computation extremely costly. In industrial settings (e.g., ad targeting, subscription propensity models), explanations must be generated for millions of samples, making SHAP computation a severe bottleneck.

Limitations of existing methods:
- **KernelShap / LinRegShap**: Approximate the exponential sum via random subset sampling; each sample to be explained requires solving a costly least-squares optimization problem.
- **TreeShap**: Restricted to white-box settings with tree-based models.
- **FastShap**: Trains an MLP to directly output SHAP values, but accuracy depends on the MLP's function approximation capacity, and switching background datasets requires retraining.

A key insight is that many practical models exhibit **spectral bias**—neural networks tend to learn low-frequency functions, and decision tree ensembles are themselves sparse Fourier functions. This suggests that these models can be approximated with compact Fourier representations, whose structural properties can then be exploited for efficient SHAP computation.

## Method

### Overall Architecture

FourierShap is a two-stage algorithm:
1. **Stage 1**: Approximate the black-box predictor as a sparse Fourier representation (exact for tree models; approximated via sparse Walsh-Hadamard transform for black-box models).
2. **Stage 2**: Directly compute SHAP values from the Fourier representation using derived closed-form formulas, eliminating the need for exponential summation.

Stage 1 is performed only once; subsequent SHAP computation for each new sample is amortized into the efficient Stage 2 calculation—this is the core advantage of the **amortized** SHAP computation paradigm.

### Key Designs

1. **Fourier Representation and Sparsity**: For a function $h$ defined on binary inputs $\{0,1\}^n$, its Fourier expansion is $h(x) = \frac{1}{\sqrt{2^n}} \sum_{f \in \{0,1\}^n} \hat{h}(f)(-1)^{\langle f, x \rangle}$. If only $k$ Fourier coefficients are nonzero, $h$ is said to be $k$-sparse. Key observations include:

    - A decision tree ensemble of $T$ trees with depth $d$ is $O(T \cdot 4^d)$-sparse.
    - Neural networks tend to learn low-degree (hence sparse) functions due to spectral bias.
    - Functions of degree $d$ are $O(n^d)$-sparse (Proposition 2).

2. **Closed-Form SHAP Formula (Core Contribution)**: For a single Fourier basis function $\Psi_f(x) = (-1)^{\langle f, x \rangle}$, a closed-form SHAP value is derived via combinatorial argument (Lemma 1):
    $\phi_i^{\Psi_f} = -\frac{2f_i}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \mathbb{1}_{x_i \neq x_i^*} (-1)^{\langle f, x \rangle} \frac{(|A|+1) \bmod 2}{|A|+1}$
   where $A = \{j \in [n] | x_j \neq x_j^*, j \neq i, f_j = 1\}$. The key combinatorial insight is that a large number of positive and negative terms in the exponential sum cancel each other, and the count of such cancellations can be determined analytically without explicit enumeration.

3. **Overall Complexity (Theorem 2)**: Exploiting the linearity of SHAP values with respect to the function being explained, the final formula is:
    $\phi_i^h = -\frac{2}{|\mathcal{D}|} \sum_{f \in \text{supp}(h)} \hat{h}(f) \cdot f_i \cdot \sum_{(x,y) \in \mathcal{D}} \mathbb{1}_{x_i \neq x_i^*} (-1)^{\langle f, x \rangle} \frac{(|A|+1) \bmod 2}{|A|+1}$
   Computational complexity is $\Theta(n \cdot |\mathcal{D}| \cdot k)$, fully eliminating the exponential summation. Both summations can be efficiently parallelized on GPU via JAX's vmap.

### Exact Setting for Tree Models

For decision tree ensembles, the Fourier representation can be computed exactly via the recursion: $t(x) = \frac{1+(-1)^{\langle e_i, x \rangle}}{2} t_{\text{left}}(x) + \frac{1-(-1)^{\langle e_i, x \rangle}}{2} t_{\text{right}}(x)$. By linearity of the Fourier transform, the transform of an ensemble equals the average of the individual tree transforms. In this setting, FourierShap produces SHAP values identical to those of TreeShap.

## Key Experimental Results

### Main Results (Black-Box Setting)

Evaluation on four real datasets (Entacmaea $n=13$, SGEMM $n=40$, GB1 $n=80$, avGFP $n=236$), comparing SHAP value accuracy ($R^2$) and computation speed (speedup relative to KernelShap).

| Method | Type | Entacmaea Speedup | SGEMM Speedup | Accuracy Control |
|--------|------|-------------------|---------------|-----------------|
| KernelShap | Black-box | 1× (baseline) | 1× (baseline) | Requires convergence check |
| LinRegShap | Black-box | ~10× | ~10× | Requires convergence check |
| DeepLift | White-box (NN) | ~100× | ~10× | Fixed |
| FastShap | Black-box | ~1000× | ~100× | Unreliable |
| **FourierShap** | **Black-box** | **~10000×** | **~1000×** | **Finely tunable ($k$)** |

FourierShap is 10–10,000× faster than KernelShap and achieves higher accuracy than FastShap on 3 out of 4 datasets.

### Tree Model Setting

| Method | Entacmaea | SGEMM | GB1 | avGFP |
|--------|-----------|-------|-----|-------|
| TreeShap | 1× | 1× | 1× | 1× |
| FastTreeShap | ~2–5× | ~2–5× | ~2–5× | ~2–5× |
| GPU TreeShap | ~5–10× | ~5× | ~3× | ~3× |
| PLTreeShap | ~3× | ~3× | ~2× | ~2× |
| **FourierShap** | **~100×** | **~50×** | **~10–30×** | **~5–20×** |

For tree models, FourierShap also achieves order-of-magnitude speedups with results exactly matching TreeShap. Advantages are most pronounced for low-dimensional inputs and shallow trees, and diminish as depth and feature count increase.

### Key Findings

- Controlling the sparsity parameter $k$ enables **fine-grained, reliable** accuracy–speed trade-offs: larger $k$ yields more accurate function approximation at higher computational cost.
- Function approximation (Stage 1) is performed only once; subsequent SHAP computation for each new sample is orders of magnitude faster than KernelShap.
- Compared to FastShap, FourierShap offers a smoother and more controllable accuracy–speed curve, and requires no retraining when the background dataset changes.
- Experiments confirm that neural networks exhibit spectral bias—Fourier function approximation achieves $R^2 \geq 0.95$ with relatively few coefficients.

## Highlights & Insights

- The work elegantly combines spectral analysis with game-theoretic interpretability, achieving fundamental computational acceleration by eliminating the exponential sum.
- The combinatorial argument in Lemma 1 constitutes the core innovation: rather than approximating the exponential sum, the paper derives it analytically for Fourier basis functions.
- The amortized design makes FourierShap particularly well-suited for industrial settings requiring explanations for large numbers of samples.
- Surpassing DeepLift—which requires white-box model access—in the black-box setting is a noteworthy result.

## Limitations & Future Work

- The method assumes binary input features; continuous features must first be discretized into quantile bins, which may result in information loss.
- Stage 1 (sparse WHT) itself incurs overhead, and approximation quality degrades when the model is not sufficiently sparse.
- As the number of features and tree depth increase, the number of Fourier coefficients grows rapidly ($O(T \cdot 4^d)$), diminishing the advantage.
- Only Interventional SHAP values are supported; Conditional SHAP is not addressed.

## Related Work & Insights

- The spectral bias theory (Yang & Salman 2019) provides the critical theoretical foundation for this approach.
- Sparse Fourier transform algorithms (Amrollahi et al. 2019) are the key implementation tool.
- This work suggests that for structured function classes, exploiting mathematical structure (e.g., sparsity) can yield qualitative improvements in computational efficiency.
- A broader lesson: computational bottlenecks in interpretability methods can be overcome by adopting an alternative function representation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing Fourier analysis into SHAP computation is a fundamentally new direction; the closed-form formula is an important theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Both black-box and white-box settings are evaluated across four datasets with multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a core bottleneck in SHAP computation with direct applicability to industrial use cases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)

</div>

<!-- RELATED:END -->
