---
title: >-
  [Paper Note] Robustness of Mixtures of Experts to Feature Noise
description: >-
  [ICML 2026][learning_theory][Mixture-of-Experts] Under the fair "iso-parameter" setting, this paper demonstrates using a block-diagonal noisy linear regression model that MoE's sparse expert activation acts as a noise filter. This allows it to achieve lower generalization error, stronger perturbation robustness, and faster convergence compared to a dense model of equ
tags:
  - ICML 2026
  - learning_theory
  - Mixture-of-Experts
date: 2026-05-08
content_hash: a3e8c5a023cef2ac
---
# Robustness of Mixtures of Experts to Feature Noise

**Conference**: ICML 2026  
**arXiv**: [2601.14792](https://arxiv.org/abs/2601.14792)  
**Code**: To be confirmed  
**Area**: Learning Theory / MoE / Robustness  
**Keywords**: Mixture-of-Experts, Feature Noise, Sparse Activation, Generalization Error, Iso-parameter Comparison

## TL;DR
Under the fair "iso-parameter" setting, this paper demonstrates using a block-diagonal noisy linear regression model that MoE's sparse expert activation acts as a noise filter. This allows it to achieve lower generalization error, stronger perturbation robustness, and faster convergence compared to a dense model of equal size under feature noise.

## Background & Motivation

**Background**: MoE (e.g., Mixtral 8×7B) has demonstrated that "sparse activation + large total parameters" can match or even surpass dense models while maintaining much lower inference costs (Mixtral has 47B total parameters but only 13B active, yet rivals Llama-2-70B). This challenges the traditional Scaling Law intuition that "performance only follows parameter scale."

**Limitations of Prior Work**: Existing MoE theories (e.g., Chen et al. 2022) mostly explain MoE's "superior expressive power," but they often assume **each expert is as large as the dense model**, making the total parameters of MoE far exceed the dense counterpart—effectively giving MoE a hidden capacity advantage. This explains why "more parameters make it stronger" but fails to isolate the benefits of the **MoE architecture itself**, nor does it explain why MoE performs better in sample efficiency and robustness.

**Key Challenge**: To answer why MoE wins due to architecture rather than parameters, one must enforce an "iso-parameter" setting (equal total parameters) and examine what sparse activation brings to the table. This perspective has rarely been directly addressed.

**Goal**: Find a mechanism unique to MoE (absent in dense models) under a strict iso-parameter comparison and quantify its impact across generalization error, perturbation robustness, convergence speed, and sample complexity.

**Key Insight**: The authors observe that ReLU-like activations in modern LLMs commonly exhibit a "block-diagonal modular structure masked by noise"—when low-magnitude activations are zeroed using pruning like TEAL, the input activations of Llama-2-7B's layer 0 MLP reveal clear block patterns. This suggests: Actual activation = Modular signal + Feature noise.

**Core Idea**: Abstract the "MoE vs. Dense" comparison as "block-wise solving vs. global solving" for a noisy linear regression with a block-diagonal design matrix. MoE's sparse activation allows each expert to estimate only on its respective feature block, thereby **filtering out noise interference from other irrelevant feature blocks**; the dense model is forced to estimate across all dimensions, allowing noise from irrelevant dimensions to interfere throughout.

## Method

This is a purely theoretical paper that does not propose a new model but constructs an analytically tractable simplified model to prove how "sparse activation brings robustness." The argument follows the logic below.

### Overall Architecture

The authors construct a **block-diagonal linear regression** as a unified abstraction for MoE and dense models. The ground truth parameters $\beta^\star=[\beta_1^{\star T},\dots,\beta_k^{\star T}]^T\in\mathbb{R}^d$ are divided into $k$ blocks, corresponding to $k$ "experts"; the design matrix $X$ is block-diagonal, where the $i$-th expert only sees its feature block $X_i$, outputting $Y=X\beta^\star$. Crucially: we only observe a **noisy version** $\bar X = X + E$, where $E_{ij}\sim\mathcal{N}(0,\sigma^2)$ is feature noise, which can be interpreted as either input perturbation or internal activation interference in a dense network that has not yet revealed its modular structure through pruning.

Two estimators are compared in this framework: a **dense estimator** using the entire noisy matrix $\bar X$ for minimum-norm least squares $\hat\beta=(\bar X^T\bar X)^+\bar X^T Y$ (one model learns all experts' tasks simultaneously); and a **sparse (MoE-like) estimator** assuming near-perfect routing, where each expert independently estimates $\hat\beta_i=(\bar X_i^T\bar X_i)^+\bar X_i^T Y_i$ using its own $\bar X_i$ and $Y_i$. Both have the same total parameters; the only difference is "global solving vs. block-wise solving." Since analyzing minimum-norm estimators under noise is difficult, the authors first analyze their **Bayesian optimal counterparts** (the performance upper bound under infinite samples) and then separately prove convergence rates to bridge back to finite samples.

### Key Designs

**1. Iso-parameter Comparison + Block-diagonal Noisy Model: Isolating "Architectural Advantage" from "Parameter Advantage"**

This is the methodological foundation of the paper and its primary differentiator from previous MoE theories. While older theories let each expert be as large as the dense network, making the MoE victory trivial due to more parameters, **Ours** strictly aligns the total parameters of MoE with the dense model (e.g., MoE uses 4 routing experts + 1 shared expert with 1024 hidden dims, while the dense FFN uses $5\times1024=5120$ hidden dims). Under this setting, any gain must be attributed to the "block structure + sparse activation." The noisy observation $\bar X = X+E$ formalizes the empirical fact of internal activation noise—without noise, a dense model could theoretically learn the block structure perfectly, becoming equivalent to MoE. **It is the noise that creates the gap.**

**2. Sparse Activation = Noise Filter: Proving Lower Generalization Error under Iso-params**

This is the core conclusion. For Bayesian optimal estimators (Eq. 3), the generalization errors for dense and sparse models (Theorem 4.2) are respectively:

$$\mathcal{R}(\beta^{Bayes}_{Sparse})=\sum_{i=1}^k p_i\sigma^2\,\beta_i^{\star T}\Sigma_i(\Sigma_i+\sigma^2 I)^{-1}\beta_i^{\star},$$

$$\mathcal{R}(\beta^{Bayes}_{Dense})=\sum_{i=1}^k p_i\sigma^2\,\beta_i^{\star T}\Sigma_i(p_i\Sigma_i+\sigma^2 I)^{-1}\beta_i^{\star}.$$

The only difference lies in whether the signal term $\Sigma_i$ in the parentheses is multiplied by the routing probability $p_i\le 1$. Since $0<p_i\le 1$ and $\Sigma_i$ is positive semi-definite, $p_i\Sigma_i\preceq\Sigma_i$, thus $p_i\Sigma_i+\sigma^2 I\preceq\Sigma_i+\sigma^2 I$. For positive definite matrices $A\preceq B$, $B^{-1}\preceq A^{-1}$ holds, so $(\Sigma_i+\sigma^2 I)^{-1}\preceq(p_i\Sigma_i+\sigma^2 I)^{-1}$. Term-by-term comparison yields $\mathcal{R}(\beta^{Bayes}_{Sparse})\le\mathcal{R}(\beta^{Bayes}_{Dense})$. Intuition: The dense model dilutes the signal term $\Sigma_i$ by probability $p_i$, effectively fighting noise at a lower Signal-to-Noise Ratio (SNR); the sparse expert estimates only on its block where the signal is not diluted, equivalent to filtering out noise from irrelevant blocks. This is the precise mathematical expression of "Sparse activation = Noise filter."

**3. Routing as Clustering + Dichotomy of Perturbation Robustness: Grounding Conclusions on Realizability**

One might question if "near-perfect routing" is too ideal. Theorem 4.1 addresses this—under the modular (MoEfication) structure, training a router is no longer a difficult joint optimization problem but reduces to a well-defined supervised classification task, as clustering provides the "which expert to enter" labels. A simple QDA router requires only $n\ge\mathcal{O}(\text{poly}(d,\log(1/\delta)))$ samples to reduce excess risk below $\epsilon$, as the data is already geometrically well-separated. Following Liao & Kyrillidis (2026), experts converge faster than the router during joint training and "guide" the router to align, so the architectural advantage of experts still holds.

Based on this, the authors split perturbation robustness into two cases (Theorem 4.3 / 4.4). **When routing is correct** (perturbation variance $\sigma_o^2$), as long as the feature SNR for each expert is high enough ($\lambda_{\min}(\Sigma_i)>4\sigma^2$) and $\sigma_o^2>\sigma^2$, the sparse estimator's error remains lower than the dense model. **When perturbation causes misrouting** (Theorem 4.4, where input belonging to expert $i$ is dragged to expert $j$ by $\eta x_j$), the conclusion reverses: highly specialized experts can perform worse than dense models if misrouted. the authors honestly highlight this as a trade-off.

**4. Faster Convergence + Optimal Sample Complexity: Completing the Advantage from Optimization and Finite Samples**

Beyond static generalization error, the authors prove that the sparse estimator converges faster under gradient descent (Theorem 4.7). Under the assumptions of fixed $X_i$, dimension-to-sample ratio $c=d/n>1$, and singular values $\lambda_{ij}>\sqrt{c}\sigma^2$, they provide closed-form error reduction factors $\rho_{Sparse,i}$ and $\rho_{Dense}$, proving $\rho_{Sparse,i}\le\rho_{Dense}$ (at most one sparse expert is as slow as the dense model; others are faster). The intuition is breaking a large problem into better-conditioned sub-problems. Regarding sample complexity, the authors hypothesize (supported by experiments) that the sparse estimator achieves lower excess risk at the same sample size. Interestingly, synthetic data shows both risks decay at approximately $O(n^{-2})$—this is why a full proof is difficult (they cannot be distinguished by convergence order)—but the constant factor for the sparse estimator is much smaller. From a bias-variance perspective: sparse experts are affected by noise only in an $s$-dimensional subspace ($s<d$), whereas dense models are affected across all $d$ dimensions.

### Loss & Training
This paper does not train a new model; verification experiments for both MoE and dense counterparts use standard pre-training (MiniMind architecture, ViT-L vs. V-MoE) or linear probe fitting, without new training objectives.

## Key Experimental Results

The experiments aim to verify theoretical assumptions on real data rather than chasing SOTAs: ① Real LLM activations indeed have block structures; ② MoE probes are more robust to feature noise; ③ The noise filtering advantage persists even if block structures are only approximate; ④ Advantages extend to end-to-end large models.

### Main Results

The table below shows the **performance drop percentage** (lower is better) under high-intensity Gaussian noise ($\sigma=2.0$) on T5-small activations. Including Ridge / Elastic Net as baselines is crucial—it distinguishes whether robustness comes from "structural sparsity" or general "regularization."

| Dataset | Lasso | Ridge | Elastic Net | MoE |
|---------|-------|-------|-------------|-----|
| SST-2 | 10.78 | 12.27 | 10.55 | **8.60** |
| CoLA | 10.25 | 12.39 | 12.19 | **7.67** |
| MNLI | 11.61 | 10.45 | 9.31 | **7.98** |
| AG News | 5.19 | 5.98 | **3.92** | 7.75 |

MoE has the smallest drop in most tasks (especially high noise), proving that "structured sparsity" rather than "regularization" drives robustness; on AG News, dense regularized methods were superior, showing the advantage is not unconditional.

### Ablation Study

Sensitivity analysis of the "approximate block structure" assumption: gradually adding cross-block overlap $\alpha$ (interference signal of magnitude $\alpha$ across all dimensions) to merge local structures into a dense structure, using an oracle router to isolate structural effects.

| Cross-block overlap $\alpha$ | Dense Acc. | Sparse Acc. | Gap (Sparse−Dense) |
|-------|-----------|-----------|-------|
| 0.00 | 0.6059 | 0.7025 | +0.0966 |
| 0.10 | 0.6054 | 0.7036 | +0.0982 |
| 0.30 | 0.5987 | 0.6991 | +0.1004 |
| 0.50 | 0.5921 | 0.6894 | +0.0973 |
| 1.00 | 0.5659 | 0.6571 | +0.0912 |

Even with maximum overlap $\alpha=1.0$, the sparse advantage persists at approximately +0.09 accuracy, proving that the mechanism in Theorem 4.2 degrades gracefully rather than failing abruptly under "generative mismatch"—matching the reality that LLMs exist between ideal dense and ideal sparse structures.

### Key Findings
- **The root of the advantage is structural sparsity, not regularization**: This is the biggest takeaway from including Ridge/Elastic Net counterparts.
- **Block structures don't need to be perfect**: Sparse advantages remain stable as cross-block overlap $\alpha$ goes from 0 to 1, proving the mechanism works for real (approximately modular) activations.
- **Advantage scales with noise intensity**: On ImageNet-C, Sparse V-MoE-B/16 uses only ~37% of ViT-L/16's active parameters (114M vs. 307M), yet shows larger robustness gains as Gaussian noise severity increases.
- **Misrouting is the boundary**: When perturbation is large enough to change routing, the advantage can reverse, defining a clear boundary for the mechanism.

## Highlights & Insights
- **Strict "Iso-parameter" constraint**: Using total parameter alignment cleanly isolates MoE's architectural advantages from parameter scale, a methodology highly recommended for future research.
- **Explaining everything with $p_i$**: Generalization error formulas differing only by whether the signal term is multiplied by $p_i$ compresses the "sparse activation = noise filter" concept into a clean matrix inequality.
- **Honesty about boundaries**: MoE is not portrayed as unconditionally better; the paper explicitly notes trade-offs with misrouting and cases like AG News where dense models win.
- **Transferable perspective**: Viewing MoE as "a set of routed linear probes" provides theoretical grounding for "dense-to-sparse" techniques like MoEfication and LLaMA-MoE.

## Limitations & Future Work
- **Linear + Gaussian assumptions**: Core theorems rely on linear regression, block structures, and Gaussian noise. While supplemented with non-linear two-layer MoE experiments, a gap remains with real Transformers.
- **Sample complexity is still a hypothesis**: The claim that sparse estimators are more sample-efficient is currently a hypothesis + empirical fit; since both dense and sparse models follow $O(n^{-2})$, it cannot be strictly distinguished by order. ⚠️ Follow the original text.
- **Perfect routing is a simplification**: The main conclusion relies on "near-perfect routing." Although supported by QDA realizability, the exact propagation of router error into generalization bounds is not yet closed.
- **Future directions**: Extending analysis to non-Gaussian noise, expert capacity imbalance, and explicitly incorporating routing error into the bounds.

## Related Work & Insights
- **vs. Chen et al. (2022)**: They match expert size to dense network size (resulting in more MoE total params), explaining "capacity" advantages; **Ours** fixes total params to isolate the "noise robustness" mechanism.
- **vs. Chowdhury et al. (2023)**: They also use parameter matching but are limited to less common "expert-choice" routing for patch-level CNNs; **Ours** uses standard token-level routing and covers robustness and convergence.
- **vs. Puigcerver et al. (2022)**: They prove MoE has a smaller Lipschitz constant to explain adversarial robustness; **Ours** targets feature noise (error-in-variable perspective), providing a complementary view.
- **vs. Activation Sparsity (TEAL, CATS, MoEfication)**: These are engineering methods to sparsify dense LLMs; **Ours** provides a theoretical filter-based explanation for why these methods are effective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to identify and quantify the "sparse activation = feature noise filter" mechanism under strict iso-parameter conditions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three layers of validation (synthetic/probe/end-to-end), though sample complexity relies on empirical support.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain and honest boundaries, though theorem-dense.
- Value: ⭐⭐⭐⭐⭐ Provides an interpretable theoretical foundation for MoE and dense-to-sparse transitions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[ICML 2026\] Efficiently Learning Drifting Halfspaces with Massart Noise](efficiently_learning_drifting_halfspaces_with_massart_noise.md)
- [\[ICML 2026\] Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts](geometric_and_stochastic_analysis_of_discontinuities_in_sparse_mixture-of-expert.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[ICLR 2026\] Transformers as Unsupervised Learning Algorithms: A study on Gaussian Mixtures](../../ICLR2026/learning_theory/transformers_as_unsupervised_learning_algorithms_a_study_on_gaussian_mixtures.md)

</div>

<!-- RELATED:END -->
