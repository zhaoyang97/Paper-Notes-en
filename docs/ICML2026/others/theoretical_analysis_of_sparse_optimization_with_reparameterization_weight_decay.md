---
title: >-
  [Paper Note] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate
description: >-
  [ICML 2026][Sparse Optimization] Ours proposes ReWA: it reparameterizes the variables to be optimized as $\boldsymbol{x}=\boldsymbol{y}^{K}$, applies weight decay to $\boldsymbol{y}$…
tags:
  - "ICML 2026"
  - "Sparse Optimization"
  - "$\\ell_p$ Regularization"
  - "Reparameterization"
  - "Weight Decay"
  - "Adaptive Learning Rate"
date: 2026-05-08
content_hash: 5fec66bed101dcd0
---

# Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate

**Conference**: ICML 2026  
**arXiv**: [2605.25134](https://arxiv.org/abs/2605.25134)  
**Code**: https://github.com/childofcuriosity/rewa (Available)  
**Area**: Optimization Theory / Sparse Training  
**Keywords**: Sparse Optimization, $\ell_p$ Regularization, Reparameterization, Weight Decay, Adaptive Learning Rate

## TL;DR
Ours proposes ReWA: it reparameterizes the variables to be optimized as $\boldsymbol{x}=\boldsymbol{y}^{K}$, applies weight decay to $\boldsymbol{y}$, and utilizes a coordinate-level adaptive step size $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$. This equivalently transforms the non-optimizable $\ell_p\;(0<p<1)$ sparse regularization into a trainable objective with bounded gradients that avoids trapping at zero-saddle points. The approach demonstrates enhanced sparsity relative to $\ell_1$ using ResNet on CIFAR-10 / ImageNet.

## Background & Motivation

**Background**: The gold standard for sparse training is $\ell_0$ regularization; however, it is difficult to solve due to non-continuity. In practice, $\ell_1$ (the LASSO route) is typically used as a convex relaxation, for which theories and algorithms are well-matured.

**Limitations of Prior Work**: $\ell_1$ introduces estimation bias, which sacrifices excessive accuracy for over-parameterized models like neural networks. Switching to $\ell_p\;(0<p<1)$ better approximates $\ell_0$ and provides stronger sparsity, but $\ell_p$ has unbounded gradients near zero and is non-smooth. It only works in simple scenarios like linear regression and almost inevitably leads to training instability in deep networks.

**Key Challenge**: There exists a structural trade-off between sparsity strength (smaller $p$ is closer to $\ell_0$) and optimization stability (smaller $p$ leads to more divergent gradients). Existing multiplicative reparameterization $f(\boldsymbol{y}_1\odot\cdots\odot\boldsymbol{y}_K)+\lambda/2\sum\|\boldsymbol{y}_i\|_2^2$ (denoted as [Cp], corresponding to $p=2/K$) keeps gradients bounded, but $\boldsymbol{y}^{K-1}$ forms high-order saddle points near zero; once a coordinate passes through zero, it cannot escape.

**Goal**: Construct an algorithm that (i) corresponds to some $\ell_p\;(0<p<1)$ at the implicit regularization level; (ii) has bounded gradients everywhere; (iii) can escape zero-saddle points; and (iv) is stably applicable to real datasets (CIFAR-10 / ImageNet).

**Key Insight**: Tie the $K$ symmetric variables into a single $\boldsymbol{y}$, and introduce an additional coordinate adaptive step size regulated by two hyperparameters $M$ and $\epsilon$. This embeds the ability to "escape zero-saddle points" into the algorithm itself rather than relying on initialization.

**Core Idea**: Use the "Reparameterization + Weight Decay + Adaptive Learning Rate" trio (ReWA) to implicitly encode $\ell_p$ regularization into SGD updates, canceling out the zero-saddle points caused by $\boldsymbol{y}^{K-1}$ through the adaptive step size.

## Method

### Overall Architecture
ReWA performs power reparameterization $\boldsymbol{x}=\boldsymbol{y}^{K}$ (where $K$ is odd, element-wise) on the parameters during the forward pass. The network loss $f$ still takes $\boldsymbol{x}$ as input, but backpropagation only updates the latent variable $\boldsymbol{y}$. The iteration form for each step is $\boldsymbol{y}(t+1)=(1-\lambda\eta_t)\boldsymbol{y}(t)-\eta_t\frac{\boldsymbol{y}^{M}(t)}{\boldsymbol{y}^{K-1}(t)+\epsilon\mathbf{1}}\odot\boldsymbol{y}^{K-1}(t)\odot\nabla f(\boldsymbol{y}^{K}(t))$. Here, $\lambda$ is the weight decay coefficient, $\eta_t$ is the base learning rate, and $M\in[0,K-1)$ together with $\epsilon\ge 0$ determines the implicit regularization. After training, $\boldsymbol{x}(T)=\boldsymbol{y}^{K}(T)$ is taken as the final (sparse) solution. The algorithm can be layered with SGD or AdamW as the base optimizer; if the base is AdamW (which already has coordinate adaptivity), setting $M=0$ is recommended.

### Key Designs

1.  **Power Reparameterization $\boldsymbol{x}=\boldsymbol{y}^{K}$**:
    - **Function**: This equivalently rewrites non-smooth regularization like $\ell_p\;(p=2/K)$, which has unbounded gradients near zero, into $\ell_2$ weight decay on $\boldsymbol{y}$ plus a standard smooth loss.
    - **Mechanism**: Lemma 3.1 proves a one-to-one correspondence between the global optima, local optima, and (sub)stable points of [Cp] and $\ell_p$ regularization. Theorem 3.7 further proves that if "gradient clipping" is applied directly to $\ell_p$, the gradient upper bound and approximation error cannot be simultaneously small (events $\mathcal{E}_1\le\sqrt{d}$ and $\mathcal{E}_2\le d/(2e)$ cannot both hold), strictly excluding the shortcut of "clipping + original $\ell_p$".
    - **Design Motivation**: Inherit the low bias and $\ell_0$ approximation advantages of $\ell_p$ regularization, while reducing the optimization difficulty to "smooth loss + ordinary weight decay"—a prerequisite for the following two components.

2.  **Adaptive Learning Rate $\eta_t\,\boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon\mathbf{1})$**:
    - **Function**: Counteracts the zero-saddle point caused by $\boldsymbol{y}^{K-1}$ in the update, allowing the algorithm to pass through zero even when the sign of $\boldsymbol{y}(0)$ differs from the ground truth.
    - **Mechanism**: Example 3.2 uses a 1D toy problem $f(x)=(x-1)^2$ with $y(0)=-1$ as a counterexample—the non-adaptive version satisfies $|y(T)-1|\ge 1$, never escaping zero; the adaptive version (when $M=0,\epsilon\to 0$, it degenerates to $\boldsymbol{y}(t)-\eta\nabla f(\boldsymbol{y}^K(t))$) satisfies $|y(T)-1|\le 2(1-\tfrac{2\eta}{K-1})^T$, converging linearly. The numerator $\boldsymbol{y}^{M}$ controls sparsity strength (larger $M$ provides stronger suppression of small coordinates), and the denominator $\boldsymbol{y}^{K-1}+\epsilon$ cancels $\boldsymbol{y}^{K-1}$ when $\boldsymbol{y}$ is large and is stabilized by $\epsilon$ when $\boldsymbol{y}$ is small (similar to Adam's stability constant).
    - **Design Motivation**: Theorem 3.3 proves that the implicit regularization of ReWA is $R(\boldsymbol{x})=\tfrac{K}{1-M+K}\|\boldsymbol{x}\|_{1+(1-M)/K}^{1+(1-M)/K}+\epsilon\tfrac{K}{2-M}\|\boldsymbol{x}\|_{(2-M)/K}^{(2-M)/K}$. Proposition 3.4 provides a practical recipe: use Config A ($\epsilon=0,M>1$) for simple data and Config B ($\epsilon>0,M<2$) for complex data, both making the primary exponent $p=1+(1-M)/K\in(0,1)$.

3.  **Explicit Weight Decay $(1-\lambda\eta_t)\boldsymbol{y}(t)$**:
    - **Function**: Ensures sparsity of the final solution under any initialization, compensating for the failure of "implicit bias only" with large initializations.
    - **Mechanism**: Example 3.8 / Theorem 3.9 prove that under a quadratic objective $f(\boldsymbol{x})=\boldsymbol{x}^\top\Lambda\boldsymbol{x}$, if no weight decay is added and only the implicit bias of reparameterization is relied upon, an initialization exists where the solution remains frozen near the initial value, far from the sparse optimum. Adding $\ell_2$ decay ensures convergence to the origin, the sparsest global optimum.
    - **Design Motivation**: Previous works by Gunasekar, Woodworth, etc., only provided implicit sparse bias in specific scenarios like matrix factorization or small initializations. ReWA replaces the "small initialization hypothesis" with "explicit weight decay," extending theoretical guarantees from toy scenarios to general non-convex problems.

### Loss & Training
The base optimizer can be SGD or AdamW (Algorithm 2 provides the AdamW version); the learning rate supports constant or cosine decay. In practice, taking odd $K$ is most convenient ($\boldsymbol{x}=\boldsymbol{y}^K$). For even $K$, one can use $\boldsymbol{y}_1\odot\boldsymbol{y}_1-\boldsymbol{y}_2\odot\boldsymbol{y}_2$ or $\boldsymbol{x}=\mathrm{sign}(\boldsymbol{y})\cdot|\boldsymbol{y}|^K$.

## Key Experimental Results

### Main Results
Using a ResNet backbone on CIFAR-10 and ImageNet respectively, the goal is to compare sparsity rates (the lower the ratio of non-zero parameters, the better) at a fixed test accuracy. The following table summarizes the trends reported in the paper:

| Dataset | Model | Method | Sparsity Rate (Non-zero) | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| CIFAR-10 | ResNet | $\ell_1$ Regularization | Baseline | Comparable to ours |
| CIFAR-10 | ResNet | **ReWA (Config B)** | Significantly lower than $\ell_1$ | Matches $\ell_1$ |
| ImageNet | ResNet | $\ell_1$ Regularization | Baseline | Comparable to ours |
| ImageNet | ResNet | **ReWA (Config B)** | Significantly lower than $\ell_1$ | Matches $\ell_1$ |

### Ablation Study
| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Full ReWA | Stable convergence + Sparsity | All three components enabled |
| w/o Adaptive LR (Non-adaptive SGD on [Cp]) | $|y(T)-1|\ge 1$ on 1D toy; training failure on ImageNet | Validates Example 3.2 / Theorem 3.10 |
| w/o Weight Decay | Remains near initialization on quadratic objective; not sparse | Validates Example 3.8 / Theorem 3.9 |
| Direct $\ell_p$ + Grad Clipping | Gradient bound and approx error cannot be simultaneously small | Validates Theorem 3.7 |
| Changing $K, M$ (Figure 1 heatmap) | Blue region is optimizable, red is high test loss, white is illegal $M > K-1$ | Provides hyperparameter selection range |

### Key Findings
- The trio is indispensable: removing adaptive LR leads to zero-saddle point trapping, removing weight decay loses sparsity, and removing reparameterization runs back into the non-optimizability of $\ell_p$.
- Configuration A vs B: The authors explicitly suggest using $\epsilon=0$ (more aggressive $\ell_p$) for simple data and $\epsilon>0$ (using a mild $\ell_q\;(q>1)$ as a stability constant) for complex data, with $\epsilon$ serving a similar role to Adam's $\epsilon$ in preventing denominators from being too small.
- Since AdamW already includes coordinate-level adaptive step sizes, setting $M=0, \epsilon \ne 0$ when layering ReWA avoids redundant sparsity suppression.

## Highlights & Insights
- **Makes "Algorithm = Implicit Regularization" explicit**: Through carefully designed update rules, it embeds an unsolvable $\ell_p$ constraint into the trajectory of SGD in a provable manner. This "using iterative formats to realize non-convex regularization" approach can be transferred to other difficult non-convex constraints.
- **Theorem 3.7's hard impossibility result is elegant**: It demonstrates that "clipping $\ell_p$ gradients" will always trade off between stability and fidelity under dimension $d$, justifying why a reparameterization route must be taken instead of simply adding gradient clipping.
- **The distinction between Configuration A/B is of engineering value**: Mapping hyperparameter selection directly to "dataset complexity" provides a ready-to-use recipe for downstream LLM or diffusion model pruning.

## Limitations & Future Work
- Experiments were limited to ImageNet + ResNet and have not been validated at the scale of Transformers or LLMs. Current LLM pruning typically relies on structured sparsity (head/channel level), whereas ReWA provides unstructured sparsity.
- Theorem 3.3 assumes $M$ is even (ensuring update symmetry and ease of analysis); in practice, $M$ can take continuous values, but theoretical guarantees are only provided for even values. This is discussed in Appendix Remark C.3 but not fully resolved.
- Increasing $K$ worsens the numerical conditioning of multiplicative reparameterization (high powers of small values easily underflow). Maintaining precision under FP16/BF16 training is an engineering gap to be filled.
- Empirical comparisons with other non-convex methods like SCAD, MCP, or adaptive Lasso were only discussed in Appendix B rather than through end-to-end benchmarking.

## Related Work & Insights
- **vs $\ell_1$ / LASSO**: $\ell_1$ is convex and easy to optimize but biased; ReWA uses $\ell_p\;(0<p<1)$ to reduce bias, at the cost of requiring reparameterization for stable training.
- **vs PowerPropagation (Schwarz et al., 2021)**: PowerPropagation similarly uses $\boldsymbol{y}^K$ reparameterization but without weight decay, relying only on implicit bias for sparsity under small initialization. ReWA removes the "small initialization hypothesis" using explicit weight decay and adds adaptive step sizes to solve zero-saddle points.
- **vs Direct $\ell_p$ + grad clip**: This paper's Theorem 3.7 provides a hard impossibility result, directly refuting this baseline.
- **vs AdamW**: AdamW implicitly performs coordinate adaptivity via $1/\sqrt{v_t}$, which can be seen as an approximation of ReWA when $M=0$. The difference is that ReWA explicitly controls $K$ and $M$ to impose an $\ell_p$ bias.

## Rating
- Novelty: ⭐⭐⭐⭐ Consolidation of existing [Cp] reparameterization into a unified framework of "adaptive step size + explicit decay," filling the theoretical gap of zero-saddle point escape.
- Experimental Thoroughness: ⭐⭐⭐ CIFAR-10 / ImageNet + ResNet are sufficient to validate the claims, but LLMs and Transformers are missing.
- Writing Quality: ⭐⭐⭐⭐ Uses 1D toy examples to string together all theoretical results; Theorem 3.7's impossibility proof is concise and powerful.
- Value: ⭐⭐⭐⭐ Provides a clean route for "engineering non-convex sparse regularization," which is valuable for the pruning and compressed sensing communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)
- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ICML 2026\] DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](disjunctivenet_neural_symbolic_learning_via_differentiable_convexified_optimizat.md)

</div>

<!-- RELATED:END -->
