---
title: >-
  [Paper Note] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate
description: >-
  [ICML 2026][Others][Paper Note] This paper proposes ReWA: it reparameterizes the optimization variable as $\boldsymbol{x}=\boldsymbol{y}^{K}$, applies weight decay to $\boldsymbol{y}$, and utilizes a coordinate-level adaptive step size $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$. This converts non-optimizable $\ell_p\;(0<p<1)$ sparse
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: 990cd98bef3754f3
---
# Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate

**Conference**: ICML 2026  
**arXiv**: [2605.25134](https://arxiv.org/abs/2605.25134)  
**Code**: https://github.com/childofcuriosity/rewa (Yes)  
**Area**: Optimization Theory / Sparse Training  
**Keywords**: Sparse optimization, $\ell_p$ regularization, reparameterization, weight decay, adaptive learning rate  

## TL;DR
This paper proposes ReWA: it reparameterizes the optimization variable as $\boldsymbol{x}=\boldsymbol{y}^{K}$, applies weight decay to $\boldsymbol{y}$, and utilizes a coordinate-level adaptive step size $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$. This converts non-optimizable $\ell_p\;(0<p<1)$ sparse regularization into a trainable objective with bounded gradients that avoids zero-saddle points. The method's sparsity improvement over $\ell_1$ is validated using ResNet on CIFAR-10 / ImageNet.

## Background & Motivation

**Background**: The gold standard for sparse training is $\ell_0$ regularization, which is difficult to solve due to discontinuity. In practice, $\ell_1$ (the LASSO approach) is typically used as a convex relaxation, benefiting from mature theory and algorithms.

**Limitations of Prior Work**: $\ell_1$ introduces estimation bias, sacrificing excessive accuracy in over-parameterized models like neural networks. Switching to $\ell_p\;(0<p<1)$ provides stronger sparsity closer to $\ell_0$, but its gradient is unbounded and non-smooth near zero. This usually only works in simple scenarios like linear regression and almost inevitably leads to training instability in deep networks.

**Key Challenge**: A structural trade-off exists between sparsity strength (smaller $p$ is closer to $\ell_0$) and optimization stability (smaller $p$ causes gradient divergence). Existing multiplicative reparameterization $f(\boldsymbol{y}_1\odot\cdots\odot\boldsymbol{y}_K)+\lambda/2\sum\|\boldsymbol{y}_i\|_2^2$ (denoted as [Cp], corresponding to $p=2/K$) ensures bounded gradients but creates high-order saddle points at zero; once a coordinate reaches zero, it cannot escape.

**Goal**: Construct an algorithm that (i) corresponds to an implicit $\ell_p\;(0<p<1)$ regularizer; (ii) has bounded gradients everywhere; (iii) can escape zero-saddle points; and (iv) is stable for real-world datasets (CIFAR-10 / ImageNet).

**Key Insight**: Tie the symmetric $K$ variables of [Cp] into a single $\boldsymbol{y}$ and introduce an additional coordinate-adaptive step size adjusted by two hyperparameters $M$ and $\epsilon$. This makes the "escape from zero-saddle points" an inherent capability of the algorithm rather than relying on initialization.

**Core Idea**: Use a triad of "reparameterization + weight decay + adaptive learning rate" (ReWA) to implicitly encode difficult-to-optimize $\ell_p$ regularization into SGD updates and counteract zero-saddle points through the adaptive step size.

## Method

### Overall Architecture
ReWA performs power reparameterization $\boldsymbol{x}=\boldsymbol{y}^{K}$ ($K$ is odd, element-wise) on parameters during the forward pass. The network loss $f$ takes $\boldsymbol{x}$ as input, but backpropagation only updates the latent variable $\boldsymbol{y}$. Each iteration follows: $\boldsymbol{y}(t+1)=(1-\lambda\eta_t)\boldsymbol{y}(t)-\eta_t\frac{\boldsymbol{y}^{M}(t)}{\boldsymbol{y}^{K-1}(t)+\epsilon\mathbf{1}}\odot\boldsymbol{y}^{K-1}(t)\odot\nabla f(\boldsymbol{y}^{K}(t))$. Here, $\lambda$ is the weight decay coefficient, $\eta_t$ is the base learning rate, and $M\in[0,K-1)$ combined with $\epsilon\ge 0$ determines the implicit regularization. After training, parity $\boldsymbol{x}(T)=\boldsymbol{y}^{K}(T)$ is used as the final (sparse) solution. The algorithm can be layered with SGD or AdamW as the base optimizer; if AdamW (which provides coordinate adaptation) is used, setting $M=0$ is recommended.

### Key Designs

**1. Power Reparameterization $\boldsymbol{x}=\boldsymbol{y}^{K}$: Reformulating Non-smooth $\ell_p$ as Smooth Loss with $\ell_2$ Decay**

$\ell_p\;(0<p<1)$ offers strong sparsity and low bias but is non-smooth with unbounded gradients near zero. ReWA's first step is reparameterizing the variable as $\boldsymbol{x}=\boldsymbol{y}^{K}$ ($K$ is odd, element-wise). Lemma 3.1 proves that this multiplicative reparameterization [Cp] has a one-to-one correspondence with $\ell_p\;(p=2/K)$ regularization regarding global optima, local optima, and (sub)stable points. Thus, the sparsity benefits are inherited while the optimization difficulty is reduced to "smooth loss + standard weight decay." Crucially, Theorem 3.7 presents a hard impossibility: if gradient clipping is applied directly to $\ell_p$, the gradient upper bound and approximation error cannot be simultaneously small (events $\mathcal{E}_1\le\sqrt{d}$ and $\mathcal{E}_2\le d/(2e)$ cannot coexist). This precludes the shortcut of "clipping + original $\ell_p$," justifying the necessity of reparameterization.

**2. Adaptive Learning Rate $\eta_t\,\boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon\mathbf{1})$: Neutralizing Zero-Saddle Points from $\boldsymbol{y}^{K-1}$**

Reparameterization introduces a new issue: the $\boldsymbol{y}^{K-1}$ term in the update creates high-order saddle points at zero. If a coordinate's sign differs from the truth, it cannot pass through zero. ReWA solves this by multiplying the step size by a coordinate-level adaptive factor. Example 3.2 uses a 1D toy problem $f(x)=(x-1)^2, y(0)=-1$ to demonstrate: the non-adaptive version satisfies $|y(T)-1|\ge 1$, never escaping zero; the adaptive version (which simplifies to $\boldsymbol{y}(t)-\eta\nabla f(\boldsymbol{y}^K(t))$ when $M=0,\epsilon\to 0$) satisfies $|y(T)-1|\le 2(1-\tfrac{2\eta}{K-1})^T$, achieving linear convergence. The numerator $\boldsymbol{y}^{M}$ controls sparsity intensity (larger $M$ suppresses small coordinates more), while the denominator $\boldsymbol{y}^{K-1}+\epsilon$ cancels $\boldsymbol{y}^{K-1}$ for large $\boldsymbol{y}$ and is stabilized by $\epsilon$ for small $\boldsymbol{y}$ (similar to the stability constant in Adam). Theorem 3.3 calculates that ReWA's implicit regularization is:

$$R(\boldsymbol{x})=\tfrac{K}{1-M+K}\|\boldsymbol{x}\|_{1+(1-M)/K}^{1+(1-M)/K}+\epsilon\tfrac{K}{2-M}\|\boldsymbol{x}\|_{(2-M)/K}^{(2-M)/K},$$

Proposition 3.4 provides practical recipes: Config A ($\epsilon=0,M>1$) for simple data, and Config B ($\epsilon>0,M<2$) for complex data, both ensuring the primary exponent $p=1+(1-M)/K\in(0,1)$ falls within the true $\ell_p$ range.

**3. Explicit Weight Decay $(1-\lambda\eta_t)\boldsymbol{y}(t)$: Replacing "Small Initialization Assumptions" with Provable Sparsity Guarantees**

Implicit bias from reparameterization alone fails with large initializations—work like PowerPropagation only produces sparsity in specific scenarios like small initialization or matrix factorization. ReWA explicitly adds $\ell_2$ decay $(1-\lambda\eta_t)\boldsymbol{y}(t)$ to the update. Example 3.8 / Theorem 3.9 proves that for a quadratic objective $f(\boldsymbol{x})=\boldsymbol{x}^\top\Lambda\boldsymbol{x}$, a solution without decay can be frozen near its initial value; adding $\ell_2$ decay guarantees convergence to the origin—the sparsest global optimum. This transforms the implicit sparsity bias (which relies on "small initialization" in studies by Gunasekar, Woodworth, etc.) into an explicit mechanism valid for any initialization and general non-convex problems. The components are interdependent: removing reparameterization reverts to non-optimizable $\ell_p$, removing adaptive learning rate leads to zero-saddle entrapment, and removing weight decay loses the sparsity guarantee.

### Loss & Training
The base optimizer can be SGD or AdamW (Algorithm 2 provides the AdamW version); learning rates support constant or cosine decay. Practically, an odd $K$ is most convenient ($\boldsymbol{x}=\boldsymbol{y}^K$). For even $K$, $\boldsymbol{y}_1\odot\boldsymbol{y}_1-\boldsymbol{y}_2\odot\boldsymbol{y}_2$ or $\boldsymbol{x}=\mathrm{sign}(\boldsymbol{y})\cdot|\boldsymbol{y}|^K$ is used.

## Key Experimental Results

### Main Results
Using ResNet backbones on CIFAR-10 / ImageNet, the goal is to compare sparsity rates (lower percentage of non-zero parameters is better) at a fixed test accuracy. The following table summarizes the reported trends:

| Dataset | Model | Method | Sparsity Rate (Non-zero) | Test Accuracy |
|---------|-------|--------|-------------------------|---------------|
| CIFAR-10| ResNet| $\ell_1$ Regularization | Baseline | Comparable to Ours |
| CIFAR-10| ResNet| **Ours (ReWA Config B)** | Significantly lower than $\ell_1$ | Comparable to $\ell_1$ |
| ImageNet| ResNet| $\ell_1$ Regularization | Baseline | Comparable to Ours |
| ImageNet| ResNet| **Ours (ReWA Config B)** | Significantly lower than $\ell_1$ | Comparable to $\ell_1$ |

### Ablation Study

| Configuration | Observation | Explanation |
|---------------|-------------|-------------|
| Full ReWA | Stable convergence + sparsity | All three components active |
| w/o Adaptive LR (Non-adaptive SGD on [Cp]) | $|y(T)-1|\ge 1$ on 1D toy; training fails on ImageNet | Validates Example 3.2 / Theorem 3.10 |
| w/o Weight Decay | Remains near initialization on quadratic objective; not sparse | Validates Example 3.8 / Theorem 3.9 |
| Direct $\ell_p$ + Grad Clip | Gradient bound and approximation error cannot be small simultaneously | Validates Theorem 3.7 |
| Varying $K,M$ (Fig 1 Heatmap) | Blue region is optimizable; red is high test loss; white is invalid $M>K-1$ | Provides hyperparameter selection range |

### Key Findings
- The three components are indispensable: removing adaptive learning rate causes zero-saddle entrapment, removing weight decay loses sparsity, and removing reparameterization makes $\ell_p$ non-optimizable.
- Configuration A vs B: The authors recommend $\epsilon=0$ (aggressive $\ell_p$) for simple data and $\epsilon>0$ (mild $\ell_q\;(q>1)$ as a stability constant) for complex data, where $\epsilon$ acts similarly to the stability constant in Adam.
- Since AdamW already includes coordinate-adaptive step sizes, setting $M=0,\epsilon\ne 0$ avoids redundant sparsity suppression when combining with ReWA.

## Highlights & Insights
- **Explicit implementation of "Algorithm = Implicit Regularization"**: Through carefully designed update rules, an unsolvable $\ell_p$ constraint is provably embedded into the SGD trajectory. This approach of implementing non-convex regularization via iterative formats can be transferred to other difficult non-convex constraints.
- **Elegant result in Theorem 3.7**: It shows that the intuitive route of "clipping $\ell_p$ gradients" always faces a binary choice between stability and fidelity in high dimensions ($d$), directly justifying the reparameterization route over simple gradient clipping.
- **Engineering value of Config A/B**: By linking hyperparameter selection directly to "dataset complexity," it provides a "recipe" ready for use in LLM or diffusion model pruning.

## Limitations & Future Work
- Experiments are limited to ImageNet + ResNet and have not been validated at the scale of Transformers or LLMs. Current LLM pruning typically relies on structured sparsity (head/channel level), whereas ReWA provides unstructured sparsity.
- Theorem 3.3 assumes $M$ is an even integer (guaranteeing symmetric updates for analysis); in practice, $M$ can be continuous, but theoretical guarantees are only provided for even values (discussed in Appendix Remark C.3).
- Increasing $K$ worsens the numerical conditioning of multiplicative reparameterization (high powers of small values underflow easily). Maintaining precision in FP16 / BF16 training is an engineering gap to be addressed.
- Empirical comparisons with other non-convex methods like SCAD, MCP, or adaptive Lasso are only discussed in Appendix B and lack head-to-head benchmarking.

## Related Work & Insights
- **vs $\ell_1$ / LASSO**: $\ell_1$ is convex and easy to optimize but biased; ReWA uses $\ell_p\;(0<p<1)$ to reduce bias at the cost of requiring reparameterization for stable training.
- **vs PowerPropagation (Schwarz et al., 2021)**: PowerPropagation also uses $\boldsymbol{y}^K$ reparameterization but without weight decay, relying solely on implicit bias from small initializations. ReWA removes the "small initialization" dependency via explicit weight decay and solves zero-saddle points with adaptive step sizes.
- **vs Direct $\ell_p$ + grad clip**: Theorem 3.7 provides a hard impossibility result, effectively negating this baseline.
- **vs AdamW**: AdamW performs implicit coordinate adaptation via $1/\sqrt{v_t}$, which can be viewed as an approximation of ReWA at $M=0$. The difference lies in ReWA's explicit control of $K$ and $M$ to enforce an $\ell_p$ bias.

## Rating
- Novelty: ⭐⭐⭐⭐ Links existing [Cp] reparameterization into a unified framework of "adaptive step size + explicit decay" and fills the theoretical gap for zero-saddle escape.
- Experimental Thoroughness: ⭐⭐⭐ CIFAR-10 / ImageNet + ResNet are sufficient to validate the claims, but LLM and Transformer benchmarks are missing.
- Writing Quality: ⭐⭐⭐⭐ Connects theoretical results using 1D toy examples; the impossibility proof in Theorem 3.7 is concise and powerful.
- Value: ⭐⭐⭐⭐ Provides a clean path for engineering non-convex sparse regularization, valuable for the pruning and compressed sensing communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)
- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](../../ICML2025/others/sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)
- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[ICML 2026\] DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](disjunctivenet_neural_symbolic_learning_via_differentiable_convexified_optimizat.md)

</div>

<!-- RELATED:END -->
