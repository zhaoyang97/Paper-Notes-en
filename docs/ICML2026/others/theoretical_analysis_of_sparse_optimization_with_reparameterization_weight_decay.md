---
title: >-
  [Paper Note] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate
description: >-
  [ICML 2026][Others][Paper Note] This paper proposes ReWA: by reparameterizing the target variable as $\boldsymbol{x}=\boldsymbol{y}^{K}$, applying weight decay to $\boldsymbol{y}$, and utilizing a coordinate-wise adaptive step size $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$, it equivalently transforms the non-optimizable $\ell_p\;(0<
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: 11fe5e6fc1ba067d
---
# Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate

**Conference**: ICML 2026  
**arXiv**: [2605.25134](https://arxiv.org/abs/2505.25134)  
**Code**: https://github.com/childofcuriosity/rewa (Yes)  
**Area**: Optimization Theory / Sparse Training  
**Keywords**: Sparse Optimization, $\ell_p$ Regularization, Reparameterization, Weight Decay, Adaptive Learning Rate  

## TL;DR
This paper proposes ReWA: by reparameterizing the target variable as $\boldsymbol{x}=\boldsymbol{y}^{K}$, applying weight decay to $\boldsymbol{y}$, and utilizing a coordinate-wise adaptive step size $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$, it equivalently transforms the non-optimizable $\ell_p\;(0<p<1)$ sparse regularization into a trainable objective with bounded gradients and resistance to zero-saddle points. Sparsity improvements over $\ell_1$ are validated using ResNet on CIFAR-10 / ImageNet.

## Background & Motivation

**Background**: The gold standard for sparse training is $\ell_0$ regularization, which is difficult to solve due to discontinuity. Industrially, $\ell_1$ (LASSO approach) is typically used for convex relaxation, supported by mature theory and algorithms.

**Limitations of Prior Work**: $\ell_1$ introduces estimation bias and can sacrifice excessive accuracy in over-parameterized models like neural networks. Switching to $\ell_p\;(0<p<1)$ approximates $\ell_0$ better and provides stronger sparsity, but $\ell_p$ has unbounded gradients and is non-smooth near zero. This has historically limited its application to simple scenarios like linear regression, as it almost inevitably leads to training instability in deep networks.

**Key Challenge**: A structural trade-off exists between sparsity intensity (smaller $p$ is closer to $\ell_0$) and optimization stability (smaller $p$ causes gradient divergence). Existing multiplicative reparameterization $f(\boldsymbol{y}_1\odot\cdots\odot\boldsymbol{y}_K)+\lambda/2\sum\|\boldsymbol{y}_i\|_2^2$ (denoted as [Cp], corresponding to $p=2/K$), while making gradients bounded, creates high-order saddle points at zero where coordinates cannot escape once they cross zero.

**Goal**: Construct an algorithm that (i) corresponds to an $\ell_p\;(0<p<1)$ implicit regularization; (ii) has gradients that are bounded everywhere; (iii) can escape zero-saddle points; and (iv) is stable for real-world datasets (CIFAR-10 / ImageNet).

**Key Insight**: Tie the symmetric $K$ variables of [Cp] into a single $\boldsymbol{y}$ and introduce an additional coordinate-adaptive step size adjusted by hyperparameters $M$ and $\epsilon$. This makes the ability to "escape zero-saddle points" an inherent capacity of the algorithm rather than relying on initialization.

**Core Idea**: Use a tri-part mechanism of "Reparameterization + Weight Decay + Adaptive Learning Rate" (ReWA) to implicitly encode difficult $\ell_p$ regularization into SGD updates and counteract zero-saddle points caused by $\boldsymbol{y}^{K-1}$ through adaptive step sizes.

## Method

### Overall Architecture
ReWA applies a power reparameterization $\boldsymbol{x}=\boldsymbol{y}^{K}$ (where $K$ is odd, element-wise) in the forward pass. The network loss $f$ takes $\boldsymbol{x}$ as input, but only the latent variable $\boldsymbol{y}$ is updated in the backward pass. The iteration format is $\boldsymbol{y}(t+1)=(1-\lambda\eta_t)\boldsymbol{y}(t)-\eta_t\frac{\boldsymbol{y}^{M}(t)}{\boldsymbol{y}^{K-1}(t)+\epsilon\mathbf{1}}\odot\boldsymbol{y}^{K-1}(t)\odot\nabla f(\boldsymbol{y}^{K}(t))$. Here, $\lambda$ is the weight decay coefficient, $\eta_t$ is the base learning rate, and $M\in[0,K-1)$ combined with $\epsilon \ge 0$ determines the implicit regularization. After training, $\boldsymbol{x}(T)=\boldsymbol{y}^{K}(T)$ is taken as the final sparse solution. The algorithm can be layered on base optimizers like SGD or AdamW; when using AdamW (which has built-in coordinate adaptation), $M=0$ is recommended.

### Key Designs

**1. Power Reparameterization $\boldsymbol{x}=\boldsymbol{y}^{K}$: Reformulating Non-smooth $\ell_p$ as Smooth Loss with $\ell_2$ Decay**

While $\ell_p\;(0<p<1)$ offers strong sparsity and low bias, its unbounded gradients near zero make deep network training unstable. ReWA reparameterizes $\boldsymbol{x}=\boldsymbol{y}^{K}$ ($K$ odd, element-wise). Lemma 3.1 proves that this multiplicative reparameterization [Cp] corresponds one-to-one with $\ell_p\;(p=2/K)$ regularization regarding global optima, local optima, and (sub)stable points. Thus, sparsity benefits are inherited while optimization is reduced to "smooth loss + standard weight decay." Theorem 3.7 further presents a hard impossibility result: if gradient clipping is directly applied to $\ell_p$, the gradient upper bound and approximation error cannot be simultaneously small (events $\mathcal{E}_1\le\sqrt{d}$ and $\mathcal{E}_2\le d/(2e)$ cannot co-occur). This justifies the necessity of reparameterization over simple gradient clipping.

**2. Adaptive Learning Rate $\eta_t\,\boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon\mathbf{1})$: Neutralizing Zero-Saddle Points**

Reparameterization introduces high-order saddle points near zero via $\boldsymbol{y}^{K-1}$ in the update. Once a coordinate has a different sign than the truth, it cannot cross zero. ReWA solves this by multiplying the step size by a coordinate-level adaptive factor. Example 3.2 uses a 1D toy $f(x)=(x-1)^2$ with $y(0)=-1$ to demonstrate: non-adaptive versions satisfy $|y(T)-1|\ge 1$ and never escape zero, while the adaptive version (reducing to $\boldsymbol{y}(t)-\eta\nabla f(\boldsymbol{y}^K(t))$ when $M=0,\epsilon\to 0$) satisfies $|y(T)-1|\le 2(1-\tfrac{2\eta}{K-1})^T$, achieving linear convergence. The numerator $\boldsymbol{y}^{M}$ controls sparsity intensity, while the denominator $\boldsymbol{y}^{K-1}+\epsilon$ cancels $\boldsymbol{y}^{K-1}$ for large $\boldsymbol{y}$ and uses $\epsilon$ as a stabilizer for small $\boldsymbol{y}$ (similar to the $\epsilon$ in Adam). Theorem 3.3 computes the implicit regularization of ReWA as:

$$R(\boldsymbol{x})=\tfrac{K}{1-M+K}\|\boldsymbol{x}\|_{1+(1-M)/K}^{1+(1-M)/K}+\epsilon\tfrac{K}{2-M}\|\boldsymbol{x}\|_{(2-M)/K}^{(2-M)/K}$$

Proposition 3.4 provides practical recipes: use Config A ($\epsilon=0,M>1$) for simple data and Config B ($\epsilon>0,M<2$) for complex data, both ensuring the primary exponent $p=1+(1-M)/K\in(0,1)$ falls within the $\ell_p$ range.

**3. Explicit Weight Decay $(1-\lambda\eta_t)\boldsymbol{y}(t)$: Removing the "Small Initialization" Dependency**

Implicit bias from reparameterization alone fails with large initializations; works like PowerPropagation only yield sparsity in specific scenarios like small initialization or matrix factorization. ReWA adds explicit $\ell_2$ decay $(1-\lambda\eta_t)\boldsymbol{y}(t)$. Example 3.8 / Theorem 3.9 prove that under a quadratic objective $f(\boldsymbol{x})=\boldsymbol{x}^\top\Lambda\boldsymbol{x}$, solutions without decay can be frozen near initial values. Explicit $\ell_2$ decay ensures convergence to the origin—the sparsest global optimum. This mechanism replaces implicit sparse biases that depend on "small initialization" with an explicit mechanism valid for arbitrary initialization and general non-convex problems. All three components are necessary: removing reparameterization leads back to non-optimizable $\ell_p$, removing adaptive step sizes results in zero-saddle point traps, and removing weight decay loses sparsity.

### Loss & Training
The base optimizer can be SGD or AdamW (Algorithm 2 provides the AdamW version); learning rate supports constant or cosine decay. Practically, odd $K$ is most convenient ($\boldsymbol{x}=\boldsymbol{y}^K$). For even $K$, $\boldsymbol{y}_1\odot\boldsymbol{y}_1-\boldsymbol{y}_2\odot\boldsymbol{y}_2$ or $\boldsymbol{x}=\mathrm{sign}(\boldsymbol{y})\cdot|\boldsymbol{y}|^K$ can be used.

## Key Experimental Results

### Main Results
Using ResNet backbones on CIFAR-10 / ImageNet, the goal is to compare sparsity rates (lower percentage of non-zero parameters is better) at a fixed test accuracy.

| Dataset | Model | Method | Sparsity Rate (Non-zero) | Test Accuracy |
|--------|------|------|----------------|----------|
| CIFAR-10 | ResNet | $\ell_1$ Regularization | Baseline | Comparable to Ours |
| CIFAR-10 | ResNet | **ReWA (Config B)** | Significantly lower than $\ell_1$ | Comparable to $\ell_1$ |
| ImageNet | ResNet | $\ell_1$ Regularization | Baseline | Comparable to Ours |
| ImageNet | ResNet | **ReWA (Config B)** | Significantly lower than $\ell_1$ | Comparable to $\ell_1$ |

### Ablation Study

| Configuration | Phenomenon | Description |
|------|------|------|
| Full ReWA | Stable convergence + Sparsity | All three components enabled |
| w/o Adaptive LR (Non-adaptive SGD on [Cp]) | $|y(T)-1|\ge 1$ on 1D toy; training failure on ImageNet | Validates Example 3.2 / Theorem 3.10 |
| w/o Weight Decay | Stays near initialization on quadratic objective; not sparse | Validates Example 3.8 / Theorem 3.9 |
| Direct $\ell_p$ + Grad Clip | Gradient bound and approx error cannot be simultaneously small | Validates Theorem 3.7 |
| Varying $K,M$ (Fig 1 Heatmap) | Blue region is optimizable; red is high loss; white is $M>K-1$ (invalid) | Provides hyperparameter selection range |

### Key Findings
- The three components are indispensable: removing adaptive LR leads to zero-saddle point stagnation, removing weight decay loses sparsity, and removing reparameterization reverts to non-optimizable $\ell_p$.
- Configuration A vs B: The authors recommend $\epsilon=0$ (more aggressive $\ell_p$) for simple data and $\epsilon>0$ (using a mild $\ell_q\;(q>1)$ as a stabilizer) for complex data, where $\epsilon$ acts similarly to Adam’s stabilizer.
- Since AdamW already includes coordinate-wise adaptive steps, setting $M=0,\epsilon\ne 0$ when using ReWA with AdamW prevents redundant sparsity suppression.

## Highlights & Insights
- **Explicit Alignment of Algorithm and Implicit Regularization**: Through carefully designed update rules, an unsolvable $\ell_p$ constraint is provably embedded into the SGD trajectory. This strategy of "achieving non-convex regularization via iterative formatting" can be transferred to other difficult non-convex constraints.
- **Elegant Impossibility Result (Theorem 3.7)**: It demonstrates that "clipping $\ell_p$ gradients" will always force a choice between stability and fidelity in dimension $d$, providing a strong justification for the reparameterization route over simple gradient clipping.
- **Practical Value of Configuration A/B**: Licensing hyperparameter selection to "dataset complexity" provides a ready-to-use recipe for subsequent pruning in LLMs or diffusion models.

## Limitations & Future Work
- Experiments are limited to ImageNet + ResNet and lack validation on Transformer or LLM scales. Current LLM pruning often relies on structured sparsity (head/channel level), whereas ReWA addresses unstructured sparsity.
- Theorem 3.3 assumes $M$ is even (to ensure update symmetry and simplify analysis). In practice, $M$ can be continuous, but theoretical guarantees are only provided for even values (discussed in Appendix Remark C.3).
- Increasing $K$ worsens numerical conditions for multiplicative reparameterization (high powers of small values may underflow). Maintaining precision during FP16 / BF16 training remains an engineering challenge.
- Empirical comparisons with non-convex methods like SCAD, MCP, or adaptive Lasso are discussed in Appendix B but lack end-to-end horizontal benchmarking.

## Related Work & Insights
- **vs $\ell_1$ / LASSO**: $\ell_1$ is convex and easy to optimize but biased. ReWA uses $\ell_p\;(0<p<1)$ to reduce bias at the cost of requiring reparameterization for stable training.
- **vs PowerPropagation (Schwarz et al., 2021)**: PowerPropagation similarly uses $\boldsymbol{y}^K$ reparameterization but lacks weight decay, relying on implicit bias under small initializations for sparsity. ReWA removes the "small initialization" dependency with explicit weight decay and solves zero-saddle points via adaptive step sizes.
- **vs Direct $\ell_p$ + grad clip**: Theorem 3.7 in this paper provides a hard impossibility proof, effectively negating this baseline.
- **vs AdamW**: AdamW performs coordinate adaptation via $1/\sqrt{v_t}$, which can be viewed as an approximation of ReWA at $M=0$. ReWA differs by explicitly controlling $K,M$ to enforce an $\ell_p$ bias.

## Rating
- Novelty: ⭐⭐⭐⭐ Merges existing [Cp] reparameterization into a unified "Adaptive Step + Explicit Decay" framework and closes the theoretical gap regarding zero-saddle point escape.
- Experimental Thoroughness: ⭐⭐⭐ CIFAR-10 / ImageNet + ResNet are sufficient to validate the claims, though LLM and Transformer benchmarks are missing.
- Writing Quality: ⭐⭐⭐⭐ Effectively uses 1D toy examples to connect theoretical results; Theorem 3.7’s impossibility proof is concise and powerful.
- Value: ⭐⭐⭐⭐ Provides a clean path for engineering non-convex sparse regularization, offering insights for pruning and compressed sensing communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)
- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](../../ICML2025/others/sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)
- [\[ICML 2026\] DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](disjunctivenet_neural_symbolic_learning_via_differentiable_convexified_optimizat.md)

</div>

<!-- RELATED:END -->
