---
title: >-
  [Paper Note] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization
description: >-
  [ICML 2026][Model Compression][Zeroth-order optimization] This paper stores "stale" block-cyclic coordinate descent gradient estimates in a FIFO buffer, reusing them with momentum decay…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Zeroth-order optimization"
  - "coordinate descent"
  - "stale gradients"
  - "implicit smoothing"
  - "on-device learning"
date: 2026-05-08
content_hash: c90ba63b3874154a
---

# Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.14373](https://arxiv.org/abs/2605.14373)  
**Code**: https://github.com/chen-dylan-liang/CoCD (Available)  
**Area**: Model Compression / Zeroth-Order Optimization / Edge Training  
**Keywords**: Zeroth-order optimization, coordinate descent, stale gradients, implicit smoothing, on-device learning

## TL;DR
This paper stores "stale" block-cyclic coordinate descent gradient estimates in a FIFO buffer, reusing them with momentum decay, and proves this is equivalent to BCCD with warm-start. Simultaneously, it presents a counter-intuitive conclusion: a larger finite difference step $\epsilon$ implicitly smooths the loss landscape and reduces the effective Lipschitz constant, allowing stale gradients to achieve stable descent.

## Background & Motivation

**Background**: Current zeroth-order (ZO) optimization primarily follows two paths: classical coordinate-wise finite difference (FD), which has extremely low variance but lacks scalability due to requiring $O(d)$ function evaluations per step; and modern stochastic methods (Evolution Strategies / SPSA / random subspaces like DeepZero), which require only $O(1)$ evaluations per step but suffer from high gradient variance, necessitating very small learning rates or large batches to suppress noise.

**Limitations of Prior Work**: Both paths have fatal flaws. FD is "accurate but slow," making it unsuitable for large models. Stochastic methods are "fast but unstable," easily diverging on non-convex landscapes and even failing on certain regression tasks (e.g., SARCOS). Furthermore, there are two implicit consensuses: "stale gradients are lag/noise and should be discarded" and "the smaller the finite difference step $\epsilon$, the closer it is to the true gradient, the better." This paper directly challenges these two consensuses.

**Key Challenge**: There is a hard trade-off between the variance of a single ZO estimate and the sample efficiency of multiple estimates; meanwhile, the noise introduced by randomization tends to mask the geometric information of the landscape itself.

**Goal**: (1) Design a deterministic, $O(1)$-budget, low-variance ZO optimizer; (2) Prove that stale gradients can be "reused rather than discarded"; (3) Demonstrate that a large $\epsilon$ is a feature rather than a bug.

**Key Insight**: The authors observe "temporal coherence" in optimization trajectories—gradients in adjacent steps can only change slightly as they are constrained by Lipschitz smoothness. Consequently, the partial derivatives calculated in the previous step remain largely valid for the next step; discarding them is wasteful. Additionally, a large $\epsilon$ is equivalent to averaging within a coordinate neighborhood, performing implicit Gaussian smoothing on the objective function.

**Core Idea**: Maintain a dense gradient buffer and refresh FD for only $B$ coordinates at each step. Use decayed old values for the remaining coordinates to form a "hybrid gradient" for descent—essentially Block Cyclic Coordinate Descent (BCCD) complemented by a momentum-style warm start.

## Method

### Overall Architecture
CoCD maintains a dense gradient buffer $\hat{\mathbf{g}} \in \mathbb{R}^n$ of length $n$ (a memory allocation equal to the parameter size). Global parameters $\mathbf{x}_t$ select the next set of coordinates $i$ in cyclic order. Each step consists of two phases: first, apply decay to the entire buffer $\hat{\mathbf{g}}_{t-1}^{\text{decay}} = \gamma \cdot \hat{\mathbf{g}}_{t-1}$; second, for the $B$ selected coordinates in the current batch, use central FD to calculate new values $\tilde{\nabla}_i f(\mathbf{x}_t) = \frac{f(\mathbf{x}_t+\epsilon\mathbf{e}_i)-f(\mathbf{x}_t-\epsilon\mathbf{e}_i)}{\epsilon}$ to overwrite corresponding positions in the buffer; finally, use the entire buffer for parameter updates $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \hat{\mathbf{g}}_t$. This mechanism reduces per-step query complexity to $O(B)$ (typically $B \ll n$) while maintaining descent directions across all dimensions.

### Key Designs

1. **Coherent buffer + momentum decay $\gamma$ unifying BCCD and CoCD**:
    - **Function**: Connects BCCD (no reuse of stale gradients) and full-history CoCD (perfect memory) into a family of algorithms using a scalar $\gamma \in [0,1]$.
    - **Mechanism**: When $\gamma=0$, the buffer is cleared every step, which is equivalent to classical BCCD (valid only for the current $B$ coordinates). When $\gamma=1$, stale gradients are retained permanently until refreshed cyclically, equivalent to full-gradient descent with time-lagged gradients. For $0<\gamma<1$, it forms a "fading memory" where old gradients are exponentially suppressed, providing robustness for highly non-convex landscapes. In an extreme SARCOS stress test with $B=1$, CoCD with $\gamma=0.95$ converged to a loss of 68.6, while BCCD ($\gamma=0$) remained stuck at 188 even when $B$ was increased to 64.
    - **Design Motivation**: Traditional views treat stale gradients as a burden in distributed SGD; this paper utilizes them inversely—since gradient changes between consecutive steps are Lipschitz-constrained, stale values serve as a free "warm start," being much more stable than random estimates started from scratch each step.

2. **Implicit Smoothing: Large $\epsilon$ is actually more stable**:
    - **Function**: Transforms the finite difference "step size" from a source of error into a landscape smoother, allowing for larger learning rates and longer history.
    - **Mechanism**: Central difference is equivalent to $\tilde{\nabla}_i^\epsilon f(\mathbf{x}) = \frac{1}{2\epsilon}\int_{-\epsilon}^{\epsilon}\nabla_i f(\mathbf{x}+u\mathbf{e}_i)du$, which is the average of the true gradient within an $\epsilon$-neighborhood. Defining the effective Lipschitz constant $L_\epsilon$, it can be proved that $L_\epsilon \le L$ and it decreases monotonically as $\epsilon$ increases. The approximation error bound for CoCD is $\|\hat{\mathbf{g}}_t - \tilde{\nabla}^\epsilon f(\mathbf{x}_t)\| \le \frac{L_\epsilon \delta}{2}(BK(K-1)+2rK)$, where $K=\lfloor n/B\rfloor$. The smaller $L_\epsilon$, the larger the tolerable step size $\delta$ for the same degree of staleness.
    - **Design Motivation**: Traditional FD literature treats $\epsilon \to 0$ as the golden rule, but this path introduces high-frequency noise into gradient estimates on non-convex landscapes. By viewing $\epsilon$ as a smoothing radius, the authors provide a theoretical explanation for "beneficial large $\epsilon$." On SARCOS, $\epsilon=1$ caused BCCD to degrade but allowed CoCD to converge to its best result, validating this counter-intuitive claim.

3. **Flattened FIFO buffer + Virtualized Indexing**:
    - **Function**: Enables the cyclic coordinate logic of CoCD to run on neural networks (where parameters are tensors of various shapes) while maintaining $O(1)$ overhead and the constraint of using only "one copy of parameter memory."
    - **Mechanism**: Allocate a contiguous 1D memory (FIFO buffer of length $m$, typically $m=n$ but can be smaller for memory-constrained deployment) and maintain mappings from flat indices to tensor views using integer pointers such as `cur_param_idx` / `cur_weight_idx` / `cur_grad_idx`. During the descent phase, use temporary tensor views (without copying) for in-place subtraction $\mathbf{x} \leftarrow \mathbf{x} - \alpha \hat{\mathbf{g}}$.
    - **Design Motivation**: Naive implementations would reshape/concat parameter tensors into a flat format every step and reshape the updates back, incurring prohibitively high memory copy costs. This trick ensures the memory usage of CoCD is strictly equal to one copy of the parameters—more efficient than Adam (requiring 2 moments) or even SPSA (requiring storage of projection matrices), which is essential for on-device training scenarios.

### Loss & Training
No new loss functions are introduced; only the optimizer is replaced. On the theoretical side, CoCD is proved to achieve linear convergence under L-smooth + PŁ conditions: $f(\mathbf{x}_t)-f(\mathbf{x}^*) \le (1-\frac{2\mu C_1}{C_2})^t (f(\mathbf{x}_0)-f(\mathbf{x}^*))$, where the staleness factor is $\tau = n/B - 1$, requiring the learning rate $\alpha < \frac{2}{L(1+n\tau)}$. As $B$ increases, $\tau$ decreases, allowing for a larger learning rate; $\gamma$ and $m$ further influence the stability threshold by defining the effective LR.

## Key Experimental Results

### Main Results
Model scales range from 13k to 270k parameters (MLP/CNN/ResNet-20), using datasets SARCOS (regression), MNIST, and CIFAR-10.

| Dataset | Method | Final Metric | Time (s/epoch) |
|---------|--------|--------------|---------------|
| SARCOS | SGD (Oracle, 1st-order) | Loss 5.38 | – |
| SARCOS | BCCD ($\gamma=0$) | 188.73 | 6.23 |
| SARCOS | **CoCD (Ours)** | **31.18** | 6.12 |
| MNIST | BCCD | 27.03% | ~44.4 |
| MNIST | **CoCD** | **95.48%** | ~44.6 |
| CIFAR-10 | BCCD | 10.13% (Random) | ~77.0 |
| CIFAR-10 | **CoCD** | **45.08%** | ~77.0 |

### Ablation Study

| Configuration | Key Finding | Description |
|---------------|-------------|-------------|
| $\gamma$ scan | $\gamma=0$ is slowest; $\gamma \to 1$ is monotonically faster | Momentum is the dominant factor for convergence speed |
| $\epsilon$ scan (SARCOS) | Larger $\epsilon$ yields lower loss | Implicit smoothing outweighs FD error in highly non-convex scenarios |
| $\epsilon$ scan (MNIST) | Large $\epsilon$ yields smoother training and better stability | Consistent with the theoretical $L_\epsilon$ explanation |
| Memory budget $M=0.25n$ | Still achieves convergence | Validates the robustness of the "single parameter copy" design |
| CoCD vs ZO-SGD | Convergence + 8.1s vs 15.7s per episode | No Gaussian sampling overhead; 2x faster wall-clock time |
| CoCD vs SPSA | CoCD converges stably; SPSA diverges on SARCOS | Deterministic updates are more stable than stochastic ones |

### Key Findings
- **Momentum $\gamma$ is the most critical hyperparameter**: It directly determines the gap between CoCD and classical BCCD (e.g., 10% vs 45% on CIFAR-10).
- **The counter-intuitive benefit of large $\epsilon$ is most significant in non-convex regression tasks**; in classification tasks, it manifests more in the smoothness of training curves than in final accuracy.
- **Wall-clock performance is significantly superior to stochastic ZO**: This is because deterministic updates eliminate the overhead of Gaussian sampling and large batch sizes.

## Highlights & Insights
- **Reversal of the "Stale = Resource" perspective**: Stale gradients have always been viewed as uncontrollable delayed noise in distributed SGD literature; this paper proves that under a cyclic structure + temporal coherence, staleness is actually a warm start providing nearly free geometric information.
- **Bidirectional confirmation of theory and empirics**: The error bound $\frac{L_\epsilon \delta}{2}(\cdot)$ directly predicts that "small $L_\epsilon \to$ tolerance for larger $\delta$," which was validated by the success of $\epsilon=1$ on SARCOS.
- **The $\gamma$/$\epsilon$ dual adjustment is a transferable trick**: Any scenario using "stale approximations + estimated noise" (Federated Learning, Asynchronous SGD, Low-precision training) can benefit from these two handles: explicit decay of history and active expansion of the smoothing radius.

## Limitations & Future Work
- **Scalability is limited to small-to-medium scales**: When parameter counts exceed 270k, $B$ must be increased to impractical levels to keep coordinate-wise FD estimates accurate; it currently cannot reach LLM scales.
- **Hyperparameter sensitivity**: The interactions between $\gamma$, $\epsilon$, $B$, and $m$ are complex. The optimal values provided in the paper are manually tuned per task, lacking an adaptive scheme.
- **Strong theoretical assumptions**: Linear convergence depends on the PŁ condition, which deep networks usually do not satisfy; the paper acknowledges this as a best-case ceiling.
- **Lack of direct comparison with MeZO**: MeZO (a ZO-SGD variant) is the most popular ZO method for LLM fine-tuning. The authors leave comparison with MeZO for future work, indicating that effectiveness in LLM scenarios has not yet been verified.

## Related Work & Insights
- **vs DeepZero (Chen 2024)**: To handle models with millions of parameters, DeepZero uses pruning at initialization to select a random subspace and then performs CGE within it, essentially freezing parameters outside the subspace. CoCD performs true full-space cyclic updates on memory-constrained medium scales, avoiding such structural bias at the cost of not being directly applicable to billion-parameter models.
- **vs SPSA / Evolution Strategies**: Those methods rely on random perturbations to estimate gradients, which have high variance and require large batches. CoCD is completely deterministic, low-variance, faster in wall-clock time, and replaces the "explicit Gaussian smoothing" of stochastic methods with implicit smoothing.
- **vs WASP (Rakita 2025)**: WASP also explicitly assumes temporal coherence and treats past gradients as geometric constraints but relies on affine subspace matrix decomposition, suitable for robot control with hundreds of dimensions. CoCD relies only on vector-level buffer operations, remaining lightweight even when scaled to tens of thousands of parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ The two counter-intuitive viewpoints—"stale as warm start" and "large $\epsilon$ implicit smoothing"—are highly substantial.
- Experimental Thoroughness: ⭐⭐⭐ The model scale is relatively small (≤270k), and comparisons with MeZO/LLM fine-tuning are missing, but the results on SARCOS/MNIST/CIFAR with multiple baselines and ablations sufficiently support the arguments.
- Writing Quality: ⭐⭐⭐⭐⭐ The structure—three-part introduction, theoretical layout, and empirical validation—flows seamlessly, with formulas and tables advancing the narrative.
- Value: ⭐⭐⭐⭐ Provides transferable design ideas for edge training, black-box optimization, and asynchronous SGD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](../../ICLR2026/model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICML 2026\] GradPower: Powering Gradients for Faster Language Model Pre-Training](gradpower_powering_gradients_for_faster_language_model_pre-training.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[ICML 2026\] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)

</div>

<!-- RELATED:END -->
