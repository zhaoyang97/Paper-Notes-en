---
title: >-
  [Paper Note] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization
description: >-
  [ICML 2026][Optimization][Zeroth-Order Optimization] This paper stores "stale" block cyclic coordinate descent gradient estimates in a FIFO buffer, reusing them with momentum decay…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Zeroth-Order Optimization"
  - "Coordinate Descent"
  - "Stale Gradients"
  - "Implicit Smoothing"
  - "On-Device Learning"
date: 2026-05-08
content_hash: e8ec1a56df668dcc
---

# Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.14373](https://arxiv.org/abs/2605.14373)  
**Code**: https://github.com/chen-dylan-liang/CoCD (available)  
**Area**: Model Compression / Zeroth-Order Optimization / Edge Training  
**Keywords**: Zeroth-Order Optimization, Coordinate Descent, Stale Gradients, Implicit Smoothing, On-Device Learning

## TL;DR
This paper stores "stale" block cyclic coordinate descent gradient estimates in a FIFO buffer, reusing them with momentum decay, and proves this is equivalent to BCCD with warm-start. It also presents a counterintuitive result: a larger finite difference step size $\epsilon$ implicitly smooths the loss landscape and reduces the effective Lipschitz constant, making stale gradients yield more stable descent.

## Background & Motivation

**Background**: Current zeroth-order (ZO) optimization mainly follows two paradigms: classical coordinate-wise finite difference (FD) with very low variance but $O(d)$ function evaluations per step, making it unscalable; and modern stochastic methods (Evolution Strategies / SPSA / DeepZero's random subspace), which require only $O(1)$ evaluations per step but have high gradient estimation variance, necessitating small learning rates or large batches to suppress noise.

**Limitations of Prior Work**: Both paradigms have critical drawbacks. FD is "accurate but slow," fundamentally unsuitable for large models; stochastic methods are "fast but unstable," prone to divergence on non-convex landscapes, and can even fail on regression tasks (e.g., SARCOS). There are also two implicit assumptions: "stale gradients are lag/noise and should be discarded," and "smaller finite difference step $\epsilon$ is closer to the true gradient and thus better." This paper directly challenges both.

**Key Challenge**: There is a hard trade-off between the variance of a single ZO estimate and the sample efficiency of multiple estimates; moreover, noise introduced by randomization obscures the geometric structure of the landscape.

**Goal**: (1) Design a deterministic, O(1)-budget, low-variance ZO optimizer; (2) Prove that stale gradients can be "reused rather than discarded"; (3) Show that large $\epsilon$ is a feature, not a bug.

**Key Insight**: The authors observe "temporal coherence" in optimization trajectories—gradients between adjacent steps are constrained by Lipschitz smoothness to change only slightly. Thus, the partial derivative computed in the previous step remains nearly valid in the next, and discarding it is wasteful. Meanwhile, large $\epsilon$ is equivalent to averaging within a coordinate neighborhood, i.e., implicit Gaussian smoothing of the objective.

**Core Idea**: Maintain a dense gradient buffer, refreshing only $B$ coordinates per step using FD, while using decayed old values for the rest, forming a "hybrid gradient" for descent—essentially Block Cyclic Coordinate Descent with momentum-style warm start.

## Method

### Overall Architecture
CoCD maintains a dense gradient buffer $\hat{\mathbf{g}} \in \mathbb{R}^n$ (same size as parameters). The global parameter $\mathbf{x}_t$ selects the next group of coordinates $i$ in cyclic order. Each step has two phases: first, decay the entire buffer $\hat{\mathbf{g}}_{t-1}^{\text{decay}} = \gamma \cdot \hat{\mathbf{g}}_{t-1}$; then, for the $B$ selected coordinates in the current batch, compute new values using central FD $\tilde{\nabla}_i f(\mathbf{x}_t) = \frac{f(\mathbf{x}_t+\epsilon\mathbf{e}_i)-f(\mathbf{x}_t-\epsilon\mathbf{e}_i)}{\epsilon}$ and overwrite the corresponding buffer entries. Finally, update parameters using the entire buffer: $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \hat{\mathbf{g}}_t$. This reduces per-step query complexity to $O(B)$ (typically $B \ll n$) while maintaining a full-dimensional descent direction.

### Key Designs

1. **Coherent buffer + momentum decay $\gamma$ unifies BCCD and CoCD**:

    - **Function**: A scalar $\gamma \in [0,1]$ connects BCCD (no reuse of stale gradients) and full-history CoCD (perfect memory) into a family of algorithms.
    - **Mechanism**: When $\gamma=0$, the buffer is reset each step, equivalent to classical BCCD (only current $B$ coordinates are valid); when $\gamma=1$, stale gradients are retained until refreshed, equivalent to full-gradient descent with time-lagged gradients; for $0<\gamma<1$, "fading memory" is formed, with old gradients exponentially suppressed, providing robustness for highly non-convex landscapes. In the SARCOS $B=1$ stress test, CoCD with $\gamma=0.95$ converges to loss 68.6, while BCCD ($\gamma=0$) is stuck at 188 even with $B$ increased to 64.
    - **Design Motivation**: Traditionally, stale gradients are seen as a burden in distributed SGD; here, the opposite is leveraged—since gradient changes between consecutive steps are Lipschitz-bounded, stale values serve as a free "warm start," much more stable than random estimation from scratch each step.

2. **Implicit smoothing: larger $\epsilon$ is more stable**:

    - **Function**: Turns the finite difference "step size" from a source of error into a landscape smoother, allowing larger learning rates and longer history.
    - **Mechanism**: Central difference is equivalent to $\tilde{\nabla}_i^\epsilon f(\mathbf{x}) = \frac{1}{2\epsilon}\int_{-\epsilon}^{\epsilon}\nabla_i f(\mathbf{x}+u\mathbf{e}_i)du$, i.e., averaging the true gradient over an $\epsilon$-neighborhood. Define the effective Lipschitz constant $L_\epsilon$, and it can be shown that $L_\epsilon \le L$ and decreases monotonically with increasing $\epsilon$. The approximation error bound for CoCD is $\|\hat{\mathbf{g}}_t - \tilde{\nabla}^\epsilon f(\mathbf{x}_t)\| \le \frac{L_\epsilon \delta}{2}(BK(K-1)+2rK)$, where $K=\lfloor n/B\rfloor$. Smaller $L_\epsilon$ allows larger tolerable step size $\delta$ for the same staleness.
    - **Design Motivation**: Traditional FD literature treats "$\epsilon \to 0$" as the gold standard, but this route injects high-frequency noise into gradient estimates on non-convex landscapes. By treating $\epsilon$ as a smoothing radius, the authors provide a theoretical explanation for the benefit of large $\epsilon$—on SARCOS, $\epsilon=1$ degrades BCCD but yields the best results for CoCD, empirically validating this counterintuitive claim.

3. **Flattened FIFO buffer + virtualized indexing**:

    - **Function**: Enables CoCD's cyclic coordinate logic to work with neural networks (parameters as various-shaped tensors), while maintaining $O(1)$ overhead and strictly "one parameter copy" memory usage.
    - **Mechanism**: Allocate a contiguous 1D memory segment (FIFO buffer, length $m$, typically $m=n$ but can be smaller for memory-constrained deployment), and use integer pointers `cur_param_idx` / `cur_weight_idx` / `cur_grad_idx` to map flat indices to tensor views; during descent, use temporary tensor views (no copy) for in-place subtraction $\mathbf{x} \leftarrow \mathbf{x} - \alpha \hat{\mathbf{g}}$.
    - **Design Motivation**: Naive implementations would reshape/concat parameter tensors to flat arrays each step and reshape back, incurring prohibitive memory copy costs. This trick ensures CoCD's memory usage is strictly one parameter copy—more efficient than Adam (which needs two moment copies) and even SPSA (which stores projection matrices), making it ideal for on-device training.

### Loss & Training
No new loss is introduced; only the optimizer is replaced. Theoretically, under L-smooth + PŁ conditions, CoCD is proven to converge linearly: $f(\mathbf{x}_t)-f(\mathbf{x}^*) \le (1-\frac{2\mu C_1}{C_2})^t (f(\mathbf{x}_0)-f(\mathbf{x}^*))$, where the staleness factor $\tau = n/B - 1$, and the learning rate must satisfy $\alpha < \frac{2}{L(1+n\tau)}$. Larger $B$ yields smaller $\tau$ and allows larger learning rates; $\gamma$ and $m$ further affect the stability threshold via the effective LR.

## Key Experimental Results

### Main Results
Model sizes range from 13k–270k parameters (MLP/CNN/ResNet-20), datasets include SARCOS (regression), MNIST, and CIFAR-10.

| Dataset | Method | Final Metric | Time (s/epoch) |
|---------|--------|-------------|---------------|
| SARCOS | SGD (Oracle, 1st-order) | Loss 5.38 | – |
| SARCOS | BCCD ($\gamma=0$) | 188.73 | 6.23 |
| SARCOS | **CoCD (Ours)** | **31.18** | 6.12 |
| MNIST | BCCD | 27.03% | ~44.4 |
| MNIST | **CoCD** | **95.48%** | ~44.6 |
| CIFAR-10 | BCCD | 10.13% (random guess) | ~77.0 |
| CIFAR-10 | **CoCD** | **45.08%** | ~77.0 |

### Ablation Study

| Configuration | Key Findings | Notes |
|---------------|-------------|-------|
| $\gamma$ sweep | $\gamma=0$ is slowest, $\gamma \to 1$ monotonically faster | Momentum is the dominant factor for convergence speed |
| $\epsilon$ sweep (SARCOS) | Larger $\epsilon$ yields lower loss | Implicit smoothing outweighs FD error in highly non-convex settings |
| $\epsilon$ sweep (MNIST) | Larger $\epsilon$ leads to smoother, more stable training curves | Consistent with the $L_\epsilon$ theoretical explanation |
| Memory budget $M=0.25n$ | Still converges | Demonstrates robustness of "one parameter copy" design |
| CoCD vs ZO-SGD | Converges + 8.1s vs 15.7s per episode | No Gaussian sampling overhead, 2x faster wall-clock |
| CoCD vs SPSA | CoCD converges stably, SPSA diverges mid-training on SARCOS | Deterministic updates are more stable than stochastic ones |

### Key Findings
- **Momentum $\gamma$ is the most critical hyperparameter**: It directly determines the gap between CoCD and classical BCCD (10% vs 45% on CIFAR-10).
- **The counterintuitive benefit of large $\epsilon$ is most pronounced on non-convex regression tasks**; for classification, it mainly improves training curve smoothness rather than final accuracy.
- **Significant wall-clock advantage over stochastic ZO**: Deterministic updates avoid the overhead of Gaussian sampling and large batches.

## Highlights & Insights
- **Reframing "stale = resource"**: In distributed SGD literature, stale gradients are always seen as uncontrollable delay noise; this paper shows that under cyclic structure and temporal coherence, staleness is actually a nearly free source of geometric information—a warm start.
- **Mutual reinforcement of theory and experiment**: The error bound $\frac{L_\epsilon \delta}{2}(\cdot)$ directly predicts "$L_\epsilon$ small → tolerate larger $\delta$," and the empirical result that $\epsilon=1$ wins on SARCOS validates this.
- **Dual tuning of $\gamma$/$\epsilon$ is a transferable trick**: Any scenario involving "stale approximations + estimation noise" (federated learning, asynchronous SGD, low-precision training) can benefit from "explicit decay of history + proactive increase of smoothing radius."

## Limitations & Future Work
- **Scalability limited to small/medium scale**: For parameter counts above 270k, coordinate-wise FD estimates require impractically large $B$ for accuracy, so LLM-scale is currently out of reach.
- **Hyperparameter sensitivity**: The four knobs $\gamma$, $\epsilon$, $B$, $m$ interact complexly; optimal values are hand-tuned per task, with no adaptive scheme.
- **Strong theoretical assumptions**: Linear convergence relies on the PŁ condition, which deep networks typically do not satisfy; the paper acknowledges this as a best-case ceiling.
- **No direct comparison with MeZO**: The hottest ZO method for LLM finetuning is MeZO (a ZO-SGD variant); comparison is left for future work, so effectiveness on LLMs is unverified.

## Related Work & Insights
- **vs DeepZero (Chen 2024)**: DeepZero, for million-parameter models, uses pruning at initialization to select a random subspace and performs CGE within it, effectively freezing out-of-subspace parameters; CoCD performs true full-space cyclic updates on memory-constrained medium-scale models, avoiding such structural bias, but cannot scale to billions of parameters.
- **vs SPSA / Evolution Strategies**: These methods estimate gradients via random perturbations, resulting in high variance and large batch requirements; CoCD is fully deterministic, low-variance, faster in wall-clock, and its implicit smoothing replaces the "explicit Gaussian smoothing" of stochastic methods.
- **vs WASP (Rakita 2025)**: WASP also assumes temporal coherence and treats past gradients as geometric constraints, but uses affine subspace matrix decomposition, suitable for hundreds of dimensions in robotics control; CoCD relies only on vector-level buffer operations, remaining lightweight up to tens of thousands of parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ Both "stale as warm start" and "large $\epsilon$ as implicit smoothing" are strong counterintuitive contributions
- Experimental Thoroughness: ⭐⭐⭐ Model scale is small (≤270k), no MeZO/LLM finetuning comparison, but SARCOS/MNIST/CIFAR + multiple baselines + ablations sufficiently support the claims
- Writing Quality: ⭐⭐⭐⭐⭐ Three-part introduction/theory/empirical validation, with formulas and tables interleaved
- Value: ⭐⭐⭐⭐ Transferable design ideas for edge training, black-box optimization, and asynchronous SGD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](../../ICLR2026/model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](../../ACL2025/model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers](../../CVPR2025/model_compression/l_swag_zero_shot_nas_vision_transformers.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)

</div>

<!-- RELATED:END -->
