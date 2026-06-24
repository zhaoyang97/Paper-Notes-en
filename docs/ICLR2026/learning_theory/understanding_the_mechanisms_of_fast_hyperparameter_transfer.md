---
title: >-
  [Paper Note] Understanding the Mechanisms of Fast Hyperparameter Transfer
description: >-
  [ICLR 2026][Deep Learning Theory][Hyperparameter Transfer] This paper establishes a conceptual framework for the empirical phenomenon where hyperparameters (especially learning rates) in µP can be rapidly transferred from small to large models. The authors strictly distinguish between "weak," "fast," and "useful" transfer via convergence rates. They propose a top-k loss decomposition linearized along EMA trajectories, splitting the final loss into "width-stable top-k componen…
tags:
  - "ICLR 2026"
  - "Deep Learning Theory"
  - "Optimization Dynamics"
  - "Hyperparameter Transfer"
  - "µP"
  - "Width Scaling"
  - "Optimization Trajectory"
  - "Loss Decomposition"
date: 2026-05-08
content_hash: f8f17a6347b193bb
---

# Understanding the Mechanisms of Fast Hyperparameter Transfer

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q7mLKxQ8qk](https://openreview.net/forum?id=Q7mLKxQ8qk)  
**Area**: Deep Learning Theory / Optimization Dynamics  
**Keywords**: Hyperparameter Transfer, µP, Width Scaling, Optimization Trajectory, Loss Decomposition

## TL;DR
This paper establishes a conceptual framework for the empirical phenomenon where hyperparameters (especially learning rates) in µP can be rapidly transferred from small to large models. The authors strictly distinguish between "weak," "fast," and "useful" transfer via convergence rates. They propose a top-k loss decomposition linearized along EMA trajectories, splitting the final loss into "width-stable top-k components that determine optimal hyperparameters" and "residual components that continuously reduce loss with width but barely affect hyperparameter selection." This mechanism is validated in synthetic cases and LLM pre-training.

## Background & Motivation

**Background**: As models grow increasingly large, searching for hyperparameters (HP) via grid search becomes prohibitively expensive. A promising solution is "scale-aware hyperparameters" — defining the learning rate as a scale-independent constant $\eta$ multiplied by a scaling factor $n^{-a}$ (where $n$ is width). The Tensor Programs series formalized this perspective and derived the Maximal Update Parameterization (µP), which guarantees "optimal training" in the infinite-width limit. By specifying initialization variance and learning rate scaling for each layer, µP ensures that activations and their updates remain $\Theta(1)$ as $n\to\infty$, making the optimal $\eta$ asymptotically scale-independent.

**Limitations of Prior Work**: In practice, µP exhibits a property much stronger than what theory predicts: **fast hyperparameter transfer**. The optimal hyperparameters converge with width **much faster** than the performance metrics we actually care about (e.g., validation loss). This "hyperparameters stabilize first while loss keeps decreasing" property allows HPs selected on small proxy models to be applied to large models with almost no performance drop. However, µP theory only guarantees that the "optimal $\eta$ asymptotically exists and is scale-independent"; it **fails to explain** why convergence is so fast—a mystery acknowledged even by the original proponents of µP.

**Key Challenge**: The "existence of a scale-independent limit for optimal HPs" (a weak property) is distinct from "optimal HPs converging faster than loss, thus providing a computational advantage for transfer" (a strong property). If optimal HPs converge slowly, values selected on small models would be sub-optimal for large models (e.g., the counter-example in Figure 2 where the optimal HP shifts suddenly at $n=8192$). Existing work either confirms transfer effectiveness empirically or observes that top Hessian eigenvalues converge quickly with width, but they do not strictly link these spectral statistics to the optimal learning rate.

**Goal**: (1) Provide a quantifiable definition of whether transfer is "fast" or "useful"; (2) Identify the **structural characteristics of the training process** that enable fast transfer and link them to the convergence rate of optimal hyperparameters.

**Key Insight**: The authors start from the intuition of the "low-dimensional structure of optimization trajectories"—gradients quickly align with the top Hessian eigenspace, and significant loss reduction occurs within a very small subspace. The hypothesis is: **Loss depends on all directions, but the learning rate is determined by only a few top directions**, and the statistics of these top directions converge exceptionally fast with width.

**Core Idea**: Use a "linearization + top-k spectral decomposition" along EMA (Exponential Moving Average) smoothed trajectories. This splits the loss change into width-stable top-k components (determining the optimal HP) and width-sensitive residual components (reducing loss further but barely shifting the optimal HP). Fast transfer is explained by the rapid convergence of the former.

## Method

### Overall Architecture

The paper does not propose a new algorithm but builds an **analytical framework** to answer why hyperparameter transfer is so fast under µP. The logic progresses through three layers:

The first layer is **Formalization and Characterization**. Training hyperparameters at width $n$ are written as $H_n(\nu,\gamma) = (\nu_1 n^{-\gamma_1}, \dots, \nu_h n^{-\gamma_h})$, where $\nu$ is the scale-independent HP and $\gamma$ is the scaling exponent. Fixing the pipeline and varying only $n$, we obtain the optimal HP $\nu^\star(n) = \arg\min_\nu \varphi_n(\nu)$ and optimal value $\varphi_n^\star$. Three quantities varying with width are defined: loss gap $a_n$, HP gap $b_n$, and transfer sub-optimality $c_n$. Their **convergence rates** are used to strictly distinguish between "weak," "fast," and "useful" transfer, proving that "fast" and "useful" coincide under the condition $\beta>\alpha/2$.

The second layer uses **Synthetic Cases to Define Boundaries**. The authors construct two scenarios where $(a_n, b_n, c_n)$ rates can be calculated precisely, showing that fast transfer is **not a free lunch**. In random feature ridge regression, transfer is provably advantageous (fast), whereas in a two-layer ReLU network, µP only achieves "weak transfer," where the optimal learning rate converges as slowly as the loss. This suggests that the speed of transfer depends on structural properties emerging from the interaction of data, optimizer, and architecture.

The third layer is the **Mechanism Explanation**, the core contribution. The authors perform one-step linearization along the EMA smoothed trajectory and use an alignment matrix for top-k spectral decomposition. This splits loss changes into top-k and residual components. They propose three conditions—"top-κ strong convexity + top-κ invariance + residual flatness"—as a sufficient mechanism for fast transfer, verified on Llama-style Transformer pre-training under µP.

### Key Designs

**1. Three Convergence Quantities: Linking "Fast Transfer" and "Useful Transfer" to the Same Rate Condition**

The pain point is that "optimal HPs converging to a limit" (weak transfer) does not guarantee usefulness if convergence is too slow. The authors introduce: loss gap $a_n = |\varphi_n^\star - \varphi_\infty^\star|$, HP gap $b_n = \|\nu^\star(n) - \nu^\star(\infty)\|$, and transfer sub-optimality $c_n = |\varphi_\infty(\nu^\star(n)) - \varphi_\infty^\star|$. $c_n$ directly measures transfer efficiency.

Under the assumption of local strong convexity in $\varphi_\infty$, the authors prove $b_n = O(a_n^{1/2})$ and $c_n = \Theta(b_n^2) = O(a_n)$. **Fast transfer** is defined as $c_n = o(a_n)$, which is equivalent to $b_n = o(a_n^{1/2})$. If $a_n \sim n^{-\alpha}$ and $b_n \sim n^{-\beta}$, fast transfer occurs if and only if $\beta > \alpha/2$. Theorem 2 further links this to **computationally optimal grid search**: transfer is useful if and only if $\beta > \alpha/2$. This step shows that as long as HPs are scale-independent, transfer asymptotically never loses to direct tuning, but "why $\beta$ can exceed $\alpha/2$" remains to be explained.

**2. Synthetic Cases: Structure Matters, µP is Not Sufficient**

To show these rates are not arbitrary, the authors provide two settings. In **Random Feature Ridge Regression** (tuning penalty $\lambda$), Theorem 3 precisely calculates $a_n \sim \psi_2^{-1}$, $b_n = O(\psi_2^{-1})$, and $c_n = O(\psi_2^{-2}) \ll a_n$, where $\psi_2 = n/d$. This satisfies $\beta > \alpha/2$, providing **provable fast transfer**.

The counter-example is a **Two-layer ReLU Network** learning the indicator function of a sphere, using µP and Adam to tune the learning rate. Here, although the optimal LR has a limit, it drifts significantly. Power-law fitting gives $b_n \sim \sqrt{a_n}$, meaning the optimal HP converges **no faster** than the trivial rate—this is only weak transfer. Together, these show that **µP guarantees weak transfer; fast transfer relies on emergent structural properties**.

**3. Top-k Trajectory Decomposition: Extracting the "HP-Determining Low-Dimensional Subspace"**

The authors explicitly extract the statistics that determine the optimal HP. First, they approximate the one-step loss change $\delta L(w_t)$ using linearization $\delta\varphi(w_t) := \langle g_t, \delta w_t\rangle$ along the **Exponential Moving Average (EMA)** of the trajectory. EMA removes "edge-of-stability" oscillations, making the trajectory smooth enough for linearization to be faithful.

Second, they define an alignment matrix $S(G,\delta W) := \tfrac{1}{2}(G^\top \delta W + \delta W^\top G)$ for a layer's weight $W$ and its gradient $G$. By sorting its eigenvalues $|\lambda_1|\ge\cdots\ge|\lambda_n|$, the total loss change is $\delta\varphi(W) = \sum \lambda_i$, and the **top-k loss change** is $\delta\varphi_k(W) := \sum_{i=1}^k \lambda_i$. This decomposes the total loss curve into a **top-κ loss curve** $\varphi_n^\kappa$ and a **residual loss curve** $\varphi_n^{-\kappa}$.

Hypothesis 1 posits three verifiable conditions: ① **Top-κ strong convexity**; ② **Top-κ invariance** (top-κ loss converges quickly with width); ③ **Residual flatness** (residual is nearly flat around the optimal HP). Together, these imply $\nu^\star(n)\approx\nu^\star(\infty)$. Algorithm 1 optimizes a proxy $J_{\text{proxy}}(\kappa)$ to find the truncation $\hat\kappa(n)$, which intuitively falls at the "elbow" where the top-k curve begins to diverge from the infinite-width limit.

### Example: Top-k Decomposition in Transformers

For a Llama-style Transformer under µP (width 128 to 2048), setting $k=60$ reveals: **the top-k loss curve is nearly width-invariant and accounts for the vast majority of loss reduction**, while the residual loss improves continuously with width, especially late in training. This suggests that the primary optimization improvement comes from a low-dimensional subspace, while the benefits of "adding width" manifest in the residual—i.e., width-dependent learning happens in the tail components. Algorithm 1's $\hat\kappa(n)$ validates that the residual is flatter than the top-κ loss near the optima, explaining why the optimal LR converges so quickly.

## Key Experimental Results

### Main Results

| Scenario | Loss Gap $a_n$ | HP Gap $b_n$ | Sub-optimality $c_n$ | Conclusion |
|:---|:---|:---|:---|:---|
| Random Feature RR (tune $\lambda$) | $\Theta(\psi_2^{-1})$ | $O(\psi_2^{-1})$ | $O(\psi_2^{-2})$ | **Fast Transfer** (provable) |
| 2-layer ReLU (tune LR, µP) | $n^{-\alpha}$ | $\sim\sqrt{a_n}$ | $\Theta(a_n)$ | **Weak Transfer** only |
| Llama Transformer (µP, WikiText-103) | $L = 3.137 + 13.16\,n^{-0.52}$ | Much faster than loss | Negligible | **Fast Transfer**: $n{=}128$ LR is near-optimal for all |

The reducible loss converges slowly as $n^{-0.52}$, but the optimal LR stabilizes much earlier. Applying the LR from width 128 to larger widths results in nearly overlapping loss curves (Figure 6b), confirming that the HP stabilizes before the loss finishes decreasing.

### Ablation Study (Top-k Analysis)

| Observation | Phenomenon | Mechanism |
|:---|:---|:---|
| Temporal Decomposition ($k{=}60$) | Top-k loss is width-invariant; residual improves with width | Top-κ invariance + Residual width sensitivity |
| Width Decomposition ($\hat\kappa(n)$) | Top-κ loss curves overlap across LRs; total loss varies | Top-κ invariance |
| Residual Shape | Residual is flatter than top-κ loss near optima | Residual flatness |
| Top-k Profile | $\varphi^k_n$ drops then flattens; $\hat\kappa(n)$ at the "exit" point | Truncation trade-off verified |

### Key Findings
- **Slow Loss Convergence $\neq$ Slow HP Convergence**: While loss decreases as $n^{-0.52}$, the optimal LR converges faster because it is determined by the top-k subspace which stabilizes rapidly, while width gains occur in the flat residual.
- **µP Only Guarantees Weak Transfer**: The 2-layer ReLU counter-example proves µP does not automatically ensure fast transfer; structural properties are required.
- **Optimizer Sensitivity**: Switching to Muon results in less stable transfer and weaker top-k invariance; GPT-2 on FineWeb replicates the fast transfer behavior found in Llama.
- **Sample Difficulty Perspective**: A "per-sample" top-k decomposition on CIFAR-10 links transfer quality to sample difficulty, providing an interpretability angle for what top vs. tail components learn.

## Highlights & Insights
- **Formalizing Vague Empirical Phenomena**: Defining $a_n, b_n, c_n$ and the $\beta>\alpha/2$ condition turns "fast transfer" into a testable mathematical property.
- **EMA + Linearization as a Critical Trick**: Conventional linearization is destroyed by oscillations; EMA smoothing makes $\langle g_t,\delta w_t\rangle$ a faithful proxy for loss change.
- **Elegant Alignment Matrix**: $S(G,\delta W)$ is a clean metric for "which directions actually drive loss reduction," compatible with optimizers where updates aren't simply proportional to gradients.
- **Validation of Intuition**: The long-held belief that "HPs are determined by the top, loss by the whole spectrum" is transformed into a measurable top-κ/residual decomposition.

## Limitations & Future Work
- **Conjecture + Qualitative Verification**: Conditions like top-κ invariance are validated qualitatively for large models due to compute costs; exact rates for µP networks remain unproven.
- **Proxy for Infinite Width**: Conclusions rely on using $n_{\max}=2048$ as a proxy for $n=\infty$ and power-law fitting.
- **Local Strong Convexity Dependence**: The framework assumes $\varphi_\infty$ is locally strongly convex; if the loss surface is extremely flat, the significance of HP choice diminishes.
- **Limited Optimizer/Architecture Coverage**: While stable for Adam/Transformers, Muon shows instability, suggesting the need for broader systematic studies.

## Related Work & Insights
- **vs µP / µTransfer (Yang et al.)**: Yang proved µP makes transfer reliable but left the "why it's so fast" question open. This paper fills that gap by distinguishing weak/fast transfer through trajectory decomposition.
- **vs Hessian Spectral View (NMHO24)**: That work observed rapid convergence of top Hessian eigenvalues. This paper explicitly links such spectral stability to the *optimal learning rate* via the top-k loss.
- **vs Low-dimensional Trajectories (GARD18 / SAY24)**: Previous work found gradients align with top Hessian subspaces. This paper utilizes that observation to isolate the low-dimensional subspace driving HP selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide provable rate characterization and a testable trajectory decomposition for fast HP transfer.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across synthetic, LLM, GPT-2, and CIFAR, though large-scale results are qualitative.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from formalization to mechanism.
- Value: ⭐⭐⭐⭐⭐ Addresses a core mystery in high-performance LLM training with actionable theoretical insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Transfer Learning in Infinite Width Feature Learning Networks](transfer_learning_in_infinite_width_feature_learning_networks.md)
- [\[ICLR 2026\] Residual Feature Integration is Sufficient to Prevent Negative Transfer](residual_feature_integration_is_sufficient_to_prevent_negative_transfer.md)
- [\[ICLR 2026\] Understanding and Relaxing the Limitations of Transformers for Linear Algebra](understanding_and_relaxing_the_limitations_of_transformers_for_linear_algebra.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data](fast_escape_slow_convergence_learning_dynamics_of_phase_retrieval_under_power-la.md)

</div>

<!-- RELATED:END -->
