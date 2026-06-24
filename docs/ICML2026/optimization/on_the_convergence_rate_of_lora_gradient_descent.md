---
title: >-
  [Paper Note] On the Convergence Rate of LoRA Gradient Descent
description: >-
  [ICML 2026][Optimization][LoRA] This paper proves for the first time that the minimum gradient norm of original LoRA gradient descent converges at a rate of $O(1/\log T)$ without assuming bounded adapter matrices or requiring Lipschitz smoothness of the re-parameterized loss (recovering the classic $O(1/T)$ if parameter norms are bounded). Based on this, adaptive/normalized learning rates strictly corresponding to the theory are designed, with training acceleration and stabil…
tags:
  - "ICML 2026"
  - "Optimization"
  - "LoRA"
  - "Convergence Analysis"
  - "Non-Lipschitz Smoothness"
  - "Burer-Monteiro"
  - "Adaptive Learning Rate"
date: 2026-05-08
content_hash: d2d524dcc4cb9766
---

# On the Convergence Rate of LoRA Gradient Descent

**Conference**: ICML 2026  
**arXiv**: [2512.18248](https://arxiv.org/abs/2512.18248)  
**Code**: https://github.com/siqiaomu/lora  
**Area**: Optimization Theory / Efficient Fine-tuning of LLMs / LoRA  
**Keywords**: LoRA, Convergence Analysis, Non-Lipschitz Smoothness, Burer-Monteiro, Adaptive Learning Rate

## TL;DR
This paper proves for the first time that the minimum gradient norm of original LoRA gradient descent converges at a rate of $O(1/\log T)$ without assuming bounded adapter matrices or requiring Lipschitz smoothness of the re-parameterized loss (recovering the classic $O(1/T)$ if parameter norms are bounded). Based on this, adaptive/normalized learning rates strictly corresponding to the theory are designed, with training acceleration and stability improvements validated on logistic regression, ResNet-18, and TinyLlama.

## Background & Motivation

**Background**: LoRA (Low-Rank Adaptation) has become the most popular scheme for LLM fine-tuning—freezing pre-trained weights $W_0$ and training only two small matrices $A, B$ such that the new weights are $W_0 + BA$. The algorithm itself is minimalist: performing gradient descent on $A$ and $B$ simultaneously at each step.

**Limitations of Prior Work**: Although LoRA is simple and empirically effective, its convergence theory has remained a paradox. Even if the original loss $\mathcal{L}(W)$ is Lipschitz smooth, the re-parameterized $\mathcal{L}(BA)$ is no longer Lipschitz smooth with respect to $A$ and $B$ (because $\nabla_B \mathcal{L}(BA)$ contains a multiplicative factor of $A$). This directly invalidates the classic $O(1/T)$ analysis derived from the descent lemma.

**Key Challenge**: Existing LoRA theoretical analyses fall into three categories, all of which avoid this core difficulty: (1) **Infinite regime limits** (Kim 2025, Jang 2024, NTK analysis, etc.): These provide asymptotic convergence or conclusions for infinite-width neural networks but do not yield non-asymptotic rates for finite models; (2) **LoRA-like variants** (GaLore, RAC-LoRA, etc.): These update only a single adapter or add projections to maintain Lipschitz smoothness, but they do not correspond to LoRA as actually deployed; (3) **Convergence under strong assumptions** (Jiang 2024, Ghiasvand 2025): These assume the norms of $A$ and $B$ are uniformly upper-bounded by some constant, which artificially forces Lipschitz smoothness and hides the constant in the convergence bound—offering no substantial novelty in the proof process.

**Goal**: To provide a **non-asymptotic convergence rate** for original LoRA synchronous gradient descent under the weakest possible assumptions (only Lipschitz smoothness of the original loss + lower boundedness), without assuming bounded norms for $A$ and $B$.

**Key Insight**: By stacking $A$ and $B$ into a single matrix $V$, $BA$ appears in a specific block of $VV^T$—this is the classic Burer-Monteiro symmetric parameterization. From the perspective of $V$, LoRA gradient descent becomes standard gradient descent on $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$. The problem is thus reduced to "non-smooth optimization in the form of $VV^T$," allowing the application of a more refined modified descent lemma.

**Core Idea**: Stacking re-parameterization → Deriving a "Lipschitz-like" descent lemma with higher-order terms → Ensuring descent at each step by limiting the learning rate to be inversely proportional to $\|V_t\|^2$ and the current gradient; analyzing the growth as $\|V_t\|^2 = O(t)$, making $\sum \eta_t = \Theta(\log T)$, which yields an $O(1/\log T)$ convergence rate.

## Method

### Overall Architecture
The proof proceeds in three steps: (1) **Problem Reformulation**—Stacking LoRA's $A$ and $B$ into $V = [B; A^T] \in \mathbb{R}^{(m+n) \times r}$, where the original loss becomes $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$ with $E_1, E_2$ as extraction matrices; LoRA synchronous gradient descent is equivalent to standard gradient descent on $V$. (2) **Modified Descent Lemma**—Proving that $\mathcal{J}$ satisfies a "Lipschitz-like" inequality containing higher-order terms $\|V_2 - V_1\|^k$ ($k=2,3,4$) (Lemma 3.3). (3) **Learning Rate Control + Convergence**—Selecting $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ to guarantee a descent of at least $\eta_t \|\nabla\mathcal{J}(V_t)\|^2 / 4$ per step (Lemma 3.4). Combining this with the worst-case estimate $\|V_t\|^2 = O(t)$ yields $\sum_t \eta_t = \Theta(\log T)$, and finally, $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq 4(\mathcal{J}(V_0) - \mathcal{L}^*) / \sum_t \eta_t$ results in $O(1/\log T)$.

### Key Designs

**1. Burer–Monteiro Stacking Re-parameterization: Reducing non-smooth problems on two matrices to a single $VV^T$ form**

The deadlock in LoRA convergence theory is that even if the original loss $\mathcal{L}(W)$ is Lipschitz smooth, the re-parameterized $\mathcal{L}(BA)$ is not smooth with respect to $A$ and $B$ because $\nabla_B\mathcal{L}(BA)$ carries a multiplicative factor of $A$. The author's solution is to stack $A$ and $B$ into a single matrix $V=\begin{bmatrix}B\\A^T\end{bmatrix}\in\mathbb{R}^{(m+n)\times r}$, such that

$$VV^T=\begin{bmatrix}BB^T & BA\\ A^TB^T & A^TA\end{bmatrix},$$

Using extraction matrices $E_1=[I_m,0]$ and $E_2=[0,I_n]^T$, one can extract $BA=E_1VV^TE_2$. Defining $\mathcal{J}(V)=\mathcal{L}(E_1VV^TE_2)$, its gradient $\nabla\mathcal{J}(V)=2\,\mathrm{Sym}(E_1^T\nabla\mathcal{L}(E_1VV^TE_2)E_2^T)V$ contains the multiplier $V$, which is precisely the root of the non-smoothness. LoRA synchronous GD is exactly equivalent to standard GD on $V$. In this coordinate system, disparate phenomena are unified: $V=0$ automatically becomes a stationary point (regardless of the original loss structure), the $V$ factor makes gradients small when $V$ is small (flat regions near the origin), and the learning rate must decrease when the norm is large—all of which can be precisely quantified, with conclusions generalizing to general $VV^T$ parameterizations.

**2. Modified "Lipschitz-like" Descent Lemma: Coupling a single-step descent inequality with higher-order corrections for the non-smooth $\mathcal{J}$**

Since standard Lipschitz smoothness cannot be proven for $V$, the authors instead carefully expanded the Taylor form of $\mathcal{J}(V_2)-\mathcal{J}(V_1)$ and used the smoothness of the original loss to bound the coefficients of higher-order terms, obtaining Lemma 3.3:

$$\mathcal{J}(V_2)\le\mathcal{J}(V_1)+\langle\nabla\mathcal{J}(V_1),V_2-V_1\rangle_F+\sqrt{2}L\|V_2-V_1\|^2\|V_1\|^2+\sqrt{2}L\|V_2-V_1\|^3\|V_1\|+\tfrac{\sqrt{2}L}{4}\|V_2-V_1\|^4+\|\nabla\mathcal{L}(E_1V_1V_1^TE_2)\|\,\|V_2-V_1\|^2.$$

Compared to the classic descent lemma, it contains three additional higher-order terms (with $\|V_1\|^2$, $\|V_1\|$, and a standalone $\|V_2-V_1\|^4$) and a gradient-dependent term, reflecting the constraints of the $V$ norm and the original gradient on the feasible step size. This is the core trick for reducing a "non-Lipschitz problem" to a "controlled weak descent condition with higher-order corrections"—as long as the learning rate is small enough, these higher-order terms are suppressed by the first-order term, and single-step descent still holds.

**3. Position-Dependent Adaptive Learning Rate and $O(1/\log T)$ Rate: Adjusting step size based on current norm and gradient to derive the convergence rate**

The final step converts the descent lemma into an executable step-size rule. Lemma 3.4 selects $\eta_t=\min\{1/(4\sqrt{2}L(\|V_t\|^2+\|\nabla\mathcal{L}(E_1V_tV_t^TE_2)\|)),\,1\}$ so that higher-order terms are dominated by the first-order term, yielding a descent of at least $\mathcal{J}(V_{t+1})\le\mathcal{J}(V_t)-\frac{\eta_t}{4}\|\nabla\mathcal{J}(V_t)\|^2$ per step. Summing over $t$ and using $\mathcal{J}\ge\mathcal{L}^*$ gives $\min_t\|\nabla\mathcal{J}(V_t)\|^2\le 4(\mathcal{J}(V_0)-\mathcal{L}^*)/\sum_t\eta_t$. The key lies in estimating $\sum_t\eta_t$: in the worst case $\|V_t\|^2=O(t)$, making $\eta_t=\Omega(1/t)$, and the harmonic series gives $\sum_t\eta_t=\Theta(\log T)$, resulting in a convergence rate of $O(1/\log T)$ (Theorem 3.5). If $\|V_t\|\le C$ is additionally assumed, then $\sum_t\eta_t=\Theta(T)$, recovering the classic $O(1/T)$. This rate characterizes LoRA's unique "position dependence"—the learning rate must decrease when iterates are far from the origin (deceleration), while moves can be aggressive when close. The artificial stationary point at $V=0$ implies LoRA may converge to the origin even if the original full-rank optimum is far away, which is the theoretical root cause of why it yields different solutions from full-rank training. Practical schedulers $\eta^{adapt}$, $\eta^{adapt2}$, and $\eta^{norm}$ in the experiments are approximations of this formula.

### Loss & Training
The proof uses only two assumptions: the original loss $\mathcal{L}$ is $L$-Lipschitz smooth and lower bounded. The algorithm is standard LoRA synchronous gradient descent: $A_{t+1} = A_t - \eta_t \nabla_A \mathcal{L}(B_t A_t)$, $B_{t+1} = B_t - \eta_t \nabla_B \mathcal{L}(B_t A_t)$. The theoretical results naturally extend to multi-weight matrix scenarios (Lemma 3.6 proves that the block-constructed $\tilde{\mathcal{L}}$ is $2L$-Lipschitz smooth).

## Key Experimental Results

### Main Results
The purpose of the experiments is not SOTA but theoretical validation. Tasks: CIFAR-10 classification with three model tiers—logistic regression on ResNet-18 embeddings (loss known to be Lipschitz smooth), Direct ResNet-18 training (LoRA on convolutional layers, BatchNorm disabled), and TinyLlama-1.1B LoRA fine-tuning on Alpaca. Three learning rate schemes were analyzed:

| LR Scheme | Formula |
|----------|------|
| Adaptive $\eta^{adapt}$ | $\alpha / (\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)$ |
| Adaptive $\eta^{adapt2}$ | $\alpha / (\|V_t\|^2 + \sqrt{\mathcal{L}(E_1 V_t V_t^T E_2)})$ |
| Normalized $\eta^{norm}$ | $\alpha / \|\nabla\mathcal{L}(V_t)\|^{1/2}$ |

| Experiment | Key Observation |
|------|---------|
| Logistic regression (rank 4 / 20) | All three non-constant LR schemes converge faster and more stably than constant LR of the same magnitude; constant LR diverges if high and is slow if low; $\eta^{adapt}$ and $\eta^{adapt2}$ are highly correlated initially. |
| ResNet-18 + LoRA on CIFAR-10 | $\eta^{adapt2}$ and $\eta^{norm}$ significantly stabilize training; $\eta^{norm}$ performs best; $\|V_t\|$ stops growing after finite steps. |
| TinyLlama LoRA on Alpaca ($\sigma = 10^{-3}$) | Under small initialization, adaptive LR converges faster and more stably than constant LR. |
| TinyLlama LoRA on Alpaca ($\sigma = 1/r$) | Under large initialization, adaptive LR behaves like constant LR (since $\|V_t\|$ is large and grows slowly), reducing the advantage. |

### Ablation Study
Observation of long-term behavior in Logistic regression over 1000 epochs:

| Phenomenon | Explanation |
|------|------|
| Loss appears to converge early | However, $\|V_t\|$ grows monotonically across all $t$, unbounded. |
| Asymptotic convergence rate is slower than $O(1/T)$ | Validates the theory that "convergence slows to $O(1/\log T)$ as $\|V_t\| \to \infty$". |
| $\|V_t\|$ stops growing after finite steps on ResNet-18 | Falls into the bounded regime, yielding a rate of $O(1/T)$. |

### Key Findings
- **Precise Theoretical-Experimental Correspondence**: The observed chain of "$\|V_t\|$ growth ↔ LR decrease ↔ convergence deceleration" directly stems from the proof; the three LR schemes are practical approximations of formula (8).
- **LoRA is Flat Near Initialization**: Since $V=0$ is a stationary point, standard initialization ($B=0$) traps the model in a low-gradient region, requiring a large initial LR to "escape," explaining why high initial values in $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ followed by decay stabilize training.
- **Position Dependence is Unique to LoRA**: Standard GD convergence rates do not depend on parameter norms, but LoRA decelerates far from the origin and accelerates close to it due to the $V$ factor—a direct manifestation of how low-rank re-parameterization reshapes the loss landscape geometry.
- **Diminishing Adaptive Advantage in High Dimensions**: When $\|V_t\|$ is extremely large and changes slowly (high-dimensional LLMs), $\eta^{adapt}$ degenerates toward a constant LR, appearing as $O(1/T)$, though the asymptotic behavior remains $O(1/\log T)$.
- **Convergence Rate is Independent of Rank $r$**: $r$ does not directly enter the rate calculation except indirectly through $\|V_0\|^2$, as gradient descent itself is dimension-free.

## Highlights & Insights
- **First Non-Asymptotic LoRA Convergence Proof without Artificial Bounded Assumptions**: All previous non-asymptotic results bypassed the failure of Lipschitz smoothness by "assuming $\|A\|, \|B\|$ are uniformly bounded"; this paper solves the core difficulty via $V$ re-parameterization + modified descent lemma.
- **Geometric Insight that $V = 0$ is Always a Stationary Point**: Reveals that LoRA may converge to the origin (even if the original optimum is far), theoretically explaining why LoRA and full-rank training yield different solutions.
- **Theory Produces Practical Algorithms**: The three LR formulas $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ were derived directly from the theoretical $\eta_t$ selection in Lemma 3.4 rather than being post-hoc heuristic adjustments; this tight "theory → algorithm" coupling is rare in optimization papers.
- **Unification via Burer-Monteiro Perspective**: The authors also prove that for general Lipschitz smooth functions, gradient descent on the Burer-Monteiro form $\min_V f(VV^T)$ converges to a stationary point, embedding LoRA theory into the broader framework of low-rank parameterization optimization.
- **$O(1/\log T)$ "Position-Dependent Deceleration" is Intrinsic**: This is not due to a loose proof but reflects actual behavior induced by LoRA’s geometric structure, as observed in experiments.

## Limitations & Future Work
- **Covers only Deterministic GD**: Actual LoRA uses SGD (stochastic batches). Generalizing to stochastic noise requires additional tools like four-moment bounds, which the authors leave for future work.
- **No Analysis of Convex/Strongly Convex Cases**: Classic GD has $O(1/T)$ or $O((1-\mu/L)^T)$ rates under convexity; LoRA re-parameterization destroys original convexity, and whether faster convergence exists remains unclear.
- **LR Formulas Depend on Original Gradient $\nabla\mathcal{L}(E_1 V_t V_t^T E_2)$**: Real-world LoRA usually computes $xA^T B^T$ to save memory without explicitly constructing $BA$, making $\eta^{adapt}$ computationally expensive; $\eta^{adapt2}$ mitigates this using loss instead of gradient but still requires extra evaluation.
- **Lack of Lower Bound for $O(1/\log T)$**: The authors did not prove that this rate is tight—it could potentially be faster; the theory currently provides only an upper bound.
- **Exclusion of Common Variants (QLoRA, ReLoRA, LoRA+)**: Theoretical analysis of these variants is not covered.
- **Relaxed Constants in Multi-matrix Extension**: Lemma 3.6 yields $\tilde{\mathcal{L}}$ as $2L$-Lipschitz smooth rather than $L$; accumulated constants may not be optimal at scale.

## Related Work & Insights
- **vs Jiang 2024 / Ghiasvand 2025**: Both assume uniform bounds on $A, B$ norms, essentially forcing Lipschitzness; this paper removes this assumption with a novel proof for non-Lipschitz challenges.
- **vs NTK-like Analysis (Jang 2024, Hayou 2024)**: These provide only asymptotic properties in infinite-width limits, whereas this work provides non-asymptotic rates for finite models.
- **vs RAC-LoRA / Bernoulli-LoRA**: These variants update only one adapter to maintain Lipschitz smoothness and do not match actual LoRA; this paper analyzes original synchronous LoRA directly.
- **vs GaLore / RSO / LDAdam**: These are LoRA-like low-rank memory optimizers with different update rules; this work proves results for LoRA itself but provides a similar analytical paradigm.
- **vs General Burer-Monteiro Theory**: This paper's proof covers GD convergence for BM parameterization, bridging to the BM optimization and semidefinite programming literature.
- **vs LoRA+ (Hayou 2024)**: LoRA+ proposes different learning rates for $A$ and $B$; the position-dependent LR in this work is an independent strategy, and the two could be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves a core difficulty in LoRA convergence theory with new proving techniques ($V$ re-parameterization + modified descent lemma).
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across three tiers (Logistic regression + ResNet-18 + TinyLlama); sufficient to support the theory given it is an optimization paper.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear three-step proof structure with well-explained motivations and natural transitions from theory to experiments.
- Value: ⭐⭐⭐⭐ High theoretical value (clean answer for LoRA convergence); moderate practical value (adaptive LR schemes are worth testing in real fine-tuning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](../../ICLR2026/optimization/on_the_convergence_direction_of_gradient_descent.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICLR 2026\] Fast Convergence of Natural Gradient Descent for Over-parameterized Physics-Informed Neural Networks](../../ICLR2026/optimization/fast_convergence_of_natural_gradient_descent_for_over-parameterized_physics-info.md)
- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->
