---
title: >-
  [Paper Note] On the Convergence Rate of LoRA Gradient Descent
description: >-
  [ICML 2026][Optimization][LoRA] This work provides the first convergence proof for vanilla LoRA gradient descent without assuming bounded adapter matrices or Lipschitz smoothness of the reparameterized loss. It demonstra…
tags:
  - "ICML 2026"
  - "Optimization"
  - "LoRA"
  - "Convergence Analysis"
  - "Non-Lipschitz Smoothness"
  - "Burer-Monteiro"
  - "Adaptive Learning Rate"
date: 2026-05-08
content_hash: 999cbf338bd75432
---

# On the Convergence Rate of LoRA Gradient Descent

**Conference**: ICML 2026  
**arXiv**: [2512.18248](https://arxiv.org/abs/2512.18248)  
**Code**: https://github.com/siqiaomu/lora  
**Area**: Optimization Theory / LLM Efficient Fine-tuning / LoRA  
**Keywords**: LoRA, Convergence Analysis, Non-Lipschitz Smoothness, Burer-Monteiro, Adaptive Learning Rate

## TL;DR
This work provides the first convergence proof for vanilla LoRA gradient descent without assuming bounded adapter matrices or Lipschitz smoothness of the reparameterized loss. It demonstrates that the minimum gradient norm converges at a rate of $O(1/\log T)$ (recovering the classical $O(1/T)$ when parameter norms are bounded) and introduces adaptive/normalized learning rates directly derived from the theory, which enhance training speed and stability on logistic regression, ResNet-18, and TinyLlama.

## Background & Motivation

**Background**: LoRA (Low-Rank Adaptation) has become the most popular scheme for LLM fine-tuning—freezing pre-trained weights $W_0$ and training only two small matrices $A, B$ such that the new weights are $W_0 + BA$. The algorithm itself is minimal: simultaneous gradient descent is applied to $A$ and $B$ at each step.

**Limitations of Prior Work**: Despite its simplicity and empirical effectiveness, LoRA's convergence theory has remained a paradox. Even if the original loss $\mathcal{L}(W)$ is Lipschitz smooth, the reparameterized loss $\mathcal{L}(BA)$ with respect to $A$ and $B$ is no longer Lipschitz smooth (since $\nabla_B \mathcal{L}(BA)$ contains a multiplicative factor of $A$), which invalidates classical $O(1/T)$ analysis based on the descent lemma.

**Key Challenge**: Existing LoRA theoretical analyses fall into three categories that bypass this core difficulty: (1) **Infinite limit regimes** (Kim 2025, Jang 2024, NTK analysis, etc.): provide only asymptotic convergence or results for infinite-width networks, lacking non-asymptotic rates for finite models; (2) **LoRA-like variants** (GaLore, RAC-LoRA, etc.): update only a single adapter or utilize projections to maintain Lipschitz smoothness, which do not correspond to vanilla LoRA; (3) **Convergence under strong assumptions** (Jiang 2024, Ghiasvand 2025): assume the norms of $A$ and $B$ are uniformly bounded by a constant, effectively forcing Lipschitz smoothness and hiding the constant in the convergence bound—offering little proof novelty.

**Goal**: To provide a **non-asymptotic convergence rate** for vanilla LoRA synchronous gradient descent under the weakest possible assumptions (original loss is Lipschitz smooth and lower-bounded) without assuming bounded norms for $A$ and $B$.

**Key Insight**: By stacking $A$ and $B$ into a single matrix $V$, the term $BA$ appears in a specific block of $VV^T$, corresponding to the classical Burer-Monteiro symmetric parameterization. From the perspective of $V$, LoRA gradient descent becomes standard gradient descent on $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$. The problem is reduced to "non-smooth optimization in $VV^T$ form," which allows the application of a refined modified descent lemma.

**Core Idea**: Stack reparameterization $\to$ derive a "Lipschitz-like" descent lemma with high-order terms $\to$ ensure descent at each step by setting the learning rate inversely proportional to $\|V_t\|^2$ and the current gradient. Analyzing the growth $\|V_t\|^2 = O(t)$ leads to $\sum \eta_t = \Theta(\log T)$, yielding an $O(1/\log T)$ convergence rate.

## Method

### Overall Architecture
The proof follows three steps: (1) **Problem Reformulation**—Stack LoRA's $A$ and $B$ into $V = [B; A^T] \in \mathbb{R}^{(m+n) \times r}$. The original loss becomes $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$, where $E_1, E_2$ are extraction matrices. LoRA synchronous gradient descent is equivalent to standard gradient descent on $V$. (2) **Modified Descent Lemma**—Prove that $\mathcal{J}$ satisfies a "Lipschitz-like" inequality containing high-order terms $\|V_2 - V_1\|^k$ ($k=2,3,4$) (Lemma 3.3). (3) **Learning Rate Control & Convergence**—Select $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ to ensure a per-step descent of at least $\eta_t \|\nabla\mathcal{J}(V_t)\|^2 / 4$ (Lemma 3.4). Combining this with the worst-case estimate $\|V_t\|^2 = O(t)$ yields $\sum_t \eta_t = \Theta(\log T)$, eventually leading to $O(1/\log T)$ from $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq 4(\mathcal{J}(V_0) - \mathcal{L}^*) / \sum_t \eta_t$.

### Key Designs

1. **Burer-Monteiro Stacking Reparameterization**:

    - Function: Transforms the non-convex, non-smooth LoRA problem involving two matrices $A$ and $B$ into a problem involving a single matrix $V$, where $BA$ appears in the upper-right block of $VV^T$. This enables the use of established tools for $VV^T$-form optimization.
    - Mechanism: Define $V = \begin{bmatrix} B \\ A^T \end{bmatrix} \in \mathbb{R}^{(m+n) \times r}$, then $VV^T = \begin{bmatrix} BB^T & BA \\ A^T B^T & A^T A \end{bmatrix}$. Extraction matrices $E_1 = [I_m, 0]$ and $E_2 = [0, I_n]^T$ are used to retrieve $BA = E_1 V V^T E_2$. Define $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$. By the chain rule, $\nabla \mathcal{J}(V) = 2\,\mathrm{Sym}(E_1^T \nabla\mathcal{L}(E_1 V V^T E_2) E_2^T) V$, where the multiplicative factor $V$ is the root of non-smoothness. LoRA synchronous gradient updates are equivalent to standard gradient descent on $V$.
    - Design Motivation: Under the $V$ perspective, several disparate phenomena have a unified explanation—$V = 0$ automatically becomes a stationary point (regardless of the original loss structure); the $V$ factor in the gradient yields small gradients for small $V$ ("flat region near the origin"); and the learning rate must decrease when the norm is large. These can be precisely quantified in the new coordinate system. The Burer-Monteiro form also allows the conclusions to generalize to general $VV^T$-type parameterizations.

2. **Modified "Lipschitz-like" Descent Lemma**:

    - Function: Provides a single-step descent inequality for the non-Lipschitz smooth function $\mathcal{J}$, containing a first-order term, three high-order terms, and a gradient-dependent term, proving that "if the learning rate is small enough, single-step descent is guaranteed."
    - Mechanism: Lemma 3.3 proves $$\mathcal{J}(V_2) \leq \mathcal{J}(V_1) + \langle \nabla\mathcal{J}(V_1), V_2 - V_1 \rangle_F + \sqrt{2}L\|V_2 - V_1\|^2 \|V_1\|^2 + \sqrt{2}L\|V_2 - V_1\|^3 \|V_1\| + \frac{\sqrt{2}L}{4}\|V_2 - V_1\|^4 + \|\nabla\mathcal{L}(E_1 V_1 V_1^T E_2)\|\|V_2 - V_1\|^2$$. Compared to the classical descent lemma, it includes three high-order terms ($\|V_1\|^2$, $\|V_1\|$, and $\|V_2 - V_1\|^4$ alone) and a gradient-dependent term, reflecting the constraints of the $V$ norm and the original gradient on the descent step size.
    - Design Motivation: Proving Lipschitz smoothness directly on $V$ is impossible due to the multiplicative factor $V$ in the gradient. However, by carefully expanding the Taylor series of $\mathcal{J}(V_2) - \mathcal{J}(V_1)$ and bounding high-order coefficients using the original loss's Lipschitz smoothness, a "weak descent condition with high-order corrections" is obtained. This is the key technique for reducing a non-Lipschitz problem to a manageable form.

3. **Position-dependent Adaptive Learning Rate and $O(1/\log T)$ Rate**:

    - Function: Automatically adjusts the learning rate based on the norm and gradient of the current iteration point to guarantee descent at every step and derives a non-asymptotic convergence rate by controlling the growth of $\|V_t\|$.
    - Mechanism: Lemma 3.4 chooses $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ so that the first-order term dominates the high-order terms, resulting in the descent $\mathcal{J}(V_{t+1}) \leq \mathcal{J}(V_t) - \frac{\eta_t}{4}\|\nabla\mathcal{J}(V_t)\|^2$. Summing over $t$ and using $\mathcal{J} \geq \mathcal{L}^*$, we get $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq \frac{4(\mathcal{J}(V_0) - \mathcal{L}^*)}{\sum_t \eta_t}$. The key is estimating $\sum_t \eta_t$—in the worst case $\|V_t\|^2 = O(t)$, making $\eta_t = \Omega(1/t)$. The harmonic series gives $\sum_t \eta_t = \Theta(\log T)$, leading to the $O(1/\log T)$ rate (Theorem 3.5). If $\|V_t\| \leq C$ is assumed additionally, $\sum_t \eta_t = \Theta(T)$, recovering $O(1/T)$.
    - Design Motivation: Theoretically, this rate reflects the "position-dependence" of LoRA—the learning rate must decrease as the iteration point moves away from the origin (increasing $\|V\|$), slowing down convergence; it can be aggressive near the origin. $V=0$ is an artificially created stationary point, so LoRA might converge to the origin (even if the original full-rank optimum is far away)—this is the theoretical root of why LoRA and full-rank training yield different solutions. In experiments, the authors designed three practical schedules ($\eta^{adapt}$, $\eta^{adapt2}$, $\eta^{norm}$) by translating this theoretical formula into deployable strategies.

### Loss & Training
The proof relies on only two assumptions: the original loss $\mathcal{L}$ is $L$-Lipschitz smooth and lower-bounded. The algorithm follows standard LoRA synchronous gradient descent: $A_{t+1} = A_t - \eta_t \nabla_A \mathcal{L}(B_t A_t)$, $B_{t+1} = B_t - \eta_t \nabla_B \mathcal{L}(B_t A_t)$. Theoretical results naturally extend to cases with multiple weight matrices (Lemma 3.6 proves that the block-constructed $\tilde{\mathcal{L}}$ is $2L$-Lipschitz smooth).

## Key Experimental Results

### Main Results
Experiments focus on theoretical validation rather than SOTA. Tasks: CIFAR-10 classification with three model scales—logistic regression on ResNet-18 embeddings (loss known to be Lipschitz smooth), direct ResNet-18 training (LoRA on convolutional layers, BatchNorm disabled), and TinyLlama-1.1B LoRA fine-tuning on Alpaca. Three learning rate schemes:

| LR Scheme | Formula |
|----------|------|
| Adaptive $\eta^{adapt}$ | $\alpha / (\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)$ |
| Adaptive $\eta^{adapt2}$ | $\alpha / (\|V_t\|^2 + \sqrt{\mathcal{L}(E_1 V_t V_t^T E_2)})$ |
| Normalized $\eta^{norm}$ | $\alpha / \|\nabla\mathcal{L}(V_t)\|^{1/2}$ |

| Experiment | Key Findings |
|------|---------|
| Logistic regression (rank 4 / 20) | All three non-constant LR schemes converge faster and more stably than constant LRs of similar magnitude; constant LR diverges if large and is slow if small; $\eta^{adapt}$ and $\eta^{adapt2}$ are highly correlated initially |
| ResNet-18 + LoRA on CIFAR-10 | $\eta^{adapt2}$ and $\eta^{norm}$ significantly stabilize training; $\eta^{norm}$ performs best; $\|V_t\|$ stops growing after finite steps |
| TinyLlama LoRA on Alpaca ($\sigma = 10^{-3}$) | With small initialization, adaptive LRs converge faster and more stably than constant LRs |
| TinyLlama LoRA on Alpaca ($\sigma = 1/r$) | With large initialization, adaptive LRs approach constant LRs (as $\|V_t\|$ is large and grows slowly), reducing the advantage |

### Ablation Study
Observations of long-term behavior in Logistic regression (1000 epochs):

| Phenomenon | Explanation |
|------|------|
| Loss appears to converge early | However, $\|V_t\|$ grows monotonically for all $t$ and remains unbounded |
| Asymptotic convergence is indeed slower than $O(1/T)$ | Validates the theory that "convergence slows to $O(1/\log T)$ as $\|V_t\| \to \infty$" |
| $\|V_t\|$ on ResNet-18 stops growing after finite steps | Falls into the bounded regime, with a rate of $O(1/T)$ |

### Key Findings
- **Precise correspondence between theory and experiment**: The observed chain of "growth of $\|V_t\|$ $\leftrightarrow$ decrease in LR $\leftrightarrow$ convergence slowdown" is derived directly from the proof. The three LR schemes are practical approximations of Equation (8).
- **LoRA has a flat region near initialization**: Since $V=0$ is a stationary point, standard initialization ($B=0$) traps the model in a low-gradient region, requiring a large initial learning rate to "escape." This explains why starting with high values for $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ before decaying stabilizes training.
- **Position-dependence is unique to LoRA**: Standard GD convergence does not depend on parameter norms, but LoRA experiences slowdown far from the origin and acceleration nearby due to the $V$ factor—this is a direct manifestation of how low-rank reparameterization alters the loss landscape geometry.
- **Reduced advantage of adaptive LR in high dimensions**: When $\|V_t\|$ is extremely large and changes slowly (as in high-dimensional LLMs), $\eta^{adapt}$ degrades to something close to a constant learning rate, appearing like $O(1/T)$, though the asymptotic behavior remains $O(1/\log T)$.
- **Convergence rate is independent of rank $r$**: $r$ does not directly enter the rate except indirectly through $\|V_0\|^2$, as gradient descent itself is dimension-free.

## Highlights & Insights
- **First non-asymptotic LoRA convergence proof to fully remove artificial bounded assumptions**: Previous non-asymptotic results relied on "assuming $\|A\|, \|B\|$ are uniformly bounded" to bypass Lipschitz smoothness failure. This work genuinely solves this core difficulty using $V$ reparameterization + modified descent lemma.
- **Geometric insight on $V=0$ as a permanent stationary point**: Reveals the counter-intuitive phenomenon that "LoRA may converge to the origin even if the original optimum is far away," theoretically explaining why LoRA and full-rank training yield different solutions.
- **Theory directly yields practical algorithms**: The three learning rate formulas $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ were derived directly from the choice of $\eta_t$ in Lemma 3.4 rather than being post-hoc heuristic. Each term has a clear theoretical meaning; such tight "theory $\to$ algorithm" correspondence is rare in optimization papers.
- **Unity of the Burer-Monteiro perspective**: The authors also proved that gradient descent for the Burer-Monteiro form $\min_V f(VV^T)$ converges to stationary points for any Lipschitz smooth function, embedding LoRA convergence theory into the broader framework of low-rank parameterization optimization.
- **$O(1/\log T)$ "position-dependent slowdown" is an intrinsic LoRA property**: It is not a symptom of a loose proof but reflects the actual behavior of LoRA's geometric structure, with corresponding slowdown observed in experiments.

## Limitations & Future Work
- **Limited to deterministic gradient descent**: Practical LoRA training uses SGD (stochastic batches). Stochastic noise requires additional tools like fourth-moment bounds for generalization, which the authors leave for future work.
- **No analysis of convex/strongly convex cases**: While classical GD yields $O(1/T)$ or $O((1-\mu/L)^T)$ in convex/strongly convex settings, LoRA reparameterization destroys original convexity. Whether faster convergence is possible remains unclear.
- **LR formula depends on original gradient $\nabla\mathcal{L}(E_1 V_t V_t^T E_2)$**: In practice, LoRA usually computes $xA^T B^T$ without explicitly constructing $BA$ to save memory. Calculating $\eta^{adapt}$ incurs high overhead; $\eta^{adapt2}$ mitigates this by using loss instead of gradient but still requires extra evaluations.
- **Lack of $O(1/\log T)$ lower bound**: The authors did not prove that this rate is tight—it could potentially be faster; the theory provides only an upper bound.
- **Common variants (QLoRA, ReLoRA, LoRA+) are not considered**: These variants are not covered by the current theoretical analysis.
- **Extended constants for multiple matrices are somewhat loose**: Lemma 3.6 shows $\tilde{\mathcal{L}}$ is $2L$-Lipschitz smooth rather than $L$; accumulated constants might not be optimal as scale increases.

## Related Work & Insights
- **vs Jiang 2024 / Ghiasvand 2025**: Both assume uniformly bounded norms for $A$ and $B$, essentially forcing the problem to be Lipschitz at the cost of proof novelty; this work removes that assumption and confronts non-Lipschitz difficulties directly.
- **vs NTK-like analysis (Jang 2024, Hayou 2024)**: Provides only asymptotic properties at the infinite-width limit; this work provides non-asymptotic rates for finite models.
- **vs RAC-LoRA / Bernoulli-LoRA**: These variants update only a single adapter to maintain Lipschitz smoothness, which does not represent actual LoRA; this work analyzes original synchronous LoRA directly.
- **vs GaLore / RSO / LDAdam**: These are LoRA-like low-rank memory optimization methods with different update rules; this work proves LoRA itself, providing a similar analysis paradigm despite not being directly transferable.
- **vs General Burer-Monteiro Theory**: This proof also covers convergence for BM-parameterized gradient descent, connecting with literature on BM optimization and semi-definite programming.
- **vs LoRA+ (Hayou 2024)**: LoRA+ proposes different learning rates for $A$ and $B$; this work's position-dependent LR is an independent approach, and the two could be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Truly solves the core difficulty of LoRA convergence theory; the proof technique ($V$ reparameterization + modified descent lemma) is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three levels of validation (Logistic regression + ResNet-18 + TinyLlama) are sufficient to support the theory given the primary goal isn't SOTA.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear three-step proof structure with well-explained motivations and natural transitions from theory to experiments.
- Value: ⭐⭐⭐⭐ High theoretical value (clean answer for LoRA convergence); moderate practical value (three LR schemes worth trying in practical fine-tuning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->
