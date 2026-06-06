---
title: >-
  [Paper Note] On the Convergence Rate of LoRA Gradient Descent
description: >-
  [ICML 2026][Optimization][LoRA] This work is the first to prove that original LoRA gradient descent achieves a minimum gradient norm convergence rate of $O(1/\log T)$ without assuming bounded adapter matrices or requirin…
tags:
  - "ICML 2026"
  - "Optimization"
  - "LoRA"
  - "convergence analysis"
  - "non-Lipschitz smoothness"
  - "Burer-Monteiro"
  - "adaptive learning rate"
date: 2026-05-08
content_hash: 600d30029fe317dc
---

# On the Convergence Rate of LoRA Gradient Descent

**Conference**: ICML 2026  
**arXiv**: [2512.18248](https://arxiv.org/abs/2512.18248)  
**Code**: https://github.com/siqiaomu/lora  
**Area**: Optimization Theory / Efficient LLM Fine-tuning / LoRA  
**Keywords**: LoRA, convergence analysis, non-Lipschitz smoothness, Burer-Monteiro, adaptive learning rate

## TL;DR
This work is the first to prove that original LoRA gradient descent achieves a minimum gradient norm convergence rate of $O(1/\log T)$ without assuming bounded adapter matrices or requiring the reparameterized loss to be Lipschitz smooth (recovers the classical $O(1/T)$ rate if parameter norms are bounded). Based on this, strictly theoretically-motivated adaptive/normalized learning rates are designed and empirically validated for accelerated and stabilized training on logistic regression, ResNet-18, and TinyLlama.

## Background & Motivation

**Background**: LoRA (Low-Rank Adaptation) has become the most popular approach for LLM fine-tuning—freeze pretrained weights $W_0$, train only two small matrices $A, B$ so that the new weights are $W_0 + BA$. The algorithm is extremely simple: at each step, perform gradient descent on both $A$ and $B$.

**Limitations of Prior Work**: Despite LoRA's simplicity and empirical effectiveness, its convergence theory has been paradoxical—even if the original loss $\mathcal{L}(W)$ is Lipschitz smooth, the reparameterized $\mathcal{L}(BA)$ with respect to $A, B$ is no longer Lipschitz smooth (since $\nabla_B \mathcal{L}(BA)$ contains a multiplicative $A$ factor), making the classical "descent lemma yields $O(1/T)$" analysis inapplicable.

**Key Challenge**: Existing LoRA theoretical analyses fall into three categories, all sidestepping this core difficulty—(1) **Infinite Regularization Limit** (Kim 2025, Jang 2024, NTK analyses, etc.): only provide asymptotic convergence or infinite-width neural network results, not non-asymptotic rates for finite models; (2) **LoRA Variants** (GaLore, RAC-LoRA, etc.): update only a single adapter or add projections to maintain Lipschitz smoothness, but do not correspond to practical LoRA deployments; (3) **Convergence under Strong Assumptions** (Jiang 2024, Ghiasvand 2025): assume the norms of $A, B$ are uniformly bounded by some constant, effectively imposing Lipschitz smoothness and hiding the constant in the convergence bound—no substantial new proof techniques.

**Goal**: Under the weakest possible assumptions (only original loss is Lipschitz smooth and lower bounded), provide a **non-asymptotic convergence rate** for original LoRA synchronous gradient descent, without assuming bounded norms for $A, B$.

**Key Insight**: Stack $A, B$ into a single matrix $V$, so $BA$ appears as a specific block in $VV^T$—this is the classic Burer-Monteiro symmetric parameterization. From the $V$ perspective, LoRA gradient descent becomes standard gradient descent on $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$, reducing the problem to "non-smooth optimization in $VV^T$ form," allowing the use of refined, corrected descent lemmas.

**Core Idea**: Stack reparameterization → derive a "Lipschitz-like" descent lemma with higher-order terms → by setting the learning rate inversely proportional to $\|V_t\|^2$ and the current gradient, guarantee descent at each step; analyze $\|V_t\|^2 = O(t)$ growth, so $\sum \eta_t = \Theta(\log T)$, yielding $O(1/\log T)$ convergence.

## Method

### Overall Architecture
The proof proceeds in three steps: (1) **Problem Reformulation**—stack LoRA's $A, B$ into $V = [B; A^T] \in \mathbb{R}^{(m+n) \times r}$, so the original loss becomes $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$, where $E_1, E_2$ are extraction matrices; LoRA synchronous gradient descent is equivalent to standard gradient descent on $V$. (2) **Corrected Descent Lemma**—prove that $\mathcal{J}$ satisfies a "Lipschitz-like" inequality with higher-order terms in $\|V_2 - V_1\|^k$ ($k=2,3,4$) (Lemma 3.3). (3) **Learning Rate Control + Convergence**—choose $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ to guarantee at least $\eta_t \|\nabla\mathcal{J}(V_t)\|^2 / 4$ descent per step (Lemma 3.4); combine with the worst-case estimate $\|V_t\|^2 = O(t)$ to obtain $\sum_t \eta_t = \Theta(\log T)$, and finally $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq 4(\mathcal{J}(V_0) - \mathcal{L}^*) / \sum_t \eta_t$, yielding $O(1/\log T)$.

### Key Designs

1. **Burer-Monteiro Stacked Reparameterization**:

    - **Function**: Transforms LoRA's non-convex, non-smooth problem in two matrices $A, B$ into a problem in a single matrix $V$, with $BA$ appearing in the upper-right block of $VV^T$, enabling the use of established tools for $VV^T$-form optimization.
    - **Mechanism**: Define $V = \begin{bmatrix} B \\ A^T \end{bmatrix} \in \mathbb{R}^{(m+n) \times r}$, so $VV^T = \begin{bmatrix} BB^T & BA \\ A^T B^T & A^T A \end{bmatrix}$; use extraction matrices $E_1 = [I_m, 0]$, $E_2 = [0, I_n]^T$ to extract $BA = E_1 V V^T E_2$. Define $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$; by the chain rule, $\nabla \mathcal{J}(V) = 2\,\mathrm{Sym}(E_1^T \nabla\mathcal{L}(E_1 V V^T E_2) E_2^T) V$, where the multiplicative $V$ factor is the source of non-smoothness. LoRA synchronous gradient updates are equivalent to standard gradient descent on $V$.
    - **Design Motivation**: From the $V$ perspective, several seemingly disparate phenomena are unified—$V = 0$ is always a stationary point (regardless of the original loss structure), the $V$ factor in the gradient causes small gradients near the origin ("flat region near zero"), and the learning rate must decrease as the norm grows; all these can be precisely quantified in the new coordinate system. The Burer-Monteiro form also allows the conclusions to generalize to arbitrary $VV^T$-type parameterizations.

2. **Corrected "Lipschitz-like" Descent Lemma**:

    - **Function**: Provides a one-step descent inequality for the non-Lipschitz smooth function $\mathcal{J}$, including first-order, three higher-order, and one gradient-dependent term, ensuring descent if the learning rate is sufficiently small.
    - **Mechanism**: Lemma 3.3 proves $\mathcal{J}(V_2) \leq \mathcal{J}(V_1) + \langle \nabla\mathcal{J}(V_1), V_2 - V_1 \rangle_F + \sqrt{2}L\|V_2 - V_1\|^2 \|V_1\|^2 + \sqrt{2}L\|V_2 - V_1\|^3 \|V_1\| + \frac{\sqrt{2}L}{4}\|V_2 - V_1\|^4 + \|\nabla\mathcal{L}(E_1 V_1 V_1^T E_2)\|\|V_2 - V_1\|^2$. Compared to the classical descent lemma, there are three additional higher-order terms ($\|V_1\|^2$, $\|V_1\|$, and $\|V_2 - V_1\|^4$) and a gradient-dependent term, reflecting the constraints imposed by the $V$ norm and the original gradient on the descent step.
    - **Design Motivation**: Directly proving Lipschitz smoothness for $\mathcal{J}$ in $V$ is impossible (the gradient has a multiplicative $V$ factor), but by carefully expanding the Taylor form of $\mathcal{J}(V_2) - \mathcal{J}(V_1)$ and bounding higher-order coefficients using the original loss's Lipschitz smoothness, a "higher-order corrected" weak descent condition is obtained. This is the key technique for reducing the non-Lipschitz problem to a controllable form.

3. **Position-dependent Adaptive Learning Rate and $O(1/\log T)$ Rate**:

    - **Function**: Automatically adjusts the learning rate based on the current iterate's norm and gradient, ensuring descent at each step and deriving a non-asymptotic convergence rate by controlling the growth of $\|V_t\|$.
    - **Mechanism**: Lemma 3.4 chooses $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ so that higher-order terms are dominated by the first-order term, yielding one-step descent $\mathcal{J}(V_{t+1}) \leq \mathcal{J}(V_t) - \frac{\eta_t}{4}\|\nabla\mathcal{J}(V_t)\|^2$. Summing over $t$ and using $\mathcal{J} \geq \mathcal{L}^*$ gives $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq \frac{4(\mathcal{J}(V_0) - \mathcal{L}^*)}{\sum_t \eta_t}$. The key is estimating $\sum_t \eta_t$—in the worst case, $\|V_t\|^2 = O(t)$, so $\eta_t = \Omega(1/t)$, and the harmonic series gives $\sum_t \eta_t = \Theta(\log T)$, yielding the $O(1/\log T)$ rate (Theorem 3.5). If $\|V_t\| \leq C$ is additionally assumed, then $\sum_t \eta_t = \Theta(T)$, recovering the classical $O(1/T)$ rate.
    - **Design Motivation**: Theoretically, this rate reflects LoRA's "position dependence"—when the iterate is far from the origin ($\|V\|$ large), the learning rate must decrease, slowing convergence; near the origin, it can be aggressive. $V = 0$ is an artificially created stationary point, so LoRA can converge to the origin (even if the original full-rank optimum is far away)—this is the theoretical root of LoRA and full-rank training yielding different solutions. In experiments, the authors design three practical schedules, $\eta^{adapt}$, $\eta^{adapt2}$, and $\eta^{norm}$, directly translating the theory into deployable learning rate strategies.

### Loss & Training
The proof uses only two assumptions: the original loss $\mathcal{L}$ is $L$-Lipschitz smooth and lower bounded. The algorithm is standard LoRA synchronous gradient descent: $A_{t+1} = A_t - \eta_t \nabla_A \mathcal{L}(B_t A_t)$, $B_{t+1} = B_t - \eta_t \nabla_B \mathcal{L}(B_t A_t)$. The theoretical results naturally extend to the multi-weight-matrix case (Lemma 3.6 proves that the block-constructed $\tilde{\mathcal{L}}$ is $2L$-Lipschitz smooth).

## Key Experimental Results

### Main Results
The experiments aim not for SOTA but to validate the theory. Task: CIFAR-10 classification, with three model tiers—logistic regression on ResNet-18 embeddings (loss is known to be Lipschitz smooth), direct training of ResNet-18 (LoRA added to convolutional layers, BatchNorm off), and LoRA fine-tuning of TinyLlama-1.1B on Alpaca. Three learning rate schemes:

| Learning Rate Scheme | Formula |
|---------------------|---------|
| Adaptive $\eta^{adapt}$ | $\alpha / (\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)$ |
| Adaptive $\eta^{adapt2}$ | $\alpha / (\|V_t\|^2 + \sqrt{\mathcal{L}(E_1 V_t V_t^T E_2)})$ |
| Normalized $\eta^{norm}$ | $\alpha / \|\nabla\mathcal{L}(V_t)\|^{1/2}$ |

| Experiment | Key Observations |
|------------|-----------------|
| Logistic regression (rank 4 / 20) | All three non-constant learning rates converge faster and more stably than constant lr of the same scale; large constant lr diverges, small is slow; $\eta^{adapt}$ and $\eta^{adapt2}$ are highly correlated in early stages |
| ResNet-18 + LoRA on CIFAR-10 | $\eta^{adapt2}$ and $\eta^{norm}$ significantly stabilize training, $\eta^{norm}$ performs best; $\|V_t\|$ stops growing after a finite number of steps |
| TinyLlama LoRA on Alpaca ($\sigma = 10^{-3}$) | With small initialization, adaptive learning rates converge faster and more stably than constant lr of the same scale |
| TinyLlama LoRA on Alpaca ($\sigma = 1/r$) | With large initialization, adaptive learning rates approach constant lr (since $\|V_t\|$ is very large and grows slowly), advantage diminishes |

### Ablation Study
Long-term behavior of logistic regression trained for 1000 epochs:

| Phenomenon | Explanation |
|------------|-------------|
| Loss appears to converge early | But $\|V_t\|$ grows monotonically and unbounded for all $t$ |
| Asymptotic convergence is indeed slower than $O(1/T)$ | Verifies the theoretical "$\|V_t\| \to \infty$ leads to $O(1/\log T)$$" |
| On ResNet-18, $\|V_t\|$ stops growing after finite steps | Falls into the bounded case, rate is $O(1/T)$ |

### Key Findings
- **Precise correspondence between theory and experiment**: The observed "$\|V_t\|$ growth ↔ learning rate decay ↔ convergence slowdown" chain directly follows from the proof; the three learning rate schemes are practical approximations of formula (8).
- **LoRA is flat near initialization**: Since $V=0$ is a stationary point, standard initialization ($B=0$) traps the model in a low-gradient region, requiring a large initial learning rate to "escape," explaining why $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ start high and then decrease for stable training.
- **Position dependence is unique to LoRA**: Standard GD convergence rate does not depend on parameter norm, but LoRA's $V$ factor causes slowdown far from the origin and acceleration near it—this directly reflects how low-rank reparameterization alters the loss landscape geometry.
- **Adaptive advantage diminishes in high dimensions**: When $\|V_t\|$ is very large and changes slowly (high-dimensional LLMs), $\eta^{adapt}$ degenerates to near-constant lr, appearing as $O(1/T)$, but the asymptotic behavior remains $O(1/\log T)$.
- **Convergence rate is independent of rank $r$**: Except for indirect appearance via $\|V_0\|^2$, $r$ does not directly enter the rate—because gradient descent itself is dimension-free.

## Highlights & Insights
- **First non-asymptotic LoRA convergence proof without artificial boundedness assumptions**: All previous non-asymptotic results relied on "assume $\|A\|, \|B\|$ are uniformly bounded" to sidestep the failure of Lipschitz smoothness; this work uses $V$ reparameterization + corrected descent lemma to truly resolve the core difficulty.
- **Geometric insight that $V = 0$ is always a stationary point**: Reveals the counterintuitive phenomenon that "LoRA may converge to the origin (even if the original optimum is far away)," theoretically explaining why LoRA and full-rank training yield different solutions.
- **Theory directly yields practical algorithms**: The three learning rate formulas $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ are not ad hoc but directly derived from Lemma 3.4's $\eta_t$ selection, with each term having clear theoretical meaning; such tight "theory → algorithm" correspondence is rare in optimization papers.
- **Unifying Burer-Monteiro perspective**: The authors also prove that for general Lipschitz smooth functions, gradient descent on Burer-Monteiro form $\min_V f(VV^T)$ converges to stationary points, embedding LoRA convergence theory into the broader low-rank parameterization optimization framework.
- **$O(1/\log T)$ "position-dependent slowdown" is intrinsic to LoRA**: This is not a loose proof, but reflects the actual behavior dictated by LoRA's geometry, and the corresponding slowdown is observed in experiments.

## Limitations & Future Work
- **Covers only deterministic gradient descent**: Actual LoRA training uses SGD (mini-batches); stochastic noise requires additional tools such as fourth-moment bounds for extension, which the authors explicitly leave for future work.
- **No analysis for convex/strongly convex cases**: Classical GD achieves $O(1/T)$ or $O((1-\mu/L)^T)$ rates under convexity/strong convexity, but LoRA reparameterization destroys original convexity, so whether faster rates are possible remains unclear.
- **Learning rate formulas depend on original gradient $\nabla\mathcal{L}(E_1 V_t V_t^T E_2)$**: In practice, LoRA implementations often compute $xA^T B^T$ for memory efficiency without explicitly constructing $BA$, so $\eta^{adapt}$ is costly; $\eta^{adapt2}$ mitigates this by using loss instead of gradient but still requires extra evaluation.
- **No lower bound for $O(1/\log T)$**: The authors do not prove this rate is tight—it may be faster in practice; the theory only provides an upper bound.
- **Does not cover common variants (QLoRA, ReLoRA, LoRA+)**: Theoretical analysis of these variants is not included, and is left for future work.
- **Constants in multi-matrix extension are somewhat loose**: Lemma 3.6 gives $\tilde{\mathcal{L}}$ as $2L$-Lipschitz smooth instead of $L$; as scale increases, constant accumulation may be suboptimal.

## Related Work & Insights
- **vs Jiang 2024 / Ghiasvand 2025**: Both assume uniformly bounded $A, B$ norms, essentially forcing Lipschitzness; proofs use no new techniques. This work removes this assumption and provides a proof that directly addresses the non-Lipschitz challenge.
- **vs NTK-type analyses (Jang 2024, Hayou 2024)**: Only provide asymptotic properties in the infinite-width limit; this work gives non-asymptotic rates for finite models.
- **vs RAC-LoRA / Bernoulli-LoRA**: These variants update only one adapter to maintain Lipschitz smoothness, not corresponding to actual LoRA; this work directly analyzes original synchronous LoRA.
- **vs GaLore / RSO / LDAdam**: These are LoRA-like low-rank memory optimization methods with different update rules; this work proves results for LoRA itself, but provides a similar analysis paradigm.
- **vs General Burer-Monteiro Theory**: This work's proof also covers convergence of gradient descent on BM parameterizations, connecting with BM optimization and semidefinite programming literature.
- **vs LoRA+ (Hayou 2024)**: LoRA+ proposes different learning rates for $A, B$; this work's position-dependent learning rate is an independent idea, and the two can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Truly resolves the core theoretical challenge of LoRA convergence; proof techniques ($V$ reparameterization + corrected descent lemma) are new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three levels of validation (logistic regression, ResNet-18, TinyLlama); not aiming for SOTA, but sufficient to support the theory.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear three-step proof structure, each step well-motivated, smooth transition from theory to experiment.
- Value: ⭐⭐⭐⭐ High theoretical value (clean answer for LoRA convergence), moderate practical value (three learning rate schemes worth trying in real fine-tuning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[ICLR 2026\] Convergence of Muon with Newton-Schulz](../../ICLR2026/optimization/convergence_of_muon_with_newton-schulz.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](../../ICLR2026/optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)
- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](../../CVPR2026/optimization/fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)

</div>

<!-- RELATED:END -->
