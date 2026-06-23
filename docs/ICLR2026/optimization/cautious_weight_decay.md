---
title: >-
  [Paper Note] Cautious Weight Decay
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper proposes Cautious Weight Decay (CWD), a one-line, optimizer-agnostic modification: weight decay is applied only on coordinates where the "optimizer update direction" aligns with the "parameter sign." This preserves the original loss objective (no longer implicitly optimizing a regularized/constrained proxy o
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 1ce76514dfcccccf
---
# Cautious Weight Decay

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=Gwe6gbGng5](https://openreview.net/forum?id=Gwe6gbGng5)  
**Code**: To be confirmed  
**Area**: optimization  
**Keywords**: Weight decay, optimizer, implicit regularization, sliding mode dynamics, Lyapunov analysis

## TL;DR
This paper proposes Cautious Weight Decay (CWD), a one-line, optimizer-agnostic modification: weight decay is applied only on coordinates where the "optimizer update direction" aligns with the "parameter sign." This preserves the original loss objective (no longer implicitly optimizing a regularized/constrained proxy objective) and generates sliding mode dynamics upon reaching the stationary manifold, leading towards a local Pareto-optimal small-norm solution. Without adding new hyperparameters, CWD consistently reduces final loss and improves accuracy for language model pre-training and ImageNet across ADAMW / LION / MUON.

## Background & Motivation
**Background**: Modern Large Language Model (LLM) training almost exclusively employs decoupled weight decay (Loshchilov & Hutter 2019). This mechanism applies the decay term directly to the parameters, with the update rule expressed as $x_{t+1} = (1-\eta_t\lambda)x_t - \eta_t u_t$, where $u_t$ is the update vector constructed by the optimizer (often sign-normalized). SOTA optimizers such as ADAMW, LION, and MUON are built on this mechanism to stabilize training and improve generalization.

**Limitations of Prior Work**: Decoupled weight decay is completely insensitive to whether the update direction $u_t$ and the parameter $x_t$ are in the same direction. When $u_t$ and $x_t$ have the same sign at a specific coordinate, the decay pulls the parameter toward zero, which is beneficial for regularization. However, when they have opposite signs, the optimizer intends to push the parameter toward the optimum while the decay pulls it back toward zero in the opposite direction, **actively canceling out beneficial updates**.

**Key Challenge**: A deeper issue is that decoupled weight decay **implicitly rewrites the objective function**. The paper restates established conclusions: SGD with decoupled decay is equivalent to SGD on an $\ell_2$ regularized objective $f(x)+\tfrac{\lambda}{2}\|x\|_2^2$; LION-K converges to a stationary point of the regularized objective $f(x)+\tfrac1\lambda K^*(\lambda x)$, while LION / MUON correspond to **constrained optimization** within $\|x\|_\infty \le 1/\lambda$ and $\|X\|_{op}\le 1/\lambda$. Similarly, ADAMW approximates solving a box-constrained problem. In essence, while the user intends to minimize $f$, the optimizer actually minimizes a **$\lambda$-dependent proxy objective**, biasing the optimal solution.

**Goal**: Can the benefits of weight decay (regularization, faster training, smaller parameter norms) be retained while allowing the optimizer to truly minimize the **original** $f$ instead of a distorted proxy objective?

**Key Insight**: The authors observe that "harmful decay" only occurs on coordinates where $u_t$ and $x_t$ have opposite signs. By **applying decay only on coordinates with the same sign**, the parts that cancel beneficial updates are disabled while the regularizing parts are preserved.

**Core Idea**: Use a per-coordinate sign gate $\mathbb{I}(u_t \odot x_t \ge 0)$ to multiply the weight decay term—apply decay if signs match, skip if they differ. This is implemented in a single line of code without introducing new hyperparameters.

## Method

### Overall Architecture
CWD modifies the standard decoupled weight decay update rule to:

$$x_{t+1} = x_t - \eta_t\big(u_t + \lambda\,\mathbb{I}(u_t \odot x_t \ge 0)\odot x_t\big),$$

where $\odot$ denotes element-wise multiplication and $\mathbb{I}(\cdot)$ is a coordinate-wise indicator function (1 if signs match, 0 otherwise). Compared to the standard form $x_{t+1}=x_t-\eta_t(u_t+\lambda x_t)$, the only difference is the sign gate applied to the decay term $\lambda x_t$. Since $u_t$ can be the update vector from any optimizer (e.g., $D_t^{-1}\hat m_t$ for ADAMW, $-\nabla K(\tilde m_t)$ for LION-K), CWD is an **optimizer-agnostic** drop-in modification.

This modification introduces two qualitative shifts: (1) **Unbiased Optimization**: So long as the base optimizer (without decay) converges, every accumulation point $x^\star$ of CWD satisfies $\nabla f(x^\star)=0$, meaning it converges to a stationary point of the **original loss** rather than a regularized proxy. (2) **Sliding Mode Dynamics**: Upon reaching the stationary manifold, the optimizer generates sliding dynamics along the manifold to minimize the parameter norm, eventually stopping at a local Pareto optimum.

### Key Designs

**1. Selective Sign Gating: Decay Only on Same-Sign Coordinates**

This constitutes the entire mechanism of CWD. Standard decay $\lambda x_t$ indiscriminately pulls every parameter toward zero, opposing beneficial updates when $u_t$ and $x_t$ have opposite signs. CWD adds a per-coordinate switch: decay is applied when $u_t$ and $x_t$ have the same sign ($u_t x_t \ge 0$, where decay aligns with the update direction), and disabled (set to zero) when signs differ (where decay would oppose the optimizer).

This is effective because the gating ensures decay "only helps and never hinders"—it operates only when it does not conflict with the primary objective. Note on sign conventions: for ADAMW/SGD, this is written as $\mathbb{I}(u_t x_t \ge 0)$, whereas for LION-K, due to different direction conventions for $u_t=\nabla K(m_t)$, the gate is $\mathbb{I}(m_t x_t \le 0)$. The essence remains the same: apply decay only if its direction aligns with the update direction.

**2. Lyapunov Proof of Unbiasedness: CWD Does Not Rewrite the Loss Landscape**

To demonstrate that "CWD minimizes the original $f$ and not a proxy," the authors analyze continuous-time dynamics using Lyapunov functions. Taking SGD+CWD as an example, the ODE is $\dot x_t = -\nabla f(x_t) - \lambda\,\mathbb{I}(\nabla f(x_t)x_t\ge 0)x_t$. Using $H(x)=f(x)$ as the Lyapunov function:

$$\frac{dH}{dt} = -\|\nabla f(x_t)\|_2^2 - \lambda\big\|(\nabla f(x_t)x_t)_+\big\|_1 \le 0,$$

where $(\cdot)_+=\max(0,\cdot)$. Crucially, the second term is **non-positive**: the gate ensures only same-sign coordinates contribute to decay, and this contribution is always non-negative before being subtracted. According to LaSalle's Invariance Principle, the accumulation points of the trajectory lie in the set $\{x\mid \nabla f(x)=0\}$—the stationary points of the **original loss**. This is the fundamental difference from standard decay, where $H=f$ is no longer monotonic (as decay terms can contribute negatively), causing convergence to a regularized proxy. The authors extend this to SGDM, LION-K, and ADAM (Table 1 provides respective Lyapunov functions, e.g., $H(x,m)=\beta f(x)+\tfrac12\|m\|^2+\lambda\|(mx)_+\|_1$ for SGDM+CWD).

**3. Sliding Mode Dynamics: Decay as a "Norm Reduction" Sub-objective Along the Manifold**

Unbiasedness only shows CWD does not deviate from the stationary manifold. What distinguishes it from "no decay at all"? The difference appears after entering the manifold. Without decay, momentum $m$ decays to zero, and dynamics stop at an arbitrary point on the manifold. With CWD, once the stationary manifold $M=\{x\mid\nabla f(x)=0\}$ is reached, the residual dynamics become:

$$\dot x_t = -\lambda\, s_t \odot x_t,\quad s_t\in[0,1]^d,$$

where $s_t$ represents the selectors of the indicator function on the switching surface $\{[\nabla f]_i=0\}$ in the sense of Filippov. Intuitively, the decay term no longer affects the loss (already at a stationary point) but instead **slides along the manifold, shrinking parameters coordinate-wise toward zero** until they cannot be further reduced across all coordinates simultaneously. This is the sliding mode: the trajectory is constrained to the manifold but continues to move. The endpoint is the **local Pareto front** $P$ of the manifold—points that cannot be dominated given the partial order of "every coordinate being smaller." In other words, among all equivalent zero-gradient solutions, CWD prefers those with smaller norms.

### Loss & Training
CWD does not change the training objective, only a single line in the optimizer's parameter update. In experiments, hyperparameters (batch size, learning rate, weight decay $\lambda$, warmup ratio) are grid-searched for the baselines (ADAMW/LION/MUON). Subsequently, **CWD directly reuses the tuned baseline settings without further tuning**—a critical validation of its "drop-in" utility.

## Key Experimental Results

### Main Results
For language modeling, Gemma-like Transformers (338M / 986M / 2B) were trained on C4 according to Chinchilla optimality (20 tokens/parameter). Additionally, over-training was performed on the OLMo codebase (OLMo-1B, 100B tokens, 100 TPP). Total experiments consumed approximately 20,000 H100 GPU hours. CWD consistently reduced final validation loss and improved downstream accuracy across all scales and optimizers.

ImageNet (300 epochs, standard augmentation) Validation Top-1 Accuracy (%):

| Model | Optimizer | Base | +CWD |
|------|--------|------|------|
| ViT-S/16 (22M) | ADAMW | 78.84 | 79.45 |
| ViT-S/16 | LION | 79.29 | 79.82 |
| ViT-S/16 | MUON | 79.35 | 79.91 |
| ResNet-50 (25.6M) | ADAMW | 76.30 | 76.68 |
| ViT-B/16 (86.6M) | ADAMW | 80.15 | 80.71 |
| ViT-B/16 | MUON | 80.83 | 81.04 |

OLMo-1B (100B tokens) downstream zero-shot accuracy (selected): ADAMW improved ARC-Easy from 0.50→0.53, PIQA 0.67→0.69, MMLU 0.23→0.25; MUON improved PIQA 0.68→0.71, ComQA 0.30→0.33. CWD generally provided a +1~3 percentage point Gain.

### Ablation Study
OLMo-1B (100B tokens) validation loss for different "selective decay" masking strategies (lower is better):

| Optimizer | Baseline (Std WD, Tuned λ) | Ours (Update Mask $\mathbb{I}(ux\ge0)$) | Random (Same Sparsity Mask) | Gradient ($\mathbb{I}(gx\ge0)$) | No WD (λ=0) |
|--------|------|------|--------|----------|-------|
| ADAMW | 2.65 | **2.56** | 2.82 | 2.75 | 2.70 |
| MUON | 2.51 | **2.42** | 2.73 | 2.74 | 2.62 |

### Key Findings
- **Structured Decay vs. Less Decay**: Replacing the CWD mask with a random Bernoulli mask of equal sparsity significantly degraded loss (ADAMW 2.56→2.82, MUON 2.42→2.73). This indicates that simply reducing decay frequency is ineffective; the key is **selection by sign**.
- **Update direction $u$ vs. Gradient direction $g$**: Replacing $u$ with the raw gradient $g$ in the gating ($\mathbb{I}(gx\ge0)$) also performed worse, showing that alignment should be with the **optimizer's actual update direction**.
- **Regularization remains necessary**: $\lambda=0$ (no decay) was still inferior to tuned weight decay. CWD represents "using regularization more selectively" rather than "disabling regularization." CWD maintained lower loss throughout training and achieved an intermediate final parameter norm ($\lambda=0$ had the largest norm and earliest convergence stagnation; standard ADAMW had the smallest norm).
- **Scale Stability**: From 111M to 2B parameters, the loss advantage of CWD over ADAMW remained stable or slightly expanded, while CWD exhibited lower RMS normalized gradient norms.
- **Instruction Tuning (vs. SPD)**: When fine-tuning TinyLlama / Mistral-7B on Alpaca GPT-4, element-wise CWD matched or exceeded SPD and the "inner-product version" of CWD across most MMLU/AGIEval/WinoGrande metrics.

## Highlights & Insights
- **Rewriting Optimizer Objectives with One Line**: Replacing $\lambda x_t$ with $\lambda\mathbb{I}(ux\ge0)x_t$ is a prime example of "minimal form, profound semantics," pulling the implicit regularization proxy back to the original loss.
- **Weight Decay as a "Sub-objective"**: Standard decay rewrites the function you intend to minimize. CWD relegates decay to a secondary role, performing "norm reduction" only when it doesn't interfere with the primary objective, precisely characterized via Pareto optimality. This perspective can be migrated to other regularizers.
- **Theory-Engineering Alignment**: The continuous-time analysis (Lyapunov + LaSalle + Filippov) explains the differences in toy trajectories and predicted the "smaller norms / lower gradient norms" observed in large-scale LLM experiments.
- **Zero-tuning Migration**: Reusing the baseline $\lambda$ directly yields gains, and the optimal $\lambda^\star$ remains largely unchanged, making the deployment cost virtually zero.

## Limitations & Future Work
- The sliding mode endpoint $P$ is generally not a single point; the final Pareto point reached **depends on initialization and discretization**, meaning the limit point is theoretically non-unique and difficult to control.
- The unbiasedness analysis for ADAMW lacks a rigorous Lyapunov function (the paper admits ADAMW results in an inability to establish formal convergence using this framework).
- Gains are relatively modest (a few percentage points in loss, +0.3~1% in accuracy)—it is a "free lunch" rather than a disruptive leap. Stability at scales larger than 2B or non-Transformer architectures requires further validation.
- While the paper correlates smaller parameter norms with better generalization (via lower gradient norms and loss), it does not provide a formal causal explanation within generalization theory.

## Related Work & Insights
- **vs. Standard Decoupled Weight Decay (ADAMW / LION-K / MUON)**: These implicitly optimize a $\lambda$-dependent proxy. CWD maintains an unbiased original loss through sign gating, achieving lower loss with the same $\lambda$.
- **vs. SPD (Tian et al. 2024)**: Both share the "selective/structured decay" philosophy, but CWD provides a theoretical characterization via Lyapunov/sliding modes and matches or outperforms SPD in instruction tuning.
- **vs. No Decay (λ=0)**: Ablations show that λ=0 drops fast mid-training but stagnates early with higher final loss and the largest norms.
- **vs. Random/Gradient Masks**: Random masks or raw gradient masks are significantly worse, highlighting that the "per-coordinate selection by optimizer update direction" is the specific source of gain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extremely simple code change with a completely new and self-consistent theoretical explanation using sliding modes/Pareto/Lyapunov.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers LLM pre-training, ImageNet, and instruction tuning across multiple scales (111M-7B) and optimizers with thorough ablation; however, gains at scales >2B are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic from motivation to mechanism to theory and experiments is clear; toy examples correlate perfectly with LLM results.
- Value: ⭐⭐⭐⭐⭐ Zero-tuning, drop-in, optimizer-agnostic, and easily deployable in existing training stacks; highly practical for LLM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_µp_for_learning_rate_transfer_in_practice.md)
- [\[ICLR 2026\] Cautious Optimizers: Improving Training with One Line of Code](cautious_optimizers_improving_training_with_one_line_of_code.md)
- [\[ICLR 2026\] Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay](incorporating_expert_priors_into_bayesian_optimization_via_dynamic_mean_decay.md)
- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](../../ICML2026/optimization/limits_of_convergence-rate_control_for_open-weight_safety.md)

</div>

<!-- RELATED:END -->
