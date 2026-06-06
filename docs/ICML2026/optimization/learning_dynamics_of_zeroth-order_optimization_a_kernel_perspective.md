---
title: >-
  [Paper Note] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective
description: >-
  [ICML 2026][Optimization][Zeroth-order optimization] This paper uses empirical NTK as a unified perspective to prove that the eNTK induced by zeroth-order SGD is equivalent to projecting the first-order eNTK onto a rando…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Zeroth-order optimization"
  - "eNTK"
  - "Johnson-Lindenstrauss"
  - "perturbation count"
  - "dimension-independence"
date: 2026-05-08
content_hash: 1032acbcadfeaa1d
---

# Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective

**Conference**: ICML 2026  
**arXiv**: [2605.03373](https://arxiv.org/abs/2605.03373)  
**Code**: Not mentioned  
**Area**: Optimization Theory / LLM Fine-tuning / Learning Dynamics  
**Keywords**: Zeroth-order optimization, eNTK, Johnson-Lindenstrauss, perturbation count, dimension-independence

## TL;DR
This paper uses empirical NTK as a unified perspective to prove that the eNTK induced by zeroth-order SGD is equivalent to projecting the first-order eNTK onto a random subspace spanned by perturbations. This, via the Johnson-Lindenstrauss lemma, explains why ZO methods still work on billion-parameter LLMs: the error depends only on output dimension $V$ and perturbation count $P$, and is independent of model dimension $d$.

## Background & Motivation
**Background**: Zeroth-order (ZO) optimization estimates gradients using only function value differences. Due to its memory efficiency and black-box attack capability, it has recently been widely used for LLM fine-tuning (MeZO series, ZO-LoRA, etc.).

**Limitations of Prior Work**: Classical optimization theory (Ghadimi-Lan 2013, Nesterov-Spokoiny 2017, Shamir 2017) consistently predicts that ZO convergence rate slows linearly with model dimension $d$, and the variance of single-perturbation estimators is also proportional to $d$. By this logic, ZO should be unusably slow on billion-parameter LLMs; however, experiments such as MeZO show ZO can approach SGD performance on OPT-13B. **Theory and experiment are completely inconsistent.**

**Key Challenge**: Viewing learning as "loss value" (a scalar) misses what ZO truly affects—the loss descent rate does depend on $d$, but the model's prediction changes on specific samples (learning dynamics) may not. The "low effective rank" hypothesis of Malladi et al. 2023 offers an explanation, but cannot be computed or verified on LLMs.

**Goal**: (1) Find an "intermediate quantity" that characterizes both ZO and FO; (2) Prove that its difference depends only on $P$ and $V$, not $d$.

**Key Insight**: Shift the perspective from loss function to function space, leveraging Jacot et al. 2018's eNTK; ZO updates can be seen as applying a low-rank random projection $U_{t,P} U_{t,P}^\top$ to the FO eNTK. This is equivalent to the inner product preservation version of the Johnson-Lindenstrauss lemma, which states that the projection dimension only needs to be $\mathcal{O}(\ln n / \epsilon^2)$, independent of the original dimension.

**Core Idea**: **ZO-eNTK is a random projection of FO-eNTK; the JL lemma guarantees that as long as the perturbation count $P$ matches the output dimension $V$, the difference in learning dynamics between ZO and FO is independent of model dimension $d$.**

## Method

### Overall Architecture
The paper is purely theoretical with no new algorithms. The core pipeline: (1) Derive the change in log-probability after one ZO-SGD update, explicitly expressing the difference between FO and ZO as "FO eNTK − projected eNTK" multiplied by two model-dependent matrices; (2) Apply the JL lemma to the projected kernel difference; (3) Compare Gaussian/Rademacher perturbations from both optimization (variance + convergence rate) and eNTK (projection error) perspectives; (4) Discuss reasonable scales for $P$; (5) Validate with experiments on LeNet/MNIST, OPT-125M / 1.3B, Mistral-7B, etc.

### Key Designs

1. **One-step learning dynamics and eNTK equivalence**:

    - Function: Explicitly expresses the impact of a one-step ZO-SGD parameter update on the model's log-probability at another data point $\mathbf{x}_o$.
    - Mechanism: Perform a first-order Taylor expansion of $\Delta \log \pi_t(y \mid \mathbf{x}_o)$ and substitute the ZO-SGD update rule to obtain $\Delta \log \pi \approx -\eta \mathcal{A}_t(\mathbf{x}_o) \mathcal{K}_t(\mathbf{x}_o, \mathbf{x}_u; U_{t,P}) \mathcal{G}_t(\mathbf{x}_u, \mathbf{y}_u)$, where the projected kernel $\mathcal{K}_t(\mathbf{x}_o, \mathbf{x}_u; U_{t,P}) = \nabla_\theta z(\mathbf{x}_o)^\top U_{t,P} U_{t,P}^\top \nabla_\theta z(\mathbf{x}_u)$; for FO, $U_{t,P} U_{t,P}^\top$ is replaced by the identity matrix. The difference is clear: ZO introduces a random projection matrix $U_{t,P} \in \mathbb{R}^{d \times P}$ formed by perturbations.
    - Design Motivation: This "only differs by a projection matrix" equivalence directly connects to the JL lemma, making the "dimension-independence" proof a straightforward JL corollary.

2. **Johnson-Lindenstrauss projection bound**:

    - Function: Controls the ZO vs FO kernel difference $\|\Delta \mathcal{K}\|_F$ as a function of $\epsilon$, and explicitly provides a dimension-free bound.
    - Mechanism: Express $\Delta\mathcal{K}[i,j]$ as the difference between the original inner product $\langle \nabla_\theta z_i(\mathbf{x}_o), \nabla_\theta z_j(\mathbf{x}_u)\rangle$ and the projected inner product $\langle U_{t,P}^\top \nabla_\theta z_i, U_{t,P}^\top \nabla_\theta z_j\rangle$; the JL lemma guarantees that as long as $P \geq (2\ln n + \ln(1/\delta))/(c(\mathcal{Q})\epsilon^2)$, all inner products are preserved within $1 \pm \epsilon$ after projection, where $c(\mathcal{Q})$ is the concentration constant of the distribution. Substituting into the Frobenius norm of the kernel difference yields $\|\Delta\mathcal{K}\|_F^2 \leq \frac{\epsilon^2 V}{2}(\|\nabla_\theta z(\mathbf{x}_o)\|_F^2 + \|\nabla_\theta z(\mathbf{x}_u)\|_F^2)^2$, **with no $d$ on the right**.
    - Design Motivation: This is precisely the "dimension-free" result sought—so long as $V$ (vocabulary/class count) does not explode, scaling from LeNet to LLaMA does not cause ZO and FO learning trajectories to diverge significantly.

3. **Gaussian vs Rademacher perturbation comparison**:

    - Function: Explains "why in practice, binary Rademacher often performs as well as Gaussian".
    - Mechanism: From the optimization perspective, the second moment of the single-perturbation estimator is $(d+2)\|\nabla\ell\|^2$ for Gaussian and $d\|\nabla\ell\|^2$ for Rademacher—**both proportional to $d$**, consistent with the traditional "inefficient in high dimensions" view; but from the eNTK perspective, both have JL concentration constants $\approx 1/4$, **and the bound does not depend on $d$**. Thus, the two are nearly identical in "projection quality", termed "distribution robustness"—the fidelity of ZO is determined by $P$, not the distribution.
    - Design Motivation: Bridges the gap between engineering experience (Rademacher and Gaussian both work well) and theoretical intuition (difference should scale with $d$); also provides theoretical support for more aggressive binary/ternary perturbations in the future.

### Loss & Training
No new training strategies; the theoretical section provides the optimization convergence rate for ZO-SGD as $\mathcal{O}(\sqrt{dL/(PT)})$ when learning rate $\eta = \mathcal{O}(\sqrt{P/(dLT)})$ (still contains $d$), contrasting with the dimension-free bound from the eNTK perspective, reminding readers that "convergence rate vs learning trajectory similarity" are distinct concepts.

## Key Experimental Results

### Main Results
The authors validate the theory with three experimental setups:

| Setting | Model | Data | Observation |
|---|---|---|---|
| ZO vs FO eNTK Frobenius error | LeNet ($d{=}29{,}624$) | MNIST | For high semantic similarity pairs (4,9), error at $P{=}125$ is $\approx 0.338$; for low similarity pairs (0,1), significant residual remains even at $P{=}125$ |
| Gaussian vs Rademacher | LeNet | MNIST | Frobenius / CKA / Wasserstein curves are nearly identical |
| Large model ZO trajectory | OPT-125M → OPT-1.3B | SST-2 | As $P$ increases, ZO trajectories of different model sizes approach FO at similar rates, confirming "dimension-independence" |

### Ablation Study

| Factor | Impact |
|---|---|
| Perturbation count $P$ | Error decays as $\mathcal{O}(\sqrt{\ln V / P})$, matching JL theory |
| Perturbation distribution (Gaussian vs Rademacher) | Almost no effect, confirming distribution robustness |
| Input pair similarity | ZO converges faster for high similarity pairs; low similarity pairs require more $P$ |
| Model dimension $d$ (OPT-125M → 1.3B) | For the same $P$, ZO trajectory deviation from FO is similar for both models |

### Key Findings
- Validates that "perturbation count $P$ is the key, not $d$"—the first kernel-level explanation for a long-standing engineering observation.
- "Sample pair similarity determines convergence speed" is a new insight: ZO estimators are better at fine-grained distinctions for semantically similar inputs, but approximate poorly for structurally dissimilar inputs.
- Classical optimization bound $\mathcal{O}(\sqrt{dL/(PT)})$ and kernel bound $\mathcal{O}(\sqrt{\ln V / P})$ coexist: loss descent speed still depends on $d$, but similarity of model prediction trajectories does not.

## Highlights & Insights
- Proposes a new framework for analyzing ZO optimization from the function-space perspective; previous analyses were stuck in parameter space and limited by $d$.
- Uses the JL lemma to shift projection dependence from model parameter dimension $d$ to output space dimension $V$, exemplifying theory driven by engineering phenomena (MeZO working on LLMs).
- Incidentally explains why Rademacher and Gaussian perturbations are nearly equivalent—previously only an empirical observation.

## Limitations & Future Work
- The analysis is a one-step + small step-size local approximation, not covering cumulative error over the full training trajectory.
- The cost of "dimension-independence" is introducing $V$ (output dimension); for modern LLMs with vocabulary $V \sim 10^5$, the $V$ factor in the bound is not small—whether it can be tightened remains open.
- No practical guideline for "how large $P$ should be"; only states "$P$ just needs to be large enough".
- Does not discuss the relationship between $d_{\text{eff}}$ and $P$ in LoRA/partial parameter perturbation scenarios.

## Related Work & Insights
- **vs Malladi et al. 2023b (MeZO's low effective rank hypothesis)**: MeZO explains dimension-independence via low-rank Hessian, but this cannot be computed/verified on LLMs; this paper does not rely on low-rank assumptions and gives a strict bound via JL.
- **vs Spall / Nesterov classical ZO analysis**: Their frameworks focus on optimization convergence rate (involving $d$); this paper uses the eNTK perspective to provide a $d$-decoupled metric.
- **vs Achlioptas 2003 (sparse JL projection)**: This paper uses the inner product preservation version of JL, not the distance preservation version, which better fits eNTK.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The equivalence "ZO eNTK = random projection of FO eNTK" is a very elegant observation
- Experimental Thoroughness: ⭐⭐⭐ Experiments are mainly for validation, not full LLM fine-tuning
- Writing Quality: ⭐⭐⭐⭐ Clear derivation chain, formulas (6)(8)(17) are tightly connected
- Value: ⭐⭐⭐⭐⭐ Provides the first trick-free, dimension-free explanation for ZO in LLM fine-tuning; theoretical framework is easily extensible

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[ICML 2026\] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)

</div>

<!-- RELATED:END -->
