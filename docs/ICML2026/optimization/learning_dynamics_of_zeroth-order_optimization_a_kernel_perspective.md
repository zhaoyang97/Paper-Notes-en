---
title: >-
  [Paper Note] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective
description: >-
  [ICML 2026][Optimization][Zeroth-order optimization] This paper adopts empirical NTK (eNTK) as a unified perspective to prove that the eNTK induced by zeroth-order (ZO) SGD is equivalent to projecting the first-order (FO…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Zeroth-order optimization"
  - "eNTK"
  - "Johnson-Lindenstrauss"
  - "perturbation count"
  - "dimension independence"
date: 2026-05-08
content_hash: 3aa0469c5dcdb0e5
---

# Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective

**Conference**: ICML 2026  
**arXiv**: [2605.03373](https://arxiv.org/abs/2605.03373)  
**Code**: Not mentioned  
**Area**: Optimization Theory / LLM Fine-tuning / Learning Dynamics  
**Keywords**: Zeroth-order optimization, eNTK, Johnson-Lindenstrauss, perturbation count, dimension independence

## TL;DR
This paper adopts empirical NTK (eNTK) as a unified perspective to prove that the eNTK induced by zeroth-order (ZO) SGD is equivalent to projecting the first-order (FO) eNTK onto a random subspace spanned by perturbations. By utilizing the Johnson-Lindenstrauss Lemma, the authors explain why ZO methods remain effective for billion-parameter LLMs: the error depends only on the output dimension $V$ and the number of perturbations $P$, remaining independent of the model dimension $d$.

## Background & Motivation
**Background**: Zeroth-order (ZO) optimization estimates gradients using only function value differences. Due to its memory efficiency and black-box applicability, it has recently been widely applied to LLM fine-tuning (e.g., MeZO series, ZO-LoRA).

**Limitations of Prior Work**: Classical optimization theories (Ghadimi-Lan 2013, Nesterov-Spokoiny 2017, Shamir 2017) consistently predict that the ZO convergence rate slows down linearly with the model dimension $d$, and the variance of single-perturbation estimators is also proportional to $d$. According to these theories, ZO should be practically unusable for billion-parameter LLMs; however, experiments like MeZO show that ZO can approach SGD performance on OPT-13B. **There is a significant mismatch between theory and experiment.**

**Key Challenge**: Compressing learning into a scalar "loss value" perspective fails to capture the true impact of ZO—while the loss reduction rate is indeed related to $d$, the change in model predictions on specific samples (learning dynamics) may be independent of $d$. The "low effective rank" hypothesis from Malladi et al. 2023 provides one explanation but is difficult to verify computationally on LLMs.

**Goal**: (1) Identify an "intermediate quantity" that can characterize both ZO and FO dynamics; (2) Prove that the discrepancy in this quantity depends only on $P$ and $V$, independent of $d$.

**Key Insight**: Move the perspective from the loss function to the function space using the eNTK framework (Jacot et al. 2018). The ZO update can be viewed as the FO eNTK subjected to a low-rank random projection $U_{t,P} U_{t,P}^\top$. This is equivalent to an inner-product-preserving version of the Johnson-Lindenstrauss (JL) Lemma, which states that the projection dimension only needs to be $\mathcal{O}(\ln n / \epsilon^2)$ to maintain fidelity, independent of the original dimension.

**Core Idea**: **ZO-eNTK is a random projection of FO-eNTK; the JL Lemma guarantees that as long as the perturbation count $P$ is adapted to the output dimension $V$, the difference between ZO and FO learning dynamics is independent of the model dimension $d$.**

## Method

### Overall Architecture
The paper is a purely theoretical analysis without a new algorithm. The core pipeline includes: (1) Deriving the change in log-probability after one step of ZO-SGD, explicitly writing the difference between FO and ZO as "FO eNTK minus projected eNTK" multiplied by two model-dependent matrices; (2) Applying the JL Lemma to the projected kernel difference; (3) Comparing Gaussian vs. Rademacher perturbations from both optimization (variance + convergence rate) and eNTK (projection error) perspectives; (4) Discussing the appropriate magnitude of $P$; (5) Validating with experiments on LeNet/MNIST, OPT-125M / 1.3B, and Mistral-7B.

### Key Designs

1. **Equivalence of One-step Learning Dynamics and eNTK**:
    - **Function**: Explicitly formulates the impact of a single ZO-SGD parameter update on the model's log-probability for a different data point $\mathbf{x}_o$.
    - **Mechanism**: Applying a first-order Taylor expansion to $\Delta \log \pi_t(y \mid \mathbf{x}_o)$ and substituting the ZO-SGD update rule yields $\Delta \log \pi \approx -\eta \mathcal{A}_t(\mathbf{x}_o) \mathcal{K}_t(\mathbf{x}_o, \mathbf{x}_u; U_{t,P}) \mathcal{G}_t(\mathbf{x}_u, \mathbf{y}_u)$, where the projected kernel is $\mathcal{K}_t(\mathbf{x}_o, \mathbf{x}_u; U_{t,P}) = \nabla_\theta z(\mathbf{x}_o)^\top U_{t,P} U_{t,P}^\top \nabla_\theta z(\mathbf{x}_u)$. The FO version simply replaces $U_{t,P} U_{t,P}^\top$ with the identity matrix. The difference is clear: ZO introduces a random projection matrix $U_{t,P} \in \mathbb{R}^{d \times P}$ formed by perturbations.
    - **Design Motivation**: This structure allows the use of the JL Lemma to transform the "dimension independence" proof into a few JL corollaries.

2. **Johnson-Lindenstrauss Projection Bounds**:
    - **Function**: Bounds the kernel difference $\|\Delta \mathcal{K}\|_F$ as a function of $\epsilon$ and provides an explicit dimension-free bound.
    - **Mechanism**: The term $\Delta\mathcal{K}[i,j]$ is defined as the difference between the original inner product $\langle \nabla_\theta z_i(\mathbf{x}_o), \nabla_\theta z_j(\mathbf{x}_u)\rangle$ and the projected inner product $\langle U_{t,P}^\top \nabla_\theta z_i, U_{t,P}^\top \nabla_\theta z_j\rangle$. The JL Lemma ensures that if $P \geq (2\ln n + \ln(1/\delta))/(c(\mathcal{Q})\epsilon^2)$, all inner products are preserved within $1 \pm \epsilon$. Substituting this back into the Frobenius norm of the kernel difference yields $\|\Delta\mathcal{K}\|_F^2 \leq \frac{\epsilon^2 V}{2}(\|\nabla_\theta z(\mathbf{x}_o)\|_F^2 + \|\nabla_\theta z(\mathbf{x}_u)\|_F^2)^2$. **The right-hand side has no $d$**.
    - **Design Motivation**: This provides the "dimension-free" conclusion: as long as $V$ (vocabulary or number of classes) does not explode, scaling the model from LeNet to LLaMA does not significantly diverge the ZO and FO learning trajectories.

3. **Gaussian vs. Rademacher Perturbation Comparison**:
    - **Function**: Explains why binary Rademacher perturbations often perform as well as Gaussian perturbations in practice.
    - **Mechanism**: From an optimization perspective, the second moment of a single-perturbation estimator is $(d+2)\|\nabla\ell\|^2$ for Gaussian and $d\|\nabla\ell\|^2$ for Rademacher; both are **proportional to $d$**, aligning with traditional views on high-dimensional inefficiency. However, from the eNTK perspective, both have JL concentration constants $\approx 1/4$, and the bounds are **independent of $d$**. Thus, there is nearly no difference in "projection quality," a property termed "distribution robustness."
    - **Design Motivation**: To reconcile empirical experience (Rademacher and Gaussian are both effective) with theoretical intuition (gap should be proportional to $d$) and support the use of aggressive binary/ternary perturbations.

### Loss & Training
No new training strategies are introduced. The theoretical section provides the optimization-perspective convergence rate for ZO-SGD as $\mathcal{O}(\sqrt{dL/(PT)})$ when the learning rate $\eta = \mathcal{O}(\sqrt{P/(dLT)})$. This still contains $d$, contrasting with the dimension-free eNTK bound and reminding readers that "convergence rate" and "learning trajectory similarity" are distinct metrics.

## Key Experimental Results

### Main Results
The authors validate the theory using three experimental setups:

| Setup | Model | Data | Observation |
|---|---|---|---|
| ZO vs FO eNTK Frobenius Error | LeNet ($d{=}29{,}624$) | MNIST | High semantic similarity pairs (e.g., 4, 9) have an error $\approx 0.338$ at $P{=}125$; low similarity pairs (0, 1) retain significant residual error. |
| Gaussian vs Rademacher | LeNet | MNIST | Curves for Frobenius, CKA, and Wasserstein metrics overlap almost perfectly. |
| LLM ZO Trajectories | OPT-125M → OPT-1.3B | SST-2 | As $P$ increases, the speed at which ZO trajectories approach FO is consistent across different model sizes, confirming "dimension independence." |

### Ablation Study

| Factor | Influence |
|---|---|
| Perturbation count $P$ | Error decays at $\mathcal{O}(\sqrt{\ln V / P})$, consistent with JL theory. |
| Perturbation distribution | Almost no impact, verifying distribution robustness. |
| Input pair similarity | High-similarity pairs result in faster ZO convergence; low-similarity pairs require higher $P$. |
| Model dimension $d$ | At the same $P$, the deviation of ZO trajectories from FO is similar for both OPT-125M and 1.3B. |

### Key Findings
- Confirmed that "perturbation count $P$ is the dominant factor, rather than $d$"—providing the first kernel-level explanation for long-standing empirical observations.
- Discovered that "sample pair similarity determines convergence speed": ZO estimators are better at making fine-grained distinctions between semantically similar inputs but struggle with structurally diverse inputs.
- The classical optimization bound $\mathcal{O}(\sqrt{dL/(PT)})$ and the kernel bound $\mathcal{O}(\sqrt{\ln V / P})$ coexist: loss reduction speed still depends on $d$, but the similarity of model prediction trajectories does not.

## Highlights & Insights
- Proposed a new framework analyzing ZO optimization through a function-space perspective; prior analyses were confined to parameter space and hindered by $d$.
- Used the JL Lemma to shift the dependence of projection dimensionality from the model parameter dimension $d$ to the output space dimension $V$.
- Provided a theoretical explanation for the near-equivalence of Rademacher and Gaussian perturbations, which was previously only an empirical observation.

## Limitations & Future Work
- The analysis relies on a one-step, small step-size local approximation and does not cover the cumulative error over a complete training trajectory.
- The "dimension-free" property comes at the cost of introducing $V$ (output dimension). For modern LLMs where $V \sim 10^5$, the $V$ factor in the bound is not negligible; whether this can be tightened remains an open question.
- No practical guideline for the "recommended value of $P$" is provided beyond "sufficiently large."
- Does not discuss the relationship between $d_{\text{eff}}$ and $P$ in the context of LoRA or partial parameter perturbation.

## Related Work & Insights
- **vs. Malladi et al. 2023b (MeZO effective rank hypothesis)**: MeZO uses Hessian low-rankness to explain dimension independence, which is hard to verify on LLMs. This paper provides rigorous bounds via JL without relying on low-rank assumptions.
- **vs. Spall / Nesterov classical ZO analysis**: Their metrics focus on optimization convergence rates (containing $d$); this paper uses an eNTK perspective to provide metrics decoupled from $d$.
- **vs. Achlioptas 2003 (Sparse JL projection)**: This work utilizes the inner-product-preserving version of JL rather than the distance-preserving version, making it more compatible with eNTK.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The observation that "ZO eNTK = Randomly Projected FO eNTK" is highly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments are mainly for theoretical validation; major LLM fine-tuning benchmarks were not the primary focus.
- **Writing Quality**: ⭐⭐⭐⭐ Derivation logic is clear; the links between equations (6), (8), and (17) are tight.
- **Value**: ⭐⭐⭐⭐⭐ Provides the first "trick-free" dimension-free explanation for using ZO in LLM fine-tuning; the theoretical framework is easily extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](../../NeurIPS2025/optimization/private_zeroth-order_optimization_with_public_data.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)

</div>

<!-- RELATED:END -->
