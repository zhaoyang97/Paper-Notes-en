---
title: >-
  [Paper Note] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective
description: >-
  [ICML 2026][Optimization & Theory][eNTK] This paper adopts empirical NTK (eNTK) as a unified perspective to prove that the eNTK induced by zeroth-order (ZO) SGD is equivalent to projecting the first-order (FO) eNTK onto a random subspace spanned by perturbations. Using the Johnson-Lindenstrauss (JL) Lemma, the authors explain why ZO methods remain effective f
tags:
  - ICML 2026
  - Optimization & Theory
  - eNTK
  - Johnson-Lindenstrauss
date: 2026-05-08
content_hash: 4c948ada11d06ed5
---
# Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective

**Conference**: ICML 2026  
**arXiv**: [2605.03373](https://arxiv.org/abs/2605.03373)  
**Code**: Not mentioned  
**Area**: Optimization Theory / LLM Fine-tuning / Learning Dynamics  
**Keywords**: Zeroth-Order Optimization, eNTK, Johnson-Lindenstrauss, Perturbation Count, Dimension Independence

## TL;DR
This paper adopts empirical NTK (eNTK) as a unified perspective to prove that the eNTK induced by zeroth-order (ZO) SGD is equivalent to projecting the first-order (FO) eNTK onto a random subspace spanned by perturbations. Using the Johnson-Lindenstrauss (JL) Lemma, the authors explain why ZO methods remain effective for billion-parameter LLMs: the error depends only on the output dimension $V$ and the perturbation count $P$, and is independent of the model dimension $d$.

## Background & Motivation
**Background**: Zeroth-order (ZO) optimization estimates gradients using only function value differences. Due to its memory efficiency and black-box nature, it has recently been widely applied to LLM fine-tuning (e.g., MeZO series, ZO-LoRA).

**Limitations of Prior Work**: Classical optimization theories (Ghadimi-Lan 2013, Nesterov-Spokoiny 2017, Shamir 2017) consistently predict that the ZO convergence rate slows down linearly with the model dimension $d$, and the variance of single-perturbation estimators is also proportional to $d$. According to these theories, ZO should be too slow for billion-parameter LLMs. However, experiments like MeZO demonstrate that ZO can approach SGD performance on OPT-13B. **Theory and experiments are completely misaligned.**

**Key Challenge**: Analyzing learning through the "scalar" lens of loss values fails to capture what ZO truly affects. While the rate of loss descent is indeed related to $d$, the change in model predictions for specific samples (learning dynamics) may be independent of $d$. The "low effective rank" assumption by Malladi et al. 2023 provides one explanation but is difficult to compute or verify in LLMs.

**Goal**: (1) Find an "intermediate quantity" capable of characterizing both ZO and FO; (2) Prove that their difference relies only on $P$ and $V$, independent of $d$.

**Key Insight**: Shift the perspective from the loss function to the function space using the eNTK (Jacot et al. 2018). The ZO update can be viewed as a low-rank random projection $U_{t,P} U_{t,P}^\top$ of the FO eNTK. This is equivalent to the inner-product preserving version of the Johnson-Lindenstrauss Lemma, which indicates that the projection dimension only needs to be $\mathcal{O}(\ln n / \epsilon^2)$, regardless of the original dimension.

**Core Idea**: **ZO-eNTK is a random projection of FO-eNTK; the JL Lemma guarantees that as long as the perturbation count $P$ is adapted to the output dimension $V$, the difference between ZO and FO learning dynamics is independent of the model dimension $d$.**

## Method

### Overall Architecture
The paper is a pure theoretical analysis without a new algorithm. The core pipeline includes: (1) Deriving the change in log-probability after a one-step ZO-SGD update, explicitly expressing the difference between FO and ZO as "FO eNTK minus projected eNTK" multiplied by two model-dependent matrices; (2) Applying the JL Lemma to the projection kernel difference; (3) Comparing Gaussian and Rademacher perturbations from both optimization (variance + convergence) and eNTK (projection error) perspectives; (4) Discussing the appropriate magnitude of $P$; (5) Verifying with experiments on LeNet/MNIST, OPT-125M / 1.3B, and Mistral-7B.

### Key Designs

**1. Equivalence of One-step Learning Dynamics and eNTK: Encapsulating the ZO-FO difference into a projection matrix**

The root of the theory-experiment gap is that the scalar perspective of loss masks the true impact of ZO. The authors move to the function space, applying a first-order Taylor expansion to the change in log-probability of the model on another data point $\mathbf{x}_o$ after a ZO-SGD update:

$$\Delta\log\pi\approx-\eta\,\mathcal{A}_t(\mathbf{x}_o)\,\mathcal{K}_t(\mathbf{x}_o,\mathbf{x}_u;U_{t,P})\,\mathcal{G}_t(\mathbf{x}_u,\mathbf{y}_u),$$

where the projected kernel is $\mathcal{K}_t=\nabla_\theta z(\mathbf{x}_o)^\top U_{t,P}U_{t,P}^\top\nabla_\theta z(\mathbf{x}_u)$. The FO version simply replaces $U_{t,P}U_{t,P}^\top$ with the identity matrix. The difference is clear: ZO adds a random projection $U_{t,P}\in\mathbb{R}^{d\times P}$ formed by perturbations. This equivalence links the dimension-free proof directly to the JL Lemma.

**2. Johnson-Lindenstrauss Projection Bound: Controlling kernel difference as a function independent of $d$**

With the projection form established, $\Delta\mathcal{K}[i,j]$ is the difference between the original and projected inner products. The JL Lemma ensures that if $P\ge(2\ln n+\ln(1/\delta))/(c(\mathcal{Q})\epsilon^2)$, all projected inner products are maintained within $1\pm\epsilon$. Substituting this back into the kernel difference yields:

$$\Bottom\|\Delta\mathcal{K}\|_F^2\le\frac{\epsilon^2 V}{2}\big(\|\nabla_\theta z(\mathbf{x}_o)\|_F^2+\|\nabla_\theta z(\mathbf{x}_u)\|_F^2\big)^2,$$

The right side contains only the output dimension $V$ and lacks the model dimension $d$. This achieves "dimension-free" scaling: as long as the vocabulary or class count $V$ does not explode, scaling from LeNet to LLaMA will not significantly diverge the ZO and FO learning trajectories.

**3. Gaussian vs Rademacher Perturbations: Fidelity is determined by $P$, not the distribution**

In practice, binary Rademacher perturbations often perform as well as Gaussian, though traditional variance analysis suggests a gap proportional to $d$. The authors clarify this: in optimization, the second moment of a single-perturbation estimator is $(d+2)\|\nabla\ell\|^2$ for Gaussian and $d\|\nabla\ell\|^2$ for Rademacher (both proportional to $d$). However, in the eNTK perspective, both have JL concentration constants of approximately $1/4$, and their bounds do not depend on $d$. This "distribution robustness" means $P$ is the dominant factor for ZO fidelity, bridging the gap between theory and empirical observation.

### Loss & Training
No new training strategy is proposed. The theoretical section confirms that when the learning rate $\eta = \mathcal{O}(\sqrt{P/(dLT)})$, the optimization convergence rate for ZO-SGD is $\mathcal{O}(\sqrt{dL/(PT)})$. This still contains $d$, contrasting with the dimension-free eNTK bound and reminding readers that "convergence rate" and "learning trajectory similarity" are distinct concepts.

## Key Experimental Results

### Main Results
The authors verify the theory across three settings:

| Setting | Model | Data | Observation |
|---|---|---|---|
| ZO vs FO eNTK Frobenius Error | LeNet ($d{=}29{,}624$) | MNIST | High semantic similarity pair (4,9) error $\approx 0.338$ at $P{=}125$; low similarity pair (0,1) maintains residual error even at $P{=}125$. |
| Gaussian vs Rademacher | LeNet | MNIST | Frobenius / CKA / Wasserstein metrics curves almost completely overlap. |
| Large Model ZO Trajectory | OPT-125M → OPT-1.3B | SST-2 | As $P$ increases, ZO trajectories approach FO at similar rates across model sizes, validating "dimension independence." |

### Ablation Study

| Factor | Impact |
|---|---|
| Perturbation Count $P$ | Error decays at $\mathcal{O}(\sqrt{\ln V / P})$, consistent with JL theory. |
| Distribution (Gaussian vs Rademacher) | Negligible impact, confirming distribution robustness. |
| Input Pair Similarity | High-similarity pairs converge faster; low-similarity pairs require larger $P$. |
| Model Dimension $d$ (OPT-125M → 1.3B) | Similar deviation levels between ZO and FO trajectories for the same $P$ across models. |

### Key Findings
- Validated that "perturbation count $P$ is the dominant factor, not $d$," providing the first kernel-level explanation for long-standing engineering observations.
- Discovered that "sample pair similarity determines convergence speed": ZO estimators are better at making fine-grained distinctions between semantically similar inputs.
- The coexistence of the classical optimization bound $\mathcal{O}(\sqrt{dL/(PT)})$ and the kernel bound $\mathcal{O}(\sqrt{\ln V / P})$: while loss descent speed still depends on $d$, the similarity of prediction trajectories does not.

## Highlights & Insights
- Proposed a "function-space perspective" for analyzing ZO optimization, bypassing the $d$-dependent bottlenecks inherent in parameter-space analysis.
- Leveraged the JL Lemma to shift dependency from parameter dimension $d$ to output space dimension $V$, serving as a model for deriving theory from engineering phenomena (MeZO's success in LLMs).
- Explained the near-equivalence of Rademacher and Gaussian perturbations, which was previously only an empirical observation.

## Limitations & Future Work
- The analysis is a local approximation based on one-step and small step-sizes, without covering cumulative errors across a full training trajectory.
- The "dimension-free" benefit introduces $V$ (output dimension); for modern LLMs where $V \sim 10^5$, the $V$ factor is non-trivial. Whether a tighter bound exists remains an open question.
- No practical guideline for the "optimal $P$" is provided beyond "larger $P$ is better."
- Scenarios involving LoRA or partial parameter fine-tuning and the relationship between $d_{\text{eff}}$ and $P$ are not discussed.

## Related Work & Insights
- **vs Malladi et al. 2023b (MeZO)**: MeZO explains dimension independence via Hessian low-rank assumptions, which are hard to verify in LLMs. This paper provides rigorous bounds via JL without such assumptions.
- **vs Spall / Nesterov**: Their frameworks focus on optimization convergence rates (containing $d$), whereas this paper uses eNTK to provide metrics decoupled from $d$.
- **vs Achlioptas 2003**: This paper utilizes the inner-product preserving version of JL rather than the distance-preserving version, making it more suitable for eNTK analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The equivalence "ZO eNTK = Random projection of FO eNTK" is a brilliant observation.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are mainly for theoretical verification and do not include full LLM fine-tuning benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The derivation chain is clear, and equations (6), (8), and (17) are tightly linked.
- Value: ⭐⭐⭐⭐⭐ Provides the first "trick-free" dimension-free explanation for ZO in LLM fine-tuning, with a highly extensible theoretical framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](../../NeurIPS2025/optimization/private_zeroth-order_optimization_with_public_data.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)

</div>

<!-- RELATED:END -->
