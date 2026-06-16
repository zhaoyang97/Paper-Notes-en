---
title: >-
  [Paper Note] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective
description: >-
  [ICML 2026][Optimization & Theory][eNTK] This paper adopts empirical NTK (eNTK) as a unified perspective to prove that the eNTK induced by zeroth-order (ZO) SGD is equivalent to projecting the first-order (FO) eNTK onto a random subspace spanned by perturbations. Utilizing the Johnson-Lindenstrauss lemma, it explains why ZO methods remain effective for billio
tags:
  - ICML 2026
  - Optimization & Theory
  - eNTK
  - Johnson-Lindenstrauss
date: 2026-05-08
content_hash: 43d4323871949ac3
---
# Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective

**Conference**: ICML 2026  
**arXiv**: [2605.03373](https://arxiv.org/abs/2605.03373)  
**Code**: Not mentioned  
**Area**: Optimization Theory / LLM Fine-tuning / Learning Dynamics  
**Keywords**: Zeroth-Order Optimization, eNTK, Johnson-Lindenstrauss, Perturbation Count, Dimension Independence

## TL;DR
This paper adopts empirical NTK (eNTK) as a unified perspective to prove that the eNTK induced by zeroth-order (ZO) SGD is equivalent to projecting the first-order (FO) eNTK onto a random subspace spanned by perturbations. Utilizing the Johnson-Lindenstrauss lemma, it explains why ZO methods remain effective for billion-parameter LLMs: the error depends only on the output dimension $V$ and the number of perturbations $P$, rather than the model dimension $d$.

## Background & Motivation
**Background**: Zeroth-order (ZO) optimization estimates gradients using only function value differences. Due to its memory efficiency and black-box nature, it has recently been widely applied to LLM fine-tuning (e.g., MeZO series, ZO-LoRA).

**Limitations of Prior Work**: Classical optimization theories (Ghadimi-Lan 2013, Nesterov-Spokoiny 2017, Shamir 2017) consistently predict that the ZO convergence rate slows down linearly with the model dimension $d$, and the variance of single-perturbation estimators is also proportional to $d$. According to these theories, ZO should be too slow for billion-parameter LLMs; however, experiments like MeZO demonstrate that ZO can approach SGD performance on OPT-13B. **Theory and experiments are completely misaligned.**

**Key Challenge**: Analyzing learning dynamics solely through the scalar "loss value" fails to capture the true impact of ZO—while the rate of loss decrease indeed depends on $d$, the changes in model predictions on specific samples (learning dynamics) might be independent of $d$. The "low effective rank" hypothesis by Malladi et al. 2023 provides one explanation, but it is computationally unverifiable for LLMs.

**Goal**: (1) Identify an "intermediate quantity" that characterizes both ZO and FO; (2) Prove that its discrepancy depends only on $P$ and $V$, independent of $d$.

**Key Insight**: Shift the perspective from the loss function to the function space using the eNTK (Jacot et al. 2018). The ZO update can be viewed as the FO eNTK passing through a low-rank random projection $U_{t,P} U_{t,P}^\top$. This is equivalent to the inner-product-preserving version of the Johnson-Lindenstrauss (JL) Lemma, which states that the projection dimension only needs to be $\mathcal{O}(\ln n / \epsilon^2)$, independent of the original dimension.

**Core Idea**: **ZO-eNTK is a random projection of FO-eNTK; the JL Lemma guarantees that if the perturbation count $P$ is matched with the output dimension $V$, the difference in learning dynamics between ZO and FO is independent of the model dimension $d$.**

## Method

### Overall Architecture
The paper consists of purely theoretical analysis without new algorithms. The core pipeline: (1) Derive the change in log-probability after a one-step update of ZO-SGD, explicitly writing the difference between FO and ZO as "FO eNTK minus projected eNTK" multiplied by two model-dependent matrices; (2) Apply the JL Lemma to the projection kernel discrepancy; (3) Compare Gaussian and Rademacher perturbations from both optimization (variance + convergence rate) and eNTK (projection error) perspectives; (4) Discuss the reasonable magnitude of $P$; (5) Verify with experiments on LeNet/MNIST, OPT-125M / 1.3B, and Mistral-7B.

### Key Designs

**1. Equivalence of One-step Learning Dynamics and eNTK: Encapsulating the ZO-FO Difference in a Projection Matrix**

The root cause of the theory-experiment gap is that the scalar loss perspective overlooks the actual influence of ZO. The authors move to the function space, performing a first-order Taylor expansion on the log-prob change of the model at another data point $\mathbf{x}_o$. Substituting the ZO-SGD update yields:

$$\Delta\log\pi\approx-\eta\,\mathcal{A}_t(\mathbf{x}_o)\,\mathcal{K}_t(\mathbf{x}_o,\mathbf{x}_u;U_{t,P})\,\mathcal{G}_t(\mathbf{x}_u,\mathbf{y}_u),$$

where the projected kernel is $\mathcal{K}_t=\nabla_\theta z(\mathbf{x}_o)^\top U_{t,P}U_{t,P}^\top\nabla_\theta z(\mathbf{x}_u)$. The FO version simply replaces $U_{t,P}U_{t,P}^\top$ with the identity matrix. The difference is clear: ZO introduces a random projection $U_{t,P}\in\mathbb{R}^{d\times P}$ composed of perturbations. This equivalence links the proof of dimension independence directly to the JL Lemma.

**2. Johnson-Lindenstrauss Projection Bounds: Controlling Kernel Discrepancy Independent of $d$**

With the projection form established, $\Delta\mathcal{K}[i,j]$ is defined as the difference between the original and projected inner products. The JL Lemma guarantees that if $P\ge(2\ln n+\ln(1/\delta))/(c(\mathcal{Q})\epsilon^2)$, all projected inner products are preserved within $1\pm\epsilon$. Applying this to the kernel discrepancy gives:

$$\|\Delta\mathcal{K}\|_F^2\le\frac{\epsilon^2 V}{2}\big(\|\nabla_\theta z(\mathbf{x}_o)\|_F^2+\|\nabla_\theta z(\mathbf{x}_u)\|_F^2\big)^2,$$

The right side contains only the output dimension $V$ and is entirely free of the model dimension $d$. This achieves dimension-free analysis: as long as the vocabulary or class count $V$ remains manageable, scaling from LeNet to LLaMA will not significantly diverge the learning trajectories of ZO and FO.

**3. Gaussian vs Rademacher Perturbations: Fidelity Determined by $P$, Not Distribution**

In practice, binary Rademacher perturbations are often as effective as Gaussian ones, despite traditional variance analysis suggesting a difference proportional to $d$. The authors analyze this from two perspectives: in optimization, the second moment of a single-perturbation estimator is $(d+2)\|\nabla\ell\|^2$ for Gaussian and $d\|\nabla\ell\|^2$ for Rademacher, both proportional to $d$. However, in the eNTK perspective, both share a JL concentration constant of approximately $1/4$, and neither bound depends on $d$. Thus, they are nearly identical in projection quality, a property termed "distribution robustness."

### Loss & Training
No new training strategies were introduced. The theoretical section provides the convergence rate for ZO-SGD from an optimization perspective as $\mathcal{O}(\sqrt{dL/(PT)})$ when the learning rate $\eta = \mathcal{O}(\sqrt{P/(dLT)})$. This remains dependent on $d$, contrasting with the dimension-free kernel bound and reminding readers that "convergence rate" and "learning trajectory similarity" are distinct concepts.

## Key Experimental Results

### Main Results
The authors verify the theory using three experimental setups:

| Setup | Model | Data | Observation |
|---|---|---|---|
| ZO vs FO eNTK Frobenius Error | LeNet ($d{=}29{,}624$) | MNIST | High semantic similarity pairs (4,9) have error $\approx 0.338$ at $P{=}125$; low similarity pairs (0,1) maintain significant residual error even at $P{=}125$. |
| Gaussian vs Rademacher | LeNet | MNIST | Curves for Frobenius, CKA, and Wasserstein metrics overlap almost perfectly. |
| Large Model ZO Trajectory | OPT-125M → OPT-1.3B | SST-2 | As $P$ increases, the speed at which ZO trajectories approach FO is consistent across model sizes, verifying dimension independence. |

### Ablation Study

| Factor | Impact |
|---|---|
| Perturbation count $P$ | Error decays at $\mathcal{O}(\sqrt{\ln V / P})$, consistent with JL theory. |
| Perturbation distribution | Negligible impact, verifying distribution robustness. |
| Input pair similarity | Highly similar pairs converge faster in ZO; dissimilar pairs require higher $P$. |
| Model dimension $d$ | For the same $P$, the deviation of ZO from FO is similar for both OPT-125M and 1.3B. |

### Key Findings
- Confirmed that "perturbation count $P$ is the dominant factor, not $d$"—providing the first kernel-level explanation for long-standing empirical observations.
- "Sample pair similarity determines convergence speed" is a new insight: ZO estimators are better at making fine-grained distinctions between semantically similar inputs.
- The classical optimization bound $\mathcal{O}(\sqrt{dL/(PT)})$ and the kernel bound $\mathcal{O}(\sqrt{\ln V / P})$ coexist: while loss descent speed still depends on $d$, the similarity of prediction trajectories does not.

## Highlights & Insights
- Proposed a novel "function-space perspective" for analyzing ZO optimization; previous analyses were hindered by $d$ in the parameter space.
- Leveraged the JL Lemma to shift the dependency of projection dimensionality from the parameter dimension $d$ to the output dimension $V$.
- Explained why Rademacher and Gaussian perturbations are nearly equivalent, which was previously only an empirical observation.

## Limitations & Future Work
- The analysis is a local approximation based on one-step and small step-sizes, not covering cumulative errors over a full training trajectory.
- The "dimension-free" property comes at the cost of introducing $V$ (output dimension); for modern LLMs where $V \sim 10^5$, the $V$ factor is non-trivial.
- No practical guideline for the "recommended $P$" was provided beyond "$P$ should be large enough."
- Relationships between $d_{\text{eff}}$ and $P$ in LoRA or partial parameter fine-tuning scenarios were not discussed.

## Related Work & Insights
- **vs Malladi et al. 2023b (MeZO Low Effective Rank)**: MeZO relies on Hessian low-rank assumptions which are hard to verify; this paper provides a rigorous bound via JL without such assumptions.
- **vs Spall / Nesterov (Classical ZO Analysis)**: Their frameworks focus on optimization convergence rates (containing $d$), whereas this work identifies a $d$-decoupled metric via eNTK.
- **vs Achlioptas 2003 (Sparse JL Projection)**: This work utilizes the inner-product-preserving version of JL rather than the distance-preserving version, making it more suitable for eNTK analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The equivalence "ZO eNTK = Random Projection of FO eNTK" is an elegant observation.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are mainly for theoretical verification; no full-scale LLM fine-tuning benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation chain; Equations (6), (8), and (17) are tightly integrated.
- Value: ⭐⭐⭐⭐⭐ Provides the first "dimension-free" explanation for ZO in LLM fine-tuning without extra tricks, offering an extensible theoretical framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](../../NeurIPS2025/optimization/private_zeroth-order_optimization_with_public_data.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)

</div>

<!-- RELATED:END -->
