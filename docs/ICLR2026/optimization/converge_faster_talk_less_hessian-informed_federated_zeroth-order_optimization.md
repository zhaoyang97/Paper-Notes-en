---
title: >-
  [Paper Note] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization
description: >-
  [ICLR 2026][Optimization][Federated Learning] This paper proposes HiSo (Hessian-informed Scalar-only communication), which leverages a global diagonal Hessian approximation to accelerate convergence in federated zeroth-order optimization while strictly maintaining scalar communication without transmitting any second-order information. Theoretical analysis proves that, under low effective rank and whitening assumptions, the convergence rate is independent of the Lipschitz constant $L$ and model dimension $d$. Experiments on OPT-350M/1.3B/2.7B fine-tuning demonstrate a 1.4–5.4× reduction in communication rounds, with total communication cost remaining at the KB level.
tags:
  - ICLR 2026
  - Optimization
  - Federated Learning
  - zeroth-order optimization
  - Hessian preconditioning
  - scalar communication
  - LLM fine-tuning
date: 2026-05-08
content_hash: 8f15017f7a46a887
---

# Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization

**Conference**: ICLR 2026
**arXiv**: [2506.02370](https://arxiv.org/abs/2506.02370)
**Code**: To be confirmed
**Area**: Optimization / Federated Learning
**Keywords**: Federated Learning, zeroth-order optimization, Hessian preconditioning, scalar communication, LLM fine-tuning

## TL;DR

This paper proposes HiSo (Hessian-informed Scalar-only communication), which leverages a global diagonal Hessian approximation to accelerate convergence in federated zeroth-order optimization while strictly maintaining scalar communication without transmitting any second-order information. Theoretical analysis proves that, under low effective rank and whitening assumptions, the convergence rate is independent of the Lipschitz constant $L$ and model dimension $d$. Experiments on OPT-350M/1.3B/2.7B fine-tuning demonstrate a 1.4–5.4× reduction in communication rounds, with total communication cost remaining at the KB level.

## Background & Motivation

**Background**: Federated LLM fine-tuning faces a severe communication bottleneck — FedAvg requires approximately 1–5 TB of communication per client for OPT-1.3B. DeComFL exploits scalar-seed representations of zeroth-order gradients to achieve dimension-free communication (TB → KB), but suffers from extremely slow convergence.

**Limitations of Prior Work**: ZO-SGD estimates gradients using isotropic random directions ($u \sim \mathcal{N}(0, I)$), completely ignoring the heterogeneous curvature of LLM parameter spaces — high- and low-curvature directions are searched with equal weight, resulting in large gradient estimation variance and a convergence rate of $\mathcal{O}(\sqrt{Ld/mR})$ that depends on both dimension $d$ and Lipschitz constant $L$. Conventional Hessian preconditioning requires $O(d)$ or $O(d^2)$ communication, directly breaking the scalar communication framework.

**Key Challenge**: Curvature information can significantly accelerate convergence (as demonstrated by Adam and second-order methods), yet transmitting any Hessian-related information within a scalar communication framework incurs linear or quadratic overhead — fundamentally contradicting the goal of dimension-free communication.

**Goal**: How can Hessian information be exploited to accelerate convergence in federated zeroth-order optimization while strictly maintaining scalar communication (i.e., transmitting only a single gradient scalar $g$ and a random seed per round)?

**Key Insight**: The key observation is that the globally aggregated zeroth-order gradient update $\Delta x_r$ can already be reconstructed from scalars (as used in the model reconstruction step), so a diagonal Hessian approximation can be computed via Adam-style EMA from these existing variables at zero additional communication cost.

**Core Idea**: The random perturbation directions are warped by the inverse square root of the Hessian to concentrate search along high-curvature directions, while the Hessian itself is computed for free from the existing global gradient scalars — achieving curvature-accelerated optimization with zero additional communication overhead.

## Method

### Overall Architecture

The paper first introduces a general scalar-communication FL framework (Algorithm 1) that decouples scalar communication from any specific optimizer — rather than being tied to ZO-SGD, it admits any optimization algorithm whose update direction can be represented via a scalar and a shared state. Within this framework, HiSo replaces ZO-SGD updates with Hessian-guided zeroth-order updates: the perturbation direction is changed from isotropic $u \sim \mathcal{N}(0, I)$ to $z \sim \mathcal{N}(0, H_r^{-1})$, and the expected update shifts from gradient descent $\nabla f$ to Newton-style descent $H_r^{-1}\nabla f$.

### Key Design 1: Hessian-Guided Zeroth-Order Gradient Estimation

- **Function**: Upgrades ZO gradient estimation from isotropic search to curvature-guided directional search.
- **Mechanism**: Local updates are formalized as a subproblem minimizing gradient estimation error (Eqs. 5–6), and the optimal ascent direction is derived under the scalar representation constraint. The solution is $\Delta x = \frac{1}{\mu}[f(x + \mu H_r^{-1/2}u) - f(x)] \cdot H_r^{-1/2}u$, whose expectation satisfies $\mathbb{E}[\Delta x] \approx H_r^{-1}\nabla f(x)$ — equivalent to natural gradient / Newton descent. Crucially, each update still produces only one scalar $g$, fully preserving dimension-free communication.
- **Design Motivation**: $H_r^{-1/2}$ transforms the isotropic search into a non-uniform search aligned with the Hessian eigendirections — finer resolution along high-curvature directions and coarser along low-curvature ones — substantially reducing gradient estimation variance.

### Key Design 2: Zero-Communication-Cost Global Diagonal Hessian Learning

- **Function**: Synchronously maintains a global diagonal Hessian approximation on both the server and clients without any additional communication overhead.
- **Mechanism**: An Adam-style EMA is applied to the globally aggregated updates: $H_{r+1} = (1-\nu)H_r + \nu \,\text{Diag}([\Delta x_r]^2 + \epsilon I)$. The key observation is that $\Delta x_r$ can be fully reconstructed from scalars and random seeds (information already used in the model reconstruction step), making Hessian computation entirely free — requiring no extra communication and no additional function evaluations.
- **Design Motivation**: The diagonal approximation avoids $d^2$ storage; EMA smoothing prevents noisy oscillations; $\epsilon$ ensures positive definiteness — conceptually consistent with RMSProp but applied to the federated zeroth-order setting.

### Key Design 3: Variance Compression Theory via Low Whitening Rank

- **Function**: Provides rigorous theoretical justification for the acceleration brought by Hessian guidance.
- **Mechanism**: The paper introduces the "whitening rank" $\zeta = \text{Tr}(H^{-1/2}\Sigma H^{-1/2})$ to measure the quality of the Hessian approximation. When $H$ is a good approximation, $\zeta \ll L\kappa \ll Ld$, and the ZO gradient variance is compressed from the standard $Ld$ to $\zeta$ — improving the convergence rate from $\mathcal{O}(\sqrt{Ld/mR})$ to $\mathcal{O}(\sqrt{\zeta/mR})$, independent of both $d$ and $L$.
- **Design Motivation**: The Hessian eigenvalue spectrum of LLMs exhibits a heavy-tailed distribution (with effective rank far smaller than dimension); the whitening operation induced by the diagonal Hessian approximation effectively flattens this distribution.

### Loss & Training

Standard federated learning objective: $\min_x f(x) = \frac{1}{M}\sum_{i=1}^M f_i(x)$. A subset of clients is uniformly sampled per round; each client performs $\tau$ local update steps before uploading a scalar. The Hessian is updated once per round via $\tau$-step EMA at the beginning of each round.

## Key Experimental Results

### Main Results: HiSo Communication Round Reduction (Table 2)

| Model | Method | SST-2 Rounds | SST-2 Speedup | QQP Rounds | QQP Speedup | SQuAD Rounds | SQuAD Speedup |
|-------|--------|-------------|--------------|-----------|------------|-------------|--------------|
| OPT-350M | DeComFL | 550 | 1× | 775 | 1× | 1350 | 1× |
| | **HiSo** | **275** | **2×** | **425** | **1.8×** | **250** | **5.4×** |
| OPT-1.3B | DeComFL | 1500 | 1× | 1125 | 1× | 350 | 1× |
| | **HiSo** | **1075** | **1.4×** | **750** | **1.5×** | **175** | **2×** |
| OPT-2.7B | DeComFL | 1250 | 1× | 1475 | 1× | 450 | 1× |
| | **HiSo** | **775** | **1.6×** | **975** | **1.5×** | **200** | **2.3×** |

Communication cost savings: 29%–80% (e.g., OPT-350M SQuAD reduces from 52.73 KB to 9.77 KB, an 81% reduction).

### Comprehensive Baseline Comparison: LLM Fine-Tuning Accuracy and Communication Cost (Table 3)

| Model | Method | SST-2 Acc | SST-2 Comm. | QQP Acc | QQP Comm. | SQuAD F1 | SQuAD Comm. |
|-------|--------|----------|------------|---------|----------|----------|------------|
| OPT-125M | FedAvg | 87.63% | 0.15 TB | 61.21% | 0.08 TB | 37.27 | 0.05 TB |
| | FedAdam | 88.29% | 0.30 TB | 63.18% | 0.06 TB | 37.98 | 0.03 TB |
| | DeComFL | 85.21% | 22.92 KB | 60.11% | 32.17 KB | 34.12 | 17.42 KB |
| | **HiSo** | **85.55%** | **14.69 KB** | **60.72%** | **21.23 KB** | **35.26** | **7.12 KB** |
| OPT-350M | FedAvg | 89.79% | 0.58 TB | 63.32% | 0.31 TB | 43.38 | 0.12 TB |
| | DeComFL | 86.72% | 21.56 KB | 60.58% | 30.35 KB | 38.20 | 52.73 KB |
| | **HiSo** | **87.50%** | **17.33 KB** | **62.49%** | **18.63 KB** | **39.13** | **20.51 KB** |
| OPT-1.3B | FedAvg | 90.48% | 0.63 TB | 65.77% | 0.32 TB | 60.39 | 0.41 TB |
| | FedAdam | 92.86% | 0.79 TB | 64.59% | 1.10 TB | 61.56 | 0.27 TB |
| | FedZO | 90.01% | 4.73 TB | 62.91% | 3.53 TB | 57.26 | 1.10 TB |
| | DeComFL | 90.22% | 58.59 KB | 63.25% | 43.95 KB | 57.14 | 13.67 KB |
| | **HiSo** | **90.34%** | **49.18 KB** | **64.20%** | **96.67 KB** | **57.58** | **7.81 KB** |

### Key Findings

- **HiSo consistently outperforms DeComFL**: Across all model scales (OPT-125M to 2.7B) and all tasks, HiSo achieves higher accuracy with fewer communication rounds. Maximum speedup is 5.4× (OPT-350M SQuAD); minimum is 1.4× (OPT-1.3B SST-2).
- **Communication gap versus first-order methods is enormous**: HiSo operates at the KB level (7–97 KB), while first-order methods require TBs (0.03–4.73 TB) — a gap of up to **90 million times** (e.g., FedZO at 4.73 TB vs. HiSo at 49.18 KB).
- **Accuracy gap versus first-order methods is acceptable**: HiSo achieves 90.34% on OPT-1.3B SST-2 vs. FedAdam at 92.86% — a 2.5 percentage point gap — while reducing communication from 0.79 TB to 49.18 KB.
- **Hessian EMA parameter $\nu$ is robust**: Different values of $\nu$ have negligible impact on convergence and final accuracy.
- **Learned diagonal Hessian exhibits a heavy-tailed distribution**: Consistent with the low effective rank assumption, empirically validating the theoretical foundation.

## Highlights & Insights

- **"Free lunch" design philosophy**: Hessian information is extracted from existing global scalars (already used for model reconstruction) at zero additional communication cost — turning the "constraint" (scalar communication) into an "advantage" (free Hessian).
- **Theoretical breakthrough**: This is the first result in zeroth-order FL achieving simultaneously dimension-free and Lipschitz-free convergence rates ($\mathcal{O}(\sqrt{\zeta/mR})$), while also resolving the open problem of providing low effective rank guarantees under multi-step local updates in DeComFL.
- **General framework contribution**: Algorithm 1 decouples scalar communication from specific optimizers, enabling future integration of techniques such as momentum and variance reduction.
- **"Whitening rank" concept**: $\zeta = \text{Tr}(H^{-1/2}\Sigma H^{-1/2})$ provides a tighter variance upper bound than the effective rank $\kappa$, theoretically explaining why ZO convergence in practice is far faster than the $\mathcal{O}(d)$ worst-case bound.

## Limitations & Future Work

- The diagonal Hessian is a coarse approximation — off-diagonal curvature information (e.g., inter-parameter interactions) is entirely ignored, which may reduce effectiveness for highly coupled parameter spaces.
- Whether the low effective rank and whitening assumptions hold across all LLM layers remains difficult to verify formally; the paper relies on indirect experimental support.
- Although HiSo generally reduces communication, the cost on OPT-1.3B QQP (96.67 KB) exceeds that of DeComFL (43.95 KB), indicating that the speedup is not uniform across all settings.
- Evaluation is limited to classification and QA tasks; generative tasks (e.g., text generation, dialogue) are not tested.
- The current formulation does not include a momentum term (resembling RMSProp rather than Adam); the paper mentions this as a possible extension but provides no experiments.

## Related Work & Insights

- **vs. DeComFL**: HiSo strictly generalizes DeComFL (reducing to DeComFL when $H_r \equiv I$); both use scalar communication, but HiSo achieves 1.4–5.4× speedup and provides theoretical coverage for multi-step local updates.
- **vs. FedAdam/FedYogi**: First-order adaptive FL methods achieve higher accuracy but require $O(d)$ communication. HiSo approximately achieves similar adaptive effects under the scalar communication constraint.
- **vs. Hessian-aware ZO (single-machine)**: Ye et al. (2018) and Zhao et al. (2025) validate the effectiveness of Hessian-guided ZO in single-machine settings; HiSo is the first to extend this to federated scalar communication and resolves the Hessian transmission problem.
- **Insights**: The generality of the scalar communication framework suggests that variance reduction, momentum, and other techniques could potentially be incorporated at no additional communication cost, leaving substantial room for improvement in communication-constrained federated optimization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — An elegant combination of scalar communication and Hessian guidance; the "free Hessian" observation is insightful; theoretical contributions ($\zeta$ and whitening rank) are rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive grid across three OPT scales × three tasks, with full comparison against first-order and ZO baselines; generative task evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ — Framework derivation and theoretical analysis are clear; the logical chain from subproblem to algorithm to theory is complete.
- Value: ⭐⭐⭐⭐⭐ — Directly applicable to communication-constrained federated LLM fine-tuning; the general framework opens up substantial future research directions.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)

<!-- RELATED:END -->
