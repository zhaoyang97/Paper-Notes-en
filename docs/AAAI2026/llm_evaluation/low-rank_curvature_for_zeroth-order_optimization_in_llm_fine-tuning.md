---
title: >-
  [Paper Note] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning
description: >-
  [AAAI 2026][LLM Evaluation][zeroth-order optimization] This paper proposes LOREN, a curvature-aware zeroth-order optimization method that captures the anisotropic curvature of the loss landscape via a low-rank block-diagonal preconditioner, combined with REINFORCE Leave-One-Out (RLOO) variance reduction. LOREN achieves higher accuracy and faster convergence in LLM fine-tuning while reducing peak memory by up to 27.3% compared to MeZO-Adam.
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "zeroth-order optimization"
  - "curvature-aware"
  - "low-rank preconditioner"
  - "variance reduction"
  - "memory-efficient"
date: 2026-05-08
content_hash: 225a795120c01692
---

# Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning

**Conference**: AAAI 2026
**arXiv**: [2511.07971](https://arxiv.org/abs/2511.07971)  
**Code**: [GitHub](https://github.com/hseung88/loren)  
**Area**: LLM Evaluation
**Keywords**: zeroth-order optimization, curvature-aware, low-rank preconditioner, variance reduction, memory-efficient

## TL;DR

This paper proposes LOREN, a curvature-aware zeroth-order optimization method that captures the anisotropic curvature of the loss landscape via a low-rank block-diagonal preconditioner, combined with REINFORCE Leave-One-Out (RLOO) variance reduction. LOREN achieves higher accuracy and faster convergence in LLM fine-tuning while reducing peak memory by up to 27.3% compared to MeZO-Adam.

## Background & Motivation

**Background**: Zeroth-order (ZO) optimization methods estimate gradients using only forward passes, eliminating the need for backpropagation and intermediate activation storage, thereby substantially reducing memory requirements for LLM fine-tuning. MeZO established this paradigm, and subsequent works such as HiZOO and LOZO have introduced further improvements.

**Limitations of Prior Work**: Existing ZO methods suffer from two fundamental shortcomings: (1) finite-difference gradient estimates exhibit extremely high variance in high-dimensional spaces, leading to unstable parameter updates; and (2) they are oblivious to the anisotropic curvature of the loss landscape, causing oscillation along high-curvature directions, stagnation along low-curvature directions, and susceptibility to convergence at saddle points.

**Key Challenge**: Incorporating curvature information (e.g., the Hessian) typically incurs additional memory and computational overhead, which directly conflicts with the memory-efficiency objective of ZO methods. The core challenge is: how can curvature awareness be introduced with virtually no additional memory cost?

**Goal**: To simultaneously address the high-variance and curvature-agnostic problems in ZO optimization, significantly improving convergence speed and fine-tuning accuracy while maintaining memory efficiency.

**Key Insight**: The gradient preconditioning problem is reformulated as adaptively estimating the covariance matrix of the perturbation distribution. Leveraging the Natural Evolution Strategies (NES) framework and Kronecker factorization, the covariance is parameterized as a low-rank structure.

**Core Idea**: Gradient preconditioning in ZO optimization is equivalent to sampling perturbation vectors from an anisotropic Gaussian distribution. A low-rank Kronecker-factorized covariance matrix approximates the inverse Hessian, and RLOO variance reduction enables efficient curvature-aware updates.

## Method

### Overall Architecture

At each iteration, LOREN: (1) samples $K$ perturbation vectors from a parameterized anisotropic Gaussian distribution; (2) evaluates the loss via $K$ forward passes with perturbed parameters; (3) computes low-variance gradient estimates and covariance parameter gradients using the RLOO estimator; and (4) simultaneously updates both the model parameters $x$ and the covariance parameters $a$.

### Key Designs

1. **Preconditioning as Anisotropic Perturbation Distribution**: Standard ZO-SGD samples perturbations from an isotropic Gaussian $u \sim \mathcal{N}(0, I)$, ignoring curvature differences. LOREN's core insight is that the preconditioned gradient update $\tilde{H}^{-1}\nabla f(x)$ is equivalent to sampling perturbations from $\mathcal{N}(0, \tilde{H}^{-1})$ and applying finite-difference estimation. Thus, learning an appropriate covariance matrix $\Sigma = \tilde{H}^{-1}$ naturally achieves curvature-aware preconditioning.

2. **Low-Rank Block-Diagonal Hessian Approximation**: For each layer parameter matrix $X \in \mathbb{R}^{m \times n}$, the Hessian is approximated as $\tilde{H} = I_m \otimes (\rho I_n + aa^T)$, where $\rho$ is a damping factor and $a \in \mathbb{R}^n$ is a learnable vector. This approximation admits closed-form inverses and square-root inverses, with an additional memory cost of only $O(n)$ (for storing vector $a$), far less than the $O(mn)$ cost of MeZO-Adam. This is the first method to employ block-diagonal Hessian approximation in ZO optimization.

3. **REINFORCE Leave-One-Out Variance Reduction**: Unlike conventional ZO methods that use SPSA (two-point finite differences), LOREN adopts the RLOO estimator: for each of the $K$ perturbation samples, the mean loss of the remaining $K-1$ samples serves as a control variate baseline, effectively reducing variance. Experiments demonstrate a significant reduction in MSE.

4. **Natural Evolution Strategies (NES) Framework**: The optimization objective is reformulated as minimizing the expected loss under the search distribution $J(\theta) = \mathbb{E}_{z \sim p(z;\theta)}[f(z)]$. Score function estimators are used to compute gradients with respect to the covariance parameters $a$ without additional forward passes. Parameters $x$ and $a$ are updated jointly at each step.

### Loss & Training

Full-parameter fine-tuning is performed without prompts. Each step involves $K=6$ forward passes for perturbation evaluation; the RLOO estimator requires no backpropagation. Early stopping is applied to prevent overfitting. Heavy-ball momentum accelerates convergence. The convergence rate is $O(1/\sqrt{T})$.

## Key Experimental Results

### Main Results

Accuracy (%) of GPT-2-XL (1.5B) on GLUE:

| Method | MNLI | QNLI | SST-2 | CoLA | Avg. |
|--------|------|------|-------|------|------|
| MeZO | 39.1 | 58.8 | 73.8 | 65.4 | 59.3 |
| MeZO-Adam | 50.9 | 72.3 | 91.2 | 71.6 | 71.5 |
| HiZOO | 48.6 | 66.3 | 89.6 | 71.5 | 69.0 |
| **LOREN** | **51.2** | **74.6** | 89.8 | **72.0** | **71.9** |

On OPT-13B (SuperGLUE), LOREN achieves 73.7% on CB, outperforming all ZO baselines across the board.

### Ablation Study

Peak GPU memory comparison (OPT-13B, BF16):

| Method | Memory (GB) | Relative to MeZO |
|--------|-------------|------------------|
| MeZO | 32.9 | 1.00× |
| MeZO-Adam | 76.0 | 2.31× |
| HiZOO | 59.6 | 1.81× |
| LOREN | ~41 | ~1.25× |

LOREN's additional memory overhead is only $O(n)$, far lower than MeZO-Adam's $O(mn)$.

### Key Findings

- LOREN consistently outperforms or matches all ZO baselines across model scales ranging from DistilBERT (66M) to OPT-13B.
- On RoBERTa-large, LOREN achieves an average accuracy of 70.1%, substantially surpassing MeZO (58.4%) and HiZOO (64.2%).
- The RLOO variance reduction consistently yields lower gradient estimation MSE than standard ZO-SGD on 1000-dimensional test functions.
- On the monkey saddle function, LOREN is the only ZO method that successfully escapes the saddle point region.

## Highlights & Insights

- Unifying "preconditioning" and "perturbation distribution learning" is an elegant theoretical contribution.
- The low-rank Kronecker factorization parameterizes the covariance via a single vector $a$, incurring minimal memory overhead.
- LOREN is the first ZO method to simultaneously achieve curvature awareness and variance reduction.
- A convergence guarantee of $O(1/\sqrt{T})$ is provided.

## Limitations & Future Work

- Comparison with parameter-efficient methods such as LoRA under the full-parameter fine-tuning setting is absent.
- Each step requires $K=6$ forward passes, resulting in higher total computational cost than MeZO, which requires only 2.
- The block-diagonal approximation assumes independence of parameters across layers, neglecting cross-layer curvature correlations.
- Performance gains on LLaMA-3-8B are less pronounced than on smaller models; scalability to larger models remains to be validated.

## Related Work & Insights

- **vs. MeZO**: LOREN augments MeZO with both curvature information and variance reduction, improving average accuracy on GPT-2-XL from 59.3 to 71.9.
- **vs. HiZOO**: HiZOO only estimates the diagonal Hessian and requires additional forward passes; LOREN captures richer curvature via a block-diagonal approximation at lower memory cost.
- **vs. LOZO**: LOZO performs low-rank gradient estimation while LOREN performs low-rank preconditioning — the two approaches are complementary.
- **vs. MeZO-SVRG**: SVRG requires storing a full-batch reference gradient, incurring high memory; RLOO has no such requirement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of NES, low-rank Kronecker factorization, and RLOO appears for the first time; the preconditioning-as-distribution-learning perspective is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers models from 66M to 13B and multiple GLUE/SuperGLUE tasks, but evaluation on generation tasks is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and complete, though the dense notation presents a non-trivial reading barrier.
- **Value**: ⭐⭐⭐⭐ Improving ZO fine-tuning performance in memory-constrained settings is practically meaningful, though the application window for ZO fine-tuning is narrowing as LoRA becomes more prevalent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](../../ICML2026/llm_evaluation/agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[AAAI 2026\] Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment](gaming_the_answer_matcher_examining_the_impact_of_text_manipulation_on_automated.md)
- [\[ICLR 2026\] Sci2Pol: Evaluating and Fine-tuning LLMs' "Science-to-Policy Brief" Generation Capabilities](../../ICLR2026/llm_evaluation/sci2pol_evaluating_and_fine-tuning_llms_on_scientific-to-policy_brief_generation.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)

</div>

<!-- RELATED:END -->
