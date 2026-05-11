---
title: >-
  [Paper Note] Rethinking Losses for Diffusion Bridge Samplers
description: >-
  [NeurIPS 2025][LLM Evaluation][diffusion bridge samplers] This paper identifies theoretical flaws in the widely used Log Variance (LV) loss for diffusion bridge samplers—namely…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "diffusion bridge samplers"
  - "loss functions"
  - "reverse KL divergence"
  - "Log Variance loss"
  - "learnable diffusion coefficients"
date: 2026-05-08
content_hash: 1842a6de83edf087
---

# Rethinking Losses for Diffusion Bridge Samplers

**Conference**: NeurIPS 2025
**arXiv**: [2506.10982](https://arxiv.org/abs/2506.10982)
**Code**: [GitHub](https://github.com/sanokows/RethinkingLossesForDiffusionBridgeSamplers)
**Area**: LLM Evaluation
**Keywords**: diffusion bridge samplers, loss functions, reverse KL divergence, Log Variance loss, learnable diffusion coefficients

## TL;DR

This paper identifies theoretical flaws in the widely used Log Variance (LV) loss for diffusion bridge samplers—namely, that it violates the data processing inequality and its gradients are not equivalent to those of the reverse KL (rKL)—and proposes computing rKL gradients via the log-derivative trick (rKL-LD). The proposed approach consistently outperforms LV loss across multiple benchmarks while exhibiting more stable training and reduced sensitivity to hyperparameters.

## Background & Motivation

Sampling from unnormalized distributions is a fundamental problem in computational physics, chemistry, and Bayesian inference. Diffusion bridge samplers (e.g., DBS and CMCD) represent the current state of the art, learning transport paths between forward and reverse diffusion processes to enable sampling.

Two main training losses exist in the literature:

**rKL-R (rKL via reparameterization trick)**: Prone to vanishing/exploding gradients in multi-step diffusion; performs poorly in practice.

**LV loss (Log Variance)**: Does not require backpropagating through expectations; widely regarded as superior to rKL-R.

However, the authors identify a critical overlooked issue: the gradient equivalence between LV and rKL losses **holds only when learning parameters of the reverse diffusion process exclusively**. When diffusion bridges are involved (i.e., the forward process also has learnable parameters) or when diffusion coefficients are learned, this equivalence no longer holds. More importantly, the LV loss violates the data processing inequality (DPI), undermining its theoretical justification as a training objective for latent variable models.

These findings motivate a reexamination of the rKL loss, leading the authors to adopt the log-derivative trick for gradient estimation (rKL-LD), which avoids the gradient pathologies of reparameterization while retaining the theoretical advantages of rKL.

## Method

### Overall Architecture

The core idea is to replace the LV training loss in diffusion bridge samplers with rKL-LD, while additionally introducing learnable diffusion coefficients to adaptively regulate stochasticity during sampling. The approach is compatible with both mainstream diffusion bridge architectures: DBS and CMCD.

### Key Designs

1. **rKL-LD Gradient Estimator**: The log-derivative trick (also known as the REINFORCE or score function trick) is used to compute gradients of the rKL divergence with respect to model parameters. For reverse process parameters $\alpha$, the gradient is:

$$\nabla_\alpha^{LD} D_{KL}(q_{\alpha,\nu} \| p_{\phi,\nu}) = \mathbb{E}_{X_{0:T} \sim q_{\alpha,\nu}} \left[ \left(\log \frac{q_{\alpha,\nu}(X_{0:T})}{p_{\phi,\nu}(X_{0:T})} - b \right) \nabla_\alpha \log q_{\alpha,\nu}(X_{0:T}) \right]$$

   where $b$ is a control variate for variance reduction. For forward process parameters $\phi$, the gradient simplifies to $-\mathbb{E}[\nabla_\phi \log p_{\phi,\nu}]$. The key design motivations are: (a) avoiding vanishing/exploding gradients from reparameterization; (b) preserving the theoretical guarantees of rKL via the data processing inequality.

2. **Theoretical Analysis of LV Loss and Gradient Discrepancy**: The authors derive the gradient of the LV loss with respect to shared parameters $\nu$ and show that it is equivalent to the gradient of the Jeffrey divergence at the optimum—not the rKL gradient. Specifically, for shared parameters:

$$\nabla_\nu D_{LV}^{q_{\alpha,\nu}^*}(q_{\alpha,\nu}, p_{\phi,\nu}) = \mathbb{E}\left[\left(\log \frac{q_{\alpha,\nu}}{p_{\phi,\nu}} - b\right) \nabla_\nu \log \frac{q_{\alpha,\nu}}{p_{\phi,\nu}}\right]$$

   This differs from the rKL-LD gradient. The authors further construct a counterexample demonstrating that LV violates the data processing inequality $D_f(\pi_0(X_0) \| q_{\alpha,\nu}(X_0)) \leq D_f(p_{\phi,\nu}(X_{0:T}) \| q_{\alpha,\nu}(X_{0:T}))$, thereby questioning the validity of LV as a training objective for diffusion bridges.

3. **Learnable Diffusion Coefficients**: The SDE diffusion coefficient $\sigma_{\text{diff}} \in \mathbb{R}^N$ is treated as a learnable parameter (with independent learning per dimension), adaptively balancing exploration and exploitation. Larger coefficients increase stochasticity for better coverage of multimodal distributions; smaller coefficients reduce noise to improve precision. A key finding is that learnable diffusion coefficients consistently improve performance under rKL-LD, while frequently causing training instability or divergence under LV loss.

### Loss & Training

The training loss is the rKL divergence, with gradients computed via the log-derivative trick and variance reduced using control variates. For DBS, independent neural networks are used for the forward and reverse processes; for CMCD, a shared control function $u_\gamma$ is used alongside a learnable interpolation function $\eta(t)$. Training proceeds for 40,000 iterations with a batch size of 2,000 and 128 diffusion steps, with grid search over learning rate, initial $\sigma_{\text{diff}}$, and prior variance.

## Key Experimental Results

### Main Results (Bayesian Learning Benchmarks)

| Method | Seeds (26d) | Sonar (61d) | Credit (25d) | Brownian (32d) | LGCP (1600d) |
|--------|-------------|-------------|--------------|----------------|--------------|
| CMCD: LV ☆ | -74.13 | -109.53 | -504.91 | -0.05 | 460.84 |
| CMCD: LV (learn σ) | -73.53 | -109.66 | **-628.39†** | **-6.05†** | **447.74†** |
| CMCD: rKL-LD ☆ | -74.10 | -109.25 | -504.88 | 0.36 | 466.73 |
| CMCD: rKL-LD (learn σ) | **-73.45** | **-108.83** | **-504.58** | **1.06** | 465.80 |
| DBS: LV ☆ | -74.12 | -110.66 | -506.21 | -9.39† | 460.48 |
| DBS: rKL-LD (learn σ) | **-73.50** | **-108.88** | **-504.71** | **0.85** | **469.89** |

*ELBO (↑ higher is better); ☆ = diffusion coefficient not learned; † = training diverged*

### Synthetic Target Benchmarks

| Method | GMM-40 Sinkhorn(↓) | GMM-40 ELBO(↑) | MoS-10 Sinkhorn(↓) | MoS-10 ELBO(↑) |
|--------|---------------------|-----------------|---------------------|-----------------|
| CMCD: LV ☆ | 2559.20 | -37.37 | 1263.78 | -52.52 |
| CMCD: rKL-LD (learn σ) | **2301.16** | **-21.94** | **915.52** | -34.93 |
| DBS: LV ☆ | 2073.09 | -35.45 | 1220.27 | -57.49 |
| DBS: rKL-LD (learn σ) | 2133.50 | **-30.44** | **1051.34** | **-43.66** |

### Key Findings

1. **rKL-LD consistently outperforms LV**: On Bayesian tasks, CMCD+rKL-LD significantly surpasses LV on 4 out of 5 tasks; DBS+rKL-LD surpasses LV on 4 out of 5 tasks.
2. **LV + learnable diffusion coefficients = catastrophic failure**: Learning $\sigma_{\text{diff}}$ under LV loss causes training divergence on 3/5 tasks for CMCD and 4/5 tasks for DBS.
3. **rKL-LD + learnable diffusion coefficients = consistent gains**: Learning $\sigma_{\text{diff}}$ under rKL-LD never degrades performance and frequently yields substantial improvements.
4. **Hyperparameter robustness**: rKL-LD is insensitive to the initial value of $\sigma_{\text{diff}}$, converging to similar optimal solutions across different initializations.

## Highlights & Insights

- The theoretical contributions are rigorous: a counterexample is provided to prove LV violates DPI, and gradient discrepancies between LV and rKL-LD are systematically analyzed across three parameter types ($\alpha$, $\phi$, $\nu$).
- Optimal diffusion coefficients are found to vary substantially across dimensions (Figure 1, middle panel), confirming the value of dimension-wise adaptive noise regulation.
- The work addresses a longstanding practical frustration in the diffusion bridge community: why LV loss frequently requires careful tuning and exhibits unstable training.

## Limitations & Future Work

- rKL-LD remains susceptible to mode collapse due to the mode-seeking nature of rKL, particularly under suboptimal hyperparameters.
- Only time-invariant diffusion coefficients are considered; time-dependent $\sigma_{\text{diff}}(t)$ is identified as a promising future direction.
- A comprehensive comparison with LV in settings combining off-policy buffers and MCMC updates has not been conducted.

## Related Work & Insights

- A direct connection exists with the Trajectory Balance loss in GFlowNets (which is essentially the LV loss), and the theoretical analysis presented here carries implications for the GFlowNet community.
- Discrete-domain diffusion samplers have successfully employed rKL-LD (e.g., in combinatorial optimization and spin-lattice statistical physics); this work extends the approach to continuous-domain diffusion bridges.
- Insight: the choice of loss function must account for parameter sharing structure—special care is required when parameters are shared between forward and reverse processes.

## Rating

- Novelty: ⭐⭐⭐⭐ — The theoretical finding (LV violates DPI) is meaningful, though rKL-LD itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers two architectures (CMCD/DBS), three losses (rKL-R/LV/rKL-LD), Bayesian and synthetic targets, with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, though the notation is dense and requires careful cross-referencing.
- Value: ⭐⭐⭐⭐⭐ — Provides important practical guidance for the diffusion bridge sampling community with directly applicable improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Evaluation of Infrared Small Target Detection](rethinking_evaluation_of_infrared_small_target_detection.md)
- [\[ICCV 2025\] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography](../../ICCV2025/llm_evaluation/degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](../../ICCV2025/llm_evaluation/lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[NeurIPS 2025\] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories](not_all_splits_are_equal_rethinking_attribute_generalization_across_unrelated_ca.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](../../ICCV2025/llm_evaluation/rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)

</div>

<!-- RELATED:END -->
