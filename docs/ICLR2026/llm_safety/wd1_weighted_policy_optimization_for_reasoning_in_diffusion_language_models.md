---
title: >-
  [Paper Note] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models
description: >-
  [ICLR 2026][LLM Safety][Diffusion Language Models] This paper proposes wd1, a ratio-free weighted log-likelihood policy optimization method for RL fine-tuning of diffusion language models (dLLMs). By combining positive-s…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Diffusion Language Models"
  - "Reinforcement Learning"
  - "Policy Optimization"
  - "Reasoning"
  - "dLLM"
date: 2026-05-08
content_hash: b7eeb46eb4315eb3
---

# wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models

**Conference**: ICLR 2026
**arXiv**: [2507.08838](https://arxiv.org/abs/2507.08838)  
**Code**: [https://github.com/xiaohangt/wd1](https://github.com/xiaohangt/wd1)  
**Area**: Image Restoration
**Keywords**: Diffusion Language Models, Reinforcement Learning, Policy Optimization, Reasoning, dLLM

## TL;DR
This paper proposes wd1, a ratio-free weighted log-likelihood policy optimization method for RL fine-tuning of diffusion language models (dLLMs). By combining positive-sample weighting with negative-sample penalization, wd1 avoids the bias and high variance introduced by policy ratio estimation in GRPO, achieving state-of-the-art performance of +59% on Sudoku and 84.5% on GSM8K over LLaDA-8B.

## Background & Motivation

**Background**: Diffusion language models (dLLMs) such as LLaDA and Dream have approached the text generation performance of autoregressive (AR) models. RL methods like RLHF and GRPO have substantially improved reasoning in AR models (e.g., DeepSeek-R1), but how to apply RL fine-tuning to dLLMs remains an open problem.

**Limitations of Prior Work**: The likelihood function of dLLMs is intractable and can only be approximated. Existing methods (e.g., d1, UniGRPO) that adapt GRPO to dLLMs must approximate the policy ratio as $r_i^k \approx \exp(\phi^{\pi_\theta} - \phi^{\pi_{old}})$, which introduces three issues: (a) approximation errors are amplified exponentially; (b) the ELBO estimate has high variance; (c) the likelihoods of three policies (current, old, reference) must be approximated simultaneously, incurring large computational overhead.

**Key Challenge**: The policy ratio is central to PPO/GRPO, yet the intractable likelihood of dLLMs makes ratio approximation unreliable. The core challenge is how to perform effective policy optimization without computing policy ratios.

**Goal**: To design an RL method that does not rely on policy ratios, requires only a single approximation of the current policy's likelihood, and makes full use of both positive and negative samples.

**Key Insight**: Starting from reverse-KL regularized policy optimization, the paper derives an analytic form of the optimal policy and then minimizes $D_{KL}(\pi^* \| \pi_\theta)$, converting optimization into weighted log-likelihood maximization—entirely free of policy ratios.

**Core Idea**: The RL objective is reformulated as a weighted log-likelihood (WLL), where weights are determined by the exponential of the advantage function. A negative-sample penalty term ($w^-$) is further introduced to actively reduce the likelihood of low-advantage completions, yielding wd1. The paper theoretically shows that wd1 is equivalent to energy-guided discrete diffusion training combined with negative-sample forgetting.

## Method

### Overall Architecture

Given a prompt $q$, the policy $\pi_\theta$ generates $G$ completions $\{o_i\}$, which are scored by a reward function $R(q, o_i)$. Group-relative advantages $\hat{A}_i = R(q, o_i) - \text{mean}(R)$ are computed, and the policy is updated via a weighted log-likelihood objective without computing any policy ratio.

### Key Designs

1. **Weighted Log-Likelihood Objective (WLL → wd1)**

    - **Function**: Weights log-likelihood training by the exponential of the advantage function.
    - **Mechanism**: Derived from reverse-KL constrained optimization, the optimal policy takes the form $\pi^* \propto \pi_{old}^{\lambda/(\lambda+\beta)} \cdot \pi_{ref}^{\beta/(\lambda+\beta)} \cdot \exp(A/(\lambda+\beta))$. Minimizing $D_{KL}(\pi^* \| \pi_\theta)$ yields the WLL objective. However, WLL has two drawbacks: low-advantage samples receive near-zero weight and are wasted; and even when all samples share the same reward, WLL still increases their likelihood. A negative-sample penalty is therefore introduced: $\mathcal{L}_{wd1} = \sum_i (-w^+ + w^-) \log \pi_\theta(o_i|q)$, where $w^+ \propto \exp(\psi \hat{A}_i)$ reinforces high-advantage samples and $w^- \propto \exp(-\psi \hat{A}_i)$ penalizes low-advantage ones.
    - **Design Motivation**: To avoid the exponential error amplification and high variance caused by policy ratios in GRPO. When all completions share equal advantage, $w^+ = w^-$ and optimization stops automatically, resolving the degenerate case of WLL.

2. **wd1++: Step-Level Weighted Policy Optimization**

    - **Function**: Leverages intermediate completions produced during the dLLM denoising process for training.
    - **Mechanism**: Standard wd1 uses only the final completion $o_i$; wd1++ extends the group to $O_i = \{x_{0|l}\}_{l=1}^L$, incorporating intermediate predictions from each denoising step. The DCE-based step-level objective is: $\mathcal{L}_{wd1++} = \frac{L}{Gl} \sum_i \sum_{x_{0|l}} (-w^+ + w^-) \log \pi_\theta(x_{0|l} | x_l, q)$.
    - **Design Motivation**: To fully exploit intermediate denoising products, substantially improving data efficiency—achieving better performance with 10× fewer rollouts.

3. **Theoretical Interpretation: Energy-Guided Diffusion + Negative-Sample Forgetting**

    - **Function**: Provides a theoretical foundation for wd1.
    - **Mechanism**: WLL is shown to be equivalent to advantage-weighted denoising cross-entropy (AW-DCE), i.e., training an energy-guided discrete diffusion model with the negative advantage as the energy function. The negative-sample penalty term is equivalent to minimizing the ELBO for data forgetting.
    - **Design Motivation**: To unify RL fine-tuning for dLLMs with the theoretical framework of energy-guided diffusion sampling.

### Loss & Training

- **wd1 Loss**: $\mathcal{L}_{wd1} = \frac{1}{G} \sum_{i=1}^G (-w^+(q,o_i) + w^-(q,o_i)) \cdot \log \pi_\theta(o_i | q)$
- LoRA fine-tuning is applied to LLaDA-8B-Instruct.
- In practice, $\beta=0, \lambda=1$ (reference policy regularization is removed).
- Likelihood approximation follows the d1 method: $\log \pi_\theta(x_0|q) \approx \sum_k \log \pi_\theta(x_0^k | x_1, q')$.
- $\mu=8$ gradient updates per step; weights are normalized across all groups to stabilize training.

## Key Experimental Results

### Main Results

| Method | Sudoku (256) | Countdown (256) | GSM8K (512) | MATH500 (512) |
|--------|-------------|-----------------|-------------|---------------|
| LLaDA-8B-Instruct | 6.7% | 19.5% | 78.2% | 36.2% |
| + diffu-GRPO | 16.1% | 27.0% | 80.7% | 39.0% |
| + d1 (SFT+GRPO) | 17.6% | 25.8% | 82.0% | 38.0% |
| + **wd1** | **76.4%** | **51.2%** | 82.3% | 39.0% |
| + **wd1++** | - | - | **84.5%** | **44.2%** |
| + MDPO | - | - | 83.7% | 43.8% |
| + TCR | - | - | 83.0% | 41.4% |

### Ablation Study

| Configuration | Sudoku | Countdown | Notes |
|---------------|--------|-----------|-------|
| wd1 (full) | 76.4% | 51.2% | Full model |
| $w^+$ only (WLL) | 50.2% | 39.5% | No negative penalty, −26% |
| $w^-$ only | 15.3% | 22.1% | Penalty only, no reinforcement |
| d1 | 17.6% | 25.8% | Baseline |

Training cost comparison (4×A100):
- d1: SFT 2.01h + RL 103.5s/step, FLOPs 9.92e15/step, NFEs (μ+2)/step
- wd1: No SFT + RL 81.16s/step, FLOPs 8.89e15/step, NFEs μ/step

### Key Findings
- **wd1 outperforms d1 by 59% on Sudoku** (76.4% vs. 17.6%) and by 25% on Countdown, demonstrating the substantial advantage of ratio-free methods on constrained reasoning tasks.
- **The negative-sample penalty is critical**: removing $w^-$ drops Sudoku accuracy from 76.4% to 50.2%, confirming that actively "forgetting" low-quality completions is essential.
- **wd1++ achieves SOTA with 10× fewer rollouts**: 84.5% on GSM8K and 44.2% on MATH500 in only 20 training steps.
- **No SFT stage required**: wd1 starts RL directly from the Instruct model, eliminating the 2-hour SFT phase required by d1.
- Per-step compute is reduced by ~22% (81.16s vs. 103.5s), as approximating the old and reference policy likelihoods is no longer needed.

## Highlights & Insights
- **Elegance of the ratio-free design**: By switching the KL direction (forward → reverse), the ratio dependency of TRPO/PPO is converted into weighted likelihood, an idea that may also be valuable for AR models.
- **Dual design of $w^+$/$w^-$**: The combination of positive-sample weighting (increasing the probability of good outputs) and negative-sample penalization (decreasing the probability of bad outputs) automatically halts when advantages are equal—a clever self-balancing mechanism.
- **Unified theory via energy-guided diffusion**: Interpreting RL fine-tuning as energy-guided diffusion training provides a new theoretical framework for understanding and improving RL in dLLMs.
- **Exploitation of intermediate steps (wd1++)**: Training on intermediate denoising products is a unique advantage of dLLMs that AR models cannot exploit.

## Limitations & Future Work
- Validation is limited to LLaDA-8B; generalization to other dLLMs (e.g., Dream, DiffuCoder) remains to be tested.
- Likelihood approximation still uses the biased d1 method ($t=1$ sampling); better approximations may yield further gains.
- The potential for combining wd1 with RLHF (human feedback) has not been explored.
- wd1++ requires storing intermediate denoising completions, increasing memory overhead.

## Related Work & Insights
- **vs. d1 (Zhao et al., 2025)**: d1 adapts GRPO to dLLMs but retains policy ratio computation. wd1 eliminates ratios entirely, reducing both error and computation.
- **vs. UniGRPO (Yang et al., 2025)**: UniGRPO estimates the likelihood via DCE by sampling multiple values of $t$, which is more accurate but slower. wd1 requires only a single approximation of the current policy.
- **vs. MDPO (He et al., 2025)**: MDPO adopts DPO-style preference optimization. wd1++ slightly outperforms MDPO on both GSM8K and MATH500.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Innovations on three levels: ratio-free design, unification with energy-guided diffusion theory, and exploitation of intermediate denoising steps.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark evaluation, ablation studies, and compute cost analysis, though only a single base model is used.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, though the paper is dense.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core technical bottleneck in RL for dLLMs, delivering SOTA performance with significant compute savings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)
- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](../../ACL2026/llm_safety/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
