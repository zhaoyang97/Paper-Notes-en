---
title: >-
  [Paper Note] QuRL: Efficient Reinforcement Learning with Quantized Rollout
description: >-
  [ICLR 2026][Reinforcement Learning][quantized inference] This paper proposes QuRL, a method that quantizes the actor model to accelerate the rollout phase in RL training. It introduces Adaptive Clipping Range (ACR) to ad…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "quantized inference"
  - "RL acceleration"
  - "PPO"
  - "GRPO"
  - "importance sampling"
date: 2026-05-08
content_hash: 6fd74c695acaa7e5
---

# QuRL: Efficient Reinforcement Learning with Quantized Rollout

**Conference**: ICLR 2026
**arXiv**: [2602.13953](https://arxiv.org/abs/2602.13953)  
**Code**: None  
**Area**: Reinforcement Learning / Model Quantization
**Keywords**: quantized inference, RL acceleration, PPO, GRPO, importance sampling

## TL;DR

This paper proposes QuRL, a method that quantizes the actor model to accelerate the rollout phase in RL training. It introduces Adaptive Clipping Range (ACR) to address training collapse caused by quantization, and Update-Aware Quantization (UAQ) to resolve the scale mismatch between weight updates and quantization error. QuRL achieves 20%–80% inference throughput improvement without performance degradation.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become the dominant paradigm for training reasoning-oriented LLMs (e.g., DeepSeek-R1, OpenAI-O1), yet the rollout phase—due to the sequential dependency of autoregressive decoding—consumes approximately 70% of total training time.

**Limitations of Prior Work**: (1) Rollout is the efficiency bottleneck of RL training, further exacerbated by the longer chain-of-thought sequences required in reasoning tasks; (2) directly applying quantization to rollout introduces importance sampling bias and training instability; (3) the trust-region constraint in RL results in weight update magnitudes ($\sim 10^{-7}$) far smaller than quantization error, causing the quantized model to be nearly insensitive to training dynamics.

**Key Challenge**: While quantization can significantly accelerate inference, the policy divergence between the quantized actor and the full-precision actor undermines the importance sampling and clipping mechanisms of PPO/GRPO.

**Goal**: To exploit quantization for accelerating rollout inference while preserving RL training quality.

**Key Insight**: Building on Decoupled PPO, which separates the behavior policy from the proximal policy, the paper addresses two unique challenges introduced by quantization: clipping instability and weight updates being masked by quantization noise.

**Core Idea**: Use a quantized model for rollout while retaining the full-precision model for clipping and gradient updates; bridge the quantization gap via adaptive clipping range and invariant scaling techniques.

## Method

### Overall Architecture

Within the RL training loop: (1) the old actor $\theta_{\text{old}}$ is quantized to $\hat{\theta}_{\text{old}}$ for rollout response generation; (2) the full-precision $\theta_{\text{old}}$ is used to compute the proximal policy clipping ratio $R_{i,t}$; (3) training proceeds with the Decoupled PPO objective, with ACR dynamically adjusting the clipping range; (4) UAQ preprocesses weights via invariant scaling to reduce quantization error and amplify weight updates.

### Key Designs

1. **Adaptive Clipping Range (ACR)**:
    - Function: Resolves long-term training collapse caused by quantized rollout.
    - Mechanism: In Decoupled PPO, the behavior policy (quantized actor $\pi_{\hat{\theta}_{\text{old}}}$) is separated from the proximal policy (full-precision $\pi_{\theta_{\text{old}}}$). FlashRL's TIS method stabilizes training by truncating $\min(\pi_{\theta_{\text{prox}}}/\pi_{\theta_{\text{behav}}}, C)$, but this implicitly introduces a scaling factor $r_{i,t}$ that over-clips sequences with positive advantage. ACR adjusts the upper bound to $(1+\epsilon)/r_{i,t}$: $\mathcal{J}_{\text{ACR}} = \tilde{\mathbb{E}}[\min(\pi_{\text{prox}}/\pi_{\text{behav}}, C) \cdot \min(R_{i,t}A_{i,t}, \text{clip}(R_{i,t}, 1-\epsilon, (1+\epsilon)/r_{i,t})A_{i,t})]$
    - Design Motivation: In late training (>1000 steps), the KL divergence between the quantized and full-precision actors grows from 0.002 to 0.025 (a 12× increase), and TIS's fixed truncation leads to biased gradient estimates. ACR dynamically relaxes the clipping upper bound based on policy divergence, allowing more positive-advantage tokens to contribute to training.

2. **Update-Aware Quantization (UAQ)**:
    - Function: Resolves the scale mismatch between weight quantization changes and weight update magnitudes.
    - Mechanism: Exploits the invariant scaling property of linear layers, $WX = (W/s) \cdot (sX)$. Choosing $s > 1$ reduces quantization error by a factor of $s$ (error $\propto |\theta|/(s \cdot 2^b)$) while amplifying weight updates by $s$ (update $\propto s \cdot \alpha G$), yielding an $s^2$ improvement in signal-to-noise ratio. The scaling factor $s$ is applied column-wise to $W$ and row-wise to the preceding layer's activations (absorbable into LayerNorm).
    - Design Motivation: In RL, the learning rate $\alpha \sim 10^{-6}$ and gradient $G \sim 0.1$–$1.0$ result in weight update magnitudes of $\sim 10^{-7}$, far smaller than quantization error (weight norm $\sim 0.001$–$0.1$). Experiments show that INT8 quantization effectively masks all weight updates, leaving the quantized model functionally frozen.

3. **System Integration and Engineering**:
    - Function: Integrates ACR and UAQ into the VeRL RL training framework.
    - Mechanism: UAQ is a one-time weight preprocessing step executed before RL training, incurring no additional training overhead. ACR modifies only the clipping logic with negligible computational cost. Inference is accelerated via vLLM's INT8/FP8 matrix multiplication kernels.
    - Design Motivation: QuRL occupies a novel position between PTQ and QAT—unlike QAT, it does not explicitly optimize for quantization, yet parameters are implicitly updated through gradients derived from quantized model outputs, necessitating a purpose-designed quantization strategy.

### Loss & Training

Decoupled variants of GRPO/DAPO objectives are adopted. Channel-wise weight quantization combined with token-wise activation quantization (INT8 or FP8) is applied. The UAQ scaling factor is set to $s = 1.5$. The TIS truncation threshold $C$ follows the FlashRL configuration.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (QuRL) | FlashRL | BF16 Baseline | Notes |
|--------|------|------|------|------|------|
| GSM8K (INT8) | Accuracy | 53.55 | 51.40 | 55.35 | Gap narrowed from 4% to 1.8% |
| GSM8K (FP8) | Accuracy | 54.28 | 53.60 | 55.35 | Gap reduced to only 1.1% |
| AIME2024 (INT8) | Avg@32 | 31.25 | 30.29 | 31.67 | Nearly lossless w/ UAQ |
| AIME2024 (FP8) | Avg@32 | 33.27 | 32.60 | 31.67 | FP8 surpasses BF16! |
| DeepScaleR (INT8) | Avg 5 tasks | 55.48 | 53.80 | 56.40 | Gap narrowed from 4.1% to 0.9% |
| DeepScaleR (INT8) | AIME24 | 40.52 | 36.77 | 40.73 | Nearly matches full precision |

### Ablation Study

| Configuration | AIME24 Avg@32 | Notes |
|------|------|------|
| QuRL w/o UAQ (INT8) | 30.63 | Baseline |
| QuRL w/ UAQ s=1.5 | 31.25 | +0.62, optimal scaling |
| QuRL w/ s=2.0 | 29.15 | Excessive scaling causes instability |
| Direct LR increase 1.5× | 29.06 | Less effective than UAQ |
| Direct LR increase 2× | 26.66 | Severe degradation |

### Key Findings

- INT8 quantization accelerates 7B models by 20–30% and 32B models by 70–90% on H100; larger models benefit more due to compute-bound matrix multiplication.
- Naïve INT8 RL causes reward to collapse to zero on DAPO tasks; ACR is critical for stable training.
- UAQ's $s^2$ signal-to-noise improvement narrows the performance gap from 1.61% to 0.92% on 7B + DeepScaleR.

## Highlights & Insights

- The paper precisely diagnoses two core failure modes of quantized RL training: clipping breakdown and weight updates being overwhelmed by quantization noise. The latter—weight update magnitudes of $10^{-7}$ versus quantization errors of $10^{-3}$–$10^{-1}$—is a fundamental challenge that had not been previously recognized.
- UAQ's design is elegant: by leveraging invariant scaling to simultaneously "shrink the denominator and amplify the numerator," it achieves an $s^2$ improvement through a single operation with virtually zero computational overhead.

## Limitations & Future Work

- Validation is limited to INT8 and FP8 precisions; 4-bit quantization could yield larger speedups but poses greater challenges and remains unexplored.
- FP8 KV cache quantization is not optimized in the current vLLM implementation, limiting practical acceleration.
- Experiments cover models up to 32B; applicability to larger models (e.g., 70B+) has not been verified.

## Related Work & Insights

- **vs. FlashRL**: FlashRL proposes TIS + Decoupled PPO for quantized rollout but still exhibits performance degradation in late training; QuRL's ACR resolves this long-term training collapse.
- **vs. Standard PTQ/QAT**: QuRL occupies a novel intermediate position—weights are quantized once per round but are implicitly influenced by gradients, requiring purpose-designed quantization strategies.
- **vs. Complex PTQ methods (e.g., GPTQ)**: Although GPTQ may capture finer-grained weight changes, per-step recalibration is prohibitively expensive for RL settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of quantization effects on RL training; ACR and UAQ are cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three algorithms (PPO/GRPO/DAPO), multiple model scales, and precision formats.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis with clear visualization of failure modes.
- Value: ⭐⭐⭐⭐ Directly addresses the core efficiency bottleneck of RL training with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EchoRL: Reinforcement Learning via Rollout Echoing](../../ICML2026/reinforcement_learning/echorl_reinforcement_learning_via_rollout_echoing.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICML 2026\] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](../../ICML2026/reinforcement_learning/darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)

</div>

<!-- RELATED:END -->
