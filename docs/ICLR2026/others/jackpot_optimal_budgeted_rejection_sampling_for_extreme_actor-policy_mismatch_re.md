---
title: >-
  [Paper Note] Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL
description: >-
  [ICLR 2026][rejection sampling] This paper proposes the Jackpot framework, which applies Optimal Budget Rejection Sampling (OBRS) to accept or reject rollout tokens at the token level within a controllable acceptance bud…
tags:
  - "ICLR 2026"
  - "rejection sampling"
  - "actor-policy mismatch"
  - "decoupled RL"
  - "importance sampling"
  - "OBRS"
  - "off-policy RL"
date: 2026-05-08
content_hash: 4db98fae3a72da58
---

# Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL

**Conference**: ICLR 2026
**arXiv**: [2602.06107](https://arxiv.org/abs/2602.06107)  
**Code**: [Infini-AI-Lab/jackpot](https://github.com/Infini-AI-Lab/jackpot)  
**Area**: Others
**Keywords**: rejection sampling, actor-policy mismatch, decoupled RL, importance sampling, OBRS, off-policy RL

## TL;DR

This paper proposes the Jackpot framework, which applies Optimal Budget Rejection Sampling (OBRS) to accept or reject rollout tokens at the token level within a controllable acceptance budget, and reweights the remaining samples. The method is theoretically proven to strictly reduce the KL divergence between the actor and policy under any budget. Combined with joint training and distillation of the rollout model, Jackpot enables a small model (e.g., Qwen3-1.7B) to serve as the rollout model for training a large model (e.g., Qwen3-8B), achieving performance close to the on-policy baseline.

## Background & Motivation

- **RL training bottleneck**: Approximately 80% of the computational cost in LLM RL training comes from rollout (autoregressive trajectory generation), which is the primary bottleneck for RL scaling.
- **Motivation for decoupling rollout**: Replacing the policy model (e.g., 8B) with a smaller and faster model (e.g., 1.7B) for rollout could substantially reduce training costs, but introduces extreme distribution mismatch.
- **Failure of existing methods**: Post-hoc correction methods such as Truncated Importance Sampling (TIS) and IceProp collapse during training under extreme mismatch (where KL divergence is an order of magnitude higher), because they only apply post-hoc reweighting and cannot reduce the distributional gap at the source.
- **Infeasibility of standard RS**: Classical rejection sampling can theoretically match the target distribution exactly, but LLM vocabularies exceed 100k tokens. Probability ratio spikes $p_i/q_i$ on a small number of tokens cause the normalization constant $\lambda$ to be extremely large, driving the acceptance rate toward zero.
- **Growing distributional gap**: In naïve decoupled training, the policy model is continuously updated while the rollout model remains fixed, causing the distributional gap to worsen progressively throughout training.
- **Core insight**: Rather than correcting the distributional gap post-hoc, it is preferable to reduce it at the source — replacing hard accept/reject rules with optimal rejection sampling under a budget constraint to maximally align distributions with a controlled sample loss.

## Method

### Overall Architecture

Jackpot consists of three core components:

1. **OBRS token rejection and reweighting**: After rollout sampling and before backpropagation, an accept/reject decision is made for each token; rejected tokens are masked out from the loss computation.
2. **Joint training objective**: The policy model (PPO with OBRS) and the rollout model (standard PPO + distillation) are optimized simultaneously.
3. **Efficient system implementation**: Top-k probability estimation with batch-level bias correction avoids full-vocabulary computation.

### Key Design 1: OBRS Acceptance Rule

For a token $x$ sampled from the rollout model $p_{\text{inf}}$, the acceptance probability is:

$$a(x) = \min\left(1, \frac{p_{\text{target}}(x)}{\lambda \cdot p_{\text{inf}}(x)}\right)$$

where $\lambda > 0$ is a user-specified budget parameter (smaller values lead to more rejections and more precise alignment). Unlike classical RS, which requires $\lambda \geq \max_i p_i/q_i$, OBRS permits arbitrary $\lambda$ while theoretically guaranteeing that the posterior distribution $\tilde{q}$ is strictly closer to the target distribution $p$:

$$D_{\text{KL}}(p \| \tilde{q}) \leq D_{\text{KL}}(p \| q)$$

The reweighted distribution after acceptance is:

$$P_{\text{OBRS}}(x) = \frac{\min\left(p_{\text{inf}}(x), \frac{p_{\text{target}}(x)}{\lambda}\right)}{Z}$$

### Key Design 2: Joint Training Objective

The total loss function comprises three terms:

$$\mathcal{L}^{\text{Jackpot}}(\theta, \omega) = \underbrace{\mathcal{L}^{\text{PPO-OBRS}}(\theta)}_{\text{Policy model RL}} + \underbrace{\mathcal{L}^{\text{PPO}}(\omega)}_{\text{Rollout model RL}} + \lambda_{\text{distill}} \underbrace{\mathcal{L}^{\text{distill}}(\omega)}_{\text{Online distillation}}$$

- **Policy model loss**: A PPO objective applied after OBRS masking and reweighting; rejected tokens do not contribute to gradient computation.
- **Rollout model PPO**: Standard PPO loss, enabling the rollout model to learn from rewards.
- **Distillation loss**: Forward KL $D_{\text{KL}}(\text{SG}(p_{\theta_{\text{new}}}) \| p_\omega)$, which causes the rollout model to continuously track improvements in the policy model, preventing the distributional gap from widening during training.

### Key Design 3: Top-k Approximation and Bias Correction

Computing the normalization constant $Z$ requires summation over the entire vocabulary (>100k tokens), which incurs prohibitive memory overhead. The proposed solution is:

- **Top-k approximation**: Summation is restricted to $\mathcal{V}_k = \text{top-k}(p_{\text{inf}}) \cup \text{top-k}(p_{\text{new}})$, exploiting the property that LLM output probabilities are concentrated on a small number of tokens.
- **Bias correction**: $Z_{\text{approx}}$ systematically underestimates the true $Z$. By leveraging the fact that $Z$ equals the expected acceptance rate $\bar{\alpha}$, a correction factor is computed using the empirical batch-level acceptance rate:

$$\kappa = \frac{\hat{\bar{\alpha}}}{\frac{1}{B}\sum_{i=1}^{B} Z_{\text{approx}}^{(i)}}$$

### Loss & Training

- **Target distribution selection**: $p_{\text{target}}$ can be set to either the reference policy $p_{\text{ref}}$ or the latest policy $p_{\text{new}}$; the latter is preferred for large-batch or asynchronous training.
- **No additional rollouts required**: All three loss terms share the same batch of rollout trajectories, incurring no additional sampling overhead.
- **No vLLM modification required**: The method is implemented directly on standard vLLM without custom operators or kernels.
- **No trajectory resampling**: Unlike speculative decoding, the remaining trajectory is retained unchanged after token rejection.

## Key Experimental Results

### Main Results: Joint Training under Extreme Actor-Policy Mismatch

| Training Configuration | GSM8K | MATH-500 | AMC22/23 | AMC12 | AIME24 Mean@4 | AIME25 Mean@4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-1.5B → 3B** (MATH-8k, 14k steps) | | | | | | |
| 3B On-policy | 85.00 | 63.90 | 37.65 | 26.11 | – | – |
| TIS + Reverse KL | 82.50 | 60.45 | 32.53 | 24.44 | – | – |
| **Jackpot** | **84.28** | **62.75** | **38.55** | **27.78** | – | – |
| **Qwen3-1.7B → 4B** (DeepScaleR, 20k steps) | | | | | | |
| 4B On-policy | 92.56 | 80.82 | 58.13 | 51.66 | 25.00 | 21.56 |
| TIS + Reverse KL | 91.21 | 73.65 | 46.39 | 32.77 | 13.33 | 10.41 |
| **Jackpot** | **92.15** | **80.52** | **59.49** | **53.88** | **23.50** | **20.83** |
| **Qwen3-1.7B → 8B** (DeepScaleR, 15k steps) | | | | | | |
| 8B On-policy | 93.29 | 79.50 | 61.14 | 53.33 | 24.37 | 16.87 |
| TIS + Reverse KL | 93.61 | 76.45 | 56.62 | 37.22 | 17.70 | 15.41 |
| **Jackpot** | **93.57** | **82.65** | **62.04** | **54.44** | **25.00** | **19.16** |

### Ablation Study: When Jackpot Helps and When It Does Not

| Scenario | Method | MATH-500 | AMC22/23 | AIME24 Mean@16 | AIME25 Mean@16 | Conclusion |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| Large batch (64×) | Off Policy | 81.55 | 60.54 | 27.50 | 23.12 | Small gap; Jackpot offers no additional benefit |
| Large batch (64×) | Jackpot | 81.95 | 59.94 | 27.71 | 22.70 | ≈ Comparable |
| FP8 KV quantization | TIS | 83.65 | 60.84 | 25.83 | 22.70 | TIS is sufficient |
| FP8 KV quantization | Jackpot | 81.30 | 62.35 | 24.79 | 22.29 | ≈ Comparable |
| **No PPO clip (128×)** | Off Policy | 60.20 | 33.00 | 8.00 | 5.00 | Severe training degradation |
| **No PPO clip (128×)** | No-Clip | 19.10 | 7.80 | 1.00 | 1.00 | Training collapse |
| **No PPO clip (128×)** | **Jackpot** | **80.00** | **51.20** | **19.16** | **18.52** | **Significant advantage** |

### Key Findings

1. **Jackpot substantially outperforms baselines under extreme mismatch**: With 1.7B rollout training an 8B policy, Jackpot matches or exceeds on-policy performance, whereas TIS falls 6 points behind on MATH-500 and 17 points behind on AMC12.
2. **KL divergence reduced by an order of magnitude**: Numerical simulations show that OBRS compresses KL divergence by 10× while maintaining acceptance rates above 90%.
3. **Training stability**: Unaligned baselines collapse within tens of steps; TIS experiences KL explosion after 100 steps; Jackpot trains stably for 300 steps.
4. **No benefit when the distributional gap is small**: When PPO clipping already sufficiently constrains the update step size, or when the gap introduced by FP8 quantization is minor, Jackpot performs on par with TIS.
5. **Stable without PPO clip**: Jackpot tolerates larger policy update steps without collapse, accelerating convergence.

## Highlights & Insights

- **Theoretically optimal closed-form solution**: Under a fixed acceptance budget, OBRS is the unique acceptance rule that minimizes $D_{\text{KL}}(p \| \hat{q})$, with rigorous proof provided.
- **Source-level problem resolution**: Unlike post-hoc corrections such as TIS, Jackpot filters mismatched tokens during the sampling stage itself, complementing importance sampling.
- **Engineering-friendly**: No modifications to the inference framework (vLLM) are required; no additional rollouts are needed; Top-k approximation with batch-level bias correction keeps memory overhead manageable.
- **Practical cost implications**: Using a 1.7B model for rollout to train an 8B policy can reduce inference costs by approximately 4× without performance degradation.
- **Elegant joint training design**: The policy model and rollout model are trained simultaneously; the distillation loss prevents the gap from widening; all three loss terms share the same data batch.

## Limitations & Future Work

- **Validation limited to mathematical reasoning**: All experiments are conducted on mathematical benchmarks such as GSM8K, MATH, and AIME; generalization to code generation, open-domain dialogue, and other tasks remains unverified.
- **Restricted to the same model family**: Both actor and policy models are drawn from the Qwen series; feasibility across architectures (e.g., Llama actor + Qwen policy) is unknown.
- **Policy model forward pass still required**: During training, a forward pass through the large model is still needed to compute $p_{\text{ref}}$ and $p_{\text{new}}$; the savings are primarily on rollout inference, while training-side computation is not reduced.
- **Hyperparameter sensitivity**: Parameters including $\lambda$ (acceptance budget), $\lambda_{\text{distill}}$ (distillation weight), and clip thresholds $c_1, c_2$ require careful tuning.
- **No benefit when the gap is small**: When the distributional gap is already minor (e.g., due to PPO clipping or FP8 quantization), Jackpot offers no advantage over simple TIS.

## Related Work & Insights

- **RL training systems for LLMs**: Frameworks such as Verl, AReal, TRL, and OpenRLHF optimize throughput but assume same-model rollout.
- **Distribution mismatch correction**: AReal uses IS ($F(x)=x$); Flash-RL/Llama-RL use TIS ($F(x)=\min(x,C)$); IceProp uses bidirectional truncation.
- **Inference acceleration approaches**: Asynchronous training, FP8 quantization, and speculative decoding reduce rollout costs but still require target model participation.
- **Rejection sampling theory**: Verine et al. (2024) introduced the original OBRS theory; this paper is the first to apply it to actor-policy alignment in LLM RL training.
- **Policy distillation**: The idea of online distillation of the rollout model draws on knowledge distillation, but is innovatively combined with OBRS to prevent gap expansion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing OBRS theory to actor-policy alignment in LLM RL is a clear and compelling contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Validation on mathematical reasoning tasks is thorough, but diverse tasks such as code generation and dialogue are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear; figures are informative; algorithmic pseudocode is complete.
- **Value**: ⭐⭐⭐⭐ The work has direct engineering significance for reducing LLM RL training costs and provides a viable path for decoupled rollout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)
- [\[ICLR 2026\] Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)
- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)

</div>

<!-- RELATED:END -->
