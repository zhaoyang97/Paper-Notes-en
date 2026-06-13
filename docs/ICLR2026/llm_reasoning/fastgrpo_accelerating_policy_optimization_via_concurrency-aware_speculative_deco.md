---
title: >-
  [Paper Note] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning
description: >-
  [ICLR 2026][LLM Reasoning][GRPO Acceleration] Targeting the severe bottleneck in GRPO training where the generation phase consumes 91%–98% of total training time…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "GRPO Acceleration"
  - "Speculative Decoding"
  - "Concurrency-aware"
  - "Online Draft Learning"
  - "Reinforcement Learning Training"
date: 2026-05-08
content_hash: 48dc53b0f9534846
---

# FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning

**Conference**: ICLR 2026
**arXiv**: [2509.21792](https://arxiv.org/abs/2509.21792)  
**Code**: [GitHub](https://github.com/yedaotian9/GRPO_speculative)  
**Area**: LLM Reasoning
**Keywords**: GRPO Acceleration, Speculative Decoding, Concurrency-aware, Online Draft Learning, Reinforcement Learning Training

## TL;DR

Targeting the severe bottleneck in GRPO training where the generation phase consumes 91%–98% of total training time, this work proposes a concurrency-aware speculative decoding strategy (dynamically adjusting draft tree parameters to accommodate the real-time shift from high to low concurrency) and online draft model learning (continuously adapting to distribution drift using hidden states produced by the target model). The combined approach achieves 2.35×–2.72× end-to-end training speedup without degrading reasoning quality.

## Background & Motivation

- **Background**: GRPO is the predominant RL framework for enhancing LLM reasoning capabilities (e.g., DeepSeek-R1, DAPO), but its throughput is far lower than SFT training, severely hindering experimental iteration speed.
- **Bottleneck Quantification**: The generation phase (rollout sampling) accounts for 91%–98% of total GRPO training time. More critically, as model capability grows and outputs become longer, the generation-to-update time ratio increases from 6× to over 20×, causing the problem to continuously worsen.
- **High-Concurrency Challenge for Speculative Decoding**: Standard speculative decoding is effective at low concurrency ($B=1$) but provides little to no speedup (speedup $<$ 1.0×) under the high-concurrency (large-batch) settings typical of GRPO. The reason is that the additional computational overhead shifts the system from memory-bound to compute-bound.
- **GRPO's Unique Dynamic Concurrency Property**: Effective concurrency changes dynamically during generation—starting at a high batch size, sequences finish at different times (with length variation up to 3–5×), causing effective concurrency to gradually decrease toward 1.
- **Distribution Drift Problem**: As the target model is continuously updated during training, the distributional gap between the target and the fixed draft model widens, causing the speculative acceptance rate to decline and the speedup to diminish over training steps.
- **Limitations of Prior Work**: EAGLE-2/HASS/EAGLE-3 achieve only 1.1×–1.3× speedup within the GRPO framework, far below what is achieved in standard inference scenarios.

## Method

### Overall Architecture

FastGRPO = Concurrency-aware Speculative Decoding (accelerating the generation phase) + Online Draft Learning (maintaining the draft model during the parameter update phase). The two components work synergistically: the former maximizes hardware utilization across varying concurrency levels, while the latter ensures that speedup does not decay over the course of training.

### Key Designs

**1. Concurrency-aware Speculative Decoding**:
- **Core Idea**: Keep the effective batch size at the verification stage consistently at the hardware-optimal concurrency level $C_{\text{peak}}$ (the inflection point at which the GPU transitions from memory-bound to compute-bound).
- **Verification token count**: $N_{\text{verify}} = C_{\text{peak}} / B$, where $B$ is the current effective batch size.
- **Draft tree width**: $K_{\text{draft}} = \min(N_{\text{verify}}-1, K^{\max})$
- **Draft tree depth**: $L_{\text{draft}} = \min(\lfloor\log_2(N_{\text{verify}}/\alpha)\rfloor, L^{\max})$, where $\alpha$ encodes the draft model quality.
- **Runtime Behavior**: Early in training, high concurrency leads to conservative speculation (small trees) to avoid compute bottlenecks; later, as sequences complete and concurrency drops, aggressive speculation (large trees) fully exploits idle compute resources.
- **Theoretical Basis**: An operational intensity analysis elegantly connects GEMM characteristics to speculative decoding hyperparameters: $I_{\text{GEMM}} \approx 2B/s$.

**2. Online Draft Learning**:
- At each GRPO iteration, the draft model is updated using hidden states produced by the current target model as supervision signals.
- Additional computational overhead is only 2%–3%, since hidden states are naturally produced and cacheable during the generation phase—effectively a "free" supervision signal.
- Effect: The accepted length continuously increases throughout training, whereas a fixed draft model exhibits continuous decline.
- Even when pre-training is entirely skipped, online learning converges from scratch to a comparable acceptance rate within 1–2 epochs.

### Loss & Training

- Alternating freezing: the target model is frozen during draft training, and the draft model is frozen during GRPO updates.
- The draft model adopts the EAGLE architecture, pre-trained for 10 epochs on ShareGPT-68K with learning rate 1e-4 (AdamW).
- Rollout data with zero reward, while unusable for target model updates, can still serve as valid training signal for the draft model, reducing data waste.

## Key Experimental Results

### Main Results

| Model | Method | GSM8K E2E SR | SimpleRL E2E SR | DAPO E2E SR | Avg. |
|-------|--------|:-----------:|:---------------:|:-----------:|:----:|
| Qwen2.5-7B-I | EAGLE-3 | 1.26× | 1.20× | 1.13× | 1.20× |
| Qwen2.5-7B-I | **FastGRPO** | **2.43×** | **2.52×** | **2.53×** | **2.49×** |
| Llama3.1-8B-I | EAGLE-3 | 1.31× | 1.28× | 1.23× | 1.27× |
| Llama3.1-8B-I | **FastGRPO** | **2.51×** | **2.69×** | **2.67×** | **2.62×** |
| DS-R1-Qwen-7B | **FastGRPO** | **2.69×** | — | — | — |

### Ablation Study

| Configuration | Generation SR | End-to-End SR | Notes |
|---------------|:------------:|:-------------:|-------|
| FastGRPO (full) | 2.91 | 2.52 | Optimal configuration |
| w/o Online Draft Learning | 2.16 | 2.01 | Online learning contributes 0.5× speedup |
| w/o Concurrency-aware | 2.59 | 2.30 | Concurrency-awareness contributes 0.2× speedup |
| Vanilla + early termination | 1.68 | 1.61 | Baseline comparison |

### Key Findings

- Comprehensive validation across 5 models (Qwen2.5-7B/1.5B-I, Llama3.1-8B-I, DS-R1-Qwen-7B, Qwen2.5-Math-7B) × 3 datasets; FastGRPO consistently outperforms all baselines by more than 2×.
- Mathematical reasoning accuracy after training is largely on par with standard GRPO, and even slightly higher in some cases—acceleration does not compromise quality.
- The method is equally effective on GRPO variants (DAPO/GPG), achieving end-to-end speedups exceeding 2×.
- The contribution of online draft learning (0.5×) exceeds that of concurrency-awareness (0.2×), indicating that distribution drift is the larger bottleneck.

## Highlights & Insights

- **Discovery of GRPO's Unique Dynamic Concurrency Property**: This represents a fundamental distinction between GRPO and standard inference scenarios, yet has been overlooked by all prior work.
- **Elegance of the Theoretical Analysis**: Connecting hardware characteristics to speculative decoding hyperparameters via operational intensity analysis provides a principled basis for strategy design.
- **Ingenuity of Online Draft Learning**: Reusing already-available hidden states as supervision signals incurs only 2%–3% extra overhead—essentially a "free lunch."
- **Rapid Deployment Capability**: Even without pre-training the draft model, online learning reaches full effectiveness within 1–2 epochs.

## Limitations & Future Work

- $C_{\text{peak}}$ requires empirical profiling for each GPU/model combination; automated profiling tools would improve usability.
- Validation is limited to mathematical reasoning datasets; effectiveness on code generation, general reasoning, and dialogue scenarios remains unknown.
- The draft model architecture is fixed to the EAGLE family; adaptation to other architectures such as Medusa has not been explored.
- Interactions between communication overhead and the concurrency-aware strategy in multi-node distributed training settings are not discussed.
- The $\alpha$ hyperparameter encodes draft model quality; dynamic adjustment across training stages may yield further improvements.

## Related Work & Insights

- **vs. EAGLE-2/HASS/EAGLE-3**: These methods provide limited speedup under GRPO's high-concurrency settings; FastGRPO's core innovation lies in dynamically adapting to concurrency variation.
- **vs. Standard Inference Acceleration**: Speculative decoding has traditionally targeted low-concurrency inference deployment; FastGRPO is the first to adapt it to the high-concurrency setting of RL training.
- **Insight**: RL training and inference deployment are fundamentally different (dynamic concurrency, distribution drift); designing acceleration strategies specifically for these characteristics yields substantial gains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The combination of concurrency-awareness and online draft learning is highly insightful)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 models × 3 datasets + ablations + variant validation)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical analysis is clear; experimental layout is well-organized)
- Value: ⭐⭐⭐⭐⭐ (A plug-and-play solution for GRPO training acceleration with extremely high engineering value)

## Rating

- Novelty: ⭐⭐⭐⭐ Targeted combination of concurrency-awareness and online learning
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 3 datasets + GRPO variant transfer + comprehensive ablation
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from motivation → observation → method
- Value: ⭐⭐⭐⭐⭐ Directly reduces GRPO training cost by 2–3×; extremely high engineering value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICML 2026\] UCPO: Uncertainty-aware Policy Optimization](../../ICML2026/llm_reasoning/ucpo_uncertainty-aware_policy_optimization.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](../../ACL2026/llm_reasoning/calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
