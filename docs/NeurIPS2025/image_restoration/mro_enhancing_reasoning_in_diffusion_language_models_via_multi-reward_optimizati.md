---
title: >-
  [Paper Note] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization
description: >-
  [NeurIPS 2025][Image Restoration][Diffusion language models] This work presents the first systematic analysis of the root cause behind the reasoning gap in diffusion language models (DLMs)—namely…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "Diffusion language models"
  - "multi-reward optimization"
  - "token correlation"
  - "reinforcement learning"
  - "SGRO"
date: 2026-05-08
content_hash: 52993b4eb57dcb6d
---

# MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.21473](https://arxiv.org/abs/2510.21473)  
**Code**: N/A  
**Area**: Image Restoration
**Keywords**: Diffusion language models, multi-reward optimization, token correlation, reinforcement learning, SGRO

## TL;DR

This work presents the first systematic analysis of the root cause behind the reasoning gap in diffusion language models (DLMs)—namely, the independent generation of tokens during denoising, which disrupts both intra- and inter-sequence correlations. A multi-reward optimization framework, MRO, is proposed and consistently improves reasoning performance of LLaDA-8B across test-time scaling, reject sampling, and RL paradigms, raising MATH500 accuracy from 34.4% to 37.4%.

## Background & Motivation

- **Background**: Diffusion language models (DLMs) such as LLaDA have emerged as a promising alternative to autoregressive LLMs. They generate text by iteratively denoising a fully masked sequence, restoring a subset of tokens at each step until the full output is recovered. Unlike autoregressive models that generate one token at a time, DLMs support parallel denoising and global planning.
- **Limitations of Prior Work**: DLMs still lag significantly behind autoregressive LLMs of comparable scale on reasoning tasks, and performance degrades further as the number of denoising steps decreases.
- **Key Challenge**: Through empirical analysis, the authors identify that the core bottleneck lies in the **independent generation** of masked tokens within each denoising step, which fails to capture dependencies among tokens. Reasoning tasks require strict logical chains and cross-step consistency, both of which are undermined by this independence assumption.
- **Goal**: The authors define two critical forms of token correlation: **intra-sequence correlation**—dependencies among tokens at different positions within the same denoising step—and **inter-sequence correlation**—consistency and cooperation among token sequences generated across different denoising steps. Experiments verify that enhancing both types of correlation improves reasoning, providing the theoretical foundation for the MRO framework.

## Method

### Overall Architecture

LLaDA-8B serves as the base DLM. Multiple reward signals are designed to capture intra- and inter-sequence correlations respectively → three complementary optimization strategies (Test-time Scaling: evaluation only, no model update; Reject Sampling: fine-tuning on high-reward samples; RL: direct reward optimization) → Step-wise Group Reward Optimization (SGRO) to reduce variance over long denoising trajectories → improved reasoning performance with reduced denoising steps.

### Key Designs

1. **Multi-Reward Signal Design**:
    - **Function**: Measure different aspects of token correlation throughout the denoising process.
    - **Token Verification Reward (TVR, intra-sequence)**: For the output sequence at a given denoising step, each token is individually masked and re-predicted; the average leave-one-out log-probability is computed. The paper theoretically shows that maximizing TVR approximates maximizing the average pairwise mutual information (PMI) among masked tokens, thereby promoting intra-sequence correlation.
    - **Perplexity Reward (PPL, intra-sequence)**: A lightweight external model (GPT-2-small) computes the perplexity of the generated sequence, capped at 100. This measures local fluency and coherence of the generated text.
    - **Answer Correctness + Format Reward ($R_0^q$, inter-sequence)**: A delayed reward applied only at the final denoising step, evaluating whether the answer is correct and properly formatted. This encourages all denoising steps to cooperate toward producing a high-quality final output.
    - **Design Motivation**: No single reward can simultaneously capture multi-dimensional token correlations. Ablation studies confirm that combining multiple rewards consistently outperforms any individual reward.

2. **Three Optimization Strategies**:
    - **Test-time Scaling (TTS)**: Multiple denoising trajectories are sampled at inference time with an increased computational budget; the reward function selects the best output. Model parameters are not updated; this strategy validates the effectiveness of the reward signals.
    - **Reject Sampling (RS)**: A large number of trajectories are sampled and high-reward examples are retained for fine-tuning the DLM, converting reward signals into training data.
    - **Reinforcement Learning (RL)**: The denoising process is modeled as an MDP, and policy gradients (REINFORCE++) are used to directly optimize the multi-reward objective. This is the most effective strategy but requires careful variance management.

3. **Step-wise Group Reward Optimization (SGRO)**:
    - **Function**: Reduce reward variance arising from long denoising trajectories in RL training.
    - **Mechanism**: The $T$ denoising steps are partitioned into groups of $w$ steps; rewards and gradients are computed at the group level rather than step-by-step via importance weighting. Steps within a group share the same group reward. Theoretically, SGRO reduces the dominant covariance term by increasing the temporal interval between evaluation points in potential-based reward shaping.
    - **Design Motivation**: Standard importance sampling suffers from exploding variance over long ($T$-step) denoising trajectories, which is the primary bottleneck for applying RL to DLMs. Experiments confirm that SGRO outperforms simple reward normalization.

### Loss & Training

- **Training data**: DeepScaleR mixed with Countdown/Sudoku tasks.
- RL training uses the REINFORCE++ algorithm.
- TVR computation is optimized via batch-parallelized masked evaluation (packing $[\_ BC, A\_C, AB\_]$ into a single batch for one forward pass).
- PPL reward is computed using GPT-2-small (lightweight) to reduce overhead.
- After training, inference follows the standard LLaDA procedure with no additional computational cost.

## Key Experimental Results

### Main Results

| Method | MATH500 (512 steps) | GPQA (512 steps) | Countdown (512 steps) |
|--------|--------------------|-----------------|-----------------------|
| LLaDA (baseline) | 34.4 | 30.3 | 14.1 |
| LLaDA-TTS + MRO | 36.0 | 34.6 | - |
| LLaDA-RS + MRO | 34.2 (+1.8) | 32.1 (+2.4) | - |
| LLaDA-RL + MRO | **37.4 (+3.0)** | **33.8 (+3.5)** | **27.2 (+13.1)** |

**Comparison with other RL methods (512 steps)**:

| Method | MATH500 | GPQA | Countdown |
|--------|---------|------|-----------|
| d1-LLaDA | 40.2 | - | - |
| RM-Baseline (REINFORCE++) | 36.4 | 31.7 | 22.7 |
| RL-Baseline (GRPO) | 35.6 | 32.1 | 21.4 |
| **LLaDA + MRO** | **37.4** | **33.8** | **27.2** |

**General tasks (RL mode)**:

| Model | MMLU | HumanEval | AlpacaEval2 | Arena-Hard |
|-------|------|-----------|-------------|------------|
| LLaDA | 65.5 | 47.6 | 16.3 | 10.0 |
| LLaDA-RS + MRO | 67.5 | 48.1 | 20.2 | 12.3 |
| LLaDA-RL + MRO | **68.2** | **50.0** | **19.4** | **15.7** |

### Ablation Study

| Reward Configuration | MATH500 | GPQA | Countdown |
|----------------------|---------|------|-----------|
| TVR only ($R_t^{tv}$) | 36.2 | 32.7 | 25.3 |
| PPL only ($R_t^{ppl}$) | 33.6 | 30.8 | 18.9 |
| $R_0^q$ only | 34.8 | 31.2 | 23.5 |
| MRO (all) | **37.4** | **33.8** | **27.2** |
| SGRO vs. Reward Normalization | 36.2 vs. 35.0 | 34.3 vs. 32.8 | - |

### Key Findings

- TVR is the most effective single reward, yet the multi-reward combination consistently outperforms any individual reward.
- SGRO significantly outperforms simple reward normalization, confirming its variance-reduction effect.
- The reasoning improvements also benefit inference under reduced denoising steps (faster sampling)—models trained with MRO exhibit smaller performance degradation when the step count is reduced.
- The inference-time computational overhead of MRO is identical to that of baseline LLaDA, as reward computation occurs only during training.
- The PPL reward cap parameter is robust across the range of 80–130, showing low sensitivity to this hyperparameter.

## Highlights & Insights

- **First systematic analysis of DLM reasoning bottlenecks from a token-correlation perspective**: The problem is attributed to the lack of intra- and inter-sequence correlations, providing actionable optimization directions.
- **Three strategies form a progressive hierarchy**: TTS (validate signals without modifying the model) → RS (data-driven fine-tuning) → RL (direct optimization), offering a complete pathway from simple to sophisticated.
- **Theoretical analysis of TVR is noteworthy**: Leave-one-out log-probability approximately maximizes pairwise mutual information, providing theoretical grounding for intra-sequence correlation optimization.
- **SGRO is a DLM-specific RL technique**: Tailored to address the variance problem in long denoising trajectories, it resolves a core technical obstacle in applying RL to DLMs.

## Limitations & Future Work

- Some mathematical notation is imprecise (reviewers noted that Eq. 3 directly equates marginal and joint distributions), and parts of the formal exposition require improvement.
- The boundary between intra-sequence and inter-sequence reward categories is ambiguous—answer correctness rewards also implicitly depend on intra-sequence correlation.
- Validation is limited to LLaDA-8B; performance on larger models (13B+) is unknown, as the largest publicly available DLM is currently 8B.
- A significant gap remains relative to autoregressive LLMs: LLaDA-RL+MRO achieves 37.4% on MATH500 versus 71.9% for Qwen2.5-7B.
- TVR computation requires a forward pass for each masked token; although it can be batched, it still increases training-time computational cost.

## Related Work & Insights

- **vs. d1-LLaDA**: MRO consistently outperforms d1-LLaDA under identical training settings (the only concurrent RL work for DLMs).
- **vs. RLHF/GRPO for autoregressive LLMs**: Autoregressive models enhance reasoning primarily through prompt engineering; DLMs must address token independence at the training level, representing a fundamentally different technical path. Autoregressive models already benefit from RLHF pre-training, leaving limited room for improvement, whereas DLMs have never been RL-trained, resulting in more substantial gains.
- **vs. Diffusion of Thoughts (DoT)**: DoT performs chain-of-thought reasoning within the diffusion framework, while MRO directly optimizes token correlation at the training level; the two approaches are complementary.
- **Insight**: The parallel generation capability of DLMs combined with MRO's correlation optimization holds promise for closing—or even surpassing—the gap with autoregressive models on long-chain reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to define intra/inter token correlation for DLM reasoning and propose a systematic RL optimization framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark, multi-strategy comparisons with comprehensive ablations; general-task evaluations and additional baselines added during rebuttal.
- Writing Quality: ⭐⭐⭐ Core ideas are clearly presented, but mathematical notation is insufficiently rigorous and reward categorization is ambiguous.
- Value: ⭐⭐⭐⭐ DLMs are an emerging direction, and MRO establishes the first systematic reasoning enhancement pathway for this model class.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/image_restoration/consistent_diffusion_language_models.md)
- [\[NeurIPS 2025\] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/image_restoration/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
