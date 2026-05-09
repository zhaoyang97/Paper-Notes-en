---
title: >-
  [Paper Note] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation
description: >-
  [ICLR 2026][LLM Evaluation][prompt optimization] This paper proposes MetaTuner, a framework that employs a shared meta-encoder to simultaneously generate query-specific prompts and LoRA parameters, enabling mutual reinforcement between prompt optimization and fine-tuning. A supervised regularization loss is designed to address the mixed discrete-continuous optimization problem. MetaTuner consistently outperforms standalone prompt optimization and fine-tuning methods on MATH, GSM8K, HotpotQA, and CosmosQA.
tags:
  - ICLR 2026
  - LLM Evaluation
  - prompt optimization
  - fine-tuning
  - joint optimization
  - LoRA
  - discrete-continuous optimization
date: 2026-05-08
content_hash: 64e5a88a0dffab68
---

# Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation

**Conference**: ICLR 2026
**arXiv**: [2509.24245](https://arxiv.org/abs/2509.24245)
**Code**: [GitHub](https://github.com/BoXiaohe/MetaTuner)
**Area**: LLM Evaluation
**Keywords**: prompt optimization, fine-tuning, joint optimization, LoRA, discrete-continuous optimization

## TL;DR

This paper proposes MetaTuner, a framework that employs a shared meta-encoder to simultaneously generate query-specific prompts and LoRA parameters, enabling mutual reinforcement between prompt optimization and fine-tuning. A supervised regularization loss is designed to address the mixed discrete-continuous optimization problem. MetaTuner consistently outperforms standalone prompt optimization and fine-tuning methods on MATH, GSM8K, HotpotQA, and CosmosQA.

## Background & Motivation

### Two Paradigms for LLM Enhancement

Two complementary strategies exist for enhancing large language models:

- **Prompt Optimization**: Optimizes inputs via explicit natural language to activate the model's existing knowledge; suitable for generalization but struggles with complex patterns in large-scale data.
- **Fine-tuning**: Adapts models through implicit parameter updates; effective at learning complex patterns but highly sensitive to the choice of prompt during training.

### Core Problem

Both methods have inherent limitations, and **prior work treats them independently**, leaving their synergistic potential unexplored:

- Fine-tuning under suboptimal prompts can degrade performance significantly—sometimes below pure prompt optimization.
- Knowledge encoded in prompts may conflict with model parameters.
- Their optimization spaces differ (prompts are discrete, parameters are continuous), posing technical challenges for joint optimization.

### Unified Formulation

Prompts and parameters are unified under a single optimization objective:

$$\min_{\theta, p_i} \sum_{i=1}^{N} \mathcal{L}(\mathcal{M}_\theta(p_i, x_i), y_i)$$

where $\theta$ denotes model parameters and $p_i$ denotes input-specific prompts. Prompts are treated as "special parameters," and the goal is to find the optimal prompt-parameter combination.

## Method

### Overall Architecture

MetaTuner consists of three core components:

1. **Meta Encoder**: A shared base encoder $\phi_s$ that encodes input queries.
2. **Prompt Decoder**: Parameters $\phi_p$ that generate query-specific natural language prompts from encoded representations.
3. **Parameter Decoder**: Parameters $\phi_q$ that generate query-specific LoRA parameters from encoded representations.

### Continuous Relaxation of Prompt Generation

To address the discrete optimization problem, an initial prompt $\tilde{p}$ is first specified and rewritten by an LLM $\mathcal{G}$:

$$p_i = \mathcal{G}_\phi(\tilde{p}, x_i)$$

The objective is then reformulated as optimization over continuous parameters $\phi$:

$$\min_{\theta, \phi} \sum_{i=1}^{N} \mathcal{L}(\mathcal{M}_\theta(\mathcal{G}_\phi(\tilde{p}, x_i), x_i), y_i)$$

### Key Designs: Shared-Private Parameter Mechanism

The prompt generator $\mathcal{G}$ and the parameter generator $\mathcal{F}$ share an encoding layer $\phi_s$:

$$\min_{\phi_s, \phi_p, \phi_q} \sum_{i=1}^{N} \mathcal{L}(\mathcal{M}_{\mathcal{F}_{(\phi_s, \phi_q)}(\tilde{p}, x_i)}(\mathcal{G}_{(\phi_s, \phi_p)}(\tilde{p}, x_i), x_i), y_i)$$

- Shared parameters $\phi_s$: enable mutual regularization between the two methods.
- Private parameters $\phi_p, \phi_q$: preserve independent exploration spaces for each branch.

### Parameter Decoder Implementation

LoRA matrices are derived from hidden states $h_i$ as follows:

$$\theta_i^b = \text{MM}(\text{ReLU}(\text{MM}(W_d^b, h_i)), W_u^b)$$
$$\theta_i^a = \text{MM}(\text{ReLU}(\text{MM}(W_d^a, h_i)), W_u^a)$$

where $\text{MM}$ denotes matrix multiplication and $W_d, W_u$ are learnable parameters of the parameter decoder.

### Loss & Training: Supervised Regularization

To address the non-differentiability of prompt generation, two loss terms are designed:

$$\min_{\phi_s, \phi_p, \phi_q} \underbrace{\sum_{(x_i, y_i) \in D_1} \mathcal{L}(\mathcal{M}_{\mathcal{F}(\tilde{p}, x_i)}(\mathcal{G}_{(\phi_s, \phi_p')}(\tilde{p}, x_i), x_i), y_i)}_{\text{Main Task Loss (frozen } \phi_p' \text{)}} + \underbrace{\sum_{(x_i, p_i) \in D_2} \alpha \cdot \mathcal{L}(\mathcal{G}_{(\phi_s, \phi_p)}(\tilde{p}, x_i), p_i)}_{\text{Supervised Regularization}}$$

- **First term**: Task loss with frozen $\phi_p'$; gradients flow through the parameter decoder.
- **Second term**: Supervises the prompt decoder using an expert dataset $D_2$ (optimal prompts generated via model rollouts).
- $\phi_p$ is periodically synchronized to $\phi_p'$ at fixed intervals.

### Framework Specifications

- Prompt generator $\mathcal{G}$: Qwen2.5-7B, with the first $k$ layers as the meta-encoder and subsequent layers as the prompt decoder.
- Downstream model $\mathcal{M}$: Qwen2.5-3B, fine-tuned with generated LoRA parameters.
- Two training strategies: MetaTuner-I (alternating optimization) and MetaTuner-J (joint optimization).

## Key Experimental Results

### Main Results: Comprehensive Comparison on 4 Datasets

| Method | MATH | GSM8K | HotpotQA | CosmosQA |
|--------|------|-------|----------|----------|
| **Prompt Optimization** | | | | |
| RLPrompt | 31.33 | 53.15 | 43.00 | 81.20 |
| BPO | 32.67 | 58.00 | 43.90 | 82.05 |
| OPRO | 22.00 | 75.06 | 25.55 | 69.10 |
| **Fine-tuning** | | | | |
| SFT | 41.33 | 61.41 | 43.20 | 82.65 |
| DPO | 43.78 | 63.68 | 44.70 | 87.90 |
| PPO | 41.78 | 62.02 | 45.85 | 84.10 |
| **Hybrid Methods** | | | | |
| BetterTogether | 41.56 | 67.93 | 52.30 | 89.80 |
| **MetaTuner-I** | 48.22 | 78.54 | **55.75** | 92.15 |
| **MetaTuner-J** | **48.67** | **78.92** | 54.56 | **92.25** |

MetaTuner achieves substantial improvements across all four datasets: +4.89 on MATH (vs. DPO), +10.99 on GSM8K (vs. BetterTogether), and +3.45 on HotpotQA.

### Ablation Study

| Variant | MATH | GSM8K | HotpotQA | CosmosQA |
|---------|------|-------|----------|----------|
| MetaTuner (w/o F) — no fine-tuning | 48.00 | 77.79 | 54.05 | 91.10 |
| MetaTuner (w/o P) — no prompt | 46.22 | 78.54 | 53.90 | 91.00 |
| MetaTuner (w/o S) — no sharing | 46.67 | 77.86 | 53.65 | 91.50 |
| **MetaTuner (Full)** | **48.67** | **78.92** | **54.56** | **92.25** |

Removing any single component causes approximately 1% absolute accuracy degradation; the shared mechanism contributes most significantly.

### Additional Key Experiments

**Sharing Depth Analysis**: The optimal sharing ratio for a 7B model is $K/4$, while for a 3B model it is $3K/4$. When model capacity is sufficient, less sharing preserves specialization; smaller models require more sharing to maintain consistency.

**Gumbel-Softmax Comparison**: Supervised regularization substantially outperforms Gumbel-Softmax, whose continuous relaxation introduces gradient bias.

**Generalization Experiment**: Trained on MATH + HotpotQA + CosmosQA and evaluated on GSM8K, MetaTuner still surpasses all baselines.

### Key Findings

1. **Hybrid methods consistently outperform standalone approaches**, validating the complementarity of prompt and parameter optimization.
2. **MetaTuner improves over BetterTogether by 10–17%**, attributable to shared encoding and supervised regularization.
3. **Joint optimization (J) marginally outperforms alternating optimization (I)**, though alternating may be more stable on complex tasks.
4. **Excessive rollout sampling is counterproductive**, as over-exploration leads to frequent policy oscillation and overfitting.

## Highlights & Insights

1. **Novel unified perspective**: Treating prompts as "special parameters" connects discrete and continuous optimization under a unified loss function.
2. **Elegant shared-private mechanism**: Ensures knowledge sharing while preserving independent exploration for each branch.
3. **Supervised regularization effectively addresses non-differentiability**: Substantially outperforms alternatives such as Gumbel-Softmax.
4. **End-to-end trainable**: A complete generation pipeline from input to prompts and parameters.
5. **Effective OOD generalization**: Cross-dataset generalization demonstrates that the method learns transferable prompt-parameter co-adaptation strategies.

## Limitations & Future Work

1. The framework introduces considerable additional computational overhead (meta-encoder plus dual decoders); deployment costs are not thoroughly analyzed.
2. The quality of the initial prompt $\tilde{p}$ may affect final performance, yet its selection is not systematically addressed.
3. Experiments are conducted only at the 3B/7B scale; applicability to larger models (70B+) remains unknown.
4. Generating query-specific LoRA parameters at each inference step incurs additional latency, which warrants further investigation.
5. Construction of $D_2$ relies on model rollouts, which may yield low-quality supervision signals during early training when the policy is weak.

## Related Work & Insights

- **OPRO (Yang et al. 2024)**: Uses an LLM as a prompt optimizer; MetaTuner substantially surpasses it through parameter co-optimization.
- **BetterTogether (Soylu et al. 2024)**: The most directly comparable hybrid method, but lacks a knowledge-sharing mechanism.
- **LoRA (Hu et al. 2022)**: MetaTuner's parameter generation is built on LoRA but conditioned on individual queries.
- **Insight**: Prompts and parameters should not be optimized in isolation; future LLM training frameworks should natively support their co-adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to unify prompt optimization and fine-tuning in an end-to-end trainable framework.
- **Theoretical Depth**: ⭐⭐⭐☆ — Formalization is clear, but theoretical contributions are limited; the work is primarily an engineering design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, 10+ baselines, and extensive ablation and analysis experiments.
- **Practical Value**: ⭐⭐⭐⭐ — Significant performance gains, though deployment complexity is relatively high.
- **Overall**: ⭐⭐⭐⭐☆ — Addresses an important problem with an elegant solution and comprehensive evaluation; offers valuable insights for LLM post-training.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](spectral_attention_steering_for_prompt_highlighting.md)
- [\[ICLR 2026\] Soft Quality-Diversity Optimization](soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)

<!-- RELATED:END -->
