---
title: >-
  [Paper Note] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models
description: >-
  [NeurIPS 2025][Reinforcement Learning][Diffusion Language Models] This paper proposes the Diffusion Chain of Lateral Thought (DCoLT), which treats each intermediate step in the reverse process of a diffusion language model as a latent "thinking" action and optimizes the entire reasoning trajectory via outcome-based reinforcement learning. DCoLT achieves state-of-the-art performance on mathematics and code generation benchmarks with both SEDD and LLaDA diffusion language model…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Diffusion Language Models"
  - "Lateral Thinking"
  - "GRPO"
  - "Plackett-Luce"
date: 2026-05-08
content_hash: 32c4a0c0ffc2c522
---

# Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.10446](https://arxiv.org/abs/2505.10446)  
**Code**: [GitHub](https://github.com/maple-research-lab/LLaDOU)  
**Area**: Image Generation
**Keywords**: Diffusion Language Models, Lateral Thinking, Reinforcement Learning, GRPO, Plackett-Luce

## TL;DR

This paper proposes the Diffusion Chain of Lateral Thought (DCoLT), which treats each intermediate step in the reverse process of a diffusion language model as a latent "thinking" action and optimizes the entire reasoning trajectory via outcome-based reinforcement learning. DCoLT achieves state-of-the-art performance on mathematics and code generation benchmarks with both SEDD and LLaDA diffusion language models.

## Background & Motivation

- **Background**: The reasoning capabilities of existing large language models primarily rely on chain-of-thought (CoT) prompting, which decomposes problems into sequential intermediate steps. However, CoT is constrained by the causal attention mechanism of autoregressive models, forcing reasoning to unfold linearly in a single direction. This diverges from human cognition — in early stages of thought, ideas emerge in a fragmented and nonlinear manner before being organized into coherent expressions. This form of nonlinear, creative reasoning is referred to as *lateral thinking*.
- **Limitations of Prior Work**: Diffusion language models (DLMs) are naturally suited to simulate lateral thinking: their bidirectional attention allows each token to attend to all others freely; intermediate steps need not conform to grammatical rules; and tokens can be generated at multiple positions simultaneously rather than left-to-right. Nevertheless, existing DLM training approaches (e.g., DoT using annotated CoT data for SFT) still encourage sequential reasoning, failing to leverage the full potential of lateral thinking.
- **Key Challenge**: No existing method jointly optimizes the entire diffusion reverse process as a complete lateral thinking chain.
- **Goal**: To develop a reinforcement learning framework that treats the full diffusion reverse trajectory as a Chain of Lateral Thought (DCoLT) and optimizes it end-to-end using only final-answer reward signals.

## Method

### Overall Architecture

DCoLT defines the diffusion reverse process $x_{0:N}$ as a complete chain of lateral thought. At each step $n$, generating $x_n$ from $x_{n-1}$ is treated as an "action," and the entire chain is jointly optimized via outcome-based RL. No explicit supervision is imposed on intermediate steps; reward signals (1 for correct, 0 for incorrect) are derived solely from the correctness of the final answer $x_N$. The advantage function is computed using GRPO:

$$A^g = \frac{r^g - \text{mean}(r^{1:G})}{\text{std}(r^{1:G})}$$

The loss at each step is:

$$\mathcal{L}_{\theta,n} = -\frac{1}{G}\sum_{g=1}^{G}\frac{\pi_{\theta,n}(x_n^g|x_{n-1}^g)}{\pi_{\text{old},n}(x_n^g|x_{n-1}^g)}A^g$$

Gradients are accumulated across all steps before a single parameter update.

### Key Designs

1. **DCoLT-SEDD (Continuous-Time DLM)**

   SEDD learns the concrete score $s_\theta(x,t)_y \approx p_t(y)/p_t(x)$ to estimate the reverse transition rate. Using τ-leaping, the sequence-level policy is defined as the product of per-token transition probabilities:

   $\pi_{\theta,n}(x_n|x_{n-1}) = \prod_{i=1}^{|x_n|}p_{\theta,t_n}(x_n^i|x_{n-1})$

   Each token's transition probability can be computed in closed form from the concrete score and the transition rate matrix $Q_t$, making the trajectory policy probability differentiable and directly trainable with GRPO.

2. **DCoLT-LLaDA + Unmask Policy Module (UPM)**

   LLaDA is a discrete-time masked diffusion model that at each step selects a subset of masked tokens to reveal. The authors find that **the unmasking order is critical for reasoning** — tokens with higher confidence should be revealed first. To this end, an Unmask Policy Module (UPM) is introduced:

    - UPM predicts a ranking score $h_{\theta,n}^i$ for each token and defines an ordering-sampling policy using the Plackett-Luce model:

   $\pi_{\theta,n}^{\text{unmask}}(\mathcal{U}_n|x_{n-1}) = \prod_{k=1}^{K}\frac{\exp(h_{\theta,n}^{u_n(k)})}{\sum_{j=k}^{K}\exp(h_{\theta,n}^{u_n(j)}) + \sum_{j \in \mathcal{M}_n}\exp(h_{\theta,n}^{u_n(j)})}$

    - Token generation policy: $\pi_{\theta,n}^{\text{token}}(x_n|x_{n-1},\mathcal{U}_n) = \prod_{i \in \mathcal{U}_n}p_{\theta,n}(x_n^i|x_{n-1})$

    - The full policy is the product of the two: $\pi_{\theta,n}(x_n|x_{n-1}) = \pi_{\theta,n}^{\text{unmask}} \cdot \pi_{\theta,n}^{\text{token}}$

   UPM consists of a single Transformer block with adaptive layer normalization (AdaLN) to embed the diffusion step $n$ and mask indicator information, incurring minimal computational overhead. The resulting model is named LLaDOU.

3. **Step-wise Gradient Accumulation**

   Fully unrolling the computation graph across all reverse steps leads to GPU memory overflow. The paper adopts a strategy of performing backpropagation at each step individually, accumulating gradients, and applying a unified parameter update at the end.

### Loss & Training

- **Reward function**: Rule-based verification (answer correctness for math; unit test pass rate for code)
- **RL algorithm**: GRPO (Group Relative Policy Optimization)
- **Training data**: 15K publicly available problems from GSM8K and MATH (no reasoning process annotations required)
- **Hardware**: 16 × H800 GPUs

## Key Experimental Results

### SEDD 400M Results

| Model | Post-training | Sudoku 4×4 | GSM8K-Aug |
|-------|---------------|-----------|-----------|
| GPT2 + CoT | SFT | 71.5% | 43.9% |
| GPT2 + CoT | RL | 74.6% | — |
| SEDD + DoT | SFT | 79.4% | 53.5% |
| **SEDD + DCoLT** | **RL** | **96.2%** | **57.0%** |

### LLaDOU 8B Results

| Model | Post-training | GSM8K | MATH | HumanEval | MBPP |
|-------|---------------|-------|------|-----------|------|
| LLaDA 8B | baseline | 78.3% | 38.9% | 39.6% | 40.2% |
| d1-LLaDA | SFT+RL | 82.1% | 40.2%‡ | — | — |
| **LLaDOU** | **RL** | **88.1%** | **44.6%** | **59.1%** | **51.6%** |
| DeepseekMath-RL 7B | SFT†+RL† | 88.2% | 51.7% | — | — |

### Ablation Study

| Training Components | UPM | LLaDA | GSM8K |
|--------------------|-----|-------|-------|
| No training | × | × | 47.27% |
| UPM only (w/ AdaLN) | ✓ (AdaLN) | × | 69.24% |
| UPM (w/o AdaLN) + LLaDA | ✓ (w/o AdaLN) | ✓ | 80.53% |
| UPM (w/ AdaLN) + LLaDA | ✓ (AdaLN) | ✓ | **81.06%** |

### Key Findings

1. **Critical role of UPM**: Training UPM alone (with LLaDA frozen) improves performance from 47.27% to 69.24%, demonstrating that the unmasking order is critical for reasoning.
2. **Data efficiency**: LLaDOU trained on only 15K public problems approaches the performance of DeepseekMath (776K SFT + 144K RL problems).
3. **Length extrapolation**: Training with a fixed generation length of 256 tokens and increasing to 384 at inference yields an additional 1.6% improvement on MATH, with more pronounced gains on harder problems.
4. **Lateral thinking outperforms CoT**: At the same 400M model scale, DCoLT surpasses CoT/DoT on Sudoku by approximately 17–25 percentage points.

## Highlights & Insights

1. **Paradigm innovation**: This is the first work to treat the diffusion reverse process as a complete reasoning chain and optimize it end-to-end with RL, rather than training each step independently.
2. **Learnable unmasking order**: The Plackett-Luce model provides an elegant mathematical framework for incorporating discrete ranking problems into differentiable policy optimization.
3. **Unsupervised emergence of reasoning**: Without any reasoning process annotations, the model learns effective lateral reasoning patterns solely from final-answer correctness signals.
4. **Progressive generation emergence**: After training, a natural progression from easy to hard tokens emerges (high-certainty tokens are filled in first), consistent with human cognitive habits.

## Limitations & Future Work

- Validation is limited to verifiable tasks (math and code); generalization to open-domain settings requires a reward model.
- Limited training data and computational resources leave room for further performance gains.
- Sequence length is constrained to 256–512 tokens, limiting applicability to long-form reasoning tasks.
- Multi-step computation graph unrolling incurs substantial GPU memory overhead.

## Related Work & Insights

- **Diffusion language models**: SEDD (continuous-time discrete diffusion), LLaDA (masked diffusion), Dream
- **Reasoning with reinforcement learning**: DeepSeek-R1 (autoregressive RL-based reasoning), GRPO
- **Diffusion-based reasoning**: DoT (training diffusion models for CoT reasoning via SFT)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — DCoLT establishes a new paradigm for reasoning with diffusion language models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ — Multi-task and multi-ablation comparisons are comprehensive, though validation at larger model scales is lacking.
- **Writing Quality**: ⭐⭐⭐⭐☆ — Well-structured with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ — Demonstrates that non-autoregressive models can achieve strong reasoning capabilities through lateral thinking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](../../ICML2026/reinforcement_learning/learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ACL 2026\] Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/beyond_fully_random_masking_attention-guided_denoising_and_optimization_for_diff.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)

</div>

<!-- RELATED:END -->
