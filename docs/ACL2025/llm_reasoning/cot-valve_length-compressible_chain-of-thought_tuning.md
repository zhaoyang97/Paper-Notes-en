---
title: >-
  [Paper Note] CoT-Valve: Length-Compressible Chain-of-Thought Tuning
description: >-
  [ACL 2025][Reasoning][Chain-of-Thought Compression] This paper proposes CoT-Valve, a method to elastically control the length of reasoning chains by identifying a "length-control direction" in the parameter space (implemented with LoRA). It requires only a single training run to generate reasoning paths of varying lengths, compressing the GSM8K reasoning chain on QwQ-32B-Preview from 741 to 225 tokens with only a 0.15% drop in accuracy (95.07% to 94.92%).
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Chain-of-Thought Compression"
  - "Reasoning Length Control"
  - "LoRA"
  - "Parameter Space Direction"
  - "Test-Time Computational Efficiency"
date: 2026-05-08
content_hash: 6fecd3120d7aa45a
---

# CoT-Valve: Length-Compressible Chain-of-Thought Tuning

**Conference**: ACL 2025  
**arXiv**: [2502.09601](https://arxiv.org/abs/2502.09601)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought Compression, Reasoning Length Control, LoRA, Parameter Space Direction, Test-Time Computational Efficiency

## TL;DR
This paper proposes CoT-Valve, a method to elastically control the length of reasoning chains by identifying a "length-control direction" in the parameter space (implemented with LoRA). It requires only a single training run to generate reasoning paths of varying lengths, compressing the GSM8K reasoning chain on QwQ-32B-Preview from 741 to 225 tokens with only a 0.15% drop in accuracy (95.07% to 94.92%).

## Background & Motivation
Chain-of-Thought (CoT) reasoning significantly enhances model reasoning capabilities, but at the cost of high reasoning expenses due to excessively long reasoning chains. A core observation is that **reasoning models allocate too many tokens to simple tasks, while potentially allocating insufficient tokens to complex tasks**. For instance, QwQ spends an average of 741 tokens on GSM8K (simple math) but 6827 tokens on AIME (competition math).

Prior reasoning chain compression methods face the following challenges:
- **Directly removing intermediate steps followed by training** degrades performance
- **Prompt-based control** has limited effectiveness—even when requested to keep "under 20 words", the model may still output 350+ tokens, failing to generate truly short reasoning chains
- **Distillation to System 1** does not show improvement when omitting intermediate steps
- Methods like **SimPO optimization and RL pruning** require additional training and offer limited control granularity

The core idea of this paper is that **there exists a "direction" $\Delta\theta$ in the parameter space, where taking larger steps along this direction generates shorter chains, and smaller steps generate longer chains**. This direction is controlled using LoRA, acting as an adjustable "Valve" which requires only a single training run to generate reasoning chains of arbitrary lengths.

## Method

### Overall Architecture
CoT-Valve is split into two stages:
1. **Stage 1: Determining the length control direction $\Delta\theta$**: LoRA parameters are obtained via distillation or post-training, and this parameter difference serves as $\Delta\theta$.
2. **Stage 2: Enhancing control precision**: Using $\Delta\theta$ to construct the MixChain dataset (different lengths of reasoning chains for the same problem), which is then refined using two enhancement methods: CoT-Valve++ for precise control or CoT-Valve+P for progressive compression.

During inference, the scale factor $\alpha$ of LoRA is adjusted to control the chain length—$\alpha=0$ represents no LoRA (original long chain), $\alpha=1$ represents full loading (short chain), and $\alpha>1$ extrapolates to obtain even shorter chains.

### Key Designs

1. **Length Direction in Parameter Space**

    - Training Goal: Find the parameter update $\Delta\theta$ that enables the model to generate shorter reasoning chains while still obtaining correct answers.
    - $\Delta\theta$ is interpreted as a "task vector" where the task is "controlling CoT length".
    - Implemented with LoRA: A low-rank external branch, where adjusting its strength $\alpha$ controls the length of the reasoning chain.
    - Key Properties: **Interpolatable and Extrapolatable**—$\alpha \in (0,1)$ smoothly transitions between long and short chains, while $\alpha > 1$ can further compress reasoning chains to lengths unseen during training.

2. **MixChain Dataset Construction**

    - Utilizing the trained CoT-Valve to generate multiple lengths of reasoning chains for the same question under different $\alpha$ values.
    - Two construction scenarios:
        - **Cold-start (MixChain-C)**: When a labeled dataset (e.g., GSM8K) is available, the base model is trained with the ground truth first, followed by generation using different $\alpha$ values.
        - **Zero-shot (MixChain-Z)**: In the absence of labels, the parameter difference between a base LLM and its corresponding reasoning model is used as $\Delta\theta$ (e.g., LLaMA-3.1-8B vs DeepSeek-R1-Distill-Llama-8B).
    - Filter out reasoning chains with incorrect answers.

3. **CoT-Valve++: Precise Control**

    - When training on MixChain, a normalization factor $\beta$ is introduced to represent the reasoning chain length: $\beta = 1 - \frac{m - m_{min}}{m_{max} - m_{min}}$
    - The training objective requires generating correct reasoning of corresponding lengths across all $\beta$ values: $\max_{\Delta\theta'} \mathbb{E} p(a|t_{<m}, q; \theta + \beta\Delta\theta')$
    - Resolves the training-inference discrepancy in the original CoT-Valve where training occurs only at $\alpha=1$ while inference is performed across all $\alpha$ values.

4. **CoT-Valve+P: Progressive Compression**

    - Similar to the iterative pruning concept in model compression.
    - Each epoch trains the model on progressively shorter reasoning chains from MixChain, compressing gradually rather than jumping directly to the shortest.
    - Running 5 epochs sequentially using Solution 4 $\rightarrow$ 3 $\rightarrow$ 2 $\rightarrow$ 1 $\rightarrow$ 0 (ground truth) improves final accuracy from 92.19% (direct training) on the shortest chain to 94.92%.

### Loss & Training
- Employs standard language modeling loss, with the core difference in the designs of training data and LoRA scale factors.
- Most experiments utilize LoRA fine-tuning, while the LIMO experiment uses full parameter fine-tuning.
- Efficiency Metric ACU (Accuracy per Computation Unit): $\text{ACU} = \text{Accuracy} / (\text{\#Params} \times \text{\#Tokens})$

## Key Experimental Results

### Main Results (QwQ-32B-Preview on GSM8K)

| Method | Accuracy | Token Count | ACU↑ |
|------|--------|---------|------|
| Original QwQ-32B-Preview | 95.07% | 741 | 0.40 |
| Prompt-based Control (Han) | 93.6% | 355 | 0.82 |
| Overthink-SimPO | 94.8% | 326 | 0.91 |
| O1-Pruner(RL) | 96.5% | 534 | 0.56 |
| **CoT-Valve++ MixChain-C** | **94.4%** | **276** | **1.07** |
| **CoT-Valve+P MixChain-Z** | **94.9%** | **225** | **1.32** |

### AIME2024 (QwQ-32B-Preview)

| Method | Score | Token Count | ACU↑ |
|------|------|---------|------|
| Original QwQ-32B-Preview | 14/30 | 6827 | 0.021 |
| Overthink | 13/30 | 5154 | 0.026 |
| **CoT-Valve+P** | **13/30** | **4630** | **0.029** |

### Small Model Distillation (LLaMA-3.2-1B)

| Method | Accuracy | Token Count | ACU↑ |
|------|--------|---------|------|
| SFT - QwQ Distillation | 52.7% | 759 | 6.94 |
| CoT-Valve - QwQ Distillation | 55.5% | 267 | 20.79 |
| **CoT-Valve - MixChain Solution 1** | **58.9%** | **275** | **21.39** |

### Ablation Study (Progressive Compression vs. Direct Training)

| Method | Accuracy | Token Count |
|------|--------|---------|
| Direct training with shortest chain for 5 epochs | 92.19% | 250 |
| Progressive compression (4 $\rightarrow$ 3 $\rightarrow$ 2 $\rightarrow$ 1 $\rightarrow$ 0) | **94.92%** | **225** |

### Impact of Training Data Length (LLaMA-3.2-1B)

| Training Chain Length | Accuracy | Token Count |
|-----------|--------|---------|
| Ground-Truth (116 tokens) | 43.8% | 139 |
| Solution 1 (280 tokens) | **57.0%** | **288** |
| Solution 4 (497 tokens) | 52.5% | 558 |

### Key Findings
- **Shorter reasoning chains sometimes outperform longer ones**: On GSM8K, the shorter chains generated by CoT-Valve (267 tokens) achieve higher accuracy than the original QwQ long chains (741 tokens) (55.5% vs 52.7% on LLaMA-3.2-1B).
- **Not all reasoning chains are suitable for training**: Excessively short or long chains are sub-optimal, whereas moderate lengths (Solution 1, ~280 tokens) yield the best results, especially for small models.
- **Progressive compression significantly outperforms direct compression**: Accuracy increases from 92.19% to 94.92%.
- **Extrapolability of CoT-Valve**: Setting $\alpha > 1$ can generate chains even shorter than those in the training set.
- **CoT-Valve achieves shorter chains than prompts can reach**: Prompting is capped at around 355 tokens, whereas CoT-Valve can compress down to 133.8 tokens.
- **The "Long-Short-Long" strategy is effective**: Training on longer chains first before compressing them (Short-Long-Short) performs better than directly training on short chains.

## Highlights & Insights
- **The concept of a "length direction in parameter space" is highly elegant**: It frames the control of reasoning chain length as vector arithmetic within the parameter space, aligning with theoretical works like task arithmetic and model merging.
- **Intuitive analogy of LoRA as a "valve"**: Just as rotating a valve regulates flow volume, adjusting $\alpha$ controls reasoning chain length, making the methodology highly intuitive.
- **Introduction of the ACU metric**: An efficiency metric that simultaneously accounts for accuracy, parameter count, and token count, enabling a fairer comparison among reasoning models.
- **Self-generation mechanism of the MixChain dataset**: It bypasses external sampling by employing CoT-Valve itself to generate chains of varying lengths, showcasing strong self-bootstrapping capabilities.
- **The finding that "not all correct chains are suitable for training"** provides crucial insights for distillation research.

## Limitations & Future Work
- The method is currently validated only on mathematical reasoning (GSM8K, AIME), leaving domains like coding and scientific reasoning unexplored.
- The length control is applied globally to the entire chain, without fine-grained compression of different parts of a chain (e.g., compressing simpler steps while retaining complex ones).
- The optimal selection of the $\alpha$ value still requires manual tuning based on the specific task and dataset.
- There is a noticeable performance drop after compression on AIME (14/30 to 13/30), showing that reasoning chain compression on complex tasks remains challenging.
- **Research idea**: Combining the model with a reward model for adaptive chain length control—automatically selecting the $\alpha$ value based on question difficulty (e.g., using a larger $\alpha$ to generate short chains for simple problems, and a smaller $\alpha$ to retain long chains for complex ones) to achieve true "reasoning on demand".

## Related Work & Insights
- Overthinking (Chen et al., 2024): Identifies QwQ's "overthinking" issue and optimizes it using SimPO, but achieves a lower compression ratio than CoT-Valve.
- O1-Pruner (Luo et al., 2025): Compresses reasoning via RL, achieving higher accuracy but at the cost of more tokens (534 vs 225).
- Kimi K1.5: Proposes a training-free fusion of long and short CoT models, which is conceptually complementary to CoT-Valve.
- Task Arithmetic (Ilharco et al., 2022): Serves as the theoretical foundation for CoT-Valve, demonstrating that directions in parameter space can encode specific tasks.
- The core contribution of this work is demonstrating that **the length of reasoning chains can be encoded as a controllable direction in the parameter space**, establishing a new technical route for optimizing reasoning efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of a length direction in the parameter space is highly novel, and the design of CoT-Valve is elegant and simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ Spans multiple models (QwQ, R1-Distill, LLaMA, Qwen), multiple scenarios (long-to-short, short-to-long, short-long-short), and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear and intuitive method explanations and logical experimental layouts, although mathematical notations are occasionally inconsistent.
- Value: ⭐⭐⭐⭐⭐ Tackles the critical pain point of high reasoning costs in reasoning models. The ACU is significantly boosted (0.40 to 1.32), yielding high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[NeurIPS 2025\] Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](../../NeurIPS2025/llm_reasoning/transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)

</div>

<!-- RELATED:END -->
