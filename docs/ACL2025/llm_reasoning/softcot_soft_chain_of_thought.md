---
title: >-
  [Paper Note] SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs
description: >-
  [ACL 2025][Reasoning][Continuous space reasoning] This paper proposes SoftCoT, which uses a frozen small auxiliary model (e.g., LLaMA-3.2-1B) to generate instance-specific "soft thought tokens" (continuous hidden states) that are mapped into the representation space of the primary LLM via a trainable projection module, serving as a reasoning prefix. This approach achieves parameter-efficient continuous-space CoT reasoning while avoiding the catastrophic forgetting problem cau…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Continuous space reasoning"
  - "soft thought tokens"
  - "auxiliary small model"
  - "projection module"
  - "catastrophic forgetting mitigation"
date: 2026-05-08
content_hash: df567edb3c231501
---

# SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.12134](https://arxiv.org/abs/2502.12134)  
**Code**: [https://github.com/xuyige/SoftCoT](https://github.com/xuyige/SoftCoT)  
**Area**: LLM Reasoning  
**Keywords**: Continuous space reasoning, soft thought tokens, auxiliary small model, projection module, catastrophic forgetting mitigation

## TL;DR
This paper proposes SoftCoT, which uses a frozen small auxiliary model (e.g., LLaMA-3.2-1B) to generate instance-specific "soft thought tokens" (continuous hidden states) that are mapped into the representation space of the primary LLM via a trainable projection module, serving as a reasoning prefix. This approach achieves parameter-efficient continuous-space CoT reasoning while avoiding the catastrophic forgetting problem caused by full-model fine-tuning.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) reasoning improves LLM performance on complex tasks by generating intermediate reasoning steps. Traditional CoT generates reasoning chains in discrete token spaces, which are limited by vocabulary dimensions. Recent methods like Coconut and CCoT explore continuous-space reasoning, using latent representations instead of discrete token sequences.

**Limitations of Prior Work**:
   - Coconut/CCoT require full-model fine-tuning (using language modeling objectives). While effective on GPT-2, this leads to catastrophic forgetting on strong aligned models such as LLaMA-3.1-8B-Instruct—where performance after LoRA fine-tuning even falls below zero-shot CoT.
   - The token space in discrete CoT is constrained, which may not represent the optimal latent space for reasoning.
   - Multi-path decoding (e.g., self-consistency, Tree-of-Thought) incurs high computational costs.

**Key Challenge**: Continuous-space reasoning has expressiveness advantages, but existing methods require modifying LLM parameters, which causes powerful instruction-tuned models to lose their pre-acquired reasoning capabilities. How can continuous-space reasoning be introduced without altering the LLM?

**Goal**
   - How to perform continuous-space CoT reasoning while keeping the primary LLM frozen.
   - How to bridge the representation space discrepancy between the auxiliary model and the primary LLM.
   - How to achieve performance superior to zero-shot CoT via parameter-efficient training.

**Key Insight**: Inspired by prompt tuning and speculative decoding—an auxiliary frozen small model is utilized to generate instance-specific soft prompts (continuous thought tokens) to replace discrete reasoning prefixes in CoT. The primary LLM is completely frozen, and only a projection module is trained.

**Core Idea**: Instead of modifying the LLM, the raw continuous hidden states of a small auxiliary model are used as instance-adaptive "soft thought" prefixes, which are passed into the frozen primary LLM through a projection layer to enhance reasoning.

## Method

### Overall Architecture
Given a query $\mathcal{Q}$ $\rightarrow$ **Auxiliary Model** (frozen LLaMA-3.2-1B) processes [Instruction, Query, $N$ UNK tokens], extracting the last-layer hidden states of the final $N$ positions as soft thoughts $\rightarrow$ **Projection Module** (trainable linear layer) maps the soft thoughts from the auxiliary model dimension to the primary LLM dimension $\rightarrow$ **Primary LLM** (frozen LLaMA-3.1-8B) receives [Instruction, Query, Soft Thoughts] to autoregressively generate reasoning steps and the final answer.

### Key Designs

1. **Soft Thought Token Generation**:

    - **Function**: Generates instance-specific continuous reasoning prefixes for each question using a small auxiliary model.
    - **Mechanism**:
        - Input construction: $\mathbf{x}_{\text{assist}} = \text{concat}[\mathcal{I}_{\text{assist}}, \mathcal{Q}, \text{[UNK]}_{1:N}]$
        - After a forward pass through the auxiliary model, the last-layer hidden states of the $N$ [UNK] positions are extracted as $\mathbf{t}_{\text{assist}} \in \mathbb{R}^{N \times d_{\text{assist}}}$
        - Operating in continuous space avoids the information loss and gradient truncation inherent in autoregressive decoding.
    - **Design Motivation**: The [UNK] tokens act as "placeholders" that carry no semantic meaning, forcing the model to compress the understanding of the query into the hidden states of these positions. Freezing the auxiliary model avoids additional training overhead.

2. **Projection Module**:

    - **Function**: Bridges the gap in representation space and dimensionality between the auxiliary model and the primary LLM.
    - **Mechanism**: $\mathcal{T}_{\text{soft}} = \text{Linear}_\theta(\mathbf{t}_{\text{assist}})$, mapping $\mathbb{R}^{d_{\text{assist}}}$ to $\mathbb{R}^{d_{\text{LLM}}}$.
    - This is the **only trainable component**—a single linear layer with minimal parameters.
    - **Design Motivation**: Similar to the projection from visual encoders to LLMs in LLaVA, it bridges the spaces of two models with minimal training cost.

3. **Reasoning with Frozen Primary LLM**:

    - **Function**: Enhances the CoT reasoning of the primary LLM using the soft thoughts.
    - **Input**: $\mathbf{x}_{\text{LLM}} = \text{concat}[\mathcal{I}_{\text{LLM}}, \mathcal{Q}, \mathcal{T}_{\text{soft}}]$
    - The primary LLM is completely frozen, autoregressively generating the reasoning chain and the answer based on the instruction, query, and soft thoughts.
    - During training: Optimization is supervised using NLL loss over the reasoning steps $\mathcal{R}$ and answer $\mathcal{A}$, with gradients backpropagated only to the projection layer.

### Loss & Training
- Standard language modeling objective (NLL loss) is adopted, where the instruction and query parts are masked, and the loss is computed only on the reasoning chain and the answer.
- Gradients only update the parameters of the projection module.
- Parameter efficiency: Both the primary LLM and the auxiliary model are frozen.

## Key Experimental Results

### Main Results (LLaMA-3.1-8B-Instruct as primary LLM)

| Method | GSM8K | ASDiv-Aug | AQuA | StrategyQA | Date Und. | Avg |
|------|-------|-----------|------|------------|-----------|-----|
| Zero-Shot CoT | 79.61 | 86.78 | 54.65 | 65.63 | 54.40 | 68.21 |
| Zero-Shot CoT-UNK | 79.95 | 86.90 | 55.28 | 66.16 | 54.16 | 68.49 |
| Zero-Shot Assist-CoT | 80.76 | 86.96 | 55.83 | 66.55 | 58.24 | 69.67 |
| LoRA Fine-Tuning | 75.66 | 86.67 | 52.36 | - | - | - |
| Coconut (LoRA) | 76.12 | 86.80 | 53.15 | - | - | - |
| **SoftCoT** | **81.03** | **87.19** | **56.30** | **69.04** | **59.04** | **70.52** |

Key Comparison: Both LoRA fine-tuning and Coconut perform **worse** than Zero-Shot CoT on GSM8K (75.66/76.12 vs. 79.61), confirming catastrophic forgetting. SoftCoT is the only method that consistently outperforms Zero-Shot CoT across all tasks.

### Ablation Study

| Component | GSM8K | ASDiv-Aug | Description |
|------|-------|-----------|------|
| SoftCoT (Full) | 81.03 | 87.19 | Full model |
| w/o Auxiliary Model (Random UNK) | 79.95 | 86.90 | Degrades to CoT-UNK, -1.08 |
| w/o Continuous Space (Hard tokens) | 80.76 | 86.96 | Degrades to Assist-CoT, -0.27 |
| w/o Projection Module | — | — | Fails to execute due to dimension mismatch |

### Key Findings
- **Catastrophic forgetting is an actual issue**: While Coconut is effective on GPT-2, using LoRA fine-tuning on LLaMA-3.1-8B-Instruct actually performs 3.5% worse than zero-shot.
- **[UNK] tokens alone provide a slight positive impact** (resembling the "pause token" effect), which increases model computational capacity and reduces variance.
- **Soft thoughts outperform hard thoughts**: SoftCoT (70.52) > Assist-CoT (69.67), proving that continuous representations encode richer information than discrete tokens.
- Consistently gains performance by training only a mapping layer (with extremely small parameter scale).
- Consistent improvements are also verified on Qwen2.5-7B-Instruct.

## Highlights & Insights
- **The "modify inputs rather than parameters" paradigm is highly practical**: This aligns with prompt tuning but is superior, as using an auxiliary model to generate instance-adaptive soft prompts outperforms fixed learnable prompts. This design guarantees the integrity of the primary LLM's knowledge.
- **Systematic verification of catastrophic forgetting** is a major contribution: It explicitly points out that Coconut-like methods are unsuitable for already powerful instruction-tuned models, redirecting research efforts in continuous-space reasoning.
- **Speculative decoding-like architectural design**: The small model "speculates" the reasoning direction while the large model "executes" it, representing an elegant model collaboration paradigm. This concept can be transferred to other scenarios requiring auxiliary reasoning.

## Limitations & Future Work
- The soft thought length $N$ is a fixed hyperparameter and is not adaptively adjusted based on query difficulty.
- The projection module uses only a linear layer, which may limit representation mapping capacity (MLP or cross-attention could be better).
- The auxiliary model is entirely frozen; co-fine-tuning the auxiliary model has not been explored (which could further improve results but increase cost).
- Experiments are verified only at 7B-8B scales; the effectiveness on larger models (70B+) remains unknown.
- Explainability of soft thoughts is limited—they cannot be read like discrete CoT reasoning steps.

## Related Work & Insights
- **vs. Coconut**: Coconut requires full-model fine-tuning and suffers from catastrophic forgetting on strong models, while SoftCoT keeps the primary LLM completely frozen and remains consistently effective.
- **vs. Prompt Tuning**: Traditional prompt learning utilizes static learnable prompts, whereas SoftCoT's soft thoughts are instance-adaptive.
- **vs. LoRA Fine-Tuning**: LoRA performs worse than zero-shot on CoT tasks, whereas SoftCoT bypasses this issue via the external module.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of generating soft thoughts using an auxiliary model is innovative and addresses catastrophic forgetting of continuous CoT on strong models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, three reasoning categories, two LLMs, multiple baseline comparisons, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, standardized method description, and concise charts/tables.
- Value: ⭐⭐⭐⭐ Provides a practical, lightweight reasoning enhancement scheme with high parameter efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](../../NeurIPS2025/llm_reasoning/polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](../../NeurIPS2025/llm_reasoning/re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)

</div>

<!-- RELATED:END -->
