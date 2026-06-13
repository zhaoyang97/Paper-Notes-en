---
title: >-
  [Paper Note] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching
description: >-
  [ICLR 2026][Interpretability][Memory-efficient fine-tuning] This paper proposes TokenSeek, a general-purpose memory optimization plugin for Transformer fine-tuning. By combining contextual attention information with grad…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Memory-efficient fine-tuning"
  - "Token pruning"
  - "Instance-aware"
  - "Activation memory optimization"
  - "PEFT compatibility"
date: 2026-05-08
content_hash: 591c30c5725cf106
---

# TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching

**Conference**: ICLR 2026
**arXiv**: [2601.19739](https://arxiv.org/abs/2601.19739)  
**Code**: [https://runjia.tech/iclr_tokenseek (project page)](https://runjia.tech/iclr_tokenseek (project page))  
**Area**: Interpretability
**Keywords**: Memory-efficient fine-tuning, Token pruning, Instance-aware, Activation memory optimization, PEFT compatibility

## TL;DR
This paper proposes TokenSeek, a general-purpose memory optimization plugin for Transformer fine-tuning. By combining contextual attention information with gradient information for instance-level token importance estimation, TokenSeek retains only the top 10% high-value tokens for gradient updates, achieving up to 65.7% memory savings while matching or surpassing full-token fine-tuning performance.

## Background & Motivation
LLM fine-tuning faces a severe memory bottleneck. Training memory consists of three components: (I) model parameters, (II) gradients and optimizer states, and (III) activations. Among these, activations are the primary bottleneck—in Llama3 8B, activations account for 87% of memory, while in GPT-2 1.5B they reach 60 GB.

Existing optimization approaches follow three directions:
- **PEFT** (e.g., LoRA): Reduces trainable parameters (I), but activation memory still accounts for 75%+
- **Optimizer efficiency** (e.g., ZeRO): Optimizes gradients and optimizer states (II)
- **Memory-efficient fine-tuning (MEFT)**: Directly targets activation memory (III), including recomputation, compression, and reversible networks

The core limitation of existing MEFT methods is that they are **data-agnostic**—they apply a uniform strategy across all training samples, ignoring the varying informativeness of tokens across different instances. For example, TokenTune randomly selects tokens to drop gradients, leading to unstable performance.

Root Cause: How to identify truly important tokens at the instance level and efficiently save activations only for them? This requires addressing two sub-problems: (1) how to evaluate the importance of each token; and (2) how to leverage importance estimates to efficiently reduce memory.

## Method

### Overall Architecture
TokenSeek operates in two steps: (1) **Instance-Aware Token Seeking**—evaluating each token's importance by combining contextual and gradient information; and (2) **Efficient Token Ditching**—performing backpropagation only for selected tokens, discarding gradients for the remaining tokens and thereby eliminating the need to store their activations.

### Key Designs

1. **Context Information**: Utilizes column-wise cumulative attention weights from the attention mechanism, $I_1(t_j) = \sum_{i=1}^{n} \mathbf{A}_{ij}$, representing the total attention a token receives from all other tokens. Intuitively, tokens attended to by more other tokens are more important in context. However, raw attention scores suffer from the attention sink effect (artificially inflated scores for early tokens) and long-tail distribution issues.

2. **Gradient Information**: Computes the gradient norm of the pre-activation of the final layer, $I_2(t_j) = \sum_{k=1}^{d} \mathbf{G}_{jk}$, where $\mathbf{G} = \partial \mathcal{L} / \partial z^{(L-1)}$. Gradient information directly reflects a token's actual contribution to the loss function, complementing attention weights (Jain & Wallace 2019 note that attention and gradient importance are often uncorrelated). Crucially, this requires only a partial backward pass through the output head and the last decoder block, with all other layers frozen.

3. **Combined Score**: $I(t_j) = \alpha \log[I_1(t_j)] + \beta \text{Norm}[I_2(t_j)]$, where the log transform addresses the long-tail distribution and min-max normalization aligns scales. Default values $\alpha = \beta = 1$ are used, and the method is insensitive to these hyperparameters.

4. **Efficient Token Ditching**: Gradients for unselected tokens are zeroed out directly, $\sigma'(a_{\bar{t}}^{(l)}) = 0$, eliminating the need to cache activations $a_{\bar{t}}^{(l)}$ for these tokens. Selecting 10% of tokens theoretically requires only approximately 1% of activation memory.

### Loss & Training
TokenSeek is a plug-and-play plugin that is architecture-agnostic and can be combined with PEFT methods such as LoRA, LoHa, and QLoRA. Token importance is evaluated independently for each batch, ensuring instance-aware selection.

## Key Experimental Results

### Main Results (Few-shot Evaluation, Open-Platypus Fine-tuning)

| Method | Setting | Ave. Mem. | ARC | HellaSwag | MMLU | TruthfulQA | WinoGrande | Avg |
|---|---|---|---|---|---|---|---|---|
| Full Token Tuning (Llama3.2 1B) | 100% | 100% | 23.72 | 26.11 | 57.53 | 48.68 | 48.07 | 40.82 |
| + TokenTune (Random 10%) | 64.6% | - | 24.32 | 25.80 | 58.14 | 47.90 | 47.59 | 40.75 |
| + **TokenSeek (10%)** | **64.6%** | - | 23.98 | 25.73 | 58.14 | 48.09 | 49.72 | **41.13** |
| QLoRA (Llama3.2 1B) | 45.6% | - | 38.82 | 65.26 | 56.39 | 38.85 | 61.33 | 52.13 |
| + TokenTune (Random 10%) | 14.8% | - | 39.33 | 62.97 | 41.76 | 41.36 | 60.69 | 49.22 |
| + **TokenSeek (10%)** | **14.8%** | - | 39.08 | 65.98 | 58.03 | 38.65 | 61.33 | **52.61** |

TokenSeek + QLoRA achieves superior performance with only 14.8% memory (2.8 GB), surpassing both full-token fine-tuning (52.61 vs. 40.82) and QLoRA alone (52.61 vs. 52.13).

### Ablation Study (Weight Sensitivity + Token Ratio)

| Setting | MMLU | ARC | HellaSwag | TruthfulQA | WinoGrande | Avg |
|---|---|---|---|---|---|---|
| α=1, β=0 (Context only) | 57.52 | 34.56 | 50.09 | 41.51 | 58.56 | 48.45 |
| α=0, β=1 (Gradient only) | 57.62 | 30.72 | 44.20 | 43.98 | 55.41 | 46.39 |
| α=5, β=5 (Balanced) | 58.49 | 35.15 | 50.20 | 41.48 | 57.93 | 48.65 |
| α=7, β=3 | 58.59 | 35.58 | 50.10 | 41.13 | 57.22 | 48.53 |

Token ratio ablation (Llama3.2 1B + QLoRA): 10%→52.61, 20%→51.80, 30%→52.75, 40%→52.66, 50%→52.26, 100%→40.82. A ratio of 10% already reaches the optimal range with only 14.8% memory usage.

### Key Findings
- Context and gradient information are complementary: context information is biased toward early tokens (attention sink), while gradient information concentrates on later tokens (response portion); combining both yields more comprehensive coverage.
- TokenSeek performs better in PEFT settings (full-parameter fine-tuning may overfit, while the regularization effect of PEFT synergizes with token sparsification).
- Random token selection (TokenTune) exhibits high variance and potential performance collapse; instance-aware selection significantly improves stability.
- Cross-model generalization: effective across Qwen 0.5B, Llama 1B, and Llama 3B, though smaller models are more sensitive.

## Highlights & Insights
- **Extreme memory compression**: Llama3.2 1B can be fine-tuned with only 2.8 GB (QLoRA + TokenSeek), enabling fine-tuning of 3B models on consumer-grade GPUs (e.g., RTX 4090 24 GB).
- The combination of PEFT + TokenSeek outperforms either method alone; the low-rank constraint of PEFT provides regularization that works synergistically with token sparsification to mitigate overfitting.
- The advantage of instance-aware selection over uniform strategies is most pronounced in terms of stability (substantially lower variance).
- Strong interpretability: phenomena such as attention sink effects and gradient concentration in the response portion are clearly visualized.
- General applicability: architecture-agnostic design is compatible with multiple PEFT methods, functioning as a true plug-in.

## Limitations & Future Work
- Evaluating token importance requires an additional forward pass and partial backward pass, introducing computational overhead (though the paper claims this overhead is small).
- Performance degrades on very small models (e.g., Qwen 0.5B used in isolation), where insufficient representational capacity may compromise selection accuracy.
- Validation is limited to instruction tuning; the effectiveness of this approach for other fine-tuning paradigms such as continual pretraining remains unexplored.
- The fixed 10% token ratio is applied uniformly across all samples; adaptive ratio adjustment warrants further investigation.
- A comprehensive comparison with other recent MEFT methods (e.g., reversible networks, mixed-precision training) is absent.

## Related Work & Insights
- Compared to TokenTune (Simoulin et al., 2024): TokenTune selects tokens randomly, whereas TokenSeek selects based on informativeness, constituting a direct improvement.
- Compared to gradient checkpointing: the two approaches are complementary—checkpointing reduces recomputation overhead, while TokenSeek reduces the number of tokens whose activations need to be cached.
- The approach generalizes naturally to inference: if certain tokens contribute little to learning during fine-tuning, they may similarly be skippable at inference time.
- Token importance estimation may be applicable to data distillation and curriculum learning.
- TokenSeek can be combined with gradient checkpointing: TokenSeek reduces the number of cached tokens, while checkpointing reduces recomputation overhead—the two are mutually complementary.
- The approach generalizes naturally to inference: if certain tokens contribute little to learning during fine-tuning, they may similarly be skippable at inference time.

## Rating
- Novelty: ⭐⭐⭐⭐ Instance-aware token selection combining gradient and attention signals is creative, though the underlying idea of token pruning is not itself new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-PEFT combination experiments with ablations and visualizations are comprehensive, but additional task types would strengthen the evaluation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich visualizations, though slightly verbose in places.
- Value: ⭐⭐⭐⭐⭐ Fine-tuning a 1B model with 2.8 GB offers substantial engineering value, and the plugin design makes deployment straightforward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](../../AAAI2026/interpretability/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ICLR 2026\] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)

</div>

<!-- RELATED:END -->
