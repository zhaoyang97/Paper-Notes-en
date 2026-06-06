---
title: >-
  [Paper Note] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection
description: >-
  [ICLR 2026][LLM Efficiency][memory efficient fine-tuning] This paper proposes TokenSeek, a general instance-aware token seeking and discarding method that evaluates token importance by combining contextual (attention) an…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "memory efficient fine-tuning"
  - "token selection"
  - "activation optimization"
  - "instance-aware"
  - "gradient sparsification"
date: 2026-05-08
content_hash: 302968a8a2851829
---

# TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection

**Conference**: ICLR 2026
**arXiv**: [2601.19739](https://arxiv.org/abs/2601.19739)  
**Code**: [runjia.tech/iclr_tokenseek](https://runjia.tech/iclr_tokenseek)  
**Area**: LLM Efficiency
**Keywords**: memory efficient fine-tuning, token selection, activation optimization, instance-aware, gradient sparsification

## TL;DR

This paper proposes TokenSeek, a general instance-aware token seeking and discarding method that evaluates token importance by combining contextual (attention) and gradient information, updates parameters only on selected tokens, and achieves up to 65.7% reduction in activation memory while maintaining or surpassing full-token fine-tuning performance.

## Background & Motivation

LLM fine-tuning incurs substantial memory overhead, where **activations consistently dominate total memory consumption** (e.g., 87% in Llama3 8B). Existing memory-efficient fine-tuning (MEFT) methods primarily adopt three paradigms: recomputation (gradient checkpointing), compression (quantization/sparsification), and reversible networks.

**Core limitation of existing methods**: They are all **data-agnostic optimizations**—applying uniform and inflexible strategies across all instances without accounting for the rich variability within each instance. This leads to:
- **Inefficient fine-tuning**: inability to adapt memory reduction granularity to individual instances
- **Unstable fine-tuning**: large performance fluctuations

**Core challenges**:
1. How to identify salient tokens that represent critical information for each instance?
2. How to leverage them for effective and stable memory optimization?

## Method

### Overall Architecture

TokenSeek consists of two key components:
1. **Instance-Aware Token Seeking**: evaluates and scores each token
2. **Efficient Token Ditching**: updates parameters only on selected tokens, discarding gradients for the rest

### Key Design 1: Token Seeking

Token redundancy is a fundamental challenge in LLM efficiency. TokenSeek evaluates token importance by integrating two types of information:

**Contextual information** (via attention mechanism):

$$I_1(t_j) = \sum_{i=1}^{n} \mathbf{A}_{ij}$$

This represents the cumulative attention weight that token $j$ receives from all other tokens. Intuitively, tokens attended to by more other tokens are more important.

**Gradient information**:

$$I_2(t_j) = \text{Accumulate}\left[\frac{\partial \mathcal{L}}{\partial z^{(L-1)}}\right] = \sum_{k=1}^{d} \mathbf{G}_{jk}$$

This represents the sum of gradient magnitudes of the second-to-last layer activations over the hidden dimension. Tokens with larger gradients contribute more to model updates.

**Combined score**:

$$I(t_j) = \alpha \log[I_1(t_j)] + \beta \text{Norm}[I_2(t_j)]$$

A logarithm is applied to the contextual score to handle long-tail distributions caused by the Attention Sink effect, while min-max normalization is applied to the gradient score.

### Key Design 2: Token Ditching

After token selection, backpropagation is performed only on the activations of selected tokens:

$$\frac{\partial z^{(l)}}{\partial z^{(l-1)}} = [\sigma'(a_t^{(l)}), 0] W^{(l)}$$

Gradients for unselected tokens are zeroed out, so **only $a_t^{(l)}$ needs to be cached** rather than the full activation $a^{(l)}$. Theoretically, selecting only 10% of tokens requires approximately 1% of activation memory.

### Computational Overhead of Token Seeking

Only one forward pass (accounting for merely 13.3% of training memory under FP8) and one partial backward pass (with all layers frozen, computing gradients only for the output head and the final decoder block) are required.

### Loss & Training

Standard language modeling loss, computed only on selected tokens:

$$\mathcal{L} = -\sum_{j \in \text{selected}} \log P(y_j | x, y_{<j}; \theta)$$

## Key Experimental Results

### Main Results

Fine-tuning is performed on Qwen2.5 0.5B, Llama3.2 1B, and Llama3.2 3B using the Open-Platypus dataset, evaluated on MMLU, ARC, HellaSwag, TruthfulQA, and WinoGrande:

| Model | Method | Avg/Peak Memory | Avg Score |
|-------|--------|----------------|-----------|
| Llama3.2 1B | Full Token | 100%/100% | 40.82 |
| Llama3.2 1B | + TokenSeek | 64.6%/34.3% | **41.13** |
| Llama3.2 1B | LoHa | 92.3%/99.4% | 52.28 |
| Llama3.2 1B | LoHa + TokenSeek | 45.9%/28.4% | **52.58** |
| Llama3.2 1B | QLoRA | 45.6%/34.8% | 52.13 |
| Llama3.2 1B | QLoRA + TokenSeek | **14.8%/14.3%** | **52.61** |
| Llama3.2 3B | Full Token | 100%/100% | 41.53 |
| Llama3.2 3B | + TokenSeek | 73.1%/39.3% | **41.95** |
| Llama3.2 3B | QLoRA + TokenSeek | 13.3%/11.1% | 60.42 |

**Highlight**: Llama3.2 1B + QLoRA + TokenSeek achieves superior performance (52.61 vs. 40.82) over the full-token baseline while using only 14.8% of memory (2.8 GB).

### Ablation Study

| Experiment | Finding |
|------------|---------|
| α=1, β=0 (context only) | 48.45 (effective but incomplete) |
| α=0, β=1 (gradient only) | 46.39 (inferior to context-only) |
| α=5, β=5 (balanced) | Optimal combination |
| TokenTune (random selection) | Consistently below TokenSeek |
| 10% vs. 50% token ratio | More tokens reduce training loss, but too few may cause optimization collapse |

**Interpretability analysis** reveals the following token selection patterns:
- **Contextual information** favors early-position tokens — influenced by causal attention masks and the Attention Sink effect
- **Gradient information** focuses primarily on later positions — typically corresponding to the "answer" portion
- The two are complementary: contextual information selects semantically meaningful tokens, while gradient information selects tokens most important for learning

### Key Findings

1. **TokenSeek favors PEFT**: Full-parameter fine-tuning is prone to overfitting at low token ratios, whereas PEFT methods are more robust to token discarding due to the limited number of updated parameters
2. **Cross-scale generalization**: Consistently effective from 0.5B to 3B, though more sensitive on smaller models (Qwen 0.5B)
3. **Architecture-agnostic**: Relies solely on attention and gradient information, making it applicable to various Transformer architectures
4. Advantages over TokenTune (random selection) are demonstrated across all experimental settings

## Highlights & Insights

1. **"Two birds, one stone" design philosophy**: Instance-aware seeking simultaneously addresses both performance (selecting the right tokens) and memory (discarding the rest)
2. **The complementarity of context and gradient signals** carries deep implications: attention reflects "which tokens are semantically important," while gradients reflect "which tokens are important for the learning objective"
3. **The QLoRA + TokenSeek combination yields remarkable results**: the compounding of parameter efficiency and memory efficiency enables performance gains under extreme compression
4. **Interpretability analysis** revealing the impact of the Attention Sink effect and causal masking on token importance estimation provides clear directions for future research

## Limitations & Future Work

1. **Token evaluation requires an additional forward pass and partial backward pass**: although the overhead is modest, it may still warrant consideration for extremely large-scale models
2. **Selection of hyperparameters α and β**: while ablation studies are conducted, no adaptive selection strategy is provided
3. **Hard-coded 10% token ratio**: different datasets and tasks may require different ratios
4. **Lack of validation on larger-scale models (7B+)**: the largest model tested is currently 3B
5. **The relationship between training loss and downstream performance** warrants deeper analysis: why does higher training loss induced by token sparsification nevertheless improve downstream performance

## Related Work & Insights

- **TokenTune** (Simoulin et al., 2024): a pioneer in random token discarding, but data-agnostic
- **QLoRA** (Dettmers et al., 2023): a parameter-efficient method complementary to TokenSeek
- **LoRA/LoHa**: other PEFT methods, all of which can be seamlessly integrated with TokenSeek
- **Gradient checkpointing**: another class of memory optimization methods that reduce memory through recomputation

The core insight of TokenSeek: **token redundancy in fine-tuning is an exploitable property, and the key to exploiting it lies in instance-level intelligent selection rather than uniform strategies**.

## Rating

- Novelty: ⭐⭐⭐⭐ — The instance-aware token selection paradigm is novel; the combined contextual and gradient evaluation is original
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across multiple models, PEFT configurations, and ablation studies; interpretability analysis is a bonus
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich visualizations
- Value: ⭐⭐⭐⭐⭐ — Provides an immediately applicable memory optimization solution compatible with a wide range of PEFT methods

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](../../ICML2026/llm_efficiency/remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)
- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](../../NeurIPS2025/llm_efficiency/hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)
- [\[ACL 2026\] Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length](../../ACL2026/llm_efficiency/understanding_llm_performance_degradation_in_multi-instance_processing_the_roles.md)
- [\[ICML 2026\] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](../../ICML2026/llm_efficiency/efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)
- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](../../ACL2026/llm_efficiency/comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)

</div>

<!-- RELATED:END -->
