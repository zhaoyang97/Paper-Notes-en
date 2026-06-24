---
title: >-
  [Paper Note] MoH: Multi-Head Attention as Mixture-of-Head Attention
description: >-
  [ICML 2025][LLM Efficiency][Mixture-of-Experts] This paper reformulates Multi-Head Attention (MHA) into a summation form and proposes Mixture-of-Head Attention (MoH) inspired by MoE. By utilizing a router to dynamically select the most relevant subset of attention heads for each token, MoH satisfies or even surpasses standard MHA performance while activating only $50\% \sim 90\%$ of the heads. It also demonstrates that pre-trained models (such as LLaMA3-8B) can be successfull…
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Attention Head Routing"
  - "Sparse Activation"
  - "Multi-Head Attention"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 4e724b64913ec0fe
---

# MoH: Multi-Head Attention as Mixture-of-Head Attention

**Conference**: ICML 2025  
**arXiv**: [2410.11842](https://arxiv.org/abs/2410.11842)  
**Code**: [https://github.com/SkyworkAI/MoH](https://github.com/SkyworkAI/MoH)  
**Area**: LLM Efficiency / Attention Mechanisms  
**Keywords**: Mixture-of-Experts, Attention Head Routing, Sparse Activation, Multi-Head Attention, Inference Acceleration

## TL;DR

This paper reformulates Multi-Head Attention (MHA) into a summation form and proposes Mixture-of-Head Attention (MoH) inspired by MoE. By utilizing a router to dynamically select the most relevant subset of attention heads for each token, MoH satisfies or even surpasses standard MHA performance while activating only $50\% \sim 90\%$ of the heads. It also demonstrates that pre-trained models (such as LLaMA3-8B) can be successfully converted into MoH models via continue-tuning.

## Background & Motivation

**Background**: Multi-Head Attention (MHA) is a core component of Transformers, widely used in NLP and CV. The standard approach computes all attention heads in parallel and then concatenates or sums their outputs, meaning every token is processed by all heads.

**Limitations of Prior Work**: Numerous studies suggest significant redundancy in multi-head attention. Voita et al. demonstrated that many attention heads can be pruned without affecting accuracy, and Michel et al. found that even severe pruning does not significantly degrade performance. This implies that standard MHA performs a large amount of unnecessary computation during inference.

**Key Challenge**: MHA activates all attention heads indiscriminately for all tokens, whereas different tokens actually require processing from only a subset of heads. This "one-size-fits-all" design wastes computational resources and limits head specialization, as all heads train on the same data and easily learn similar features.

**Goal**: (a) How can each token dynamically select the required attention heads? (b) How can the number of activated heads be reduced while maintaining or even improving performance? (c) How can existing pre-trained models be efficiently converted into sparse-head models?

**Key Insight**: The authors notice that MHA can be rewritten from the standard concatenation form into a **summation form**, where the output equals a simple sum of each head's output. Since it is a summation, one can weight and sparsely activate these terms just like in MoE, naturally leading to the "heads as experts" analogy.

**Core Idea**: Treat attention heads as experts in MoE, use a router to select Top-K heads for each token, and enhance routing stability through shared heads and a two-stage routing mechanism.

## Method

### Overall Architecture

MoH replaces the MHA layer in standard Transformer architectures. The input remains a token sequence $\mathbf{X} \in \mathbb{R}^{T \times d_{in}}$, and the output dimension remains unchanged. Each MoH layer contains: (1) $h$ attention heads (identical to standard MHA), (2) a router, and (3) a subset of shared heads. For each token, the router selects Top-K active heads among the non-shared heads, while the shared heads are always activated. The final output is a **weighted sum** of the outputs from all active heads (rather than the equal-weight sum in standard MHA), with weights provided by the router.

### Key Designs

1. **Summation Form Representation of MHA**

    - **Function**: Reformulate standard MHA from a concatenation form to an equivalent summation form.
    - **Mechanism**: Decompose the output projection matrix $\mathbf{W}_O \in \mathbb{R}^{d_v \times d_{out}}$ row-wise into $h$ sub-matrices $\mathbf{W}_O^i$, yielding $\text{MultiHead}(\mathbf{X}, \mathbf{X}') = \sum_{i=1}^{h} \mathbf{H}^i \mathbf{W}_O^i$.
    - **Design Motivation**: The summation form reveals the independence of each head. Since the output is a simple sum of each head's contribution, one can naturally set certain terms to zero (sparsity) or re-weight them, providing a theoretical foundation for introducing the MoE routing mechanism.

2. **Heads as Experts**

    - **Function**: Treat $h$ attention heads as $h$ experts in MoE, dynamically activating Top-K heads via a router.
    - **Mechanism**: The output of MoH is $\text{MoH}(\mathbf{X}, \mathbf{X}') = \sum_{i=1}^{h} g_i \mathbf{H}^i \mathbf{W}_O^i$, where $g_i$ is the routing score and $g_i = 0$ for inactive heads. Unlike standard MoE, MoH does not increase the number of heads, and the total parameter count remains basically identical to MHA.
    - **Design Motivation**: Unlike MoE which aims at parameter scaling, the core objective of MoH is to reduce the activation of redundant heads to improve inference efficiency. Replacing the equal-weight sum with a weighted sum increases flexibility, unlocking additional performance potential.

3. **Shared Heads**

    - **Function**: Appoint $h_s$ heads as shared heads that are always activated for all tokens.
    - **Mechanism**: Shared heads capture general knowledge across contexts (such as syntactic rules), and their routing scores are computed via Softmax using an independent projection matrix $\mathbf{W}_s \in \mathbb{R}^{h_s \times d_{in}}$. Experiments show that model performance remains stable when the shared head ratio is within $13.9\% \sim 74.0\%$.
    - **Design Motivation**: Inspired by DeepSeekMoE, shared heads focus on common knowledge, allowing the routed heads to specialize more in domain- or task-specific information, thereby reducing redundancy among routed heads. Shared heads can also be viewed as a form of Soft MoE; the authors suggest a shared head ratio of $> 40\%$.

4. **Two-Stage Routing**

    - **Function**: Compute routing scores in two stages: first, normalize within each type using Softmax, and then use learnable coefficients $\alpha_1, \alpha_2$ to balance the contributions of shared and routed heads.
    - **Mechanism**: The routing score is defined as a piecewise function. For a shared head $i$: $g_i = \alpha_1 \cdot \text{Softmax}(\mathbf{W}_s \mathbf{x}_t)_i$; for an active routed head: $g_i = \alpha_2 \cdot \text{Softmax}(\mathbf{W}_r \mathbf{x}_t)_{i-h_s}$. The balancing coefficients are $[\alpha_1, \alpha_2] = \text{Softmax}(\mathbf{W}_h \mathbf{x}_t)$, where $\mathbf{W}_h \in \mathbb{R}^{2 \times d_{in}}$ is learnable.
    - **Design Motivation**: Single-stage routing cannot dynamically adjust the overall importance of shared heads relative to routed heads. The two-stage design allows the model to adaptively allocate contribution weights to both types of heads based on the input.

5. **Continue-Tuning Strategy (Pre-trained Model Conversion)**

    - **Function**: Convert an existing pre-trained MHA model (such as LLaMA3-8B) into an MoH model.
    - **Mechanism**: Alleviate three core challenges: (a) Shared head selection: directly select the first 16 heads of each layer as shared heads; (b) Parameter-free router initialization: use the $\ell_2$-norm of each head's query as the routing score, avoiding random initialization; (c) Routing score quantization: use a straight-through estimator to handle the quantized score $g_i^q = \mathbb{1}(\text{Token } \mathbf{x} \text{ selects Head } i)$, preventing the weighted sum from drastically altering the output distribution.
    - **Design Motivation**: Training from scratch is extremely expensive. By utilizing clever initialization and two-stage training (first 300B tokens to adapt to the data distribution, then 100B tokens to convert to MoH), the conversion can be completed using only about $3\%$ of the original pre-training data volume.

### Loss & Training

- **Load Balance Loss**: Prevents routing collapse (most tokens routed to a minority of heads), formulated as $\mathcal{L}_b = \sum_{i=h_s+1}^{h} P_i f_i$, where $P_i$ is the average routing probability of head $i$, and $f_i$ is the fraction of tokens that choose head $i$.
- **Overall Training Objective**: $\mathcal{L} = \mathcal{L}_{task} + \beta \mathcal{L}_b$, where $\beta = 0.01$ (unified across all tasks).
- **Non-Uniform Head Activation Budget**: Fewer heads are activated in shallower layers, and more heads are activated in deeper layers (inspired by the design of TransNeXt).
- **Two-Stage Continue-Tuning**: The first stage uses 300B tokens to adapt to the data distribution, and the second stage uses 100B tokens to train the MoH routing.

## Key Experimental Results

### Main Results 1: ViT Image Classification (ImageNet-1K)

| Method | Params (M) | Activated Head Ratio | Top-1 Acc (%) |
|------|-----------|-----------|--------------|
| TransNeXt-S | 50 | 100% | 84.7 |
| **MoH-ViT-S** | **50** | **80%** | **84.7** |
| MoH-ViT-S | 50 | 75% | 84.6 |
| TransNeXt-B | 90 | 100% | 84.8 |
| **MoH-ViT-B** | **90** | **75%** | **84.9** |
| MoH-ViT-B | 90 | 50% | 84.7 |

### Main Results 2: Continue-Tuning LLaMA3-8B (Average of 14 Benchmarks)

| Method | Activated Head Ratio | MMLU | CEVAL | CMMLU | GSM8K | TruthfulQA | HellaSwag | ARC-C | 14-Task Avg |
|------|-----------|------|-------|-------|-------|------------|-----------|-------|---------|
| LLaMA3-8B | 100% | 65.2 | 52.3 | 50.7 | 49.5 | 35.4 | 81.9 | 59.0 | 61.6 |
| **MoH-LLaMA3-8B** | **75%** | **65.8** | **61.5** | **64.4** | **56.9** | **44.0** | 80.1 | **60.1** | **64.0** |

MoH-LLaMA3-8B uses only $75\%$ of the attention heads and achieves an average gain of **2.4%** across 14 benchmarks, carrying particularly significant improvements in Chinese tasks (CEVAL +9.2, CMMLU +13.7) and mathematical reasoning (GSM8K +7.4).

### Ablation Study

| Shared Heads | Two-Stage Routing | ViT Acc (%) | DiT FID ↓ |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 75.6 | 71.97 |
| ✓ | ✗ | 78.3 | 69.54 |
| **✓** | **✓** | **78.6** | **69.42** |

### Key Findings

- **Shared Heads Contribute the Most**: Introducing shared heads improves ViT accuracy from $75.6\%$ to $78.3\%$ ($+2.7\%$), proving to be a key factor in performance improvement.
- **Fewer Heads Can Be Better in Small Models**: MoH-LLM-S activating only $50\%$ of the heads actually outperforms activating $75\%$ ($45.4\%$ vs $44.6\%$). The authors suggest that in small-model + small-data scenarios, fewer heads provide a regularization effect.
- **Image Generation Tasks Require More Heads**: The performance of DiT models degrades under a $75\%$ activation rate, as pixel-level dense prediction requires more heads to capture fine-grained relations.
- **Measurable Inference Acceleration**: With a sequence length of 512, MoH with a $50\%$ activation rate takes 0.863ms compared to MHA's 1.376ms, achieving approximately a $37\%$ speedup.
- **Visualization of Head Workload** reveals distinct head allocation patterns across different categories/tasks, demonstrating that MoH successfully achieves head specialization.

## Highlights & Insights

- **The insight of summation form is highly critical**: Reformulating MHA in a summation form seems simple, yet it opens the door to incorporating MoE routing. This reinterpretation of existing formulas is an inspiring research paradigm—many innovations do not require inventing completely new structures, but simply viewing old problems from a new perspective.
- **Ingenious continue-tuning strategy design**: The combination of a parameter-free router (using the L2-norm of queries) + quantized routing scores + straight-through estimator elegantly solves the cold-start problem when injecting new modules into pre-trained models. This strategy can be migrated to any scenario where sparse routing needs to be introduced to pre-trained models.
- **Cross-modal validation enhances credibility**: Verifying the same method across three distinct architectures (ViT, DiT, and LLM) and yielding positive gains across all of them indicates that attention head redundancy is a universal phenomenon and that MoH's solution is highly generalizable.
- **Shared heads can be viewed as Soft MoE**: This perspective connects shared heads to Soft MoE proposed by Puigcerver et al., providing theoretical support for understanding the role of shared heads.

## Limitations & Future Work

- **Relatively high activation rates**: Currently, at least $50\%$ of the heads are required to maintain performance; future work needs to explore more aggressive sparsification ($<50\%$).
- **Fixed head dimensions**: All attention heads have the same hidden size, leaving heterogeneous heads (heads of different sizes for different functions) unexplored, which represents a natural extension.
- **Considerable data requirement for continue-tuning**: Requiring 400B tokens (300B for adaptation + 100B for conversion) still imposes a relatively high demand on computational power.
- **Validation limited to decoder-only LLMs**: Encoder-only (like BERT) or encoder-decoder (like T5) models have not been tested.
- **Unexplored multi-modal scenarios**: Vision and text tokens exhibit different patterns in attention; how MoH's routing mechanism handles multimodal inputs remains an open question.
- **Inference speedup depends on sparse matrix multiplication**: Practical deployment requires operator-level support (sparse QKV), whereas optimization for sparse computations on current hardware remains insufficient.

## Related Work & Insights

- **vs MoA (Zhang et al., 2022)**: MoA also combines attention heads with MoE, but its target is parameter scaling (similar to standard MoE), and it requires sharing K/V, thereby necessitating training from scratch. MoH does not increase parameters, supports continue-tuning, and offers wider applicability.
- **vs SwitchHead (Csordás et al., 2024)**: This method similarly adopts an MoE-style activation of attention heads, but MoH additionally introduces shared heads and two-stage routing, yielding more stable training and better performance.
- **vs DuoAttention (Xiao et al., 2024)**: DuoAttention distinguishes between retrieval heads and streaming heads to optimize KV cache, focusing on long-context inference. Conversely, MoH optimizes head activation at a more fundamental level; the two ideas can be integrated.
- **vs Head Pruning Methods (Voita et al., 2019; Michel et al., 2019)**: Traditional head pruning is static (pruning the same heads for all inputs), whereas MoH is dynamic (activating different heads for different tokens), which is more flexible but introduces routing overhead.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Re-writing MHA into a summation form before introducing MoE routing is a natural yet effective innovation. The combination of shared heads + two-stage routing + continue-tuning offers significant engineering contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated across three major architectures (ViT, DiT, and LLM), spanning both training-from-scratch and continue-tuning, with thorough ablations and in-depth visual analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, smoothly transitioning from the derivation of the summation form to the proposed MoH with standardized formula formatting.
- **Value**: ⭐⭐⭐⭐ As a plug-and-play alternative to MHA, MoH possesses broad prospects for application, and the continue-tuning scheme successfully lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Mixture of Lookup Experts](mixture_of_lookup_experts.md)
- [\[ICLR 2026\] MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](../../ICLR2026/llm_efficiency/mhla_restoring_expressivity_of_linear_attention_via_token-level_multi-head.md)
- [\[ACL 2026\] BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](../../ACL2026/llm_efficiency/bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i.md)
- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)
- [\[NeurIPS 2025\] From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](../../NeurIPS2025/llm_efficiency/from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)

</div>

<!-- RELATED:END -->
