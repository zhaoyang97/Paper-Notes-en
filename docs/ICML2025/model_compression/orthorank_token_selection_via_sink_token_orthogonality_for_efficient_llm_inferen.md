---
title: >-
  [Paper Note] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference
description: >-
  [ICML2025][Model Compression][LLM Inference Acceleration] This paper proposes OrthoRank, a **training-free** dynamic token selection method. By utilizing the orthogonality between the sink token and other tokens in the hidden state space to measure token importance, OrthoRank selects the Top-K important tokens for full computation at each layer. Other tokens only participate in KV computation. OrthoRank achieves lower perplexity and higher zero-shot accuracy compared to layer…
tags:
  - "ICML2025"
  - "Model Compression"
  - "LLM Inference Acceleration"
  - "Attention Sink"
  - "Token Selection"
  - "Orthogonality"
  - "Dynamic Pruning"
date: 2026-05-08
content_hash: 85c440ab4f244961
---

# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference

**Conference**: ICML2025  
**arXiv**: [2507.03865](https://arxiv.org/abs/2507.03865)  
**Authors**: Seungjun Shin, Jaehoon Oh, Dokwan Oh  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: LLM Inference Acceleration, Attention Sink, Token Selection, Orthogonality, Dynamic Pruning  

## TL;DR

This paper proposes OrthoRank, a **training-free** dynamic token selection method. By utilizing the orthogonality between the sink token and other tokens in the hidden state space to measure token importance, OrthoRank selects the Top-K important tokens for full computation at each layer. Other tokens only participate in KV computation. OrthoRank achieves lower perplexity and higher zero-shot accuracy compared to layer pruning at the same sparsity rate.

## Background & Motivation

### Background
LLM inference incurs extremely high computational costs, especially in real-time application scenarios. Existing acceleration methods are mainly categorized as follows:
- **Layer Pruning**: Redundant layers are pruned by measuring the input-output similarity of layers or the impact of their removal on the final output. It is simple and efficient, but applies a **uniform** pruning decision to all tokens, failing to adapt to the varying computation requirements of different tokens.
- **Early Exit**: Dynamically decides whether to exit subsequent layers early based on token features. This approach is flexible but **requires training additional routers or classifiers**.
- **Mixture of Depth**: Uses a router at each layer to decide whether a token requires computation. This also **requires additional training**.

### Core Motivation
Can we determine, layer by layer, which tokens still need to be updated and which can be skipped, **without introducing any additional training**?

### Key Observation: Hidden State Behavior of Sink Tokens
Starting from the attention sink phenomenon, this paper thoroughly analyzes the relationship between the sink token (typically the first token of the sequence) and other tokens at the hidden state level, discovering two key patterns:

**Increasing Cosine Similarity**: As the layers deepen, the cosine similarity between the normalized hidden states of other tokens and that of the sink token **continues to increase**.

**Approximately Constant Sink Token Hidden State**: The normalized hidden state of the sink token remains almost unchanged across different layers.

These two observations imply that **all tokens continuously align toward the direction of the sink token across layers**. Tokens that are already close to the direction of the sink token have stabilized in terms of information and do not need further updates; tokens that remain orthogonal to the sink token's direction contain more semantic information to be extracted and require continued computation.

## Method

### Overall Architecture
The process of OrthoRank is as follows:
1. **Orthogonality Computation**: At each candidate selection layer, compute the orthogonality between the normalized hidden state of each token $\bar{h}_i^l$ and the normalized hidden state of the sink token $\bar{h}_0^l$.
2. **Ranking and Selection**: Sort the tokens in descending order of orthogonality and select the Top-K tokens to enter the full computation path.
3. **Differentiated Execution**:
    - **Selected tokens**: Undergo all operations of the layer normally (Query, Key, Value projections + Attention + FFN) and update their hidden states.
    - **Unselected tokens**: Only participate in Key and Value computation (providing context for selected tokens) but **do not update their own hidden states**, bypassing directly to the next layer.
4. **Layer Selection Strategy**: Leveraging evaluation methods from layer pruning, determine which layers are suitable for replacing full computation with token selection.

### Key Designs

#### Key Design 1: Orthogonality-Based Token Importance Measure

Let the hidden state of token $i$ at layer $l$ be $h_i^l$, and its normalized form be $\bar{h}_i^l = h_i^l / \|h_i^l\|$. Token importance is defined as its **degree of orthogonality** to the sink token:

$$\text{Importance}(i, l) = 1 - |\cos(\bar{h}_i^l, \bar{h}_0^l)| = 1 - \left|\frac{\bar{h}_i^l \cdot \bar{h}_0^l}{\|\bar{h}_i^l\| \cdot \|\bar{h}_0^l\|}\right|$$

Intuitive Explanation:
- **High Orthogonality (Important)**: The direction of the token's hidden state differs significantly from that of the sink token, indicating that the token is still moving rapidly toward the sink token's direction and contains semantic information that has not been fully processed.
- **Low Orthogonality (Unimportant)**: The token has largely aligned with the sink token, and its information has stabilized, resulting in negligible loss if its computation is skipped for this layer.

#### Key Design 2: Determining Token Selection Layers

Not all layers are suitable for token selection. This paper adopts a **layer evaluation method** similar to layer pruning to determine the optimal positions for token selection:

1. Evaluate the "replaceability" of each layer on a calibration set—observe performance changes after replacing the layer with a token selection layer.
2. Sort the layers in ascending order of performance loss and select the layers with the smallest loss as token selection layers.
3. The remaining layers undergo full computation.

This allows OrthoRank to adaptively adjust the number and locations of token selection layers under different sparsity rates.

#### Key Design 3: KV Interaction Mechanism for Unselected Tokens

Although unselected tokens do not update their own hidden states, they still participate in the Key and Value computations of the current layer. This design is crucial:

- It preserves the complete context, ensuring that the attention of selected tokens can still cover the KVs of all tokens.
- Similar to how exited tokens are handled in early-exit mechanisms, this ensures that the information flow is not broken.
- Computational overhead is primarily saved by avoiding Query projection computation, attention computation, and FFN updates for unselected tokens.

### Essential Differences from Layer Pruning

| Dimension | Layer Pruning | OrthoRank |
|------|--------|-----------|
| Granularity | Layer-level (skip entire layer) | Token-level (decide per token) |
| Adaptability | Static (shared across all inputs) | Dynamic (varies with input) |
| Information Preservation | All information of the layer is lost | Preserves KV information flow |
| Overhed / Extra Training | Not required | Not required |
| Applicable Scenario | Storage/bandwidth-constrained | Pursuit of fine-grained acceleration |

## Key Experimental Results

### Experiment 1: Language Modeling Perplexity Comparison

Comparison of Perplexity (PPL, lower is better) between OrthoRank and layer pruning methods on multiple LLMs under the same sparsity rate and comparable throughput:

| Method | Sparsity | LLaMA-2-7B PPL | LLaMA-2-13B PPL | LLaMA-3-8B PPL | Throughput Ratio |
|------|--------|----------------|-----------------|----------------|----------|
| Original Model | 0% | 5.47 | 4.88 | 6.14 | 1.0x |
| ShortGPT (Layer Pruning) | ~25% | 7.82 | 6.91 | 8.53 | ~1.3x |
| LaCo (Layer Pruning) | ~25% | 7.45 | 6.54 | 8.12 | ~1.3x |
| **OrthoRank** | **~25%** | **6.83** | **5.97** | **7.41** | **~1.3x** |
| ShortGPT (Layer Pruning) | ~40% | 12.6 | 9.87 | 15.2 | ~1.5x |
| **OrthoRank** | **~40%** | **9.14** | **7.65** | **10.8** | **~1.5x** |

OrthoRank achieves lower perplexity across all models and sparsity settings, with the advantage becoming more pronounced at higher sparsity levels.

### Experiment 2: Zero-Shot Downstream Task Accuracy

Accuracy (%, higher is better) on multiple zero-shot benchmark tasks:

| Method | Sparsity | ARC-C | HellaSwag | MMLU | WinoGrande | Average |
|------|--------|-------|-----------|------|------------|------|
| LLaMA-2-7B (Original) | 0% | 46.3 | 76.0 | 45.9 | 69.1 | 59.3 |
| ShortGPT | ~25% | 38.7 | 68.2 | 37.4 | 63.5 | 51.9 |
| LaCo | ~25% | 40.1 | 69.5 | 38.9 | 64.2 | 53.2 |
| **OrthoRank** | **~25%** | **42.8** | **72.1** | **41.5** | **66.4** | **55.7** |
| Layer Pruning (best) | ~40% | 31.2 | 57.8 | 29.6 | 56.1 | 43.7 |
| **OrthoRank** | **~40%** | **36.5** | **64.3** | **34.8** | **61.7** | **49.3** |

The average accuracy of OrthoRank on zero-shot tasks is 2–5 percentage points higher than the best layer pruning methods.

### Experiment 3: LongBench Long-Text Evaluation

OrthoRank also demonstrates advantages on LongBench, indicating that dynamic token selection is particularly effective in long-sequence scenarios—where a vast number of redundant tokens exist, and token-level selection can preserve crucial information more accurately than layer-level pruning.

## Highlights & Insights

- **Insightful**: Starting from the attention sink phenomenon, this work discovers that "all tokens continuously align toward the sink token" in the hidden state space, and translates this geometric intuition into a quantitative measure of token importance, presenting a novel and natural perspective.
- **Training-Free**: Unlike early-exit and MoD methods that require additional router training, OrthoRank computes orthogonality entirely based on the existing hidden states of the model and can be directly applied to any pretrained LLM.
- **Fine-Grained Dynamism**: Compared to layer pruning which treats all tokens uniformly, OrthoRank dynamically decides whether to update each token based on its current state, enabling finer-grained allocation of computational resources.
- **KV Preservation Design**: The design where unselected tokens still participate in KV computation cleverly balances efficiency and quality—saving most of the computation (skipping Q projection, attention, and FFN) while preserving complete contextual information.
- **Complementary to Layer Pruning**: OrthoRank leverages layer evaluation strategies from layer pruning to determine the positions of selection layers, forming a seamless integration of the two approaches.

## Limitations & Future Work

- **Simplified Definition of Sink Tokens**: The paper uniformly uses the first token as the sink token, but delimiter tokens (e.g., "." or "\n") in practice may also act as sinks. Information from multiple sink tokens has not been fully exploited.
- **Granularity of Top-K Selection**: A fixed K value is uniformly applied across all layers and inputs, without exploring adaptive K schemes.
- **Orthogonality Computation Overhead**: Although much lighter than attention/FFN computation, calculating cosine similarity for all tokens at each selection layer still introduces some overhead.
- **Decoder-Only Architecture Restriction**: Experiments are mainly validated on decoder-only models like the LLaMA series, and the applicability to encoder-decoder or encoder architectures remains unexplored.
- **Lack of Theoretical Convergence Analysis**: Theoretical upper bounds on the final output error caused by token selection are not provided.
- **Integration with Other Acceleration Techniques**: Combination with orthogonal acceleration techniques, such as quantization (e.g., GPTQ, AWQ) and KV cache compression, has not been explored.
- **Applicability to the Prefill Phase**: The current method primarily focuses on the decoding phase, and whether it remains effective in prefill phases with massive token parallel computation is worth exploring.

## Related Work & Insights

### Attention Sink
- **Xiao et al. (2024b)**: First discovered and named the attention sink phenomenon, where the first token of a sequence receives disproportionately high attention weights.
- **Sun et al. (2024), Cancedda (2024)**: Further studied the causes and properties of the sink phenomenon.
- **Ours Contribution**: Expands the research of sink tokens from the attention distribution level to the **geometric structure of the hidden state space**.

### Layer Pruning
- **ShortGPT (Men et al., 2024)**: Pruning based on inter-layer hidden state similarity.
- **LaCo (Siddiqui et al., 2024)**: Pruning based on layer importance scores.
- **Kim et al. (2024)**: One-shot pruning + LoRA fine-tuning.
- **Comparison**: OrthoRank introduces token-level granularity into the layer pruning framework, achieving a better efficiency-performance trade-off.

### Dynamic Computation
- **Schuster et al. (2022), Bae et al. (2023)**: Early-exit methods, which require additional training.
- **Raposo et al. (2024) - MoD**: Mixture of Depth, which requires router training.
- **Comparison**: OrthoRank implements a training-free dynamic token-level computation path.

## Rating

- Novelty: ⭐⭐⭐⭐ — Deriving the token importance metric from the geometric properties of sink token hidden states offers a highly novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple dimensions including perplexity, zero-shot tasks, and LongBench, with model coverage across the LLaMA-2/3 series.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation; the transition from observation to the proposed method is natural and smooth.
- Value: ⭐⭐⭐⭐ — Can be directly applied to accelerate existing LLMs without training, showing high practical value and offering a fresh perspective on token redundancy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](../../NeurIPS2025/model_compression/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)

</div>

<!-- RELATED:END -->
