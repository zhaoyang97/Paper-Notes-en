---
title: >-
  [Paper Note] Memory Mosaics at Scale
description: >-
  [NeurIPS 2025][LLM Pretraining][Memory Mosaics] Memory Mosaics v2 scales associative memory networks to 10B parameters trained on 1T tokens…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Memory Mosaics"
  - "Gaussian kernel regression"
  - "in-context learning"
  - "compositionality"
  - "large-scale scaling"
date: 2026-05-08
content_hash: 6149f3be2c0c7570
---

# Memory Mosaics at Scale

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2507.03285](https://arxiv.org/abs/2507.03285)  
**Code**: [https://github.com/facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics)  
**Authors**: Jianyu Zhang, Léon Bottou (NYU & FAIR, Meta)
**Area**: LLM Pretraining
**Keywords**: Memory Mosaics, Gaussian kernel regression, in-context learning, compositionality, large-scale scaling

## TL;DR
Memory Mosaics v2 scales associative memory networks to 10B parameters trained on 1T tokens, substantially outperforming same-scale—and even 8T-token-trained—Transformers on new-task learning and in-context learning.

## Background & Motivation
- Compositional generalization and in-context learning (ICL) have long been central objectives in machine learning, yet the underlying mechanisms in existing Transformers remain opaque.
- Prior efforts via statistical independence (ICA) and multi-environment optimization (IRM/MAML) have achieved limited success.
- Memory Mosaics (Zhang et al., 2025) replaces attention with simple key-value associative memories (without positional encoding), demonstrating superior ICL capability at GPT-2 scale on synthetic data.
- **Core Problem**: Can these advantages be preserved at large scale on real data? This work scales Memory Mosaics to LLaMA-8B size to answer this question.

## Method

### Associative Memory Fundamentals
- An associative memory stores key-value pairs $\{(k_1,v_1)\dots(k_n,v_n)\}$ and retrieves values given a query key.
- The storage set is **permutation-invariant** and can be viewed as estimating the conditional probability $P(V|K)$; retrieval computes the conditional expectation.
- Retrieval is implemented via Gaussian kernel regression: $f(k) = \sum_i \frac{e^{-\beta\|k-k_i\|^2}}{\sum_j e^{-\beta\|k-k_j\|^2}} v_i$
- When all key vectors share the same L2 norm, this reduces to standard softmax attention (dot-product form).

### Key Differences from Transformer Attention
1. **L2-normalized keys + explicit bandwidth $\beta$**: controls the bias-variance trade-off in kernel regression.
2. **Symmetric key-query formulation**: keys and queries share the same extractor, eliminating the need to learn separate $W_q$ and $W_k$.
3. **No positional encoding**: keys represent the recent past and values represent the near future; a single layer suffices to implement an induction head.

### Three Architectural Improvements in Memory Mosaics v2

#### 1. Adaptive Gaussian Kernel Bandwidth
- The original model uses a fixed $\beta$, but the optimal bandwidth depends on the number of samples $n$ (bias-variance trade-off).
- v2 adopts a learnable adaptive bandwidth: $\beta = \beta_1 n^\alpha + \beta_0$
- where $\beta_0 \geq 0$, $\beta_1 > 0$, $0 < \alpha < 1$ are all learnable parameters.
- Intuition: the more key-value pairs in memory, the smaller the bandwidth ($1/\sqrt{\beta}$), yielding finer estimation.

#### 2. Gated Time-Varying Key Feature Extractor
- The original model uses a fixed-weight leaky average: $\bar{k}_T = \tilde{k}_T + \lambda \bar{k}_{T-1}$ with fixed $\lambda$.
- Problem: "tom-and-jerry" and "tom---and---jerry" are semantically equivalent but produce different key features.
- v2 introduces an input-dependent gating mechanism:
    - $g_T = e^{W_g x_T}$ (exponential gate controlling the contribution of the current input)
    - $\lambda_T = e^{-|W_\lambda x_T|}$ (time-varying forget factor, semantically driven)
    - $\bar{k}_T = g_T \tilde{k}_T + \lambda_T \bar{k}_{T-1}$
- Inspired by recurrent architectures such as RWKV, Mamba, and xLSTM, but **used solely for key construction; the associative memory still retains all key-value pairs**.

#### 3. Three-Tier Memory Design
- **Short-term memory**: stores only key-value pairs within $h=256$ steps of position $t$, handling position-sensitive signals.
- **Long-term memory**: skips nearby tokens, storing only key-value pairs from positions before $t-m$, handling position-invariant signals.
    - During training, $m \in [64, 256]$ is sampled randomly; at inference, $m=64$ is fixed.
    - Setting $m < h$ causes short- and long-term memories to overlap, creating a soft boundary.
- **Persistent memory**: two-layer SwiGLU FFN storing global knowledge from training data (equivalent to a high-capacity associative memory).
- Outputs of multiple short- and long-term memories are concatenated and fused via a linear projection $W_o$.

### Training Configuration

| Configuration | Small (LLaMA-1.5B scale) | Large (LLaMA-8B scale) |
|---|---|---|
| Layers | 24 | 32 |
| Hidden Dimension | 2048 | 4096 |
| Attention Heads | 16 | 32 |
| Training Tokens | 200B | 1T |
| Training Context | 4096 → fine-tuned to 32768 | 4096 → fine-tuned to 32768 |

## Three-Dimensional Evaluation Framework

### Dimension 1: Persistent Knowledge Storage and Retrieval
- Evaluates the ability of persistent memory (FFN) to store knowledge from training data.
- 19 standard language benchmarks (ARC, PIQA, BoolQ, HellaSwag, MMLU, etc.).
- **Results**: MM v2 and Transformer perform comparably (52.2% vs. 52.2%), as expected given the shared persistent memory architecture.
- **Validation**: removing long-term memory leaves 13 benchmarks nearly unaffected (56.6% vs. 56.8%), confirming that these tasks rely solely on persistent knowledge.

### Dimension 2: New Knowledge Storage and Retrieval
- Evaluates the model's ability to store and retrieve new information at inference time.
- Uses the Multi-Document Question Answering task from the Ruler benchmark (concatenating multiple articles with a question).
- Substantially harder than "needle-in-a-haystack"—high information entropy, not simple exact matching.

| Model | Train ctx | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|---|
| Transformer large | 32k | 51.2 | 48.8 | 44.7 | 41.1 | × |
| MM v2 large | 32k | **58.9** | **55.5** | **54.9** | **53.4** | **46.4** |

- MM v2 outperforms Transformer by **12.3%** at 32k task length.
- MM v2 trained at 4k extrapolates to 32k without fine-tuning (Transformer fails beyond 4k→8k).
- Compressed-memory approaches such as RNN/SSM/sliding-window attention **structurally fail** on this task—they cannot retain all articles before encountering the question.

### Dimension 3: In-Context Learning (ICL)
- Uses standard multi-class classification: Banking77 (77 classes), Tacred (41 classes), GoEmotion (28 classes).
- Both semantic-label and anonymized-label variants are evaluated; the latter more rigorously tests genuine new-task learning.
- **Key Findings**:
    - MM v2 accuracy **consistently improves** as the number of shots increases.
    - Transformers exhibit an anomalous trend—more shots actually degrade performance.
    - MM v2 surpasses Transformer by **over 10%**.
- Adding separate short- and long-term attention to a Transformer does not reproduce this advantage, confirming that MM v2 is not a simple Transformer variant.

## Extended Data Comparison: 1T MM v2 vs. 8T Transformer

| Dimension | Transformer 1T | Transformer 8T | MM v2 1T |
|---|---|---|---|
| New Knowledge Storage (32k) | 41.1% | 46.9% | **53.4%** |
| Semantic-Label ICL | Lower | Close to MM v2 | **Best** |
| Anonymized-Label ICL | Low | Even lower (degrades) | **Significantly best** |

- A Transformer trained on 8× more data still trails MM v2 (1T) by approximately **6.5%** on new knowledge storage.
- On anonymized-label ICL, more training data actually degrades Transformer performance—the "more data" strategy fails completely.

## Fine-Tuning Efficiency
- MM v2 achieves a 22% accuracy gain with only **1 mini-batch** of fine-tuning.
- Optimal performance is reached with just 2 mini-batches.
- A Transformer fine-tuned for 800 mini-batches still underperforms MM v2 fine-tuned for 1 mini-batch.

## Computational Overhead

| Model | Parameters | FLOPs/token |
|---|---|---|
| Transformer large | 8.8B | 16.7B |
| MM v2 large | 9.9B | 18.9B |
| MM v2 (without long-term memory) | 8.3B | 15.6B |

- MM v2 incurs slightly higher parameter count and computation, though removing long-term memory yields a lighter model.
- The 10%+ performance advantage substantially outweighs the 13% increase in computational cost.

## Highlights & Insights
1. **Interpretable memory mechanism**: explicit key-value storage makes attention allocation transparent; attention scores over distant tokens are position-invariant (vs. position-dependent curves in Transformers).
2. **Challenging scaling-law orthodoxy**: a Transformer trained on 8T tokens still falls short of MM v2 trained on 1T tokens on new-task learning, demonstrating that architectural innovation matters more than data accumulation.
3. **Context length extrapolation**: the absence of positional encoding combined with adaptive bandwidth enables a model trained at 4k to extrapolate directly to 32k without fine-tuning.
4. **Extremely fast fine-tuning**: adaptation to a new domain requires only 1 mini-batch, offering high practical value.
5. **Clear division of labor across three memory tiers**: short-term handles local patterns, long-term handles remote retrieval, and persistent memory stores global knowledge.

## Limitations & Future Work
- FLOPs/token are slightly higher than a same-scale Transformer (18.9B vs. 16.7B), with greater overhead at long contexts.
- Long-term memory must store all historical key-value pairs, leading to large memory consumption at very long contexts.
- Techniques such as fuzzy hashing and hierarchical memory for long-context optimization have not yet been explored.
- Validation is limited to the 10B scale; whether the advantages persist at 70B/400B remains unknown.

## Related Work & Insights
- Bietti et al. (2023) analyzed that Transformer induction heads require positional encoding and asymmetric QK; MM achieves equivalent behavior with symmetric keys in a single layer.
- Olsson et al. (2022) identified the induction head mechanism as central to ICL; MM constructs this mechanism explicitly.
- Compressed-memory methods such as RWKV, Mamba, and xLSTM structurally fail on multi-document QA because they cannot fully retain all information.

## Rating
⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](../../ICCV2025/llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[ACL 2026\] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](../../ACL2026/llm_pretraining/working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)

</div>

<!-- RELATED:END -->
