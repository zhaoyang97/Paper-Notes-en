---
title: >-
  [Paper Note] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs
description: >-
  [ACL 2025][LLM Efficiency][long-context modeling] LADM proposes an attention-based long-context training data selection framework. By training a small-scale Long Attention Calculator to compute attention dependency scores between spans (PFS → AFS → CDS), it efficiently screens high-quality samples with strong long-range dependency from large-scale corpora for continual pre-training. Using only 1B tokens, it significantly enhances the long-context capabilities of LLMs.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "long-context modeling"
  - "data selection"
  - "attention mechanism"
  - "contextual dependency"
  - "continual pre-training"
date: 2026-05-08
content_hash: a22db300b44c42f9
---

# LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs

**Conference**: ACL 2025  
**arXiv**: [2503.02502](https://arxiv.org/abs/2503.02502)  
**Code**: [ZNLP/LADM](https://github.com/ZNLP/LADM)  
**Area**: LLM Efficiency  
**Keywords**: long-context modeling, data selection, attention mechanism, contextual dependency, continual pre-training  

## TL;DR

LADM proposes an attention-based long-context training data selection framework. By training a small-scale Long Attention Calculator to compute attention dependency scores between spans (PFS → AFS → CDS), it efficiently screens high-quality samples with strong long-range dependency from large-scale corpora for continual pre-training. Using only 1B tokens, it significantly enhances the long-context capabilities of LLMs.

---

## Background & Motivation

**Background:** Long-context modeling is a hot topic in the LLM field, with GPT-4 supporting 128K and Gemini 1.5 supporting 1M tokens. Continual pre-training on long-context data is the standard approach to endow LLMs with long-input processing capabilities. However, measuring the quality of training data has not received sufficient attention.

**Limitations of Prior Work:** (1) Simple concatenation of short texts results in a lack of cross-segment dependencies within samples, causing models to ignore long-range information; (2) ProLong (Chen et al., 2024a) segments samples to compute delta perplexity but ignores the intrinsic structures and relationships in the full context; (3) Similarity-based document aggregation methods (Staniszewski et al., 2023) cannot accurately measure the dependency strength within existing samples.

**Core Motivation:** The attention mechanism itself possesses an inherent retrieval capability—assigning higher weights to historical tokens that are more relevant to the current token. Leveraging this characteristic can accurately measure the cross-segment dependency relationships in long contexts.

---

## Method

### Overall Architecture

LADM consists of four steps: (1) training a Long Attention Calculator → (2) calculating the Pairwise Focus Score (PFS) → (3) aggregating into the Aggregated Focus Score (AFS) → (4) combining into the Contextual Dependency Score (CDS) for sample ranking and selection.

### Key Designs

**1. Long Attention Calculator:** A small-scale long-context model trained based on TinyLlama-1.1B, optimized with 5B randomly sampled tokens to handle a 32K context window. Validation shows that it can distinguish samples with different dependency intensities through attention scores.

**2. Pairwise Focus Score (PFS):** Measures the attention dependency of the $j$-th span in a sample on the $i$-th span:
$$\text{PFS}(i,j) = \text{Sum}\left(\text{Softmax}\left(\frac{Q_j K_{0:j}^T}{\sqrt{d_k}}\right)[:,i]\right)$$
representing the cumulative attention weight assigned from span $j$ to span $i$.

**3. Aggregated Focus Score (AFS):** For each span, aggregates its PFS with all preceding spans, excluding the first $m$ and local $n$ spans (to avoid biases from initial tokens and nearby neighbors), and incorporates distance and variance weighting:
$$\text{AFS}(j) = \sigma_j \sum_{i=m}^{j-n-1} \frac{j-i}{N} \cdot \text{PFS}(i,j)$$

### Sample-level Scoring

The Contextual Dependency Score (CDS) is obtained through a weighted sum of the AFS of all spans:
$$\text{CDS}(S) = \sum_{j=n_0}^{N-1} \frac{j}{N} \cdot \text{AFS}(j)$$

Spans at later positions are assigned higher weights, encouraging long-range dependency.

---

## Experiments

### Main Results: Perplexity Evaluation

| Model | Method | 2K | 4K | 8K | 16K | 32K |
|------|------|-----|-----|-----|------|------|
| L-7B | Random | 4.515 | 3.900 | 3.264 | 2.780 | 2.458 |
| L-7B | ProLong | 4.516 | 3.906 | 3.275 | 2.792 | 2.470 |
| L-7B | **LADM** | **4.481** | **3.878** | **3.252** | **2.773** | **2.453** |
| M-7B | Random | 4.620 | 3.936 | 3.267 | 2.775 | 2.455 |
| M-7B | ProLong | 4.293 | 3.696 | 3.095 | 2.644 | 2.346 |
| M-7B | **LADM** | **4.266** | **3.673** | **3.076** | **2.629** | **2.332** |

### LongBench Empirical Evaluation (Partial)

| Model | Method | Single-Doc QA Avg | Multi-Doc QA Avg |
|------|------|-------------------|------------------|
| L-7B | Random | 30.13 | 25.80 |
| L-7B | ProLong | 29.50 | 29.04 |
| L-7B | **LADM** | **32.24** | **31.10** |
| M-7B | Random | 24.08 | 24.63 |
| M-7B | ProLong | 23.76 | 24.43 |
| M-7B | **LADM** | **33.85** | **29.09** |

### Ablation Study & Analysis

| Analysis Dimension | Finding |
|----------|------|
| Concatenated Data Length | Average retrieval accuracy of 4K concatenation → 32K is only 0.57, whereas native 32K data achieves 0.88 |
| Attention Score Validation | Long Attention Calculator successfully distinguishes samples of different dependency intensities |
| Training Efficiency | LADM with 0.5B tokens outperforms the Random method using 1B tokens |
| Needle-in-Haystack | L-7B/13B and M-7B achieve near 100% retrieval rate |

### Key Findings

- **Dependency is Core:** Training data using concatenated short texts lacks cross-segment dependencies, significantly degrading long-context modeling performance.
- **Attention as dependency probes:** Attention distribution can effectively serve as a metric signal for long-context dependency.
- **Data Efficiency Boost:** LADM outperforms the full training of random sampling while using only half the training tokens.
- **Cross-model Robustness:** Consistently effective across four models: OpenLlama-3B, Llama2-7B, 13B, and Mistral-7B.

---

## Highlights & Insights

- Proposes a clear and actionable quality measurement framework for long-context data (PFS → AFS → CDS).
- Innovatively leverages the intrinsic retrieval capabilities of the attention mechanism to measure contextual dependency.
- Uses a low-cost TinyLlama as a data filter to achieve an efficient data selection pipeline.
- Achieves a substantial improvement in long-context capabilities with only 1B tokens of continual pre-training.

## Limitations & Future Work

- Experiments are constrained to a context length of 32K tokens; applicability to longer contexts (64K/128K) has not been verified.
- The Long Attention Calculator itself requires 5B tokens for training, introducing a certain amount of front-end cost.
- Data selection maintains the original domain distribution, without exploring potential benefits of adjusting the domain distribution.
- Analysis of parameter sensitivity for multiple hyperparameters in the CDS score ($m$, $n$, $d$, $n_0$) is currently lacking.

## Related Work & Insights

- **Long-context Modeling:** Training methods such as RoPE scaling (PI, YaRN, NTK), LongLoRA, S²-Attn, etc.
- **Long-context Data:** ProLong (Chen et al., 2024a) based on delta perplexity for data filtering.
- **Document Aggregation:** Similarity-based document composition by Staniszewski et al. (2023).
- **Attention Retrieval:** Wu et al. (2024) revealing retrieval operations within the attention mechanism.

---

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Rating | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](../../NeurIPS2025/llm_efficiency/loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)

</div>

<!-- RELATED:END -->
