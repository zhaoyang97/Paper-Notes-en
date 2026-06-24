---
title: >-
  [Paper Note] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval
description: >-
  [ACL 2025][LLM Efficiency][Attention Head Scaling] By identifying that specific attention heads/channels have positive or negative impacts on long-context retrieval, SEAL designs learnable head-level and channel-level scaling factors. Fine-tuning with only 50 synthetic samples significantly improves the long-context retrieval performance of LLMs, and these scaling factors can be merged into model weights offline to achieve zero inference overhead.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Attention Head Scaling"
  - "Long-Context Retrieval"
  - "Channel-level Tuning"
  - "Synthetic Data Fine-tuning"
  - "Zero Inference Overhead"
date: 2026-05-08
content_hash: 628d9ce7da9affb4
---

# SEAL: Scaling to Emphasize Attention for Long-Context Retrieval

**Conference**: ACL 2025  
**arXiv**: [2501.15225](https://arxiv.org/abs/2501.15225)  
**Code**: None  
**Area**: LLM Efficiency / Long-Context Retrieval  
**Keywords**: Attention Head Scaling, Long-Context Retrieval, Channel-level Tuning, Synthetic Data Fine-tuning, Zero Inference Overhead

## TL;DR

By identifying that specific attention heads/channels have positive or negative impacts on long-context retrieval, SEAL designs learnable head-level and channel-level scaling factors. Fine-tuning with only 50 synthetic samples significantly improves the long-context retrieval performance of LLMs, and these scaling factors can be merged into model weights offline to achieve zero inference overhead.

## Background & Motivation

1. **Background**: Current LLMs are designed to support ultra-long context windows (32K-128K tokens). However, in practice, even within the window limit, retrieval quality still degrades significantly as the input length increases. For example, LongChat-7B-32K exhibits severe performance degradation in the range of 19K-31K.

2. **Limitations of Prior Work**: This degradation is not caused by insufficient knowledge capacity of the model, but is more likely due to inherent biases in the training data (such as locality bias). Existing context extension methods either require high fine-tuning costs (e.g., PI, YaRN) or yield suboptimal performance (e.g., training-free methods like NTK, Self-Extend).

3. **Key Challenge**: LLMs inherently possess the capability for accurate reasoning at arbitrary lengths, but the biases embedded in the trained parameters lead to long-context performance degradation. The key challenge is—how to correct these biases at an extremely low cost?

4. **Goal**: (1) How to identify the key attention components critical for long-context retrieval? (2) How to efficiently adjust the strength of these components to improve retrieval performance? (3) How to guarantee zero additional inference overhead?

5. **Key Insight**: The authors performed a key experiment—pruning each attention head one by one. They discovered that pruning certain heads improved performance by over 20%, whereas pruning others caused a degradation of over 20%. This indicates that different attention heads contribute vastly differently to retrieval, and this difference remains consistent across lengths.

6. **Core Idea**: Enhance retrieval-friendly attention components and suppress adverse ones by learning head-level and channel-level scaling factors, achieving low-cost, high-yield long-context enhancement.

## Method

### Overall Architecture

The pipeline of SEAL: (1) Using an LLM, 50 synthetic training samples (keeping only the format while using random content) are generated based on the target task format; (2) Learnable scaling parameters are attached to the output of each attention head; (3) Gradient optimization is performed on the synthetic data to learn the optimal scaling; (4) The learned scaling factors are merged offline into the `v_proj` or `o_proj` weights, incurring zero extra computation during inference. The input is a long-context text + retrieval question, and the output is the correct retrieved answer.

### Key Designs

1. **Head-wise Pruning Analysis (Validation of Motivation)**:
    - **Function**: Pruning each attention head of LongChat-7B one by one to observe the impact on retrieval accuracy.
    - **Key Findings**: The impact of different heads varies drastically ($\pm 20\%$), displaying consistent behaviors in medium and long contexts. Structuring these based on their impact direction reveals four quadrants—heads in Q1 (performance improves after pruning) and Q3 (performance degrades after pruning) can be adjusted simultaneously for additive effects. Scaling Q1 down to 0.9 and scaling Q3 up to 1.1 allowed LongChat-7B's performance at 31K length to jump from 32% to approximately 60%+.
    - **Design Motivation**: This validates the hypothesis that "attention heads have functional division of labor," providing an experimental foundation for subsequent learning methods.

2. **SEAL-H (Head-level Scaling)**:
    - **Function**: Learn a scalar scaling factor $s_{l,h} \in \mathbb{R}$ for each attention head.
    - **Mechanism**: The output of each head in every Transformer layer is multiplied by its corresponding learnable scalar. For the entire LongChat-7B (32 layers $\times$ 32 heads), there are only 1024 learnable parameters in total. Gradient optimization is performed on synthetic data for 1 epoch using the AdamW optimizer.
    - **Design Motivation**: The parameter size is extremely small (4,000 times fewer than LoRA rank=4), but it directly targets and regulates the influence of attention heads, representing the most "precise" intervention.

3. **SEAL-C (Channel-level Scaling)**:
    - **Function**: Learn a scaling factor $s_{l,h,c} \in \mathbb{R}^{d_h}$ for each hidden layer channel under every attention head's output.
    - **Mechanism**: A more fine-grained approach—scaling is applied individually to the 128 channel dimensions inside each head. Experiments show that different channels inside the same head also have vastly different impacts (e.g., in the L1H18 head, only the 94th channel contributed to a 12% performance gain, while other channels even had negative impacts).
    - **Design Motivation**: Head-level granularity is too coarse, as different channels within the same head may serve different functions. Channel-level scaling provides finer control, is usually complementary to SEAL-H, and yields superior performance.

### Offline Merging (Zero Inference Overhead)

The learned scaling factors can be merged offline into the weight matrices of `v_proj` (along the output channel dimension) or `o_proj` (along the input channel dimension for GQA models). After merging, the model architecture remains unchanged, requiring zero additional computational overhead during inference—a major practical advantage of SEAL over methods like LoRA.

### Format-Aware Data Synthesis

SEAL is independent of the actual semantic content of the data and focuses only on the format/representation of the target task. For instance, for the Needle-in-a-Haystack task, an LLM is used to generate 50 synthetic samples with the same format but random content. This avoids data contamination while ensuring that training signals are aligned with the target task format.

## Key Experimental Results

### Main Results: Line Retrieval (LongEval)

| Model | Method | 9K | 14K | 19K | 23K | 28K | 31K |
|------|------|-----|-----|-----|-----|-----|-----|
| LongChat-7B-32K | Baseline | 0.98 | 0.96 | 0.84 | 0.54 | 0.38 | **0.32** |
| LongChat-7B-32K | SEAL-H | 1.00 | 1.00 | 0.98 | 1.00 | 0.94 | **0.80** |
| LongChat-7B-32K | SEAL-C | 0.98 | 0.96 | 0.94 | 0.92 | 0.94 | **0.88** |
| Mistral-7B-v0.2 | Baseline | 0.98 | 1.00 | 0.90 | 0.86 | 0.88 | 0.94 |
| Mistral-7B-v0.2 | SEAL-C | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.98** |
| Vicuna-13B-16K | Baseline | 0.98 | 0.98 | 0.94 | 0.88 | 0.68 | 0.42 |
| Vicuna-13B-16K | SEAL-C | 1.00 | 1.00 | 0.96 | 0.98 | 0.98 | **0.94** |

### RULER Benchmark (Llama-3.1-8B-Instruct)

| Task | Method | 4K | 8K | 16K | 32K | 64K |
|------|------|-----|-----|------|------|------|
| CWE | Baseline | 99.5 | 94.3 | 53.9 | 2.6 | 0.1 |
| CWE | SEAL-H | 100 | 99.8 | 98.7 | 95.8 | 21.8 |
| CWE | SEAL-C | 100 | 99.6 | 99.5 | **99.7** | **95.7** |
| VT | Baseline | 99.3 | 98.8 | 99.3 | 97.7 | 94.3 |
| VT | SEAL-C | 100 | 100 | 100 | 99.4 | **99.1** |

### Comparison with LoRA/DoRA

| Method | Learnable Parameters | 31K Accuracy (LongChat-7B) |
|------|------------|------------------------|
| SEAL-H | **1,024** | 0.80 |
| SEAL-C | ~131K | **0.88** |
| LoRA (r=4, QKVO) | ~4M | 0.80 |
| DoRA (r=4) | ~4M | 0.86 |

### Key Findings

- SEAL-C generally outperforms SEAL-H, validating the importance of fine-grained channel-level control.
- SEAL has almost no impact on general MMLU performance (42.53 $\to$ 42.17), indicating that it only adjusts retrieval-related attention heads rather than global capabilities.
- Extremely efficient as it requires only 50 synthetic samples and less than 1 hour of training.
- Combining SEAL with training-free context extension methods (NTK, Self-Extend) yields significant performance improvements—whereas NTK fails completely beyond 12K, adding SEAL restores performance to near short-sequence levels.

## Highlights & Insights

- **Highly insightful key discovery**: Identifying the positive and negative functional division of attention heads for long-context retrieval is deeply insightful and thoroughly validated. This suggests that LLM context degradation is not a "global collapse" but is driven by "a subset of lagging components." This implies that many LLM capability issues could be localized and repaired using similar component-level analyses.
- **Extreme parameter efficiency**: The entire 7B model requires only 1,024 parameters for SEAL-H, which is 4,000 times fewer than LoRA (r=4) while remaining competitive in performance. This design ethos of "no modification to model architecture and no extra inference computation" is highly practical and generalizable to other scenarios requiring targeted fine-tuning of specific capabilities.
- **Offline merging strategy**: Merging scaling factors offline into adjacent linear layer weights eliminates inference overhead, presenting an elegant engineering design. Compared to the extra forward pass required by adapters, this solution is much more deployment-friendly.
- **Cross-task transferability analysis**: SEAL scaling factors are found to be transferable between tasks of similar formats (e.g., CWE $\leftrightarrow$ FWE), implying that the functional division of attention heads possesses format-specific characteristics rather than being completely task-specific.

## Limitations & Future Work

- **Task specificity**: SEAL requires training scaling factors separately for each target task format, offering limited cross-format transferability. Future work could explore learning universal scaling factors under mixed training of various formats.
- **GPU memory during long-sample training**: GPU memory demands increase when training sample lengths exceed 31K, potentially requiring multi-GPU setups. Training costs will rise for scaling to 128K+ context models.
- **Sensitivity to learning rate**: The only hyperparameter in SEAL is the learning rate, but its optimal value varies across tasks, incurring tuning costs. Nonetheless, since synthetic data is used with only 50 samples, the tuning overhead remains acceptable.
- **Lack of validation on generative tasks**: Experiments heavily focus on rule-based retrieval tasks (numerical matching, keyword extraction), leaving complex real-world generative tasks like QA/summarization under-validated (only briefly discussed in the Appendix for LongBench QA).
- **Missing theoretical explanation**: Why do certain heads suppress long-context retrieval? Is it due to training data distribution or an inherent architectural property? A deep theoretical analysis is currently lacking.

## Related Work & Insights

- **vs LoRA/DoRA**: LoRA performs low-rank decomposition across all QKVO layers in attention modules, which offers a larger learning space but with higher parameter counts (4M vs 1K). SEAL designs a minimal intervention point—scaling only the attention outputs—achieving comparable performance.
- **vs NTK / Self-Extend**: These training-free context window expansion methods only solve the "window size" issue, but performance degradation within the window remains unaddressed. SEAL is complementary to them—extending the window first, then using SEAL to boost retrieval quality within the extended window.
- **vs Lost-in-the-Middle**: That study reveals the tendency of LLMs to forget information in the middle of long documents. SEAL provides a diagnostic and corrective approach from the perspective of attention head scaling, though they target different mechanisms.
- **Serves as a great baseline for long-context enhancement**, and extending the SEAL methodology to scenarios like KV cache compression / sparse attention is highly worth exploring.

## Rating

- Novelty: ⭐⭐⭐⭐ The discovery of positive/negative functional division among attention heads for retrieval is insightful, though the scaling factor design itself is quite direct.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 6 models, 3 benchmarks, comparisons with LoRA, transferability analyses, and compound experiments with context extension methods.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from motivation to methodology, with well-structured experiments.
- Value: ⭐⭐⭐⭐ Practical and lightweight, contributing both to the understanding of attention head functionality and to long-context enhancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](../../ICML2025/llm_efficiency/efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ACL 2025\] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](ladm_long_context_data.md)

</div>

<!-- RELATED:END -->
