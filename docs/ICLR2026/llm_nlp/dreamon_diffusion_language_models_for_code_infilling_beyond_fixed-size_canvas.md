---
title: >-
  [Paper Note] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas
description: >-
  [ICLR 2026][LLM/NLP][diffusion language model] DreamOn introduces two special states, [expand] and [delete], to overcome the fixed-length generation constraint of diffusion language models (DLMs)…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "diffusion language model"
  - "code infilling"
  - "variable-length generation"
  - "discrete diffusion"
  - "DLM"
date: 2026-05-08
content_hash: 9348474a7f8eaf54
---

# DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas

**Conference**: ICLR 2026  
**arXiv**: [2602.01326](https://arxiv.org/abs/2602.01326)  
**Code**: [https://github.com/DreamLM/DreamOn](https://github.com/DreamLM/DreamOn)  
**Area**: LLM/NLP  
**Keywords**: diffusion language model, code infilling, variable-length generation, discrete diffusion, DLM

## TL;DR
DreamOn introduces two special states, [expand] and [delete], to overcome the fixed-length generation constraint of diffusion language models (DLMs), enabling variable-length code infilling without any architectural modification. It achieves an average improvement of 26.4% over diffusion baselines on HumanEval-Infilling, reaching performance on par with state-of-the-art autoregressive models.

## Background & Motivation

**Background**: Diffusion language models (DLMs, e.g., LLaDA, Dream, DiffuCoder) realize flexible, arbitrary-order generation through iterative denoising, making them naturally suited for code infilling—generating missing code between a given prefix and suffix. Autoregressive models require the cumbersome Fill-in-the-Middle (FIM) hack for infilling, which disrupts the natural context structure.

**Limitations of Prior Work**: The critical bottleneck of DLMs is that **the length of the masked sequence must be specified in advance**. Inputs and outputs must share the same length, preventing the model from dynamically determining generation length. When the preset mask length does not match the true completion length: too few masks lead to incomplete completions; too many masks produce spurious erroneous code. Empirically, average performance degrades by 38%.

**Key Challenge**: The bidirectional attention of DLMs is inherently well-suited for infilling, yet the fixed-length constraint completely negates this advantage. The key question is how to enable DLMs to dynamically adjust output length while keeping the architecture unchanged.

**Goal**: Enable DLMs to autonomously decide whether to expand or contract the sequence length during generation.

**Key Insight**: Introduce two special states into the diffusion process as length-control signals—predicting [expand] indicates "more space is needed here," while predicting [delete] indicates "this position is redundant."

**Core Idea**: Encode length control as two special tokens in the diffusion vocabulary ([expand] → split into two [mask] tokens; [delete] → remove the position). The model is trained to predict these states via data augmentation, achieving variable-length generation with zero architectural changes.

## Method

### Overall Architecture
Building on standard masked diffusion, DreamOn introduces an augmented diffusion process: during training, the original sequence $\mathbf{x}_0$ is transformed into an augmented sequence $\mathbf{z}_0$ containing [expand] and [delete] tokens via span merging and delete insertion, followed by standard masked diffusion training. During inference, the model autonomously predicts [expand] (expanding the current position into two [mask] tokens) or [delete] (removing the current position) during denoising, thereby dynamically adjusting the output length.

### Key Designs

1. **[expand] and [delete] Special States**:

    - Function: Additional tokens in the diffusion vocabulary that encode length-control semantics.
    - Core Rule: Predicting [expand] → replace the position with two [mask] tokens (sequence grows); predicting [delete] → remove the position (sequence shrinks).
    - Design Motivation: These two operations form a Turing-complete set of length-adjustment primitives—any target length can be reached from any source length through repeated expand and delete operations.

2. **Augmented Data Construction (Training)**:

    - Function: Construct augmented sequences $\mathbf{z}_0$ containing [expand] and [delete] from the original sequence $\mathbf{x}_0$.
    - Mechanism: [expand] tokens are produced via span merging—consecutive [mask] tokens are merged into a single [expand]; [delete] tokens are produced by appending 0–64 random delete tokens at the end.
    - Design Motivation: A merge schedule controls the proportion of [expand] tokens (a 1:1 mixture of static and dynamic inverse schedulers), preventing an excessive number of special tokens from degrading standard performance.

3. **Weighted Training Loss**:

    - Function: Balance the disproportionate contribution of [delete] tokens.
    - Mechanism: Each [delete] token corresponds to one [mask], while multiple [mask] tokens are merged into a single [expand], causing [delete] to be over-represented in the loss. A weight $w_n$ is introduced so that the total contribution of [delete] tokens is equivalent to that of a single [mask].
    - Design Motivation: Calibrate the loss distribution to prevent the model from over-learning deletion at the expense of expansion.

4. **Broadcasting Deletion (Inference Optimization)**:

    - Function: Upon detecting [delete], simultaneously remove all consecutive [mask] tokens to its right.
    - Design Motivation: Avoids the need for the model to delete positions one by one over many forward passes, significantly accelerating inference.

### Loss & Training
- Weighted cross-entropy loss from standard masked diffusion, with the vocabulary extended to include [expand] and [delete].
- Fine-tuned from DreamCoder-7B/DiffuCoder-7B on only 110K Python code pairs for 10 epochs; approximately 5 hours on 8×H800 GPUs.
- **Training compute is only 0.15% of pretraining cost**—extremely lightweight.

## Key Experimental Results

### Main Results

| Model | HE-Single (Pass@1) | HE-Multi (Pass@1) | SantaCoder (EM) |
|------|--------------------|--------------------|-----------------|
| Qwen2.5-Coder-7B (AR) | 92.6 | 58.7 | 79.8 |
| Seed-Coder-8B (AR) | 89.7 | 59.3 | 77.2 |
| DreamCoder-7B (DLM) | 55.5 | 43.2 | 59.3 |
| **DreamCoder + DreamOn** | **92.1** (+36.6) | **63.8** (+20.6) | **79.0** (+19.7) |
| DiffuCoder + DreamOn | 92.2 (+38.5) | 63.1 (+18.1) | 77.4 (+19.4) |

DreamOn enables DLMs to **surpass** state-of-the-art autoregressive models on multi-line infilling.

### Ablation Study

| Configuration | Single-line Avg | Multi-line Avg | Notes |
|------|----------------|----------------|------|
| DreamCoder-7B Baseline | 55.3 | 26.0 | Fixed mask=64 |
| + DreamOn (Full) | **90.8** | **57.1** | Both states enabled |
| w/o Delete | 67.4 | 39.2 | Cannot shorten; degrades with excess masks |
| w/o Expand | 73.4 | 43.2 | Cannot extend; degrades with too few masks |
| Oracle (True Length) | 91.6 | 69.0 | Upper bound reference |

### Key Findings
- **DreamOn nearly reaches Oracle performance** (single-line: 90.8 vs. 91.6), demonstrating highly effective length adaptation.
- **Expand and Delete are complementary and both indispensable**: removing Delete causes severe degradation with long masks; removing Expand causes severe degradation with short masks.
- **Highly robust to initial mask length**: as the initial mask varies from 4 to 64, DreamOn performance remains nearly constant (88.7–92.1), whereas the baseline fluctuates from 24.9 to 55.5.
- Fine-tuning at only **0.15% of pretraining compute** yields substantial gains, demonstrating the method's lightweight nature.
- Broadcasting Deletion significantly reduces the number of inference steps without sacrificing accuracy.

## Highlights & Insights
- **Elegant minimalist design**: only two special tokens are added to the vocabulary alongside data augmentation, with zero architectural changes. The approach of encoding structural operations as vocabulary operations is particularly ingenious.
- **DLMs reach AR-level infilling performance for the first time**: demonstrating that the inherent infilling advantage of DLMs is fully unlocked once the length constraint is resolved.
- **Transferability**: DreamOn is effective across three different DLMs—Dream-7B, DiffuCoder-7B, and DreamCoder-7B—indicating that it is a model-agnostic enhancement.
- **Insight behind Broadcasting Deletion**: if the model predicts delete at a position, the subsequent all-mask region is most likely also redundant—this simple heuristic substantially accelerates inference.

## Limitations & Future Work
- Each [expand] can only expand into 2 [mask] tokens; large-scale length expansion requires multiple forward passes.
- Validation is currently limited to code infilling; effectiveness on natural language infilling (e.g., text editing, story continuation) remains unexplored.
- The maximum expansion length $L_{max}=128$ is an artificial constraint, which may limit performance on longer code infilling tasks.
- Training data consists of only 110K Python code samples; generalization to multilingual code infilling has not been verified.

## Related Work & Insights
- **vs. Autoregressive FIM (Qwen2.5-Coder, etc.)**: FIM requires reordering sequences during training (moving the middle segment to the end) and special prompts at inference time. DLMs with DreamOn naturally support bidirectional-context infilling without such workarounds.
- **vs. Fixed-length DLMs (LLaDA, Dream)**: DreamOn addresses the most critical practical bottleneck of DLMs, making them competitive on infilling tasks for the first time.
- **vs. Edit-based Methods**: Some approaches perform text editing through explicit insert/delete operations; DreamOn integrates these operations into the diffusion process in a more elegant and end-to-end trainable manner.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Resolving a fundamental limitation of DLMs with just two special tokens is a concise and powerful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three DLM baselines, two benchmarks, and detailed ablations, though evaluation is limited to code tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is illustrated intuitively, algorithmic descriptions are clear, and ablation analyses are thorough.
- Value: ⭐⭐⭐⭐⭐ — Resolves a fundamental bottleneck in the DLM field, enabling DLMs to compete with autoregressive models on infilling tasks for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](../../NeurIPS2025/llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](../../ICCV2025/llm_nlp/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)

</div>

<!-- RELATED:END -->
