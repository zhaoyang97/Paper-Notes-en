---
title: >-
  [Paper Note] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning
description: >-
   This paper proposes CL4D, a contrastive learning framework that adapts pretrained decoder-only code generation models to code understanding tasks (code search, clone detection) via continued pretraining, achieving performance comparable to or better than encoder-only models of equivalent scale without retraining them from scratch.
tags:

date: 2026-05-08
content_hash: 46ffb375aa006881
---

# Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning

## Metadata
- **Title**: Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning
- **Authors**: Jiayi Lin, Yanlin Wang, Yibiao Yang, Lei Zhang, Yutao Xie
- **Conference**: AAAI 2026
- **arXiv**: [2406.12326](https://arxiv.org/abs/2406.12326)
- **Code**: [GitHub](https://github.com/JiayiLin1024/CL4D)
- **Area**: Code Understanding, Contrastive Learning, Self-Supervised Learning

## TL;DR

This paper proposes CL4D, a contrastive learning framework that adapts pretrained decoder-only code generation models to code understanding tasks (code search, clone detection) via continued pretraining, achieving performance comparable to or better than encoder-only models of equivalent scale without retraining them from scratch.

## Background & Motivation

### State of the Field
- Large-scale decoder-only code generation models (e.g., StarCoder, CodeLlama, DeepSeek-Coder) have achieved remarkable success in code generation, yet perform poorly on code understanding tasks such as code search and clone detection.
- This gap is primarily attributed to the autoregressive training objective (next-token prediction), which emphasizes generation over semantic understanding.
- Encoder-only models (e.g., CodeBERT, UniXcoder, CodeSage) excel at understanding tasks but are far smaller than state-of-the-art decoder-only models.

### Root Cause
- Decoder-only models benefit from larger scale and richer training data, but their unidirectional attention mechanism limits fine-grained code understanding.
- Pretraining encoder-only models of equivalent scale from scratch is computationally prohibitive (the largest existing CodeSage is only 1.3B; CoLSBERT is only 1.5B).
- Core question: **Can the code understanding capability of existing decoder-only models be enhanced without retraining them from scratch?**

### Design Motivation
- Leverage the rich code knowledge already encoded in existing decoder-only models and transfer it via contrastive learning.
- Explore the feasibility of unifying code generation and code understanding within a single decoder-only architecture serving both task types.

## Method

### Overall Architecture: CL4D (Contrastive Learning for Decoder-only)
CL4D is a contrastive learning framework that performs continued pretraining on pretrained decoder-only code generation models to improve their representation quality.

### 1. Data Construction
- Six programming languages (Python, Java, Go, PHP, JavaScript, Ruby) are extracted from The Stack dataset.
- Tree-Sitter is used to extract bimodal data and construct (query, code) pairs, where the query is the first line of a function's docstring.
- CodeSearchNet filtering rules are applied to improve data quality.
- Millions of training samples are constructed for continued pretraining.

### 2. Model Architecture
Two strategies for extracting code representations from decoder-only models are explored:
- **Last Token**: The embedding of the last token in the final layer is used as the code representation (due to unidirectional attention, only the last token aggregates all preceding context).
- **Average**: The mean of all token embeddings in the final layer is used.

A dual-encoder architecture is adopted, where two weight-sharing Transformer decoders separately encode the query and the code.

### 3. Contrastive Learning Objective
- **In-batch Negatives**: Code snippets from other samples in the same batch serve as negative examples.
- **Hard Negatives**: UniXcoder is used to retrieve code snippets that are close in representation space but semantically different from each query, drawn from the full codebase.
- The loss function takes the InfoNCE form with temperature $\tau = 0.05$; cosine similarity is used to compute relevance scores.

### 4. Representation Extraction Strategy
Ablation experiments reveal that:
- **Right padding + average pooling over all token embeddings** is the optimal representation extraction strategy.
- With left padding, the last-token strategy performs better; with right padding, average pooling performs better.

### Training Details
- Optimizer: AdamW, learning rate $2 \times 10^{-5}$
- Trained for 2 epochs with batch size 64
- 8× A100 (80G) GPUs; maximum training time approximately 3 days (phi-1)

## Key Experimental Results

### Experimental Setup
- **Code Search**: CodeSearchNet (CSN, 6 languages) and CoSQA datasets, metric: MRR
- **Clone Detection**: POJ-104 dataset, metric: MAP
- Baselines: encoder-only models (CodeBERT / GraphCodeBERT / UniXcoder / CodeSage) and decoder-only models (CodeGPT / CodeGen / SantaCoder / phi-1 / DeepSeek-Coder)

### Table 1: Overall Fine-tuned Performance

| Method | Scale | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|--------|-------|-----------|-------------|----------------|
| CodeBERT (Enc) | 125M | 70.18 | 65.7 | 83.79 |
| GraphCodeBERT (Enc) | 125M | 72.08 | 68.4 | 85.50 |
| UniXcoder (Enc) | 125M | 74.40 | 70.1 | 89.56 |
| CodeSage (Enc) | 1.3B | 75.80 | 68.0 | 87.70 |
| CodeGPT + CL4D (Dec) | 125M | 70.20 | 69.0 | 87.96 |
| CodeGen + CL4D (Dec) | 350M | 73.30 | 71.5 | 89.68 |
| SantaCoder + CL4D (Dec) | 1.1B | 74.98 | 72.2 | 83.98 |
| phi-1 + CL4D (Dec) | 1.3B | 75.18 | 72.8 | 92.72 |
| **DeepSeek-Coder + CL4D (Dec)** | **1.3B** | **77.57** | **71.9** | **89.71** |

**Key Findings**: CL4D enables decoder-only models to outperform encoder-only models of equivalent scale by approximately 2% on most tasks; larger model scale consistently yields better understanding performance.

### Table 2: Zero-Shot Performance (No Fine-Tuning)

| Method | Scale | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|--------|-------|-----------|-------------|----------------|
| CodeBERT (Enc) | 125M | 0.10 | 0.24 | 20.38 |
| UniXcoder (Enc) | 125M | 46.40 | 42.11 | 42.08 |
| CodeSage (Enc) | 1.3B | 71.24 | 47.53 | 73.07 |
| CodeGPT (Dec) | 125M | 0.12 | 0.04 | 9.41 |
| DeepSeek-Coder (Dec) | 1.3B | 0.12 | 0.63 | 16.51 |
| CodeGPT + CL4D (Dec) | 125M | 67.56 (↑67.44) | 53.49 (↑53.45) | 25.93 (↑16.52) |
| CodeGen + CL4D (Dec) | 350M | 71.97 (↑70.55) | 51.18 (↑50.73) | 45.84 (↑32.64) |
| SantaCoder + CL4D (Dec) | 1.1B | 74.18 (↑74.11) | 52.82 (↑52.71) | 71.14 (↑55.57) |
| **DeepSeek-Coder + CL4D (Dec)** | **1.3B** | **76.02 (↑75.90)** | **48.34 (↑47.71)** | **71.18 (↑54.67)** |

**Key Findings**: CL4D improves decoder-only model performance by 40%–76% in the zero-shot setting, enabling it to match encoder-only models without any task-specific fine-tuning.

### Ablation Study
- Removing hard negatives leads to approximately 1.5% performance degradation.
- Removing in-batch negatives causes a drastic performance drop (CSN MRR from 72.00 to 1.42), confirming that contrastive learning is the core of the method.

## Highlights & Insights

1. **High Practicality**: The approach directly reuses existing decoder-only models' code knowledge without training large encoder models from scratch, substantially reducing computational cost.
2. **Unified Architecture Potential**: The work demonstrates that a decoder-only architecture can simultaneously handle both code generation and code understanding, pointing toward a unified code model paradigm.
3. **Significant Zero-Shot Gains**: CL4D achieves zero-shot improvements of up to 75.90%, matching encoder-only SOTA without any fine-tuning.
4. **Clear Scaling Effect**: Experiments clearly demonstrate that larger decoder-only models yield consistently better code understanding performance.
5. **Systematic Exploration**: A complete ablation analysis over representation extraction strategies (padding direction × pooling method) is conducted.

## Limitations & Future Work

1. **Limited Evaluation Tasks**: Only code search and clone detection are evaluated; other understanding tasks such as code summarization, bug detection, and type inference are not covered.
2. **Constrained Model Scale**: The largest model in experiments is only 1.3B; the effectiveness on larger-scale decoder-only models (e.g., 7B, 13B) remains unverified.
3. **Dependency on External Model for Hard Negatives**: Constructing hard negatives requires UniXcoder as an auxiliary ranker, introducing an additional external dependency.
4. **No Evaluation of Generation Capability**: The impact of CL4D continued pretraining on the original model's code generation ability (e.g., potential catastrophic forgetting) is not analyzed.
5. **Mainstream Language Bias**: Only 6 mainstream programming languages are covered; generalization to low-resource languages is not validated.

## Related Work & Insights

- **Code Representation Learning**: Encoder-only models such as CodeBERT, GraphCodeBERT (data flow), TreeBERT (AST), UniXcoder, CodeSage, and CoLSBERT learn code representations via objectives like MLM.
- **Code Contrastive Learning**: CoSQA (query rewriting for positive pairs), SynCoBERT/Code-MVP (cross-modal positive pairs), UniXcoder (dropout-based positive pairs), CodeRetriever/R2/CodeSage (hard negative construction).
- **Decoder-Only Code Models**: Codex, CodeGen, StarCoder, CodeLlama, DeepSeek-Coder, etc., continuously growing in scale but primarily targeting generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐ — The intuition is sound but technical novelty is limited; the work essentially applies SimCSE-style contrastive learning to decoder-only code models.
- **Value**: ⭐⭐⭐⭐ — The method is simple and efficient, requires low training cost, and can directly reuse existing models with low engineering overhead.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons across multiple models, datasets, and settings (fine-tuned / zero-shot), with complete ablation and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-defined research problem, and well-organized experiments.
- **Overall**: ⭐⭐⭐⭐ (7.5/10) — High practical value and thorough experiments, but limited technical innovation; the core contribution lies in systematically demonstrating that contrastive learning can effectively bridge the gap of decoder-only models in code understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)
- [\[AAAI 2026\] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)

</div>

<!-- RELATED:END -->
