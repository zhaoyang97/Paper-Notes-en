---
title: >-
  [Paper Note] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs
description: >-
  [ACL 2026][Information Retrieval & RAG][Code Prompt Compression] CodePromptZip is proposed as the first code-oriented prompt compression framework. It constructs training data through type-aware priority ranking and trai…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Code Prompt Compression"
  - "RAG"
  - "Type-aware Priority"
  - "Copy Mechanism"
  - "Coding Tasks"
date: 2026-05-08
content_hash: 684b5fff7f702d09
---

# CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2502.14925](https://arxiv.org/abs/2502.14925)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Code Prompt Compression, RAG, Type-aware Priority, Copy Mechanism, Coding Tasks

## TL;DR

CodePromptZip is proposed as the first code-oriented prompt compression framework. It constructs training data through type-aware priority ranking and trains a small model compressor with a copy mechanism, achieving gains of 23.4%, 28.7%, and 8.7% over the best baselines across three coding tasks.

## Background & Motivation

**Background**: RAG enhances LLM performance in coding tasks by retrieving relevant code examples. However, retrieved code often spans tens of thousands of tokens, limited by LLM context windows and API costs.

**Limitations of Prior Work**: Existing prompt compression techniques (LLMLingua, RECOMP, etc.) are designed for natural language, ignoring the unique characteristics of code—where different token types (e.g., Identifier, Symbol, Invocation) impact generation quality with varying degrees of significance.

**Key Challenge**: Natural language compression methods use heuristic information entropy or knowledge distillation to judge token importance, but these metrics fail to consider the structural information of code types, leading to sub-optimal compression.

**Goal**: Design the first code-specific prompt compression framework capable of retaining maximum information useful for downstream tasks under a specified compression ratio.

**Key Insight**: Utilize program analysis to categorize code tokens by type. Establish type-level removal priorities through ablation analysis to guide training data construction and compression model learning.

**Core Idea**: Different types of code tokens have varying impacts on tasks. Tokens are removed according to a priority sequence from least to most impactful, and a CodeT5 model with a copy mechanism is trained to learn this compression strategy.

## Method

### Overall Architecture

The framework consists of training and inference phases. Training phase: (1) Type-aware priority ranking—using JavaParser for AST analysis to ablate token types and establish removal priorities based on task performance; (2) Greedy algorithm to construct compression training samples based on priority; (3) Training a copy-enhanced CodeT5 compressor. Inference phase: The compressor accepts raw code and a target compression ratio, outputting compressed code to be embedded into the RAG prompt.

### Key Designs

1.  **Type-aware Priority Ranking**:
    - **Function**: Determines the removal priority for different types of code tokens.
    - **Mechanism**: Categorizes tokens into five types: Symbol, Signature, Invocation, Identifier, and Structure. After individual ablation, it measures $Priority(T) = \text{Compression Rate} / \text{Performance Degradation}$. Higher priority types are removed first.
    - **Design Motivation**: Observed that priority hierarchies are consistent across models but specific to tasks (e.g., Invocation has the highest priority in Assertion Generation but the lowest in Code Suggestion).

2.  **Copy-enhanced CodeT5 Compressor**:
    - **Function**: Learns to generate compressed code at a specified compression ratio.
    - **Mechanism**: Integrates a copy module into the CodeT5 encoder-decoder architecture—calculating $p_{gen}$ to decide whether to generate from the vocabulary or copy from the source sequence, with the final distribution as $P(y) = p_{gen} \cdot P_{vocab} + (1 - p_{gen}) \cdot P_{copy}$.
    - **Design Motivation**: Code compression is essentially an extractive task; a copy mechanism fits naturally and can handle un-parsable code fragments.

3.  **Flexible Compression Rate Control**:
    - **Function**: Supports user-defined target compression ratios.
    - **Mechanism**: Extends the vocabulary with special tokens like `<Ratio>` to explicitly encode the target compression ratio in the input, allowing the model to adaptively learn different compression levels.
    - **Design Motivation**: Different scenarios require different trade-offs between cost and quality, necessitating a controllable compressor rather than a fixed one.

### Loss & Training

A cross-entropy loss is used for training with the AdamW optimizer. Hyperparameters include $batch=16$, $lr=5e-5$, $warmup=1000$ steps, for 10 epochs. Training data is automatically constructed using Algorithm 1 (priority-driven greedy algorithm) across various compression ratios.

## Key Experimental Results

### Main Results

| Method | Assertion (EM%) | Bugs2Fix (CB%) | Code Suggestion (CB%) |
| :--- | :--- | :--- | :--- |
| w/o retrieval | 23.9 | 41.7 | 14.2 |
| LLMLingua | 33.8 | 41.9 | 21.8 |
| LongLLMLingua | 34.1 | 42.1 | 21.2 |
| LLMLingua-2 | 21.2 | 48.1 | 21.7 |
| RECOMP | 23.4 | 45.3 | 21.0 |
| CodePromptZip (w/o Copy) | 40.9 | 56.7 | 20.5 |
| **CodePromptZip** | **42.1** | **61.9** | **23.7** |
| Oracle (AST) | 46.2 | 66.8 | 23.8 |
| w/o compression | 50.5 | 81.4 | 24.7 |

*$\tau_{code}=0.3$, 1-shot, using GPT-3.5-turbo*

### Ablation Study

| Component | Assertion (EM%) | Bugs2Fix (CB%) | Code Suggestion (CB%) |
| :--- | :--- | :--- | :--- |
| CodePromptZip w/o Copy | 40.9 | 56.7 | 20.5 |
| CodePromptZip (full) | 42.1 (+1.2) | 61.9 (+5.2) | 23.7 (+3.2) |

**Compression Rate Control**: The actual compression ratio of CodePromptZip aligns closely with the specified ratio, whereas the version without a copy mechanism shows significantly worse control.

### Key Findings

- Performance improved over the best baseline by 23.4% (42.1 vs 34.1), 28.7% (61.9 vs 48.1), and 8.7% (23.7 vs 21.8) across three tasks.
- The copy mechanism is crucial for compression ratio control and yields performance gains across all tasks.
- Trade-off analysis shows: Under a fixed token budget, using fewer examples with low compression ratios is superior to more examples with high compression ratios.
- Cross-model generalization: Outperformed all baselines on CodeLlama-13B and Gemini-1.0-Pro.
- Un-parsable code: Removing the final 1-3% of tokens resulted in only minor performance drops (42.1% → 42.0%/41.7%), proving the robustness of the learning-based method.

## Highlights & Insights

- First to propose a code-specific prompt compression problem and solution, filling the gap between NL and code compression.
- The discovery of type-aware priority ranking is insightful: removal priorities are consistent across models but vary by task, suggesting code token importance is task-driven rather than model-driven.
- The learning-based approach elegantly addresses the limitations of Oracle methods (requiring AST parsing) which cannot handle incomplete code.
- Dynamic compression ratio control allows the framework to adapt to different cost/quality trade-off requirements.

## Limitations & Future Work

- Currently only supports Java (relies on JavaParser for building training data); generalization to other languages is not yet verified.
- The compressor is based on CodeT5-775M; model size and inference latency require consideration.
- Priority ranking in the ablation analysis needs to be repeated for every new task; automated/adaptive priority discovery is a future direction.
- Multilingual code mixture scenarios have not been considered.

## Related Work & Insights

- Direction for improvement over the LLMLingua series is clear: information entropy metrics are unsuitable for code, which requires leveraging type-structure information.
- RECOMP's GPT-3.5 distillation is high-cost and lacks controllable compression; CodePromptZip’s priority-driven method is more efficient and controllable.
- Implications for RAG system optimization: Retrieval content of different modalities should use specialized compression strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ First code-specific prompt compression framework; type-aware priority ranking is a novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks, multiple baselines, cross-model generalization, and un-parsable code testing.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, comprehensive methodology, and distinct experimental conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ACL 2026\] Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers](code-switching_information_retrieval_benchmarks_analysis_and_the_limits_of_curre.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
