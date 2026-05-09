---
title: >-
  [Paper Note] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs
description: >-
  [ACL 2026][code prompt compression] This paper proposes CodePromptZip, the first code-specific prompt compression framework, which constructs training data via type-aware priority ranking and trains a small-model compressor with a copy mechanism. It achieves improvements of 23.4%, 28.7%, and 8.7% over the best baseline across three coding tasks.
tags:
  - ACL 2026
  - code prompt compression
  - RAG
  - type-aware priority ranking
  - copy mechanism
  - coding tasks
date: 2026-05-08
content_hash: 8d3a4db5d169130b
---

# CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs

**Conference**: ACL 2026
**arXiv**: [2502.14925](https://arxiv.org/abs/2502.14925)
**Code**: None
**Area**: Information Retrieval
**Keywords**: code prompt compression, RAG, type-aware priority ranking, copy mechanism, coding tasks

## TL;DR

This paper proposes CodePromptZip, the first code-specific prompt compression framework, which constructs training data via type-aware priority ranking and trains a small-model compressor with a copy mechanism. It achieves improvements of 23.4%, 28.7%, and 8.7% over the best baseline across three coding tasks.

## Background & Motivation

**State of the Field**: RAG enhances LLM performance on coding tasks by retrieving relevant code examples, but retrieved code often spans tens of thousands of tokens, constrained by LLM context windows and API costs.

**Limitations of Prior Work**: Existing prompt compression techniques (LLMLingua, RECOMP, etc.) are designed for natural language and overlook the distinctive characteristics of code—different token types (e.g., Identifier, Symbol, Invocation) have vastly different impacts on generation quality.

**Root Cause**: Natural language compression methods rely on heuristic information entropy or knowledge distillation to assess token importance, but these metrics do not account for the type-structural information inherent in code, leading to suboptimal compression.

**Paper Goals**: Design the first code-specific prompt compression framework capable of maximally preserving task-relevant code information at a specified compression ratio.

**Starting Point**: Leveraging program analysis to categorize code tokens by type, establishing type-level removal priorities through ablation analysis, and using these priorities to guide training data construction and compressor learning.

**Core Idea**: Different types of code tokens have different impacts on downstream tasks. Tokens are removed in order of increasing impact (i.e., lowest-priority first), and a CodeT5 model augmented with a copy mechanism is trained to learn this compression strategy.

## Method

### Overall Architecture

The framework consists of two phases: training and inference. **Training phase**: (1) Type-aware priority ranking—JavaParser is used for AST analysis, the impact of each token type on task performance is ablated, and a removal priority order is established; (2) a greedy algorithm constructs compressed training samples according to priority; (3) a copy-enhanced CodeT5 compressor is trained. **Inference phase**: the compressor takes raw code and a target compression ratio as input and outputs compressed code to be embedded in the RAG prompt.

### Key Designs

1. **Type-aware Priority Ranking**:
    - Function: Determine the removal priority of different code token types.
    - Mechanism: Tokens are categorized into five types—Symbol, Signature, Invocation, Identifier, and Structure. Each type is ablated individually, and $\text{Priority}(T) = \text{compression ratio} / \text{performance degradation rate}$ is computed; types with higher priority are removed first.
    - Design Motivation: The priority hierarchy is observed to be consistent across models but task-specific (e.g., Invocation has the highest priority in Assertion Generation but the lowest in Code Suggestion).

2. **Copy-enhanced CodeT5 Compressor**:
    - Function: Learn to generate compressed code at a specified compression ratio.
    - Mechanism: A copy module is added to the CodeT5 encoder-decoder architecture, computing $p_{\text{gen}}$ to decide whether to generate from the vocabulary or copy from the source sequence. The final output distribution is $P(y) = p_{\text{gen}} \cdot P_{\text{vocab}} + (1 - p_{\text{gen}}) \cdot P_{\text{copy}}$.
    - Design Motivation: Code compression is fundamentally an extractive task (output is entirely derived from input), making the copy mechanism a natural fit; it also handles unparseable code fragments.

3. **Flexible Compression Ratio Control**:
    - Function: Support user-specified arbitrary target compression ratios.
    - Mechanism: The vocabulary is extended with special tokens such as `<Ratio>`, and the target compression ratio is explicitly encoded in the input, allowing the model to adaptively learn different compression levels.
    - Design Motivation: Different deployment scenarios require different compression ratios, necessitating a controllable compressor rather than one with a fixed compression rate.

### Loss & Training

The model is trained with cross-entropy loss using the AdamW optimizer, with batch size 16, learning rate $5 \times 10^{-5}$, 1000 warmup steps, and 10 training epochs. Training data at various compression ratios is automatically constructed by Algorithm 1 (priority-driven greedy algorithm).

## Key Experimental Results

### Main Results

| Method | Assertion (EM%) | Bugs2Fix (CB%) | Code Suggestion (CB%) |
|--------|-----------------|----------------|----------------------|
| w/o retrieval | 23.9 | 41.7 | 14.2 |
| LLMLingua | 33.8 | 41.9 | 21.8 |
| LongLLMLingua | 34.1 | 42.1 | 21.2 |
| LLMLingua-2 | 21.2 | 48.1 | 21.7 |
| RECOMP | 23.4 | 45.3 | 21.0 |
| CodePromptZip (w/o Copy) | 40.9 | 56.7 | 20.5 |
| **CodePromptZip** | **42.1** | **61.9** | **23.7** |
| Oracle (AST) | 46.2 | 66.8 | 23.8 |
| w/o compression | 50.5 | 81.4 | 24.7 |

*$\tau_{\text{code}}=0.3$, 1-shot, using GPT-3.5-turbo*

### Ablation Study

| Component | Assertion (EM%) | Bugs2Fix (CB%) | Code Suggestion (CB%) |
|-----------|-----------------|----------------|----------------------|
| CodePromptZip w/o Copy | 40.9 | 56.7 | 20.5 |
| CodePromptZip (full) | 42.1 (+1.2) | 61.9 (+5.2) | 23.7 (+3.2) |

**Compression ratio control**: CodePromptZip's actual compression ratio closely aligns with the specified target, whereas the variant without the copy mechanism exhibits significantly degraded control capability.

### Key Findings

- CodePromptZip outperforms the best baseline by 23.4% (42.1 vs. 34.1), 28.7% (61.9 vs. 48.1), and 8.7% (23.7 vs. 21.8) on the three tasks, respectively.
- The copy mechanism is critical for compression ratio control and consistently improves performance across all three tasks.
- Trade-off analysis shows that under a fixed token budget, using fewer examples with lower compression ratios outperforms using more examples with higher compression ratios.
- Cross-model generalization: CodePromptZip outperforms all baselines on CodeLlama-13B and Gemini-1.0-Pro as well.
- Unparseable code: removing the trailing 1–3% of tokens causes only a marginal performance drop (42.1% → 42.0%/41.7%), demonstrating the robustness of the learning-based approach.

## Highlights & Insights

- This work is the first to formulate and address code-specific prompt compression, bridging the gap between NL compression and code compression.
- The type-aware priority ranking finding is insightful: removal priorities are consistent across models but differ across tasks, indicating that code token importance is task-driven rather than model-driven.
- The learning-based approach elegantly resolves the limitation of the Oracle method (which requires AST parsing and cannot handle incomplete code).
- The controllable compression ratio design allows the framework to adapt to different cost/quality trade-off requirements.

## Limitations & Future Work

- Currently limited to Java code (relying on JavaParser for training data construction); generalization to other programming languages remains unverified.
- The compressor is based on CodeT5-775M; model size and inference latency warrant consideration.
- The priority ranking via ablation analysis must be redone for each new task; automated or adaptive priority discovery is a promising future direction.
- Multi-language code mixing scenarios are not considered.

## Related Work & Insights

- The improvement direction over the LLMLingua series is clear: information entropy metrics are ill-suited for code, and the type-structural information of code must be exploited.
- RECOMP's GPT-3.5 distillation approach is costly and offers no compression ratio control; CodePromptZip's priority-driven method is more efficient and controllable.
- Implication for RAG system optimization: retrieved content of different modalities should employ modality-specific compression strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ First code-specific prompt compression framework; type-aware priority ranking is a novel and well-motivated idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks, multiple baselines, cross-model generalization, and unparseable code evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, methodology is thoroughly described, and experimental conclusions are well-supported.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)

<!-- RELATED:END -->
