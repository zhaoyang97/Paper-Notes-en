---
title: >-
  [Paper Note] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion
description: >-
  [ACL 2026][Shapley value] This paper proposes RepoShapley, a coalition-aware context filtering framework based on Shapley values, which estimates the interactive contribution of retrieved code snippets in combination to determine whether each snippet should be retained or discarded, thereby significantly improving repository-level code completion quality.
tags:
  - ACL 2026
  - Shapley value
  - context filtering
  - repository-level code completion
  - retrieval-augmented generation
  - coalitional game
date: 2026-05-08
content_hash: 141f2d1805ba1595
---

# RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion

**Conference**: ACL 2026
**arXiv**: [2601.03378](https://arxiv.org/abs/2601.03378)
**Code**: [github](https://github.com/yuhuo03/RepoShapley)
**Area**: Information Retrieval / Code Completion
**Keywords**: Shapley value, context filtering, repository-level code completion, retrieval-augmented generation, coalitional game

## TL;DR

This paper proposes RepoShapley, a coalition-aware context filtering framework based on Shapley values, which estimates the interactive contribution of retrieved code snippets in combination to determine whether each snippet should be retained or discarded, thereby significantly improving repository-level code completion quality.

## Background & Motivation

**Background**: Repository-level code completion requires resolving cross-file dependencies (e.g., project APIs, shared contracts). Retrieval-augmented generation (RAG) enhances code language models by injecting cross-file evidence.

**Limitations of Prior Work**: The utility of retrieved code snippets exhibits interaction dependencies — certain snippets are individually unhelpful but become critical when paired with complementary context, while others appear relevant yet degrade generation quality when co-present with conflicting evidence. Existing methods (e.g., CODEFILTER) score each snippet independently and thus fail to capture such combinatorial effects.

**Key Challenge**: Under a fixed context budget, the utility of independently scored snippets systematically diverges from their actual utility when consumed in multi-snippet combinations.

**Goal**: Design a coalition-aware context filtering mechanism that supervises snippet selection using Shapley marginal contribution signals.

**Key Insight**: The context selection problem is modeled as a cooperative game — each retrieved snippet is treated as a player, any subset forms a coalition, and Shapley values quantify each snippet's average marginal contribution across all possible combinations.

**Core Idea**: Shapley values are approximated via a lightweight surrogate game, followed by bounded post-verification to select the optimal coalition; the verification results are then distilled into discrete control tokens to enable online inference.

## Method

### Overall Architecture

The framework consists of two stages: (1) **ChunkShapley offline annotation** — single-snippet probing → logistic surrogate game → exact Shapley values → bounded post-verification to generate keep/drop labels; (2) **RepoShapley online inference** — distilling the verification labels into control tokens (`<KEEP>`/`<DROP>`/`<NEED>`/`<DONE>`), allowing a single model to simultaneously perform retrieval triggering, snippet selection, and code generation.

### Key Designs

1. **Single-Snippet Probing and Logistic Surrogate Game**: For each candidate snippet $cc_i$, the individual teacher-forced log-likelihood gain is computed as $\Delta_i = \ell(X_{in}, \{cc_i\}) - \ell(X_{in})$, yielding a sign $y_i = \text{sign}(\Delta_i)$ and weight $\omega_i = |\Delta_i|$. The surrogate utility is defined as $v_{sur}(S) = \sigma(\beta \sum_{i \in S} \omega_i y_i) - \sigma(0)$, where the saturation property of the sigmoid naturally captures redundancy effects and negative votes ($y_i = -1$) model conflicting evidence.

2. **Exact Shapley Values and Post-Verification**: Since the surrogate utility $v_{sur}$ admits a closed-form expression, exact Shapley values can be computed by enumerating all $2^K$ subsets for retrieval sets of size $K \leq 10$. A candidate pool $\mathcal{C}$ is constructed (Shapley-ranked prefixes, $\Delta$-ranked prefixes, and size-2/3 combinations from top-$L$ snippets), and a frozen generator decodes and selects the ES/EM-optimal coalition $S^\star$.

3. **Dual-Format Distillation Training**: Format-1 supervises selection (predicting a keep/drop token sequence for each snippet); Format-2 supervises generation (FIM completion using only retained snippets). Both formats share parameters, enabling the model to learn selection and generation jointly within a unified autoregressive interface.

### Loss & Training

The training loss comprises a retrieval control loss $\mathcal{L}_R$ (predicting `<NEED>`/`<DONE>`) and a snippet selection loss $\mathcal{L}_S$ (predicting the keep/drop sequence), both using standard cross-entropy. At inference time, a threshold $t_c$ determines whether retrieval is triggered.

## Key Experimental Results

### Main Results

| Method | RepoEval Line EM | RepoEval API EM | CCLongEval Chunk ES | CCEval Line EM |
|---|---|---|---|---|
| No-Retrieve (SC-1B) | 43.14 | 38.03 | 47.29 | 18.72 |
| Full-Retrieve | 52.27 | 44.18 | 55.93 | 22.38 |
| RepoFormer | 54.71 | 45.73 | 57.69 | 25.42 |
| CODEFILTER | 57.19 | 48.37 | 59.91 | 27.81 |
| **RepoShapley** | **61.34** (+4.15) | **53.62** (+5.25) | **64.39** (+4.48) | **32.26** (+4.45) |

*Code completion performance on StarCoder-Base-1B*

### Ablation Study

RepoShapley consistently outperforms all baselines (No-Retrieve, Full-Retrieve, RepoFormer, CODEFILTER) across all 11 evaluation metrics. Consistent gains on StarCoder-Base-7B and CodeLlama-13B further demonstrate backbone-agnostic effectiveness.

### Key Findings

- Coalition-aware supervision outperforms independent scoring by 4–5 percentage points.
- Shapley-prefix selection surpasses ranking by $\Delta$ alone, validating the importance of interaction effects.
- Retrieval trigger control effectively reduces unnecessary retrievals without sacrificing performance.
- The $\beta$ parameter of the surrogate game controls the saturation scale; performance degrades when $\beta$ is either too large or too small.

## Highlights & Insights

- **Paradigm shift from independence to coalition**: Elevating context filtering from "per-snippet scoring" to "combinatorial game theory" represents a significant conceptual advance in RAG control.
- **Elegant design for computational feasibility**: A lightweight surrogate game avoids exponential generator evaluations, while bounded post-verification ensures precision — striking a strong balance between efficiency and effectiveness.
- **Distillation into control tokens**: Compressing offline combinatorial reasoning into online single-token prediction is an engineering-elegant design choice.

## Limitations & Future Work

- The retrieval set size is limited to $K \leq 10$; larger sets require sampling-based approximation.
- The sigmoid assumption of the surrogate game may not be appropriate for certain code structures.
- Post-verification requires the target sequence $Y$ and is therefore restricted to offline annotation; online updating is not supported.
- Future work may explore adaptive $\beta$ scheduling and richer interaction modeling.

## Related Work & Insights

- The framework extends the data valuation ideas of Data Shapley (Ghorbani & Zou, 2019) to the RAG setting.
- Unlike SHAP, which provides post-hoc explanations, the proposed method performs forward supervision to construct training labels.
- The control token distillation paradigm is generalizable to other scenarios requiring dynamic context selection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing Shapley values into RAG context control via a coalitional game perspective is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-benchmark and multi-backbone evaluation is thorough, with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical formalization is clear and method motivation is well articulated.
- **Value**: ⭐⭐⭐⭐⭐ Provides a systematic solution to context control in RAG with broad potential impact.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](../../AAAI2026/information_retrieval/prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)

<!-- RELATED:END -->
