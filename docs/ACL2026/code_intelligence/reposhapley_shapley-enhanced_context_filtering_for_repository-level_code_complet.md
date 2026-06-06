---
title: >-
  [Paper Note] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion
description: >-
  [ACL 2026][Code Intelligence][Shapley Values] Ours proposes RepoShapley, a coalition-aware context filtering framework based on Shapley values. It decides whether to retain or discard retrieved code snippets by estimatin…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Shapley Values"
  - "Context Filtering"
  - "Repository-level Code Completion"
  - "Retrieval-Augmented Generation"
  - "Coalition Game"
date: 2026-05-08
content_hash: 60e979d95a151d6a
---

# RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion

**Conference**: ACL 2026  
**arXiv**: [2601.03378](https://arxiv.org/abs/2601.03378)  
**Code**: [github](https://github.com/yuhuo03/RepoShapley)  
**Area**: Information Retrieval / Code Completion  
**Keywords**: Shapley Values, Context Filtering, Repository-level Code Completion, Retrieval-Augmented Generation, Coalition Game

## TL;DR

Ours proposes RepoShapley, a coalition-aware context filtering framework based on Shapley values. It decides whether to retain or discard retrieved code snippets by estimating their interactive contributions in combinations, significantly improving the quality of repository-level code completion.

## Background & Motivation

**Background**: Repository-level code completion requires parsing cross-file dependencies (e.g., project APIs, shared contracts). Retrieval-Augmented Generation (RAG) enhances code LMs by injecting cross-file evidence.

**Limitations of Prior Work**: The utility of retrieved code snippets has interactive dependencies—some snippets are useless alone but become critical when paired with complementary context; others appear relevant but reduce generation quality when coexisting with conflicting evidence. Existing methods (e.g., CODEFILTER) score each snippet independently and fail to capture these combinatorial effects.

**Key Challenge**: Under a fixed context budget, there is a systematic bias between the utility of independently scored snippets and their actual utility when consumed in multi-snippet combinations.

**Goal**: Design a coalition-aware context filtering mechanism using Shapley marginal contribution signals to supervise snippet selection.

**Key Insight**: Context selection is modeled as a cooperative game—each retrieved snippet is a player, any subset is a coalition, and Shapley values quantify the average marginal contribution of a snippet across all possible combinations.

**Core Idea**: Approximate Shapley values via a lightweight surrogate game + select the optimal coalition via bounded post-verification, then distill the verification results into discrete control tokens for online inference.

## Method

### Overall Architecture

A two-stage pipeline: (1) Offline ChunkShapley annotation: single-snippet probing $\rightarrow$ surrogate proxy game $\rightarrow$ exact Shapley values $\rightarrow$ bounded post-verification to generate keep/drop labels; (2) Online RepoShapley inference: distill verification labels into control tokens (`<KEEP>`/`<DROP>`/`<NEED>`/`<DONE>`), enabling a single model to simultaneously perform retrieval triggering, snippet selection, and code generation.

### Key Designs

1. **Single-Snippet Probing and Surrogate Proxy Game**: Calculate a standalone teacher-forced log-likelihood gain $\Delta_i = \ell(X_{in}, \{cc_i\}) - \ell(X_{in})$ for each candidate snippet $cc_i$, obtaining sign $y_i = \text{sign}(\Delta_i)$ and weight $\omega_i = |\Delta_i|$. Define surrogate utility $v_{sur}(S) = \sigma(\beta \sum_{i \in S} \omega_i y_i) - \sigma(0)$, where the saturation of sigmoid naturally captures redundancy effects, and negative votes ($y_i=-1$) model conflicts.

2. **Exact Shapley Values and Post-Verification**: Since the surrogate utility $v_{sur}$ provides a closed-form solution, $2^K$ subsets can be exactly enumerated to calculate Shapley values on a retrieval set where $K \leq 10$. Construct a candidate pool $\mathcal{C}$ (Shapley prefix + $\Delta$ prefix + size-2/3 combinations of top-L snippets) and use a frozen generator to decode and select the ES/EM optimal coalition $S^\star$.

3. **Dual-Format Distillation Training**: Format-1 supervises selection (predicting keep/drop token sequences for each snippet); Format-2 supervises generation (FIM completion using only retained snippets). Both formats share parameters, allowing the model to learn selection and generation within a unified auto-regressive interface.

### Loss & Training

The training loss includes a retrieval control loss $\mathcal{L}_R$ (predicting `<NEED>`/`<DONE>`) and a snippet selection loss $\mathcal{L}_S$ (predicting keep/drop sequences), both using standard cross-entropy. During inference, a threshold $t_c$ determines whether to trigger retrieval.

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

RepoShapley outperforms all baseline methods (including No-Retrieve, Full-Retrieve, RepoFormer, CODEFILTER) across 11 evaluation metrics. Consistent advantages are also observed on StarCoder-Base-7B and CodeLlama-13B, proving the backbone-agnostic nature of the method.

### Key Findings

- Coalition-aware supervision improves performance by 4-5 percentage points compared to independent scoring.
- Shapley prefix selection outperforms simple $\Delta$ ordering, validating the importance of interactive effects.
- Retrieval trigger control effectively reduces unnecessary retrieval without performance loss.
- The $\beta$ parameter of the surrogate game controls the saturation scale; values that are too large or too small lead to degradation.

## Highlights & Insights

- **Paradigm Shift from Independence to Coalition**: Upgrading context filtering from "individual scoring" to "combinatorial games" represents a significant evolution in RAG control logic.
- **Clever Computation Design**: Using a lightweight surrogate game avoids exponential generator evaluations, while bounded post-verification ensures accuracy, balancing efficiency and effectiveness.
- **Distillation into Control Tokens**: Compressing offline combinatorial reasoning into online single-token prediction is an elegant engineering approach.

## Limitations & Future Work

- Retrieval set size is limited to $K \leq 10$; larger sets require sampling approximations.
- The sigmoid assumption in the surrogate game may not apply to certain code structures.
- Post-verification requires target sequences $Y$ and can only be used for offline annotation, not online updates.
- Future work could explore adaptive $\beta$ adjustment and richer interaction modeling.

## Related Work & Insights

- Extension of data valuation ideas from Data Shapley (Ghorbani & Zou, 2019) to RAG scenarios.
- Difference from SHAP: This work uses forward supervision (constructing training labels) rather than backward explanation.
- The control token distillation approach can be generalized to other scenarios requiring dynamic context selection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing Shapley values into RAG context control from a coalition game perspective is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough validation across multiple benchmarks and backbones, with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical formalization and well-explained motivation.
- **Value**: ⭐⭐⭐⭐⭐ Provides a systematic solution for context control in RAG scenarios with broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](../../ICML2026/code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](../../ICLR2026/code_intelligence/improving_code_localization_with_repository_memory.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)

</div>

<!-- RELATED:END -->
