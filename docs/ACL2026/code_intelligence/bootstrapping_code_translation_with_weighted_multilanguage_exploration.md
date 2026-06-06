---
title: >-
  [Paper Note] Bootstrapping Code Translation with Weighted Multilanguage Exploration
description: >-
  [ACL 2026][Code Intelligence][Code Translation] BootTrans proposes a bootstrapping multilingual code translation approach that leverages test cases from a single pivot language (Python) as a cross-lingual verification or…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Translation"
  - "Bootstrapping Exploration"
  - "Language-aware Weighting"
  - "RLVR"
  - "Multilingual Optimization"
date: 2026-05-08
content_hash: ea3f2d86ce1ac959
---

# Bootstrapping Code Translation with Weighted Multilanguage Exploration

**Conference**: ACL 2026  
**arXiv**: [2601.03512](https://arxiv.org/abs/2601.03512)  
**Code**: [https://github.com/nju-websoft/BootTrans/](https://github.com/nju-websoft/BootTrans/)  
**Area**: Code Translation/Reinforcement Learning  
**Keywords**: Code Translation, Bootstrapping Exploration, Language-aware Weighting, RLVR, Multilingual Optimization

## TL;DR

BootTrans proposes a bootstrapping multilingual code translation approach that leverages test cases from a single pivot language (Python) as a cross-lingual verification oracle. By combining a dual-pool architecture for experience collection to expand training data with a language-aware weighting mechanism to dynamically prioritize difficult translation directions, it significantly outperforms baselines on HumanEval-X and TransCoder-Test.

## Background & Motivation

**Background**: Code translation is crucial for legacy system modernization and cross-platform interoperability. LLMs have made significant progress in coding tasks, but code translation typically relies on high-quality parallel corpora, which are rarely equipped with executable test cases.

**Limitations of Prior Work**: (1) Multilingual parallel code data is scarce and rarely equipped with cross-lingual executable test cases; (2) Unsupervised methods (e.g., those utilizing code structural information) require massive monolingual corpora and cannot be directly optimized based on functional correctness; (3) Existing RLVR methods face two major challenges: **input monotony** (verifiable seeds are limited to a single pivot language) and **optimization imbalance** (skewed learning signals due to varying difficulties across different translation directions).

**Key Challenge**: While test cases are naturally portable across languages, expanding from a single pivot language to a complete multilingual translation matrix faces the dual hurdles of data bottlenecks and optimization imbalance.

**Goal**: (1) Address the scarcity of training data in multilingual code translation; (2) Mitigate the optimization imbalance problem during joint multilingual optimization.

**Key Insight**: Utilize the cross-lingual portability of unit tests as a unified verification mechanism, and progressively expand training data to cover all translation directions through bootstrapping experience collection.

**Core Idea**: Using one language as the axis, "bootstrap" the expansion of training data through successful translations from the RL policy model itself, while dynamically adjusting the learning intensity of different translation directions using language-aware weights.

## Method

### Overall Architecture

BootTrans consists of two core components: (1) **Bootstrapping Multilingual Exploration**—leveraging a dual-pool architecture (Seed Pool + Exploration Pool) to gradually expand from a pivot language to the full translation matrix; (2) **Language-aware Weight Optimization**—dynamically adjusting the loss weight for each translation direction based on the relative performance of "sibling languages." Training is conducted using the GRPO algorithm.

### Key Designs

1.  **Dual-Pool Architecture**:

    *   **Function**: Progressively expand training data to cover all translation directions.
    *   **Mechanism**: The Seed Pool $\mathcal{D}_{\text{seed}}$ contains code-test pairs of the pivot language (Python); the Exploration Pool $\mathcal{D}_{\text{explore}}$ dynamically stores successful translations that pass all tests in the policy model's rollouts. Each training iteration prioritizes sampling from the Exploration Pool, supplementing with the Seed Pool when necessary. Successfully translated code can serve as source inputs for new translation directions in subsequent iterations (e.g., Java→Python back-translation).
    *   **Design Motivation**: Break the dependence on parallel corpora through experience collection, allowing the model to self-construct multilingual training data; FIFO queue management prevents pool overload.

2.  **Language-aware Weight Optimization**:

    *   **Function**: Mitigate the optimization imbalance across different directions in multilingual translation.
    *   **Mechanism**: For a translation from source code $x_i$ to target language $L_k$, define sibling reward $\mathcal{R}_{i,\neg k}$ as the sum of cumulative rewards for other target languages. The weight is $w_{i,k} = \frac{\mathcal{R}_{i,\neg k}}{\mathcal{R}_{i,k} + \mathcal{R}_{i,\neg k}}$. When the model performs well in other languages but poorly in $L_k$, $w_{i,k}$ increases, forcing the model to focus more on difficult directions.
    *   **Design Motivation**: The intuition is that if a model demonstrates semantic understanding through sibling languages but struggles with a specific language, the issue lies in that language's syntax or idiomatic expressions, requiring higher learning intensity.

3.  **Verification Oracle and Reward Design**:

    *   **Function**: Provide a unified cross-lingual functional correctness verification.
    *   **Mechanism**: A binary verifiable reward $R(y, T) = \mathbb{1}[y \text{ compiles and passes all tests in } T]$ is used. Test suites are converted from Python to other languages via rule-based conversion through MultiPL-E. Compilation errors, runtime errors, and timeouts all result in R=0.
    *   **Design Motivation**: Align the optimization objective with functional equivalence rather than surface form similarity.

### Loss & Training

Using the GRPO algorithm, the objective function is a language-aware weighted PPO-style objective, including clipping ratios, advantage estimates calculated within target language groups, and KL penalties. Training uses the AdamW optimizer with a learning rate of 1e-6, a rollout macro-batch size of 256, and $G=8$ candidate translations generated per source code.

## Key Experimental Results

### Main Results

**HumanEval-X CA@1 Average Score**

| Method | Avg |
|------|-----|
| Qwen3-1.7B (base) | 64.33 |
| BootTrans Qwen3-1.7B | **74.70** (+10.37) |
| Llama-3.1-8B (base) | 61.79 |
| BootTrans Llama-3.1-8B | **78.36** (+16.57) |
| Qwen2.5-7B (base) | 68.50 |
| BootTrans Qwen2.5-7B | **83.84** (+15.34) |

**Comparison with Other Methods (Qwen3-1.7B, HumanEval-X Avg)**

| Method | Avg |
|------|-----|
| CoTran | 64.03 |
| MultiPL-T | 64.74 |
| PPOCoder | 69.21 |
| OORL | 69.92 |
| BootTrans | **74.70** |

### Ablation Study

The BootTrans 1.7B model outperforms Qwen3-32B on HumanEval-X (74.70 vs 67.99), demonstrating the potential for small models to surpass larger ones through RL training. On TransCoder-Test, BootTrans also yields consistent improvements.

### Key Findings

*   Both bootstrapping exploration and language-aware weighting components contribute significantly to final performance.
*   BootTrans enables a small model with 1.7B parameters to surpass a large model with 32B parameters.
*   Consistent improvements are achieved across all six translation directions, mitigating the optimization imbalance problem.
*   The cross-lingual portability of test cases is the fundamental key to the method's success.

## Highlights & Insights

*   The bootstrapping data expansion approach is simple and effective, making full use of the cross-lingual portability of test cases.
*   The language-aware weighting mechanism provides a clear intuition, achieving adaptive difficulty adjustment based on "sibling language" comparison.
*   Experimental results showing small models surpassing large models highlight the value of RL training in code translation.
*   The FIFO management strategy for the dual-pool architecture is well-considered from an engineering perspective.

## Limitations & Future Work

*   Experiments are currently limited to three languages (C++, Java, Python) and have not yet been extended to more languages.
*   The method relies on MultiPL-E's rule-based test conversion, which may fail for certain complex test cases.
*   Training costs are high, requiring a large volume of rollouts, compilations, and executions.
*   Future work could explore extending the method to more programming languages and complex software engineering scenarios.

## Related Work & Insights

*   Compared to RL methods such as PPOCoder and OORL, the innovation of BootTrans lies in the combination of data expansion and weighting mechanisms.
*   The test conversion tools from MultiPL-E provide critical infrastructure for the method.
*   The bootstrapping training data expansion approach can be generalized to other generation tasks that require verification feedback.

## Rating

*   Novelty: ⭐⭐⭐⭐ The combined design of bootstrapping exploration and language-aware weighting is novel and practical.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across three base models, two benchmarks, and multiple baselines.
*   Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed algorithmic description.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](../../ICML2026/code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](pexa_parallel_exploration_agent_for_complex_text-to-sql.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)

</div>

<!-- RELATED:END -->
