---
title: >-
  [Paper Note] Bootstrapping Code Translation with Weighted Multilanguage Exploration
description: >-
  [ACL 2026][Reinforcement Learning][code translation] BootTrans proposes a bootstrapping multilingual code translation approach that leverages test cases from a single pivot language (Python) as cross-lingual verification…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "code translation"
  - "bootstrapping exploration"
  - "language-aware weighting"
  - "RLVR"
  - "multilingual optimization"
date: 2026-05-08
content_hash: 69b5183bd0e62380
---

# Bootstrapping Code Translation with Weighted Multilanguage Exploration

**Conference**: ACL 2026
**arXiv**: [2601.03512](https://arxiv.org/abs/2601.03512)
**Code**: [https://github.com/nju-websoft/BootTrans/](https://github.com/nju-websoft/BootTrans/)
**Area**: Code Translation / Reinforcement Learning
**Keywords**: code translation, bootstrapping exploration, language-aware weighting, RLVR, multilingual optimization

## TL;DR

BootTrans proposes a bootstrapping multilingual code translation approach that leverages test cases from a single pivot language (Python) as cross-lingual verification oracles, employs a dual-pool architecture to expand training data through experience collection, and designs a language-aware weighting mechanism to dynamically prioritize difficult translation directions, achieving significant improvements over baselines on HumanEval-X and TransCoder-Test.

## Background & Motivation

**Background**: Code translation is critical for legacy system modernization and cross-platform interoperability. LLMs have made remarkable progress on coding tasks, yet code translation typically relies on high-quality parallel corpora, which are rarely accompanied by executable test cases.

**Limitations of Prior Work**: (1) Multilingual parallel code data is scarce and seldom equipped with cross-lingual executable test cases; (2) unsupervised methods (e.g., those exploiting code structural information) require massive monolingual corpora and cannot directly optimize for functional correctness; (3) existing RLVR methods face two key challenges: **input monotonicity** (verifiable seeds are confined to a single pivot language) and **optimization imbalance** (skewed learning signals caused by difficulty disparities across translation directions).

**Key Challenge**: Although test cases are inherently portable across languages, scaling from a single pivot language to a complete multilingual translation matrix faces the dual barriers of data bottleneck and optimization imbalance.

**Goal**: (1) Address training data scarcity in multilingual code translation; (2) mitigate optimization imbalance during simultaneous multilingual optimization.

**Key Insight**: Exploit the cross-lingual portability of unit tests as a unified verification mechanism, and progressively expand training data coverage to all translation directions through bootstrapping experience collection.

**Core Idea**: Use one language as a pivot, and "bootstrap" training data expansion via successful translations produced by the RL policy itself, while dynamically modulating the learning intensity of different translation directions with language-aware weights.

## Method

### Overall Architecture

BootTrans comprises two core components: (1) **Bootstrapping Multilingual Exploration**—a dual-pool architecture (seed pool + exploration pool) that progressively extends coverage from the pivot language to the full translation matrix; and (2) **Language-Aware Weighted Optimization**—dynamically adjusting the loss weight of each translation direction based on the relative performance of "sibling languages." Training is performed with the GRPO algorithm.

### Key Designs

1. **Dual-Pool Architecture**:

    - Function: Progressively expand training data to cover all translation directions.
    - Mechanism: The seed pool $\mathcal{D}_{\text{seed}}$ contains code–test pairs in the pivot language (Python); the exploration pool $\mathcal{D}_{\text{explore}}$ dynamically stores successful translations produced by the policy model during rollouts that pass all test cases. Each training iteration preferentially samples from the exploration pool, supplementing from the seed pool when necessary. Successful translations can subsequently serve as source inputs for new translation directions (e.g., reverse translation from Java→Python).
    - Design Motivation: Break dependence on parallel corpora through experience collection, enabling the model to construct multilingual training data autonomously; a FIFO queue prevents pool overflow.

2. **Language-Aware Weight Optimization**:

    - Function: Mitigate optimization imbalance across different directions in multilingual translation.
    - Mechanism: For translating source code $x_i$ into target language $L_k$, the sibling reward $\mathcal{R}_{i,\neg k}$ is defined as the cumulative reward across all other target languages. The weight is $w_{i,k} = \frac{\mathcal{R}_{i,\neg k}}{\mathcal{R}_{i,k} + \mathcal{R}_{i,\neg k}}$; when the model performs well on sibling languages but poorly on $L_k$, $w_{i,k}$ increases, forcing the model to attend more to the difficult direction.
    - Design Motivation: The intuition is that if the model demonstrates semantic understanding via sibling languages yet struggles with a specific language, the difficulty lies in that language's syntax or idiomatic expressions, warranting increased learning intensity.

3. **Verification Oracle and Reward Design**:

    - Function: Provide unified cross-lingual functional correctness verification.
    - Mechanism: A binary verifiable reward $R(y, T) = \mathbb{1}[y \text{ compiles and passes all tests in } T]$ is used; test suites are rule-based transpiled from Python to other languages via MultiPL-E. Compilation errors, runtime errors, and timeouts all yield $R=0$.
    - Design Motivation: Align the optimization objective with functional equivalence rather than surface-level syntactic similarity.

### Loss & Training

The GRPO algorithm is employed, with a language-aware weighted PPO-style objective that incorporates clipped probability ratios, advantage estimation computed within groups sharing the same target language, and a KL penalty. Training uses the AdamW optimizer with a learning rate of 1e-6, a rollout macro-batch size of 256, and $G=8$ candidate translations generated per source program.

## Key Experimental Results

### Main Results

**HumanEval-X CA@1 Average Score**

| Method | Avg |
|--------|-----|
| Qwen3-1.7B (base) | 64.33 |
| BootTrans Qwen3-1.7B | **74.70** (+10.37) |
| Llama-3.1-8B (base) | 61.79 |
| BootTrans Llama-3.1-8B | **78.36** (+16.57) |
| Qwen2.5-7B (base) | 68.50 |
| BootTrans Qwen2.5-7B | **83.84** (+15.34) |

**Comparison with Other Methods (Qwen3-1.7B, HumanEval-X Avg)**

| Method | Avg |
|--------|-----|
| CoTran | 64.03 |
| MultiPL-T | 64.74 |
| PPOCoder | 69.21 |
| OORL | 69.92 |
| BootTrans | **74.70** |

### Ablation Study

The BootTrans 1.7B model surpasses Qwen3-32B on HumanEval-X (74.70 vs. 67.99), demonstrating the potential of small models to outperform much larger ones through RL training. Consistent improvements are also observed on TransCoder-Test.

### Key Findings

- Both bootstrapping exploration and language-aware weighting contribute significantly to overall performance.
- BootTrans enables a 1.7B-parameter model to outperform a 32B-parameter model.
- Consistent improvements are achieved across all six translation directions, alleviating optimization imbalance.
- The cross-lingual portability of test cases is a critical foundation for the method's success.

## Highlights & Insights

- The bootstrapping data expansion strategy is concise and effective, fully exploiting the cross-lingual portability of test cases.
- The language-aware weighting mechanism is intuitively grounded, achieving adaptive difficulty adjustment through sibling-language comparison.
- The experimental finding that a small model surpasses a large model underscores the value of RL training for code translation.
- The FIFO management strategy of the dual-pool architecture reflects careful engineering consideration.

## Limitations & Future Work

- Experiments are currently limited to three languages (C++, Java, Python) and have not been extended to a broader set.
- The approach relies on rule-based test transpilation via MultiPL-E, which may fail for certain complex test cases.
- Training cost is relatively high, requiring extensive rollouts and compilation executions.
- Future work could explore extending the method to more programming languages and more complex software engineering scenarios.

## Related Work & Insights

- Compared to RL-based methods such as PPOCoder and OORL, BootTrans innovates through the combination of data expansion and weighting mechanisms.
- The test transpilation tooling from MultiPL-E provides critical infrastructure for the approach.
- The idea of bootstrapping training data expansion is generalizable to other generation tasks requiring verification feedback.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of bootstrapping exploration and language-aware weighting is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three base models, two benchmarks, and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and algorithmic descriptions are thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)

</div>

<!-- RELATED:END -->
