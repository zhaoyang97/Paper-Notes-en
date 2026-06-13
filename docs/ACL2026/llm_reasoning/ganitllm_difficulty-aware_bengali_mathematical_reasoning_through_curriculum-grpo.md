---
title: >-
  [Paper Note] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO
description: >-
  [ACL 2026][LLM Reasoning][Bengali mathematical reasoning] This paper proposes GanitLLM, the first mathematical reasoning model that truly reasons in Bengali (rather than translating or reasoning in English). The authors…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Bengali mathematical reasoning"
  - "curriculum learning"
  - "GRPO cold-start"
  - "difficulty-aware"
  - "low-resource languages"
date: 2026-05-08
content_hash: ee2b6ac9fef0d31c
---

# GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.06767](https://arxiv.org/abs/2601.06767)  
**Code**: [Website](https://dipta007.github.io/GanitLLM/)  
**Area**: Low-resource language reasoning / Mathematical reasoning  
**Keywords**: Bengali mathematical reasoning, curriculum learning, GRPO cold-start, difficulty-aware, low-resource languages

## TL;DR

This paper proposes GanitLLM, the first mathematical reasoning model that truly reasons in Bengali (rather than translating or reasoning in English). The authors construct Ganit, a difficulty-annotated Bengali math dataset, and introduce Curriculum-GRPO to address the cold-start issue in GRPO training for low-resource languages. The 4B model achieves an 8-percentage-point accuracy gain on Bn-MGSM, with the proportion of Bengali reasoning tokens increasing from 14% to 88%.

## Background & Motivation

**Background**: LLMs have made significant progress in mathematical reasoning for high-resource languages like English (e.g., DeepSeek-R1, OpenAI o1), where RL methods such as GRPO have proven effective. However, reasoning progress in low-resource languages lags severely. Despite Bengali being the seventh most spoken language globally, existing LLMs either reason in English and translate the answer or fail entirely on Bengali math problems.

**Limitations of Prior Work**: (1) Even when explicitly prompted to reason in Bengali, existing LLMs tend to reason in English and only output the final answer in Bengali, which is poor for native user interpretability; (2) Standard GRPO training encounters a "cold-start problem" in low-resource languages, where the policy model fails to generate any correct solutions in rollout groups due to insufficient target language capability, leading to zero rewards, zero gradients, and ineffective training; (3) Bengali math datasets suffer from inconsistent quality and lack difficulty annotations and systematic filtering.

**Key Challenge**: GRPO requires at least some correct answers within a rollout group to calculate valid advantage values. However, low-resource language models often fail completely on difficult problems—a "chicken-and-egg" problem where the model must already know how to solve problems to learn from them.

**Goal**: Construct a high-quality, difficulty-annotated Bengali math dataset and design a training strategy that solves the cold-start problem, enabling the model to reason natively in Bengali.

**Key Insight**: The problem is decomposed into three steps: (1) Data: Building a quality-filtered and difficulty-annotated dataset; (2) SFT: Teaching the model to reason step-by-step in Bengali first (regardless of correctness); (3) GRPO: Using a curriculum learning strategy to train from easy to difficult.

**Core Idea**: Use Curriculum-GRPO to arrange training data by difficulty, ensuring the model can generate partially correct answers throughout each stage to obtain valid gradients and avoid cold starts.

## Method

### Overall Architecture

A two-stage training process: (1) SFT Stage: Teaching the model step-by-step Bengali reasoning on CoT-SFT data, focusing on language consistency rather than final correctness; (2) Curriculum-GRPO Stage: Training with GRPO on RL data sorted by difficulty, starting with simple problems and gradually increasing complexity. The Ganit dataset is derived from ~1.5M raw samples through multi-stage filtering and difficulty annotation.

### Key Designs

1.  **Difficulty-Aware Dataset (Ganit)**:
    *   **Function**: Provides high-quality, difficulty-annotated Bengali math training and evaluation data.
    *   **Mechanism**: (a) Collects ~1.5M samples from 9 public datasets; (b) Filters datasets to keep those with >95% accuracy via human evaluation (~1.1M remaining); (c) Applies rule-based filtering (numerical solutions only, >99% Bengali characters, excludes multiple-choice); (d) Performs fuzzy and MinHash deduplication; (e) Generates 32 independent solutions using Qwen3-32B and classifies samples into Easy/Medium/Hard/Olympiad based on pass@k; (f) Decontaminates against evaluation benchmarks.
    *   **Design Motivation**: Existing Bengali math datasets vary in quality, and standard benchmarks (Bn-MGSM/Bn-MSVAMP) are too simple for modern LLMs (77-86% are Easy level).

2.  **Curriculum-GRPO Training Strategy**:
    *   **Function**: Solves the cold-start problem of GRPO in low-resource settings.
    *   **Mechanism**: Uses fine-grained difficulty signals (1-32 correct generations). For each difficulty bucket, 60% of samples are drawn from the current bucket and 40% from the remaining 31 buckets (3 per bucket), sorted from easy to difficult by the primary bucket. This ensures: (a) The model gains correct experience on easy problems first; (b) Sufficient sample diversity in each stage to prevent forgetting; (c) The 60/40 ratio balances curriculum signal strength and diversity.
    *   **Design Motivation**: Naive full sorting (100% by difficulty) leads to overfitting on easy problems early on; random shuffling introduces difficult problems too early, causing cold starts.

3.  **Three-Dimensional Reward Function**:
    *   **Function**: Simultaneously optimizes format correctness, answer accuracy, and Bengali reasoning ratio.
    *   **Mechanism**: $R = R_{format} + R_{correctness} + R_{bengali}$, where $R_{format} \in \{0,1\}$ checks output format, $R_{correctness} \in \{0,1,2\}$ rewards the correct answer (with bonus points for Bengali answers), and $R_{bengali} \in \{0,1\}$ provides a reward if the proportion of Bengali tokens in reasoning is $\ge 80\%$.
    *   **Design Motivation**: Traditional GRPO only rewards final answer correctness, failing to incentivize reasoning in the target language.

### Loss & Training

The SFT stage uses standard cross-entropy loss. The GRPO stage utilizes standard GRPO loss combined with a max-length filter and token-level loss. Qwen3-4B serves as the base model.

## Key Experimental Results

### Main Results

| Model | Bn-MGSM | Bn-MSVAMP | Bengali % | Avg. Length (words) |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-4B (Base) | 69 | 78 | 14% | 943 |
| + SFT only | 73 | 81 | 82% | 210 |
| + Curriculum-GRPO | **77** | **84** | **88%** | **193** |
| Qwen3-8B | 76 | 83 | 18% | 876 |
| GPT-5-mini | 82 | 88 | 45% | 520 |

### Ablation Study

| Training Strategy | Bn-MGSM | Cold-start Rate |
| :--- | :--- | :--- |
| Random Shuffle GRPO | 72 | 35% |
| Full Sort (Easy → Hard) | 74 | 12% |
| **Curriculum-GRPO (60/40)** | **77** | **5%** |

### Key Findings

*   Curriculum-GRPO reduces the cold-start rate from 35% to 5%, which is crucial for GRPO training in low-resource languages.
*   The SFT stage is vital for language switching; Bengali rewards in GRPO alone cannot shift the reasoning language from English to Bengali.
*   Through Curriculum-GRPO, the 4B model reaches the accuracy level of the 8B base model while reducing reasoning tokens by 79.5%.
*   The difficulty distribution of Ganit-Dev is much more balanced (approx. 21-29% per level) compared to standard benchmarks (77-86% Easy), providing more discriminative evaluation.

## Highlights & Insights

*   The identification and resolution of the "cold-start problem" provide valuable insights for RL training in all low-resource languages.
*   The design of the three-dimensional reward function is elegant—it not only optimizes correctness but also explicitly incentivizes target language reasoning.
*   The 80% Bengali threshold design accounts for the language-agnostic nature of mathematical symbols, reflecting deep domain understanding.

## Limitations & Future Work

*   Validation is limited to the 4B model; cold-start issues might manifest differently in larger models.
*   The 60/40 curriculum ratio is empirically tuned and lacks theoretical guidance.
*   Difficulty labels depend on Qwen3-32B's capability and may need updates as evaluator models evolve.
*   Applicability to other reasoning tasks, such as logical or commonsense reasoning, remains unexplored.

## Related Work & Insights

*   **vs Confucius3-Math**: While the Chinese K-12 math model uses standard RL, GanitLLM must solve the cold-start problem inherent in smaller-scale Bengali training data.
*   **vs mCoT**: mCoT performs multilingual CoT tuning but does not enforce target language reasoning; GanitLLM achieves 88% native reasoning through specific Bengali rewards.
*   **vs MathOctopus**: Uses parallel corpora but reasoning remains in English; GanitLLM achieves true native-language reasoning.

## Rating

*   Novelty: ⭐⭐⭐⭐ The identification of the cold-start problem and the Curriculum-GRPO approach are novel contributions.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies, dataset quality analysis, and language ratio statistics.
*   Writing Quality: ⭐⭐⭐⭐ Clear problem definition and exhaustive description of the data construction process.
*   Value: ⭐⭐⭐⭐ Provides a practical solution for RL training in low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](../../ICLR2026/llm_reasoning/harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](semantic-aware_logical_reasoning_via_a_semiotic_framework.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ACL 2026\] Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding (CoRD)](distilling_long-cot_reasoning_through_collaborative_step-wise_multi-teacher_deco.md)

</div>

<!-- RELATED:END -->
