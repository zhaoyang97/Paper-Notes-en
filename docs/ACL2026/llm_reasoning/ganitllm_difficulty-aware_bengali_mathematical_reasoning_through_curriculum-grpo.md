---
title: >-
  [Paper Note] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO
description: >-
  [ACL 2026][LLM Reasoning][Bengali mathematical reasoning] This paper presents GanitLLM, the first mathematical reasoning model that genuinely reasons in Bengali (rather than translating or reasoning in English)…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Bengali mathematical reasoning"
  - "curriculum learning"
  - "GRPO cold start"
  - "difficulty-aware"
  - "low-resource language"
date: 2026-05-08
content_hash: 5c49d320faca4b44
---

# GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO

**Conference**: ACL 2026
**arXiv**: [2601.06767](https://arxiv.org/abs/2601.06767)  
**Code**: [Website](https://dipta007.github.io/GanitLLM/)  
**Area**: Low-Resource Language Reasoning / Mathematical Reasoning
**Keywords**: Bengali mathematical reasoning, curriculum learning, GRPO cold start, difficulty-aware, low-resource language

## TL;DR

This paper presents GanitLLM, the first mathematical reasoning model that genuinely reasons in Bengali (rather than translating or reasoning in English), together with Ganit, a difficulty-annotated Bengali math dataset. The proposed Curriculum-GRPO addresses the cold-start problem in GRPO training for low-resource languages. The 4B model achieves an 8 percentage-point accuracy gain on Bn-MGSM, and the proportion of Bengali reasoning tokens increases from 14% to 88%.

## Background & Motivation

**State of the Field**: LLMs have achieved remarkable progress in mathematical reasoning for high-resource languages (e.g., English), as demonstrated by DeepSeek-R1 and OpenAI o1, and RL methods such as GRPO have proven effective at enhancing mathematical reasoning. However, progress on low-resource languages lags far behind. Bengali is the seventh most spoken language globally, yet existing LLMs either reason in English and translate the final answer, or fail outright on Bengali math problems.

**Limitations of Prior Work**: (1) Even when explicitly prompted to reason in Bengali, existing LLMs tend to reason in English and only output the answer in Bengali, which severely hinders comprehensibility for native speakers. (2) Standard GRPO training encounters a cold-start problem in low-resource languages: the policy model, lacking sufficient target-language capability, fails to produce any correct solution within a rollout group, resulting in zero reward, zero gradient, and ineffective training. (3) Existing Bengali math datasets vary widely in quality and lack difficulty annotations and systematic quality filtering.

**Root Cause**: GRPO requires at least some correct answers within a rollout group to compute valid advantage estimates, yet low-resource language models are entirely unable to generate correct answers for difficult problems—a chicken-and-egg situation where the model must already be capable in order to learn.

**Paper Goals**: Construct a high-quality, difficulty-annotated Bengali math dataset; design a training strategy that resolves the cold-start problem; and enable the model to genuinely reason in Bengali rather than English.

**Starting Point**: The problem is decomposed into three steps: (1) Data—build a quality-filtered, difficulty-annotated dataset; (2) SFT—first teach the model to reason in Bengali (focusing on language rather than correctness); (3) GRPO—apply a curriculum learning strategy to train progressively from easy to hard.

**Core Idea**: Curriculum-GRPO arranges training data in order of increasing difficulty, ensuring that the model can produce some correct answers at each stage to obtain valid gradients, thereby avoiding the cold-start problem.

## Method

### Overall Architecture

The training consists of two stages: (1) **SFT stage**—the model is trained on CoT-SFT data to perform step-by-step reasoning in Bengali, with emphasis on language rather than correctness; (2) **Curriculum-GRPO stage**—GRPO training is applied to difficulty-sorted RL data, beginning with easy problems and gradually increasing difficulty. The dataset Ganit is derived from ~1.5M raw samples through multi-stage filtering and difficulty annotation.

### Key Designs

1. **Difficulty-Aware Dataset Ganit**

    - **Function**: Provides high-quality, difficulty-annotated training and evaluation data for Bengali mathematical reasoning.
    - **Mechanism**: (a) ~1.5M samples are collected from 9 public datasets; (b) human evaluation filters for datasets with >95% accuracy (~1.1M retained); (c) rule-based filtering retains only numerical answers, samples with >99% Bengali characters, and excludes multiple-choice questions; (d) fuzzy deduplication and MinHash deduplication are applied; (e) Qwen3-32B generates 32 independent solutions per problem, and difficulty is categorized into Easy/Medium/Hard/Olympiad levels based on pass@k; (f) benchmark contamination is removed.
    - **Design Motivation**: Existing Bengali math datasets are inconsistent in quality; standard evaluation sets (Bn-MGSM/Bn-MSVAMP) are too easy for modern LLMs (77–86% of samples are at the Easy level).

2. **Curriculum-GRPO Training Strategy**

    - **Function**: Resolves the cold-start problem in GRPO training for low-resource languages.
    - **Mechanism**: A fine-grained difficulty signal from 1 to 32 (number of correctly generated solutions) is used. For each difficulty bucket, 60% of samples are drawn from the current bucket and 40% from the remaining 31 buckets (3 samples per bucket). Batches are then ordered by the primary bucket difficulty from easy to hard. This ensures: (a) the model first gains correct experience on easy problems; (b) sufficient mixed samples at each stage prevent forgetting; (c) the 60/40 ratio balances curriculum signal strength and diversity.
    - **Design Motivation**: A naive full sort (100% difficulty-ordered) causes overfitting to easy problems in early training; random shuffling causes difficult problems to appear too early, triggering the cold-start problem.

3. **Three-Dimensional Reward Function**

    - **Function**: Simultaneously optimizes format correctness, answer accuracy, and the proportion of Bengali reasoning tokens.
    - **Mechanism**: $R = R_{format} + R_{correctness} + R_{bengali}$, where $R_{format} \in \{0,1\}$ checks output format, $R_{correctness} \in \{0,1,2\}$ rewards correct answers (with a bonus for answers in Bengali), and $R_{bengali} \in \{0,1\}$ rewards the model when the proportion of Bengali tokens in the reasoning chain is ≥80%.
    - **Design Motivation**: Conventional GRPO rewards only final answer correctness and provides no incentive for the model to reason in the target language.

### Loss & Training

The SFT stage uses standard cross-entropy loss. The GRPO stage applies standard GRPO loss with an overlength filter and token-level loss. The base model is Qwen3-4B.

## Key Experimental Results

### Main Results

| Model | Bn-MGSM | Bn-MSVAMP | Bengali% | Avg. Length (tokens) |
|---|---|---|---|---|
| Qwen3-4B (base) | 69 | 78 | 14% | 943 |
| + SFT only | 73 | 81 | 82% | 210 |
| + Curriculum-GRPO | **77** | **84** | **88%** | **193** |
| Qwen3-8B | 76 | 83 | 18% | 876 |
| GPT-5-mini | 82 | 88 | 45% | 520 |

### Ablation Study

| Training Strategy | Bn-MGSM | Cold-Start Rate |
|---|---|---|
| Random shuffle GRPO | 72 | 35% |
| Full sort (easy→hard) | 74 | 12% |
| **Curriculum-GRPO (60/40)** | **77** | **5%** |

### Key Findings

- Curriculum-GRPO reduces the cold-start rate from 35% to 5%, which is critical for effective GRPO training in low-resource languages.
- The SFT stage is essential for language switching—Bengali rewards in GRPO alone are insufficient to shift reasoning language from English to Bengali.
- The 4B model trained with Curriculum-GRPO matches the accuracy of the 8B base model while reducing reasoning tokens by 79.5%.
- The difficulty distribution of Ganit-Dev is far more balanced than standard evaluation sets (~21–29% per level vs. 77–86% Easy in standard sets), providing more discriminative evaluation.

## Highlights & Insights

- The identification and resolution of the cold-start problem offers broadly applicable insights for RL training across all low-resource languages.
- The three-dimensional reward function is an elegant design—it optimizes not only correctness but also explicitly incentivizes reasoning in the target language.
- The 80% Bengali token threshold accounts for the language-agnostic nature of mathematical symbols, reflecting a nuanced understanding of the domain.

## Limitations & Future Work

- Validation is limited to a 4B model; the cold-start problem may manifest differently at larger scales.
- The 60/40 curriculum ratio is empirically tuned and lacks theoretical justification.
- Difficulty labels depend on the capability of Qwen3-32B and may require updating as stronger evaluation models become available.
- Evaluation is restricted to mathematical reasoning; applicability to other reasoning tasks such as logical or commonsense reasoning remains unexplored.

## Related Work & Insights

- **vs. Confucius3-Math**: Confucius3-Math is a Chinese K-12 math model trained with standard RL; GanitLLM must address the cold-start problem arising from a much smaller volume of Bengali training data.
- **vs. mCoT**: mCoT performs multilingual CoT fine-tuning but does not enforce reasoning in the target language; GanitLLM achieves 88% native-language reasoning through a dedicated Bengali reward.
- **vs. MathOctopus**: MathOctopus uses parallel corpora but still reasons in English; GanitLLM achieves genuine native-language reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Curriculum-GRPO and the identification of the cold-start problem constitute novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes detailed ablations, dataset quality analysis, and language proportion statistics.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and the data construction process is described in detail.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for RL training in low-resource language settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](../../ICLR2026/llm_reasoning/harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](semantic-aware_logical_reasoning_via_a_semiotic_framework.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)

</div>

<!-- RELATED:END -->
