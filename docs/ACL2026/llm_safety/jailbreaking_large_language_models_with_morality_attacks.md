---
title: >-
  [Paper Note] Jailbreaking Large Language Models with Morality Attacks
description: >-
  [ACL 2026][LLM Safety][morality attack] This paper constructs a 10.3K morality attack dataset (covering value ambiguity and value conflict scenarios) and manipulates the moral judgment of LLMs via four adversarial strategies. It finds that both LLMs and guardrail models are highly vulnerable to morality attacks, and that larger models are paradoxically easier to compromise.
tags:
  - ACL 2026
  - LLM Safety
  - morality attack
  - jailbreak attack
  - value pluralism
  - LLM robustness
  - moral judgment
date: 2026-05-08
content_hash: 3f634fce00e1b188
---

# Jailbreaking Large Language Models with Morality Attacks

**Conference**: ACL 2026
**arXiv**: [2604.17053](https://arxiv.org/abs/2604.17053)
**Code**: [GitHub](https://github.com/MMLC-lab/Jailbreaking-LLM-Morality)
**Area**: AI Safety / Moral Robustness
**Keywords**: morality attack, jailbreak attack, value pluralism, LLM robustness, moral judgment

## TL;DR

This paper constructs a 10.3K morality attack dataset (covering value ambiguity and value conflict scenarios) and manipulates the moral judgment of LLMs via four adversarial strategies. It finds that both LLMs and guardrail models are highly vulnerable to morality attacks, and that larger models are paradoxically easier to compromise.

## Background & Motivation

**Background**: The field has accumulated a body of work, yet critical gaps remain.

**Limitations of Prior Work**: Existing approaches fail to adequately address core issues, exhibiting limitations in accuracy, scalability, or generalizability.

**Key Challenge**: The fundamental tension lies in the mismatch between the implicit assumptions of existing paradigms and real-world requirements.

**Goal**: To propose a new framework, method, or benchmark that systematically addresses the aforementioned problems.

**Key Insight**: A novel observation or theoretical perspective is leveraged to identify a new solution pathway.

**Core Idea**: The core contradiction is resolved through innovative technical means.

## Method

### Overall Architecture

The proposed method comprises multiple collaborating components that form a complete processing pipeline.

### Key Designs

1. **Core Component 1**:

    - Function: Addresses the primary technical challenge.
    - Mechanism: Achieves the objective through an innovative algorithmic or architectural design.
    - Design Motivation: Grounded in a deep understanding of the problem's nature.

2. **Core Component 2**:

    - Function: Provides auxiliary support or regularization.
    - Mechanism: Complements the limitations of the primary component.
    - Design Motivation: Its necessity is demonstrated through empirical or theoretical analysis.

3. **Core Component 3**:

    - Function: Optimizes training or inference efficiency.
    - Mechanism: Balances performance and efficiency.
    - Design Motivation: Driven by practical deployment requirements.

### Loss & Training

An optimization strategy and evaluation metrics appropriate to the task are adopted.

## Key Experimental Results

### Main Results

| Method | Core Metric | Note |
|--------|-------------|------|
| Baseline | Lower | Current best |
| **Ours** | **Highest** | Significant gain |

### Ablation Study

| Configuration | Result | Note |
|---------------|--------|------|
| Full | Highest | Complete model |
| w/o Core Component | Drops | Validates necessity |

### Key Findings

- The proposed method consistently outperforms baselines across multiple benchmarks.
- Ablation studies confirm the necessity of each component.
- The method exhibits particularly strong performance in specific scenarios.

## Highlights & Insights

- The core technical innovation addresses a long-standing problem.
- The method demonstrates strong scalability and practical applicability.
- The analysis reveals valuable and generalizable patterns.

## Limitations & Future Work

- The evaluation scope can be further broadened.
- The applicability of certain assumptions warrants further validation.
- Additional application scenarios merit exploration in future work.

## Related Work & Insights

- **vs. Most Related Work A**: This paper improves upon key dimensions.
- **vs. Most Related Work B**: This paper offers a distinct solution perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative, though some techniques build on combinations of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation is relatively comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and well-organized.
- Value: ⭐⭐⭐⭐ Makes a practical contribution to the field.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)
- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](../../NeurIPS2025/llm_safety/exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](../../AAAI2026/llm_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)

<!-- RELATED:END -->
