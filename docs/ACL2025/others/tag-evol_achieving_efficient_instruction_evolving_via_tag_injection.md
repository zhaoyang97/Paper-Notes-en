---
title: >-
  [Paper Note] Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection
description: >-
  [ACL 2025][Instruction Evolution] Tag-Evol proposes an instruction evolution framework based on knowledge tag injection. By constructing a multi-step fine-grained tag pool and a budget-controlled injection mechanism, it generates high-quality evolved instruction data of varying difficulties without iteration, significantly outperforming Evol-Instruct across multiple tasks and backbone models.
tags:
  - "ACL 2025"
  - "Instruction Evolution"
  - "Knowledge Tags"
  - "Tag Injection"
  - "Synthetic Data"
  - "Evol-Instruct"
date: 2026-05-08
content_hash: ebbd654fa87e198c
---

# Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection

**Conference**: ACL 2025  
**arXiv**: [2505.24165](https://arxiv.org/abs/2505.24165)  
**Code**: Yes ([https://github.com/fghccv/TagEvol](https://github.com/fghccv/TagEvol))  
**Area**: Others  
**Keywords**: Instruction Evolution, Knowledge Tags, Tag Injection, Synthetic Data, Evol-Instruct

## TL;DR

Tag-Evol proposes an instruction evolution framework based on knowledge tag injection. By constructing a multi-step fine-grained tag pool and a budget-controlled injection mechanism, it generates high-quality evolved instruction data of varying difficulties without iteration, significantly outperforming Evol-Instruct across multiple tasks and backbone models.

## Background & Motivation

Evol-Instruct is currently the most popular method for instruction data synthesis, enhancing instruction complexity and diversity through iterative evolution. However, two core limitations persist:

**Fixed Evolution Strategies**: Existing methods rely on a fixed set of human-designed or machine-optimized strategies (e.g., "adding constraints", "increasing reasoning steps"), which are limited in variety and difficult to adapt to new domains. Furthermore, fixed strategies restrict the diversity of generated data due to model preference biases.

**Inefficient Synthesis Mode**: To obtain highly complex samples, Evol-Instruct requires multiple rounds of iterative evolution. This not only consumes extra time but also introduces cumulative errors due to the buildup of hallucinations.

An ideal framework should possess: **diverse and concrete evolution strategies** + **the ability to efficiently and directly generate data of varying difficulties**.

The core insight of Tag-Evol stems from InsTag (Lu et al., 2023): the difficulty of an instruction can be estimated by the number of knowledge tags. Therefore, directly injecting different quantities and combinations of tags can control the evolution difficulty without requiring iteration.

## Method

### Overall Architecture

Tag-Evol consists of two phases:
1. **Tag Pool Construction**: Performs multi-step fine-grained labeling on the seed dataset to construct a diverse and specific tag pool.
2. **Tag Sampling Evolution**: Samples tag combinations based on a budget and injects them into the original instructions to achieve evolution.

### Key Designs

1. **Multi-step Fine-grained Labeling Method (Tag Pool Construction)**

    - Compared to the single-step coarse-grained labeling in InsTag, this paper proposes a two-step labeling approach:
    - **Step 1 - Aspect Generation**: The model summarizes sample features from a high-level perspective (task type, required skills, operation type, etc.) to ensure broad tag coverage.
    - **Step 2 - Tag Generation**: Concrete knowledge tags are generated based on the abstract aspects.
    - The list structure is improved to a dictionary structure, enabling the model to generate layer-by-layer from key (aspect) to value (concrete tag).
    - Effect: The number of tags is ~20 times that of InsTag and ~2 times that of single-step fine-grained labeling.
    - Motivation: Coarse-grained tags (e.g., "adding constraints") fail to provide sufficiently specific guidance for evolution, whereas concrete tags like "exponential-related" make the evolution goal clearer.

2. **Budget-Controlled Tag Injection (Tag Sampling Evolution)**

    - Assigns a difficulty budget $b$ (the number of tags to inject) to each sample.
    - Randomly samples a batch of candidate tags $cand$ from the tag pool $\mathcal{P}$.
    - Directs the evolution model $M_\theta$ to **actively select** a suitable tag subset $t$ from the candidates based on the original instruction and the budget.
    - Equation: $\hat{x}, t = M_\theta(x, b, cand)$, where $|t| = b$.
    - Four-step process: Select tag combinations $\rightarrow$ Generate injection plan $\rightarrow$ Execute plan $\rightarrow$ Rewrite to eliminate hallucinations.
    - Motivation: Allowing the model to select rather than forcing assignments reduces hallucinations caused by irrelevant tags.

3. **Multi-round Evolution Strategy**

    - Although Tag-Evol can generate samples of varying difficulties in a single step, three rounds of evolution are still conducted for fair comparison.
    - Different budgets are used (e.g., 1/3/5 tags for the math domain, 3/5/7 tags for the code domain).
    - Key difference: The input of Evol-Instruct in each round is the output of the previous round (chain-like), whereas Tag-Evol **directly evolves from the seed data** in every round.

## Key Experimental Results

### Main Results — Multi-task and Multi-backbone (Table 2 Summary)

| Method | Mistral-7B Avg | Llama3-8B Avg | Qwen2.5-7B Avg |
|------|---------------|---------------|-----------------|
| Seed | 33.9 | 41.6 | 59.0 |
| Evol-Instruct | 40.0 | 47.4 | 59.0 |
| Auto Evol-Ins | 41.4 | 48.0 | 60.4 |
| **Tag-Evol** | **43.7** | **50.4** | **61.7** |

- Comprehensive leadership across general tasks (MTBench/IFEval), mathematical reasoning (GSM8K/MATH-500), and code generation (HumanEval/MBPP).
- MTBench scores improved from 5.4-6.7 to 5.7-7.2.

### Ablation Study

| Experiment | Key Conclusion |
|------|----------|
| Multi-step vs. Single-step Labeling | Multi-step labeling improves GSM8K from 67.0 to 69.3, and MATH-500 from 34.6 to 38.0. |
| Data Scale Analysis | Tag-Evol outperforms the baseline across almost all data scales, showing more stable growth. |
| Evolution Model Scale | A 7B model is sufficient to execute Tag-Evol (benefiting from the explicit guidance provided by concrete tags). |
| n-gram Repetitive Analysis | The n-gram repetition rate of Tag-Evol is lower than that of Evol-Instruct, indicating higher data diversity. |

### Key Findings

1. **Tag-Evol consistently improves performance by 2-3 points across different backbones**, with an approximately 1.5-point gain even on the strong backbone Qwen2.5.
2. **No iterative evolution required**: Directly injecting tags produces evolutionary effects equal to or better than iterative evolution, while avoiding cumulative hallucinations.
3. **Tag diversity is crucial**: Raising the quantity of tags by 20 times compared to the baseline level of the original InsTag method leads to significant improvements in downstream performance.
4. **Applicable to small models**: A 7B model can serve as the evolution model to execute Tag-Evol, as concrete tags reduce task difficulty.
5. In the Qwen2.5 setting, Evol-Instruct barely outperforms the seed data, whereas Tag-Evol still achieves a 2-point improvement—indicating that tag injection introduces novel knowledge combinations beyond simple difficulty scaling.

## Highlights & Insights

1. **Elegant perspective shift**: Reduces the dimension of evolution strategies from "instruction-level operations" to "knowledge tag injection", simplifying complexity.
2. **Obvious efficiency advantage**: Bypasses the cumulative error issues of iterative evolution, achieving the goal in a single step through budget control.
3. **Complementarity with InsTag**: InsTag uses tags to evaluate data quality, whereas Tag-Evol uses tags to guide data synthesis—forming an "evaluation $\rightarrow$ generation" closed loop.
4. **High practicality**: The tag pool can be reused once constructed, a 7B model can execute the evolution, and the open-source framework is easy to integrate.

## Limitations & Future Work

- The quality of the tag pool depends on the capability of the labeling model, and different domains may require domain adaptation.
- The optimal value of the budget parameter $b$ needs to be determined experimentally, lacking an automated selection mechanism.
- In multi-round evolution, the seed data itself might become a bottleneck (a shared limitation with Evol-Instruct).
- The compatibility or conflict issues between tags have not been explored.

## Related Work & Insights

- Evol-Instruct (Xu et al., 2023) and Auto Evol-Instruct (Zeng et al., 2024) are direct baselines.
- InsTag (Lu et al., 2023) provides the core inspiration—that the number of tags is positively correlated with data difficulty.
- Self-Instruct (Wang et al., 2022) pioneered the paradigm of LLM self-generated instruction data.
- QFT (Ding et al., 2024) formulates instruction synthesis as a learnable task, which is complementary to the tag pool construction concept of Tag-Evol.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using tag injection as an evolution strategy is novel and practical, with a cleverly designed budget-controlled mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparison across three domains and three backbones + multi-perspective ablation (labeling method, data scale, model size, and diversity).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and detailed methodology description.
- **Value**: ⭐⭐⭐⭐ — Provides a more efficient and controllable instruction evolution framework, offering practical value for data synthesis in LLM alignment training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)
- [\[ACL 2025\] Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[CVPR 2026\] InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space](../../CVPR2026/others/instantretouch_efficient_and_high-fidelity_instruction-guided_image_retouching_w.md)

</div>

<!-- RELATED:END -->
