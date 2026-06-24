---
title: >-
  [Paper Note] Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][inverse scaling] Through the redefinition of physical/mathematical constants and measurement units (e.g., "let $\pi=500$"), this paper systematically investigates the performance of LLMs on inverse scaling tasks. The study reveals that larger models are more prone to anchoring to pre-existing memorized values and failing to follow prompt-based redefinitions, with incorrect confidence (giving wrong answers instead of abstaining) also increasing with sca…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "inverse scaling"
  - "redefinition"
  - "anchoring"
  - "memorization"
  - "reasoning flexibility"
  - "physical constants"
date: 2026-05-08
content_hash: 321f9c8cfada62ef
---

# Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models

**Conference**: ACL 2025  
**Authors**: Elena Stringli, Maria Lymperaiou, Giorgos Filandrianos, Athanasios Voulodimos, Giorgos Stamou (NTUA)  
**arXiv**: [2502.12821](https://arxiv.org/abs/2502.12821)  
**Code**: —  
**Area**: LLM/NLP, Evaluation of Reasoning  
**Keywords**: inverse scaling, redefinition, anchoring, memorization, reasoning flexibility, physical constants  

## TL;DR

Through the redefinition of physical/mathematical constants and measurement units (e.g., "let $\pi=500$"), this paper systematically investigates the performance of LLMs on inverse scaling tasks. The study reveals that larger models are more prone to anchoring to pre-existing memorized values and failing to follow prompt-based redefinitions, with incorrect confidence (giving wrong answers instead of abstaining) also increasing with scale.

## Background & Motivation

### Background

**Background**: **Core Problem**: Inverse scaling tasks, where model performance degrades as scale increases, can expose potential flaws in LLM reasoning. Redefinition tasks belong to the category of "strong priors" inverse scaling, which humans can easily perform with 100% accuracy, but LLMs often fail.

### Limitations of Prior Work

**Limitations of Prior Work**: **Prior Gaps**: Inverse scaling is understudied in existing literature. While the Inverse Scaling Prize (McKenzie et al., 2024) initially highlighted the issue, a systematic study of redefinition tasks—including the effects of different difficulty levels, response formats, and prompting strategies—remains a blank.

### Key Challenge

**Key Challenge**: **Research Motivation**: As LLMs are increasingly deployed in high-risk scenarios (such as scientific computing and engineering decisions), it is crucial to understand whether they can flexibly adapt their reasoning paths when faced with instructions that conflict with their pre-trained knowledge.

## Method

### Overall Architecture

The experimental design covers two categories of redefined objects $\times$ multiple difficulty levels $\times$ three levels of question difficulty $\times$ two response formats:
1. **Constant Redefinition**: 15 physical/mathematical constants ($\pi, e, \phi, c, G, h$, etc.), involving both assignment and swap methods.
2. **Unit Redefinition**: Modification of conversion relationships for 15 measurement units (minutes to seconds, kilograms to grams, meters to centimeters, etc.).

### Key Designs

1. **Incremental Redefinition Difficulty Levels**: Assignment is divided into three levels: $R_{a1}$ close to the original value ($\pi=4.5$), $R_{a2}$ magnitude deviation ($\pi=500$), and $R_{a3}$ implausible value ($\pi=-10$). Swap is divided into two levels: $R_{s1}$ close-value swap ($\pi \leftrightarrow \phi$) and $R_{s2}$ large-difference swap ($\pi \leftrightarrow h=6.626 \times 10^{-34}$).
2. **Incremental Question Difficulty Levels**: $Q_1$ simple retrieval ("What is the first non-zero digit of $\pi$ after redefinition?"), $Q_2$ simple computation ("$\pi \times 3 = ?$"), and $Q_3$ multi-step reasoning ("What is the surface area of the Earth?"—requiring computation with the redefined $\pi$).
3. **Two Response Formats**: Free-form (FF) and Multiple-Choice (MC), where MC includes misleading distractors.

### Loss & Training

This work is an evaluation study and does not involve model training.

## Experiments

### Main Results — Anchoring Rates Across Different Model Families and Scales

| Model Family | Scale Trend | Change in Anchoring Rate | Key Observation |
|---|---|---|---|
| GPT Series | 3.5 $\rightarrow$ 4 $\rightarrow$ 4o | **Increase** | Larger models use the original values more frequently |
| Llama Series | 8B $\rightarrow$ 70B $\rightarrow$ 405B | **Increase** | Anchoring behavior is particularly severe in the 405B model |
| Gemma Series | 2B $\rightarrow$ 9B $\rightarrow$ 27B | **Increase** | Significant anchoring exists even in the smallest models |
| Human Baseline | — | — | 100% accuracy (easily handles redefinition) |

### Ablation Study — Analysis of Contributing Factors

| Factor | Results |
|---|---|
| Question Difficulty $Q_1 \rightarrow Q_2 \rightarrow Q_3$ | Anchoring rate increases with difficulty, most severe in multi-step reasoning |
| Redefinition Difficulty $R_{a1} \rightarrow R_{a2} \rightarrow R_{a3}$ | The larger the deviation from the original value, the higher the anchoring rate |
| Assignment vs. Swap | Higher anchoring rate in swap tasks (conflict between two memorized values) |
| Free-form vs. Multiple-Choice | Multiple-choice format reduces the anchoring rate but does not eliminate it |
| Chain-of-Thought | CoT sometimes increases anchoring instead (the model "recalls" original values during reasoning) |
| System Prompt Emphasis | System prompts emphasizing "must use the new value" help but have limited effect |

### Key Findings

- Larger models not only make errors more frequently but also select "abstain" less often—they prefer to confidently provide incorrect answers based on the original values rather than acknowledging uncertainty.
- Prompting techniques (CoT, system prompts, response formats) can influence anchoring rates but cannot fundamentally eliminate the anchoring effect of prior knowledge.
- Unit redefinition is followed more easily than constant redefinition, likely because unit conversions appear less frequently than constant values in training data.
- Swap-type redefinition is the most challenging—the model needs to simultaneously override two memorized values and apply them correctly.

## Highlights & Insights

- The experimental design is systematic and comprehensive: 15 constants $\times$ 15 units $\times$ multiple difficulty levels $\times$ multiple question levels $\times$ multiple formats, providing extensive coverage.
- Reveals a counter-intuitive yet vital phenomenon: the "knowledge" of LLMs can act as a barrier to reasoning flexibility.
- The contrast between 100% human accuracy and the inverse scaling of LLMs highlights the fundamental difference between memorization and reasoning.

## Limitations & Future Work

- Only English prompts were tested; performance in multilingual settings remains unverified.
- The impact of fine-tuning or RLHF on anchoring behavior was not explored.
- Constant redefinition in the experiments is relatively simple (direct assignment); more complex conditional redefinitions were not tested.
- The API versions of some models might affect the reproducibility of the results.
- Whether anchoring behavior correlates with specific training data sources (e.g., Wikipedia entry frequencies) was not analyzed.

## Related Work & Insights

- **Inverse Scaling**: The Inverse Scaling Prize (McKenzie et al., 2024) reveals causes of inverse scaling such as strong priors and undesirable imitation.
- **LLM Reasoning**: The boundary between memorization and reasoning is blurred (Wu et al., 2024; Mahowald et al., 2024); LLM performance degrades under modified phrasing.
- **Knowledge Conflict**: When in-context information conflicts with memorized knowledge, LLMs tend to rely on training data (Xu et al., 2024).

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)
- [\[ACL 2025\] TESS 2: A Large-Scale Generalist Diffusion Language Model](tess_2_a_large-scale_generalist_diffusion_language_model.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)

</div>

<!-- RELATED:END -->
