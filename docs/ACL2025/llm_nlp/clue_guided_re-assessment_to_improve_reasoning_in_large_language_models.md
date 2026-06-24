---
title: >-
  [Paper Note] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Clue guidance] This paper proposes the "Clue Guided Re-Assessment" method, which extracts key clues during the LLM reasoning process and guides the model to reflect on and correct its initial reasoning, significantly improving the accuracy of multi-step reasoning tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Clue guidance"
  - "reflective reasoning"
  - "self-correction"
  - "reasoning enhancement"
  - "step-by-step verification"
date: 2026-05-08
content_hash: f428c78b4c0b3b18
---

# Clue Guided Re-Assessment to Improve Reasoning in Large Language Models

**Conference**: ACL 2025  
**Area**: LLM/NLP  
**Keywords**: Clue guidance, reflective reasoning, self-correction, reasoning enhancement, step-by-step verification

## TL;DR
This paper proposes the "Clue Guided Re-Assessment" method, which extracts key clues during the LLM reasoning process and guides the model to reflect on and correct its initial reasoning, significantly improving the accuracy of multi-step reasoning tasks.

## Background & Motivation

**Background**: Large language models have made significant progress in tasks such as mathematical reasoning, logical reasoning, and commonsense reasoning. Techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT) have further unlocked the reasoning potential of LLMs. However, LLMs are still prone to error accumulation in multi-step reasoning—once an intermediate step goes wrong, subsequent reasoning often builds upon this mistake, leading to an incorrect final answer.

**Limitations of Prior Work**: Existing self-correction methods (such as self-refine and self-verification) allow models to review their own reasoning and correct mistakes, but they face two main issues: (1) models tend to be overconfident and reluctant to modify existing reasoning steps; (2) there is a lack of clear correction direction, meaning models do not know which parts to focus on for correction. Numerous experiments show that simple "please check your reasoning" prompts do not effectively improve accuracy, and sometimes even alter correct answers into incorrect ones.

**Key Challenge**: LLMs possess some self-review capability but lack effective "anchors" to guide the direction of correction—not knowing "where to look" is the root cause of poor self-correction efficacy.

**Goal**: To design a clue-based guidance mechanism that extracts key information from the reasoning chain as checkpoints, systematically guiding the LLM to re-assess and correct suspicious steps.

**Key Insight**: The authors observe that reasoning errors are often concentrated in specific categories—numerical calculation errors, omitted conditions, logical jumps, etc. For each error type, corresponding "clue templates" can be designed to extract crucial information and verify whether the step is correct.

**Core Idea**: To automatically extract structured clues from the reasoning chain and utilize them to guide the LLM in re-assessing each critical step, achieving targeted self-correction.

## Method

### Overall Architecture
The method consists of three phases: (1) Initial Reasoning—using CoT to let the LLM generate a complete reasoning chain; (2) Clue Extraction—extracting critical clues for each step (involved numerical values, conditions used, logical relationships) from the reasoning chain; (3) Guided Re-Assessment—using clues to guide the LLM in progressively verifying and correcting suspicious steps in the reasoning chain. All three phases are completed during inference without requiring additional training.

### Key Designs

1. **Structured Clue Extractor**:

    - **Function**: Automatically extracts key verification information from each step of the reasoning chain.
    - **Mechanism**: Defines multiple clue types—numerical clues (extracting numerical values and calculation operations in steps), conditional clues (extracting premises and assumptions used), and consistency clues (extracting consistency constraints that should be satisfied between consecutive steps). The LLM itself is utilized to extract clues, guided by structured prompt templates (e.g., "Please list all numerical operations in this step"). The extracted results are organized in JSON format for easy subsequent usage.
    - **Design Motivation**: With explicit clues, re-assessment transforms into the verification of specific information rather than a vague holistic review, significantly improving the precision of correction.

2. **Clue-Guided Step-by-Step Verification**:

    - **Function**: Performs targeted verification on each step of the reasoning chain using the extracted clues.
    - **Mechanism**: For each reasoning step, a corresponding verification strategy is selected based on its clue type. Numerical clues trigger a recalculation (prompting the LLM to independently redo the computation and compare results), conditional clues trigger a traceback check (verifying if the conditions used are provided in the question), and consistency clues trigger a front-to-back comparison (checking if the step's conclusion contradicts prior steps). Verification results are labeled as "passed" or "suspicious".
    - **Design Motivation**: Different types of errors require different verification strategies; a unified verification prompt cannot effectively cover all error types.

3. **Selective Correction with Confidence**:

    - **Function**: Only corrects steps judged as incorrect with high confidence, preventing over-correction.
    - **Mechanism**: For steps marked as "suspicious", the LLM is asked to generate a corrected version and estimate a correction confidence $c$. The correction is only adopted when $c$ exceeds a threshold $\tau$. Concurrently, multi-path verification is introduced—multiple independent verifications are performed using different prompt templates, and the step is marked as "suspicious" only when a majority consensus deems it incorrect.
    - **Design Motivation**: Avoids the "correct-to-incorrect" issue—if the model is uncertain whether a step is erroneous, keeping the original reasoning is safer than risking correction.

### Loss & Training
This method is purely an inference-time approach that requires no training and is implemented entirely through prompt engineering. It can be combined with any LLM.

## Key Experimental Results

### Main Results

| Method | GSM8K Acc↑ | MATH Acc↑ | LogiQA Acc↑ | StrategyQA Acc↑ |
|------|-----------|-----------|-------------|-----------------|
| CoT (baseline) | 78.2 | 45.8 | 52.3 | 73.1 |
| Self-Refine | 76.8 | 44.2 | 51.8 | 72.5 |
| Self-Verify | 79.5 | 47.1 | 53.9 | 74.2 |
| PHP (Progressive Hint) | 80.3 | 48.5 | 54.2 | 75.0 |
| **Clue Guided (Ours)** | **83.7** | **52.3** | **57.8** | **77.6** |

### Ablation Study

| Configuration | GSM8K | MATH | Description |
|------|-------|------|------|
| Full Method | 83.7 | 52.3 | All components |
| w/o Clue Extraction (General Verification) | 80.1 | 48.0 | Degenerates to general self-review without clues |
| w/o Step-by-Step Verification (Holistic Verification) | 81.2 | 49.5 | Holistic verification is less precise than step-by-step verification |
| w/o Selective Correction (Full Correction) | 81.5 | 47.8 | Over-correction leads to correct answers being altered to incorrect ones |
| Numerical Clues Only | 82.4 | 51.0 | Numerical verification contributes the most |
| Conditional Clues Only | 80.8 | 49.2 | Condition checking also makes a significant contribution |

### Key Findings
- Self-correction (Self-Refine) without guided clues may even decrease accuracy, as the model often miscorrects sound reasoning.
- Clue extraction contributes the most (+3.6 on GSM8K), validating that "giving the model an explicit review direction" is more effective than "allowing the model to review freely".
- The selective correction mechanism is critical—full correction actually decreases accuracy on MATH (-4.5), demonstrating that over-correction is worse than no correction.
- Numerical clues are most effective in mathematical reasoning tasks, while conditional clues are most effective in logical reasoning tasks, validating the necessity of category-based handling.

## Highlights & Insights
- The "clue-guided" concept addresses the core issue of "not knowing where to look" in self-correction methods, akin to providing a student with a grading checklist rather than merely saying "please check your work".
- The "conservative strategy" of selective correction (preferring no change to incorrect changes) is a vital engineering insight applicable to all self-correction methods.
- Multi-path verification reduces the randomness of a single verification, similar to the idea of ensemble learning.

## Limitations & Future Work
- Clue extraction and multi-path verification increase the number of LLM calls during inference, with inference costs roughly 3-5x that of CoT.
- Clue templates are currently manually designed, requiring human effort to design new templates for novel reasoning types.
- Efficacy on open-ended reasoning tasks (e.g., creative writing, argument formulation) has not yet been verified.
- Future work could explore training a specialized clue extractor to automatically learn the optimal clue types and extraction strategies.

## Related Work & Insights
- **vs Self-Refine (Madaan et al., 2023)**: Self-Refine uses generic reflection prompts, whereas this work provides more precise correction guidance via structured clues.
- **vs Self-Verification (Weng et al., 2023)**: Self-Verification checks answers via backward verification, while this work performs step-by-step verification within the reasoning chain.
- **vs Progressive-Hint (Zheng et al., 2023)**: PHP narrows the answer scope by progressively providing hints, whereas this work focuses on correcting the reasoning process rather than approximating the answer.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of clue-guided self-correction is novel, though it represents an incremental improvement in self-correction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-task evaluation, detailed ablation, and complete error analysis are provided.
- Writing Quality: ⭐⭐⭐⭐ Sufficient motivational discussion and clear methodological description.
- Value: ⭐⭐⭐⭐ Offers practical reference value for LLM reasoning enhancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)
- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[ACL 2025\] GAMEBoT: Transparent Assessment of LLM Reasoning in Games](gamebot_transparent_assessment_of_llm_reasoning_in_games.md)
- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)

</div>

<!-- RELATED:END -->
