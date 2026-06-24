---
title: >-
  [Paper Note] RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing
description: >-
  [ACL 2025][LLM Alignment][math reasoning] RISE finds that around 75% of LLM mathematical errors are subtle inner-step errors (digit substitutions, operand swaps, step omissions). By leveraging the LLM to self-edit correct solutions by injecting predefined subtle errors, it constructs high-quality hard negative samples. Combined with error-aware DPO training, this method improves performance on GSM8K by 3.0% and on MATH by 7.9% using only 4.5K samples…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "math reasoning"
  - "error injection"
  - "preference learning"
  - "DPO"
  - "hard negatives"
  - "subtle errors"
date: 2026-05-08
content_hash: 04546fe029c87c25
---

# RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing

**Conference**: ACL 2025  
**arXiv**: [2410.06638](https://arxiv.org/abs/2410.06638)  
**Code**: [available](https://github.com)  
**Area**: LLM Alignment / Mathematical Reasoning  
**Keywords**: math reasoning, error injection, preference learning, DPO, hard negatives, subtle errors

## TL;DR

RISE finds that around 75% of LLM mathematical errors are subtle inner-step errors (digit substitutions, operand swaps, step omissions). By leveraging the LLM to self-edit correct solutions by injecting predefined subtle errors, it constructs high-quality hard negative samples. Combined with error-aware DPO training, this method improves performance on GSM8K by 3.0% and on MATH by 7.9% using only 4.5K samples, while generalizing to logical reasoning and code generation.

## Background & Motivation

**Bottleneck of LLM Mathematical Reasoning**: Although LLMs have made significant progress in mathematical reasoning, they still make frequent errors. The key question is: what types of errors do LLMs actually make? Understanding the error types is crucial for targeted improvements.

**Key Findings in Error Classification**: The authors analyzed the error distribution of LLMs on the MATH dataset and found that approximately 75% of the errors are subtle inner-step errors (miscalculations, incorrect substitutions), rather than logical jumps between steps or misunderstandings of the questions. This challenges the general assumption that "LLMs lack reasoning capabilities."

**Limitations of Rejection Sampling**: The standard approach for preference learning is using rejection sampling to generate negative samples—sampling the model's outputs multiple times, taking the correct ones as chosen and the incorrect ones as rejected. However, the resulting negative samples often differ too greatly from the positive samples, allowing the model to easily learn "shortcuts" to distinguish them rather than truly understanding the reasoning logic.

**Demand for Hard Negative Samples**: Effective preference learning requires "hard negative samples"—incorrect solutions that have only minor differences from the correct answer. This forces the model to learn precise reasoning steps rather than superficial features.

**Non-scalability of Manual Construction**: Manually constructing solutions with subtle errors for each math problem is time-consuming and requires domain expertise, making it impossible to scale to large training datasets.

**Idea of Automated Error Injection**: Leveraging the LLM itself as an editor to inject subtle errors into correct solutions according to predefined error types (REPLACE/SWAP/DELETE), and using Levenshtein distance to control the degree of "subtlety," thereby generating hard negative samples at a low cost and on a large scale.

## Method

### Overall Architecture

The workflow of RISE is divided into three stages: (1) **Error Analysis**: classifying and analyzing LLM mathematical errors to determine the primary error types; (2) **Error Injection**: instructing the LLM to perform predefined editing operations on each reasoning step of correct solutions to generate negative samples with subtle errors; (3) **Error-Aware DPO Training**: combining step-level and sequence-level preference learning with adaptive NLL to prevent probability collapse.

### Key Designs

**1. Subtle Error Classification System**

- **Function**: Classifies LLM mathematical reasoning errors into two main categories: inter-step and inner-step, and further refines them into sub-types.
- **Mechanism**: Inner-step errors include REPLACE (replacing correct numbers/variables with incorrect values), SWAP (swapping operand positions, such as changing $a-b$ to $b-a$), and DELETE (omitting certain calculation steps), while inter-step errors include logical jumps, irrelevant reasoning, etc.
- **Design Motivation**: Since 75% of errors are subtle inner-step errors, constructing training data targeting the primary error types is the most efficient way to improve the model.

**2. LLM Self-Editing Error Injection**

- **Function**: Uses the LLM itself as an editor to inject specific types of subtle errors into designated steps of correct solutions based on instructions.
- **Mechanism**: The LLM is provided with the correct solution, the target step number, and the error type (REPLACE/SWAP/DELETE) to generate the revised solution. A prompt is used to guide the LLM to modify only the target step and correctly propagate the error to subsequent steps.
- **Design Motivation**: LLMs inherently understand the structure of mathematical reasoning, allowing them to generate solutions that "look reasonable but contain subtle errors"; self-editing is more controllable than generating from scratch.

**3. Levenshtein Distance Filtering**

- **Function**: Uses edit distance to measure the degree of difference between the post-injection solution and the original correct solution, filtering out samples with excessive differences.
- **Mechanism**: Computes character-level Levenshtein distance, keeping only modifications within a certain threshold to ensure that the negative samples are indeed "subtle."
- **Design Motivation**: Since LLMs might accidentally change too much content during editing, filtering guarantees the quality of the hard negative samples.

**4. Subtle Error-Aware DPO (RISE-DPO)**

- **Function**: Designs a specialized preference learning strategy to utilize the constructed (correct solution, subtle error solution) pairs.
- **Mechanism**: A combination of three losses: (a) DPO-Edit: step-level preference learning, constructing preference pairs at the error step; (b) DPO-Full: sequence-level preference learning, pairing full correct/incorrect solutions; (c) Adaptive NLL loss: adding NLL regularization on the correct solution to prevent probability collapse due to over-rejection.
- **Design Motivation**: Step-level DPO allows the model to precisely locate the error position, sequence-level DPO provides global preference signals, and NLL prevents training instability.

### Loss & Training

The total loss is a weighted sum of three terms: DPO-Edit (step-level preference) + DPO-Full (sequence-level preference) + NLL (regularization). The weight of NLL is adaptively adjusted during training. The training data consists of only 4.5K samples, constructed based on the correct solutions of GSM8K and MATH.

## Key Experimental Results

### Main Results

| Model | Method | GSM8K | MATH | Training Data Size |
|------|------|-------|------|-----------|
| Qwen2-7B | SFT | 82.3% | 45.6% | — |
| Qwen2-7B | DPO (rejection sampling) | 83.1% | 47.2% | 4.5K |
| Qwen2-7B | **RISE** | **86.1%** | **55.1%** | **4.5K** |
| Qwen2-7B | Gain vs SFT | **+3.0%** | **+7.9%** | — |
| LLaMA-3.1-8B | SFT | 79.8% | 38.4% | — |
| LLaMA-3.1-8B | DPO (rejection sampling) | 80.5% | 39.1% | 4.5K |
| LLaMA-3.1-8B | **RISE** | **83.7%** | **41.1%** | **4.5K** |
| LLaMA-3.1-8B | Gain vs SFT | **+3.9%** | **+2.7%** | — |

### Ablation Study

| Ablation | GSM8K | MATH | Analysis |
|--------|-------|------|------|
| Full RISE | 86.1% | 55.1% | Baseline |
| w/o DPO-Edit (DPO-Full only) | 84.5% | 51.8% | Step-level preference contributes significantly |
| w/o DPO-Full (DPO-Edit only) | 85.2% | 53.4% | Sequence-level preference provides complementary signals |
| w/o NLL Regularization | 84.8% | 52.1% | Probability collapse leads to performance degradation |
| Random Errors (non-subtle) | 83.5% | 48.3% | Hard negative samples from subtle errors are crucial |
| REPLACE errors only | 85.4% | 53.7% | REPLACE is the most effective single error type |
| SWAP errors only | 84.9% | 52.8% | Operand swap is the second most effective |
| DELETE errors only | 84.2% | 51.5% | Step omission is the least effective |

### Key Findings

- Hard negative samples with subtle errors are much more effective than "easy" negative samples obtained via rejection sampling—improving MATH gains from +1.6% (DPO) to +7.9% (RISE).
- While the effects of the three error types (REPLACE/SWAP/DELETE) vary, combining them yields the best results, indicating that a diverse set of subtle errors can cover a more comprehensive range of reasoning weaknesses.
- Step-level DPO-Edit and sequence-level DPO-Full are complementary rather than redundant—with the step-level offering precise error-localization signals and the sequence-level offering global quality judgments.
- The model trained with RISE also shows improvements in logical reasoning (LogiQA +2.1%) and code generation (HumanEval +1.8%), demonstrating that the ability to "discern subtle errors" is transferable.
- Achieving such significant improvements with only 4.5K training samples indicates that data quality is far more important than data quantity.

## Highlights & Insights

- **Error-Classification-Driven Method Design**: Everything in RISE stems from the empirical finding: "75% of errors are subtle inner-step errors"—placing data analysis before method design. This research paradigm is worth learning from.
- **Ingenuity of Self-Editing**: Instead of manually writing error cases, the LLM is instructed to perform controllable editing on correct solutions—since models are best at knowing what kind of errors "look reasonable."
- **Levenshtein Distance as a Metric of Subtlety**: A simple but effective method of quality control, ensuring that negative samples do not degrade into random sampling.
- **Extremely High Data Efficiency**: 4.5K samples yielding a +7.9% boost on MATH shows that a good preference data construction strategy can leverage massive performance improvements with very little data.

## Limitations & Future Work

- The error classification system is based on manual analysis and may miss certain error patterns.
- The quality of LLM self-editing relies on the foundational model's editing capability—weaker models might generate unnatural subtle errors.
- The method has only been thoroughly validated on mathematical reasoning, while the generalization experiments on logical reasoning and coding are relatively preliminary.
- It has not been validated in combination with newer preference learning methods (e.g., KTO, ORPO).
- The Levenshtein distance threshold is manually set; automated threshold selection could be explored further.
- Error injection is concentrated at numerical and operational levels, lacking coverage for more abstract reasoning errors (such as conceptual confusion).

## Related Work & Insights

- **vs. Rejection Sampling DPO**: Standard methods obtain positive/negative samples through multiple sampling, but the negative samples usually differ too much from the positive ones, allowing the model to easily distinguish them. In contrast, RISE's hard negatives force the model to learn fine-grained reasoning capabilities.
- **vs. Step-DPO**: Step-DPO performs preference learning at the step level, but negative samples still come from sampling. RISE provides more precise step-level preference signals using constructed subtle errors.
- **vs. Math Verifiers (ORM/PRM)**: Verifiers judge correctness externally. RISE enables the generator itself to learn how to discern subtle errors from the perspective of training data. The two are complementary.
- **Insights**: The concept of error injection can be extended to other scenarios requiring "fine-grained discrimination"—such as fact-checking (injecting subtle factual errors) or code debugging (injecting subtle bugs).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of error injection to construct hard negative samples is unique and empirically supported.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model + cross-task generalization + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Convincing error analysis with clear motivation.
- Value: ⭐⭐⭐⭐⭐ Highly efficient (4.5K samples) + high effectiveness + scalable ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)
- [\[ACL 2025\] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)

</div>

<!-- RELATED:END -->
