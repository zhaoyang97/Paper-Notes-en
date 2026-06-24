---
title: >-
  [Paper Note] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation
description: >-
  [ACL 2025][LLM Alignment] Retrieval Preference Optimization (RPO) is proposed, a lightweight preference alignment method specifically designed for RAG. By implicitly integrating retrieval quality evaluation into the generation process, it enables LLMs to adaptively choose between parametric and retrieved knowledge, mitigating hallucination issues caused by knowledge conflicts without requiring additional components.
tags:
  - "ACL 2025"
  - "LLM Alignment"
date: 2026-05-08
content_hash: edbc38510bec06fb
---

# RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation


**Conference**: ACL 2025  
**arXiv**: [2501.13726](https://arxiv.org/abs/2501.13726)  
**Authors**: Shi-Qi Yan, Quan Liu, Zhen-Hua Ling (USTC / iFLYTEK)
**Code**: Not publicly available  
**Area**: LLM Alignment / RAG

## TL;DR

Retrieval Preference Optimization (RPO) is proposed, a lightweight preference alignment method specifically designed for RAG. By implicitly integrating retrieval quality evaluation into the generation process, it enables LLMs to adaptively choose between parametric and retrieved knowledge, mitigating hallucination issues caused by knowledge conflicts without requiring additional components.

## Background & Motivation

**Core Problem**: In RAG, when the retrieved non-parametric knowledge conflicts with the LLM's internal parametric knowledge, the model tends to blindly trust the retrieved content, leading to degradation in generation quality.

**Limitations of Prior Work**:

1. **Pre-Eval Methods** (e.g., CRAG): Evaluate retrieval quality using an external classifier/LLM prior to generation, requiring multiple API calls and incurring high computational overhead (~6 calls).
2. **Post-Eval Methods** (e.g., Self-RAG): Independently generate multiple answers based on multiple retrieved documents and then select the best, resulting in high inference costs (2-11 calls).
3. Both types of methods remove certain information, making the generator more dependent on the evaluator, which affects final performance.

**Three Mathematical Obstacles of Applying DPO Directly to RAG**:
1. The optimization objective of RLHF/DPO is inconsistent with the conflict mitigation objective of RAG—the KL divergence constraint prevents recovery from over-reliance on retrieval.
2. When inputs for preferred/dispreferred answers differ (with/without retrieval), the partition function of DPO cannot be canceled out, making the loss function incomputable.
3. Forging parametric knowledge answers to bypass the partition function introduces likelihood discrepancies, leaving the model still prone to selecting non-parametric answers after training.

## Method

### Overall Architecture

RPO is a two-stage training workflow:

**Phase 1: Supervised Fine-Tuning (SFT)**
- For each question x, generate answers with and without retrieval separately.
- Filter samples where knowledge conflict occurs (only one of the retrieved or non-retrieved answers is correct).
- Perform SFT on the model using the correct answers to initially activate the model's retrieval quality awareness.

**Phase 2: RPO Preference Optimization**
- Regenerate answers using the fine-tuned SFT model and filter conflicting samples again.
- Construct preference pairs, labeling which answer is superior.
- Perform alignment training using the RPO loss.

### Key Designs

The core innovation lies in the introduction of a **retrieval reward term** in the reward model. The RPO loss function consists of three parts:

1. **Preferred Generation Reward** (consistent with DPO): Rewards preferred answers.
2. **Dispreferred Generation Reward** (consistent with DPO): Penalizes dispreferred answers.
3. **Retrieval Reward** (unique to RPO): Positive sign when the non-parametric answer is superior (encouraging retrieval usage), and negative sign when the parametric answer is superior (suppressing retrieval reliance).

Key Designs:
- The sign of the retrieval reward term adaptively flips depending on which knowledge source is more accurate.
- Length normalization is performed by dividing by the length of the retrieved document to eliminate bias caused by document length.
- Retrieval relevance is implicitly represented via the model's conditional probability of the retrieved document, eliminating the need for an external evaluator.

### Data Construction Strategy

The training data consists of two complementary subsets:
- **D1 (Enhancing Retrieval Utilization)**: Samples where the parametric knowledge answer is incorrect but the non-parametric knowledge answer is correct.
- **D2 (Mitigating Retrieval Over-reliance)**: Samples where the parametric knowledge answer is correct but the answer becomes incorrect after being distracted by retrieved knowledge.

## Key Experimental Results

### Table 1: Main Results (Accuracy %)

| Method | Adaptive Category | API/LM Calls | PopQA | NQ | TriviaQA | RGB |
|--------|-------------|-----------|-------|------|----------|------|
| RAG (LLaMA2-7B) | - | 1 | 48.8 | 22.0 | 52.5 | 91.6 |
| RAG+DPO | - | 1 | 53.6 | 43.5 | 51.7 | 96.3 |
| CRAG | Pre-Eval | 6 | 54.9 | 38.4 | 59.6 | 92.0 |
| Self-RAG | Post-Eval | 2-11 | 54.9 | 42.4 | 68.9 | 92.6 |
| **RPO (LLaMA2)** | Integrated | **1** | **55.8** | **45.3** | 57.6 | **97.3** |
| RAG (LLaMA3-8B) | - | 1 | 59.0 | 41.3 | 65.8 | 96.3 |
| InstructRAG | - | 1 | 65.0 | 46.7 | 65.1 | 99.3 |
| **RPO (LLaMA3)** | Integrated | **1** | **65.4** | **51.9** | **74.4** | **100.0** |

**Key Findings**: RPO improves over the RAG baseline on LLaMA3 by 4-10% (PopQA +6.4%, NQ +10.6%, TriviaQA +8.6%, RGB +3.7%) while requiring only 1 inference call.

### Table 2: Ablation Study (Accuracy %)

| Variant | PopQA | NQ | TriviaQA | RGB |
|------|-------|------|----------|------|
| **RPO (Full)** | **55.8** | **45.3** | **57.6** | **97.3** |
| RPO w/o Retrieval Reward (=DPO) | 53.6 | 43.5 | 51.7 | 96.3 |
| RPO w/o Preference Optimization (=SFT) | 51.3 | 36.0 | 54.3 | 94.6 |
| RPO w/o SFT | 52.5 | 34.9 | 50.1 | 90.6 |

The retrieval reward term contributes +2.2 (PopQA), and preference optimization contributes +4.5 (PopQA); both stages are indispensable.

### Table 3: Robustness to Low-Quality Retrieval (When all retrieved information is incorrect)

| Method | Accuracy | Relative Gain |
|--------|----------|---------|
| RAG | 18.6 | 0.0% |
| SFT | 19.5 | +4.8% |
| DPO | 19.3 | +3.7% |
| **RPO** | **23.5** | **+26.3%** |

RPO maintains a significant advantage even in environments where all retrieval is incorrect, proving that the model has truly learned to evaluate retrieval quality rather than simply favoring one side.

## Highlights & Insights

1. **Solid Theoretical Contribution**: Mathematically proved three major limitations of applying DPO directly to RAG (inconsistent optimization objectives, in-eliminable partition functions, and over-reliance on non-parametric knowledge), providing theoretical support for the method design.
2. **Extremely High Inference Efficiency**: Integrates retrieval evaluation and generation into a single inference pass, avoiding multiple API calls required by Pre-Eval/Post-Eval methods.
3. **Ingenious Retrieval Reward Design**: By implicitly representing retrieval relevance and adaptively adjusting based on the sign, it seamlessly handles both "helpful retrieval" and "harmful retrieval" scenarios.

## Limitations & Future Work

1. **Restricted Training Data Domain**: Only trained on NQ, which may limit generalization to other domains.
2. **Under-explored Reward Function Design Space**: The authors acknowledge that more optimal reward function formulations may exist.
3. **Implicit Retrieval Evaluation**: Lacks explicit interpretability of how the model weighs the two knowledge sources during generation.
4. **Reliance on Conflicting Sample Filtering**: Data construction requires multiple forward passes to detect conflicts, increasing training costs.
5. **No Comparison with Larger-Scale Models**: Experiments are validated only on 7B/8B models.

## Related Work & Insights

- **Adaptive RAG**: CRAG (Pre-Eval, retrieval correction), Self-RAG (Post-Eval, self-reflective generation and critique), AstuteRAG (iterative knowledge selection)
- **Model Alignment**: RLHF/PPO -> DPO (simplified to closed-form solution) -> RPO (alignment designed specifically for RAG)
- **Knowledge Conflict**: Longpre et al. 2021 study entity-level conflicts; Zou et al. 2024 and Xiang et al. 2024 study conflict tendencies in RAG.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to mathematically demonstrate that DPO is not suitable for RAG, proposing a retrieval-aware preference optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets + 2 base models + comprehensive ablation and robustness analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations and complete structure.
- Value: ⭐⭐⭐⭐ — Solves key pain points of knowledge conflicts in RAG, with a highly practical and efficient method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2025\] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](synthesizeme_persona_prompts.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)

</div>

<!-- RELATED:END -->
