---
title: >-
  [Paper Note] Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context
description: >-
  [Robotics] This work proposes the DICE dataset (2066 sentences, 402 idioms) to reveal systematic flaws in LLMs when contextual understanding is required to disambiguate idioms (literal vs. figurative meanings), achieved through highly controlled contrastive evaluation with identical idiom forms.
tags:
  - "Robotics"
date: 2026-05-08
content_hash: 37749eda62798e9b
---

# Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context

| Information | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2410.16069](https://arxiv.org/abs/2410.16069) |
| Code | [github.com/mi-m1/dice](https://github.com/mi-m1/dice) |
| Area | NLP Understanding |
| Keywords | Idiom Understanding, Contrastive Evaluation, Contextual Disambiguation, LLM Evaluation, DICE |

## TL;DR

This work proposes the DICE dataset (2066 sentences, 402 idioms) to reveal systematic flaws in LLMs when contextual understanding is required to disambiguate idioms (literal vs. figurative meanings), achieved through highly controlled contrastive evaluation with identical idiom forms.

## Background & Motivation

**Research Question:** LLMs perform exceptionally well on idiom detection benchmarks, but does this success stem from true contextual understanding, or is it merely exploiting surface-level shortcuts in the datasets?

**Limitations of Prior Work:** In existing idiom disambiguation datasets (such as MAGPIE), literal usage is often realized by modifying the grammatical structure of the idiom (e.g., passivization, inserting modifiers). Consequently, models can make judgments based on surface cues (grammatical changes) rather than contextual understanding. For instance, the literal usage of "kick the bucket" is typically written as "The bucket was kicked by him", where the model only needs to detect the passive voice to classify it as literal.

**Core Hypothesis:** If models truly rely on contextual understanding, they should perform consistently on both literal and figurative usages of the same idiom; if models rely on memorization, they will exhibit a bias towards the figurative meaning.

## Method

### Overall Architecture

The core design principle of DICE (Dataset for Idiomatic Contrastive Evaluation) is to **keep the idiom form completely identical**, switching between literal and figurative meanings solely by altering the context, thereby forcing models to rely on contextual understanding for disambiguation.

### Key Designs

**1. Expression Selection:** By cross-matching from MAGPIE and SLIDE (phrasal idioms) as well as NCTTI and AStitchInLanguageModels (compound noun idioms), 402 idioms (299 phrasal expressions + 103 compound nouns) were selected, covering a scale far exceeding previous single-type datasets.

**2. Sentence Generation and Quality Assurance:**
- GPT-4 was used to generate sentences containing the idiom in a literal context (suppressing the figurative meaning), with 3 sentences per idiom.
- Validated by 4 linguistics experts (Cohen's $\kappa = 0.95$), retaining only samples where the figurative meaning was precisely suppressed.
- Figurative sentences were directly extracted from MAGPIE/AStitchInLanguageModels.
- **Strict Balance**: The number of literal and figurative sentences for each idiom is equal.

**3. Three-Tier Evaluation Scheme:**
- **Accuracy**: Computes classification accuracy separately on the literal and figurative subsets.
- **Lenient Consistency**: Whether the model can consistently make correct predictions across all literal/figurative instances of the same idiom.
- **Strict Consistency**: The most stringent metric—the model must correctly classify all variants of the same idiom across both contexts simultaneously.

**4. Frequency and Likelihood Analysis:** Idiom frequency was estimated using the enTenTen corpus (52 billion words), while exploring how sentence likelihood assigned by models correlates with performance.

## Experiments

### Main Results (Zero-shot)

| Model | Figurative Accuracy | Literal Accuracy | Overall Accuracy | Strict Consistency |
|------|-----------|-----------|-----------|-------------------|
| Llama 3.1 (405B) | 88.63% | 88.25% | 88.45% | 60.36% |
| GPT-4o | 87.05% | 87.30% | 84.33% | 48.59% |
| Llama 3 (70B) | 87.72% | 86.13% | 87.00% | 57.55% |
| Llama 3 (8B) | 79.27% | 74.01% | 76.91% | 33.83% |
| GPT-3.5 Turbo | 79.05% | 70.02% | 75.54% | 32.84% |
| Flan-T5-XXL (11B) | 77.18% | 74.91% | 76.40% | 32.92% |
| Flan-T5-Small (80M) | 0.51% | 66.72% | 50.13% | 0.00% |

### One-shot Results Comparison

| Model | Zero-shot Overall | One-shot Overall | Strict (0-shot) | Strict (1-shot) |
|------|---------------|---------------|-----------------|-----------------|
| GPT-4o | 84.33% | 89.72% | 48.59% | 63.52% |
| Llama 3.1 (405B) | 88.45% | 89.53% | 60.36% | 63.27% |
| Flan-T5-XXL | 76.40% | 52.79% | 32.92% | 1.49% |

### Key Findings

- **Sharp Decline from Accuracy to Strict Consistency:** Even the strongest model, Llama 3.1 (405B), only achieves 60.36% strict consistency, indicating that models cannot reliably handle both usages of the same idiom simultaneously.
- **Systemic Bias Towards Figurative Meaning:** For most models, figurative consistency under Lenient Consistency is significantly higher than literal consistency, suggesting that models tend to default to the figurative interpretation when encountering an idiom.
- **GPT-4o's High Accuracy is Sophistic:** Its 84.33% accuracy masks a low strict consistency of only 48.59%, implying its success stems more from broad coverage than deep understanding.
- **Limited and Inconsistent Help from One-shot:** While GPT-4o and Llama 3.1 benefit from one-shot learning, the Flan-T5 series deteriorates significantly.
- **Frequency is Not a Silver Bullet:** Highly frequent idioms are more likely to be correctly recognized, but there is a performance trade-off between literal and figurative settings.
- **Sentence Likelihood Correlates Positively with Performance:** Models perform better on sentences they assign higher likelihood, hinting at a reliance on distribution matching rather than semantic comprehension.

## Highlights & Insights

- The first contrastive evaluation dataset with strictly controlled idiom form consistency, entirely blocking surface cue shortcuts.
- Covers both phrasal idioms and compound noun idioms, with a scope far exceeding existing datasets.
- A three-tier evaluation system (Accuracy $\rightarrow$ Lenient $\rightarrow$ Strict) that gradually unmasks the models' pseudo-capabilities.
- Integrates frequency and likelihood analyses to explain the successes and failures of models from multiple perspectives.

## Limitations & Future Work

- Literal sentences are generated by GPT-4, which may introduce distributional bias (GPT-4's performance on this dataset should be interpreted with caution).
- Figurative sentences are sourced from existing datasets, and their average length (28.1 words) is significantly longer than that of literal sentences (15.4 words); this length discrepancy itself might affect model judgments.
- Evaluated only on English idioms; generalizability to other languages remains unknown.
- Did not explore mitigation strategies to improve model idiom understanding (e.g., targeted fine-tuning).

## Related Work & Insights

- **Idiom Disambiguation Datasets:** MAGPIE (56K samples, allowing form variations), VNC-Tokens, IDIX, SemEval-2013, AStitchInLanguageModels, IdioTS.
- **Contrastive Evaluation Paradigm:** Isolating specific language capabilities (e.g., grammatical judgment, semantic understanding) via minimal contrastive pairs.
- **LLMs and Memorization:** Li et al. (2022) and Coil & Shwartz (2023) found that GPT-3's processing of idioms relies predominantly on memorization rather than reasoning.
- **Context vs. Memorization:** Cheng & Bhat (2024) discovered that removing contextual information actually improves model performance on idiom reasoning.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Total Score | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning to Stop: Deep Learning for Mean Field Optimal Stopping](../../ICML2025/robotics/learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)
- [\[ACL 2025\] Vulnerability of LLMs to Vertically Aligned Text Manipulations](vulnerability_of_llms_to_vertically_aligned_text_manipulations.md)
- [\[ACL 2025\] SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations](self_percept_manipulation_detection.md)
- [\[ICLR 2026\] When would Vision-Proprioception Policies Fail in Robotic Manipulation?](../../ICLR2026/robotics/when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](../../NeurIPS2025/robotics/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)

</div>

<!-- RELATED:END -->
