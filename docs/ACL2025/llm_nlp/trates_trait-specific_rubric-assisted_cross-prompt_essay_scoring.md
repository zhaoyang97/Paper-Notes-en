---
title: >-
  [Paper Note] TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring
description: >-
  [ACL2025][LLM (Other)][Automated Essay Scoring] This work proposes the TRATES framework, redefining the role of LLMs in automated essay scoring (AES) from direct scorers to **trait-specific feature generators and extractors**. TRATES leverages LLMs to automatically convert grading rubrics into assessment questions (sub-traits). By combining these with general writing quality features and prompt-specific features, a regression model is trained. TRATES achieves SOTA across all…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Automated Essay Scoring"
  - "Cross-Prompt Generalization"
  - "Trait Scoring"
  - "Rubrics"
  - "LLM Feature Generation"
date: 2026-05-08
content_hash: 2885e93b7670e6e7
---

# TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring

**Conference**: ACL2025  
**arXiv**: [2505.14577](https://arxiv.org/abs/2505.14577)  
**Code**: [GitHub](https://github.com/Sohaila-se/TRATES)  
**Area**: LLM/NLP  
**Keywords**: Automated Essay Scoring, Cross-Prompt Generalization, Trait Scoring, Rubrics, LLM Feature Generation  

## TL;DR

This work proposes the TRATES framework, redefining the role of LLMs in automated essay scoring (AES) from direct scorers to **trait-specific feature generators and extractors**. TRATES leverages LLMs to automatically convert grading rubrics into assessment questions (sub-traits). By combining these with general writing quality features and prompt-specific features, a regression model is trained. TRATES achieves SOTA across all 8 traits on the ASAP dataset and establishes the first cross-prompt trait scoring baseline on the ELLIPSE dataset.

## Background & Motivation

**Long-term AES research bias toward holistic scoring**: Since Page (1966), holistic scoring has been dominant. Trait scoring (which assesses multiple dimensions like organization, vocabulary, sentence fluency, etc.) remains under-investigated due to its complexity, despite providing more actionable feedback to help students improve.

**Cross-prompt setting is closer to reality**: In real-world scenarios, models must generalize to unseen writing prompts. The significant variations in writing styles, topics, and structures across different prompts make this far more challenging than single-prompt scoring.

**Poor performance of direct LLM scoring**: Prior studies attempting to let GPT-4/GPT-3.5 directly score essays show that zero-shot scoring performs worse than a simple XGBoost baseline, suffering from scoring inconsistency and hallucinations.

**LLM conversational strategies fall short of baselines**: Multi-turn dialogue scoring (such as impersonation or CoT), though improved, still lags behind traditional feature engineering methods. This indicates a fundamental bottleneck in the direct scoring paradigm.

**Existing cross-prompt methods lack rubric utilization**: SOTA methods like ProTACT and Li & Ng rely on hand-crafted features or neural architectures but do not incorporate rubrics into feature design, missing critical domain priors.

**Automatic generation of trait-specific features remains a gap**: Different traits require different evaluative perspectives. Hand-crafting features for each trait is time-consuming and non-scalable, highlighting the need for an automated and general feature generation method.

## Method

### Overall Architecture: Three-Stage Hybrid Scoring Pipeline

- **Function**: Builds a unified framework to perform cross-prompt automated scoring for any writing trait.
- **Design Motivation**: Combines the powerful text analysis capability of LLMs with the stability of traditional feature engineering to mitigate the unreliability of direct LLM-based scoring.
- **Mechanism**: (1) Uses an LLM to generate trait-specific assessment questions from rubrics; (2) Uses the same LLM to answer these questions for each essay to extract trait features; (3) Concatenates trait features, prompt-specific features, and general writing quality features to train a shallow neural network regression model for score prediction.

### Key Designs 1: Rubric-Based Feature Generation

- **Function**: Automatically converts rubrics into a set of answerable assessment questions, where each question corresponds to a sub-trait.
- **Design Motivation**: Asking an LLM to directly evaluate an entire trait is too broad and unstable. Decomposing the rubric into fine-grained sub-questions enables more precise and interpretable assessments, with the same template applicable to any trait.
- **Mechanism**: Feeds the LLM with the trait name and rubric text, using a unified prompt template to generate a set of high/medium/low-tier assessment questions. For instance, the "Organization" trait might generate questions like "How strong are the logical transitions between paragraphs?". Different LLMs generate varying numbers of features: Gemma averages 8.6 features/trait, while Llama averages 20 features/trait.

### Key Designs 2: Trait Feature Extraction and Multi-Source Feature Fusion

- **Function**: Uses the LLM to answer the generated questions one by one for each essay, numericalizing the answers (high/medium/low $\to$ 3/2/1) and fusing them with other features.
- **Design Motivation**: While LLM-extracted features (LLM-F) possess predictive power, they alone are insufficient to achieve SOTA. Prompt-specific features (essay genre, expected length, grade level, etc.) and general writing quality features (length, readability, lexical variation, syntactic complexity, sentiment) are required as complements.
- **Mechanism**: Prompt-specific features (4 dimensions) are extracted from dataset metadata. General features (81 dimensions across 5 categories) cover length, readability, POS tag variation, syntactic complexity, and sentiment. All features are concatenated and fed into a shallow neural network regression model trained via leave-one-prompt-out cross-validation. Feature normalization uses the min-max values from the training set to avoid assumptions about the test set distribution.

### Key Designs 3: Cross-Prompt Score Scaling

- **Function**: Maps scores from different prompts with varying score ranges to a unified scale.
- **Design Motivation**: Different grades and essay genres have different score ranges (e.g., 0-6 vs 0-4). Simple min-max normalization ignores grade differences—a perfect score in grade 8 represents a completely different quality standard than a perfect score in grade 12.
- **Mechanism**: Proposes a grade-based incremental scaling method: using the highest grade as the anchor, the maximum score of lower grades is sequentially shifted down by one level, ensuring that scores across different grades are comparable on a unified scale. The predicted outputs are then reverse-scaled back to their original ranges for evaluation.

## Key Experimental Results

### Table 1: QWK Performance on ASAP Dataset (Across 8 Traits)

| Model | ORG | WC | SF | PA | NAR | LNG | CNV | CNT | Avg |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| ProTACT (Prev. SOTA) | 0.518 | 0.599 | 0.585 | 0.619 | 0.639 | 0.596 | 0.450 | 0.596 | 0.575 |
| Li & Ng (Prev. SOTA) | 0.478 | 0.459 | 0.452 | 0.617 | 0.637 | 0.556 | 0.439 | 0.592 | 0.529 |
| LLM-D (Gemma Zero-Shot) | 0.345 | 0.375 | 0.390 | 0.337 | 0.382 | 0.337 | 0.263 | 0.326 | 0.344 |
| LLM-F (LLM Features Only, Gemma) | 0.329 | 0.546 | 0.456 | 0.533 | 0.525 | 0.412 | 0.429 | 0.546 | 0.472 |
| **TRATES (Starling)** | 0.518 | 0.593 | **0.612** | **0.624** | **0.668** | **0.608** | **0.501** | **0.636** | **0.595** |
| **TRATES (Gemma)** | **0.547** | **0.622** | **0.612** | 0.599 | 0.600 | 0.521 | 0.556 | **0.632** | **0.586** |

**Key Findings**:
- TRATES establishes a new SOTA across all 8 traits. The Starling variant outperforms the previous SOTA on 6/8 traits, while the Gemma variant achieves this on 5/8.
- Direct LLM scoring (LLM-D) is on average 9 points lower than the LLM feature-based model (LLM-F), demonstrating that LLMs are unsuitable for direct scoring but excel as feature extractors.
- Gemma generates the fewest but most precise features (8.6 on average) and performs best in the LLM-F-only experiments. However, Starling stands out in the full TRATES framework, indicating greater complementarity between its features and general features.

### Table 2: Ablation Study - QWK Decline After Excluding a Single Feature Category

| Feature Category | Avg Size | ORG | CNT | AVG |
|----------|---------|-----|-----|-----|
| Trait-Specific Features | 18.2 | 2.23 | 8.35 | **7.60** |
| Prompt-Specific Features | 4 | 4.57 | 5.28 | 3.14 |
| Length Features | 16 | 3.39 | 3.42 | 2.29 |
| Readability Features | 12 | 0.97 | 2.58 | 1.68 |
| Text Complexity | 5 | 1.17 | 2.47 | 1.79 |
| Text Variation | 43 | 7.27 | 0.10 | 1.67 |
| Sentiment Features | 5 | 2.01 | 0.23 | 1.22 |

**Key Findings**: Trait-specific features represent the most important feature category; removing them drops the average QWK by 7.60 points, and they are the most significant category across all traits except ORG. Notably, this is the only automatically generated feature category, while all other categories require manual engineering.

### Table 3: Generalization Experiments on ELLIPSE Dataset (44 Prompts, Starling)

| Model | COH | SYN | VOC | GRM | CNV | PHR | Avg |
|------|-----|-----|-----|-----|-----|-----|-----|
| ProTACT' | 0.33 | 0.35 | 0.42 | 0.29 | 0.36 | 0.36 | 0.35 |
| GP-F (General + Prompt Features) | 0.45 | 0.49 | 0.48 | 0.40 | 0.50 | 0.46 | 0.46 |
| **TRATES** | **0.52** | **0.54** | **0.52** | **0.51** | **0.56** | **0.53** | **0.53** |

TRATES achieves the best performance across all traits on ELLIPSE, outperforming GP-F by at least 6.5 points, which demonstrates the strong generalization capability of the framework.

## Highlights & Insights

1. **Paradigm Innovation**: Shifts the role of LLMs from "direct scorers" to "feature generators and extractors", avoiding the instability of zero-shot LLM scoring while still leveraging their powerful text comprehension capabilities.
2. **General and Adaptive**: The same framework can be applied to any trait simply by replacing the grading rubrics; LLMs automatically generate corresponding sub-trait questions, eliminating the need for manual feature design.
3. **High Interpretability**: The generated assessment questions can serve directly as feedback for students, pointing out specific sub-dimensions that need improvement.

## Limitations & Future Work

1. **Only Tested on 7-9B Small Models**: The potential of larger LLMs (e.g., 70B+) to generate higher-quality trait features remains unexplored.
2. **No Extension to Holistic Scoring**: Holistic scoring criteria are typically highly prompt-dependent; it remains uncertain whether TRATES can be successfully applied to this setup.
3. **Intuition-Based Score Scaling**: The mapping of scores across different rubrics is subjectively determined, lacking a theoretical foundation or an automated approach.
4. **Generation Quality Heavily Dependent on Rubric Quality**: Low-quality or ambiguous grading rubrics lead to suboptimal generated assessment questions.
5. **Inference Latency Dominated by LLMs**: Extracting trait features takes 2-7 seconds per essay per trait; while acceptable, this could become a bottleneck in large-scale assessments.

## Related Work & Insights

| Dimension | TRATES | ProTACT (Do et al. 2023) | Li & Ng (2024b) |
|------|--------|--------------------------|-----------------|
| LLM Utilization | Feature Generation & Extraction | No LLM | No LLM |
| Trait-Specific Features | ✔ Automatically Generated | ✘ | ✘ |
| Rubric Utilization | ✔ Core Component | ✘ | ✘ |
| ASAP Avg QWK | **0.595** | 0.575 | 0.529 |
| ELLIPSE Evaluation | ✔ (First Time) | ✘ | ✘ |
| Interpretability | Sub-trait questions as feedback | Low | Medium |

- **vs Direct LLM Scoring (Yancey et al. 2023; Mansour et al. 2024)**: Zero-shot/few-shot scoring using GPT-4 falls short of the XGBoost baseline. By shifting the role of LLMs from active scorers to feature extractors, TRATES avoids the instability of direct scoring entirely, making it the first LLM-integrated AES system to surpass SOTA.
- **vs Multi Trait Specialization (Lee et al. 2024)**: This method uses multi-turn dialogue for holistic scoring but fails to surpass the baselines. TRATES focuses on trait scoring and comprehensively outperforms baselines through a hybrid architecture.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE Dataset](rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs](trimllm_layer_dropping.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions](aimscheck_modern_slavery.md)

</div>

<!-- RELATED:END -->
