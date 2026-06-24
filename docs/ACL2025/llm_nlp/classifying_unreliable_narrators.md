---
title: >-
  [Paper Note] Classifying Unreliable Narrators with Large Language Models
description: >-
  [LLM (Other)] Drawing on literary narratology theory, this work defines three different levels of unreliable narrators (intra-narrational / inter-narrational / inter-textual), constructs an expert-annotated dataset TUNa, and systematically evaluates the performance of LLMs on the task of classifying unreliable narrators.
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: 3a8084e726d3840f
---

# Classifying Unreliable Narrators with Large Language Models

| Metadata | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2506.10231](https://arxiv.org/abs/2506.10231) |
| Code | [GitHub](https://github.com/adbrei/unreliable-narrators) |
| Area | NLP / Text Classification / Narrative Analysis |
| Keywords | unreliable narrators, narratology, LLM classification, curriculum learning, expert-annotated dataset |

## TL;DR

Drawing on literary narratology theory, this work defines three different levels of unreliable narrators (intra-narrational / inter-narrational / inter-textual), constructs an expert-annotated dataset TUNa, and systematically evaluates the performance of LLMs on the task of classifying unreliable narrators.

## Background & Motivation

- **Problem Definition**: In daily life, first-person narratives (reviews, social media, cover letters, etc.) are frequently encountered, and judging whether the narrator is reliable is a critical issue for secure information transmission. An unreliable narrator refers to a narrator who **unintentionally** misleads readers (distinct from intentional deception).
- **Limitations of Prior Work**: Previously, no work utilized automated methods to analyze narrator unreliability, and no annotated datasets were available.
- **Narratological Foundation**: The authors draw on Hansen's (2007) taxonomic framework of narratology, dividing unreliability into three levels that increase progressively from concrete to abstract:
    - **(1) Intra-narrational**: The narrator exhibits "verbal tics," such as hedging language, selective memory, admission of bias, and other textual cues.
    - **(2) Inter-narrational**: A second voice contradicts the narrator (e.g., refutations in others' dialogue), or the narrator exhibits consistent unreliability over past and present.
    - **(3) Inter-textual**: The narrator conforms to classic unreliable character archetypes—the naïf, the madman, the pícaro, or the clown.
- **Key Challenge**: Unreliability cues are often subtle and context-dependent, scattered throughout the text, and sometimes require deep reasoning regarding the narrator's emotional and psychological state.

## Method

### Overall Architecture

Ours models unreliable narrator identification as three independent classification tasks:
- **Intra-narrational**: Binary classification (verbal tics vs. reliable)
- **Inter-narrational**: Three-class classification (same-unreliable-character-over-time / other-character-contradiction / reliable)
- **Inter-textual**: Five-class classification (naïf / madman / pícaro / clown / reliable)

### Key Designs

1. **TUNa Dataset Construction**: Collects first-person narrative texts from four domains: blogs (PersonaBank), Reddit posts (r/AITA), hotel reviews (Deceptive Opinion), and literary works (Project Gutenberg). Each sample is annotated by at least 2 experts majoring in English literature, achieving a Cohen's Kappa inter-annotator agreement of κ=0.71~0.75 (substantial agreement). Disagreements are resolved through discussion.
2. **Curriculum Learning**: Training samples are ranked by "degree of ambiguity"—LLMs first count the number of feature instances for each label type; samples with fewer candidate labels are considered easier. Parameter-efficient fine-tuning via LoRA is performed first on the easy subset, followed by the difficult subset, gradually improving the model's capability.
3. **Transfer from Literature to Reality**: Models are trained using training samples from the literary domain (Fiction) and tested out-of-domain on real-world domains such as blogs, Reddit, and reviews, verifying the transferability of unreliability knowledge learned from literature.

### Loss & Training

Standard classification cross-entropy loss is used, paired with LoRA adapters (8-bit quantization) for parameter-efficient fine-tuning, training for 3 epochs.

## Key Experimental Results

### Main Results (Llama3.1-8B, F1 macro)

| Task | Method | Fiction | Blog | Subreddit | Review |
|------|------|---------|------|-----------|--------|
| Intra-nar | CL | 58.51 | 53.94 | 50.04 | **67.17** |
| Intra-nar | Fine-tuned | 50.09 | 50.63 | 49.00 | 55.85 |
| Intra-nar | Zero-Shot | 45.17 | 45.56 | 47.41 | 58.46 |
| Inter-nar | CL | 34.59 | **35.92** | 30.91 | **35.29** |
| Inter-nar | Fine-tuned | 34.63 | 28.73 | 25.59 | 36.59 |
| Inter-tex | CL | 27.42 | 19.58 | 13.49 | 16.72 |
| Inter-tex | Fine-tuned | 28.59 | 18.99 | 10.85 | 17.54 |

### Ablation Study: Comparison of Different Model Sizes (Average F1 macro across all domains)

| Model | Intra-nar (CL) | Inter-nar (CL) | Inter-tex (CL) |
|------|---------------|----------------|----------------|
| Llama3.1-8B | 57.42 | 34.18 | 19.30 |
| Llama3.3-70B | 51.26 | 33.49 | 21.04 |
| Mistral-7B | 55.76 | 31.15 | **29.68** |
| ModernBERT | 39.94 | 27.07 | 16.98 |

### Key Findings

1. **Increasing Task Difficulty**: Intra-narrational is the easiest, while Inter-textual is the hardest (requiring more abstract reasoning).
2. **Effectiveness of Curriculum Learning**: CL significantly outperforms standard fine-tuning on smaller models, but the few-shot performance of large models (70B) is already comparable to CL.
3. **Feasibility of Cross-Domain Transfer**: Knowledge learned from Fiction can be reasonably transferred to real-world domains such as Blogs, Subreddits, and Reviews.
4. **Gender Bias**: Male narrators are correctly predicted at a higher rate than female narrators.
5. **Impact of Narrative Style**: Dialogue-heavy styles aid intra-narrational prediction, while descriptive styles aid inter-narrational and inter-textual prediction.

## Highlights & Insights

- Innovatively combines literary theory (narratology) with NLP tasks, defining a completely new and meaningful task—automated identification of unreliable narrators.
- The TUNa dataset covers multiple domains (literature/blogs/Reddit/reviews) with high-quality expert annotations (approx. 5 minutes of annotation time per sample).
- Cleverly designed curriculum learning strategy: defines sample difficulty based on the "number of ambiguous candidate labels" rather than traditional loss magnitude.
- Discovers that knowledge of unreliability learned from fictional works can transfer to real-world texts, opening a new direction of "learning real-world understanding from fiction."

## Limitations & Future Work

- Processes only short texts (up to 1050 tokens), without considering entire novels or long documents.
- The dataset contains only English, without covering other languages.
- The dataset size is relatively small (817 samples), limited by the high cost of expert annotation.
- Even with the best methods, the F1 score for the Inter-textual task remains very low (~30%), indicating that the task is highly challenging.

## Related Work & Insights

- Related to but fundamentally different from misinformation detection and deception detection: this work focuses on **unintentional** unreliability rather than **intentional** deception.
- A natural extension of narrative AI directions such as character understanding and protagonist emotion analysis.
- The curriculum learning strategy of "defining difficulty based on ambiguity" can be generalized to other NLP tasks containing ambiguous labels.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)

</div>

<!-- RELATED:END -->
