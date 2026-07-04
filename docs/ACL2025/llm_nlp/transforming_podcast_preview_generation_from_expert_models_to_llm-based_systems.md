---
title: >-
  [Paper Note] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems
description: >-
  [ACL 2025][LLM (Other)][podcast preview] Spotify proposes using an LLM (Gemini 1.5 Pro) to replace the legacy multi-model feature engineering pipeline for generating podcast preview clips. This approach significantly outperforms the traditional system in both offline human evaluation and online A/B testing, achieving a 4.6% increase in user engagement duration and a 5x improvement in processing efficiency.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "podcast preview"
  - "LLM application"
  - "content understanding"
  - "A/B testing"
  - "industry deployment"
date: 2026-05-08
content_hash: 2238dfcd2e68bb0c
---

# Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems

**Conference**: ACL 2025  
**arXiv**: [2505.23908](https://arxiv.org/abs/2505.23908)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: podcast preview, LLM application, content understanding, A/B testing, industry deployment

## TL;DR
Spotify proposes using an LLM (Gemini 1.5 Pro) to replace the legacy multi-model feature engineering pipeline for generating podcast preview clips. This approach significantly outperforms the traditional system in both offline human evaluation and online A/B testing, achieving a 4.6% increase in user engagement duration and a 5x improvement in processing efficiency.

## Background & Motivation

**Background**: Discovering and evaluating long-form content like podcasts is time-consuming for users. Preview clips serve as an effective way to help users quickly assess their interest in the content.

**Limitations of Prior Work**: Legacy podcast preview systems (Legacy ML) rely on complex feature engineering pipelines that integrate multiple expert models—such as topic analysis, sentiment analysis, ad detection, audio event detection, sentence boundary detection, and ranking. This results in extremely high maintenance and iteration costs.

**Key Challenge**: Whenever a new requirement is added or criteria are adjusted in legacy systems, multiple models must be retrained or their weights and aggregation logic retuned. This leads to long iteration cycles and poor flexibility.

**Goal**: Replace the entire multi-model pipeline with a single LLM + few-shot prompt, eliminating feature engineering through prompt iteration to greatly simplify the architecture.

**Key Insight**: Leverage the long-context understanding and structured reasoning capabilities of LLMs to select the optimal preview clip directly from transcripts while generating metadata (such as topic tags and recommendation explanations).

**Core Idea**: Combining LLMs, sentence indexing, and few-shot prompting can replace complex legacy feature engineering pipelines, generating better podcast previews more rapidly.

## Method

### Overall Architecture
Podcast Audio → Transcript → Sentence Segmentation and Timestamp Alignment → LLM (Gemini 1.5 Pro) Few-shot Inference to Select Preview Clip → Post-processing Curation to ~1 Minute → Output Final Preview.

### Key Designs

1. **Sentence Indexing and Timestamp Alignment (Sentencization)**

    - Segment the transcripts into sentences based on heuristic rules like punctuation, and label each sentence with start and end timestamps.
    - **Design Motivation**: LLMs require precise localization of clip boundaries; timestamp indexing serves as a critical bridge mapping the text space back to the audio space.

2. **Structured Reasoning Prompt**

    - Guide the LLM toward step-by-step reasoning: identify the main topic of the episode → evaluate the relevance and appeal of various segments → generate preview metadata (reasons for recommendation, topic tags).
    - **Design Motivation**: Structured reasoning enhances both the transparency and interpretability of decisions, thereby improving the quality of the generated previews.

3. **Preview Constraint Rules**

    - The prompt explicitly lists preview constraints: engaging hook, logical progression, exclusion of advertisements, self-contained beginning and ending, emotional resonance, and a duration of approximately one minute.
    - **Design Motivation**: Encode the domain knowledge of product design teams as prompt constraints, replacing the rule engines in legacy systems.

4. **Few-shot Learning**

    - Provide manually curated, high-quality showcase examples in the prompt.
    - **Design Motivation**: Let the LLM learn the definitions of "quality previews" through exemplars without requiring fine-tuning.

5. **Manual Prompt Iteration**

    - Product and design teams iteratively optimize prompts and validate them repeatedly on small-scale evaluation sets.
    - **Design Motivation**: Human feedback is better suited for tasks requiring aesthetic judgment compared to automated prompt engineering.

### Comparison with Legacy System

| Dimension | Legacy ML System | LLM System |
|------|-------------|----------|
| Number of Models | 6+ Expert Models | 1 LLM |
| Input Modality | Audio + Text | Text Only |
| Processing Time | ~100 seconds / episode | ~20 seconds / episode |
| Iteration Method | Model Retraining + Feature Adjustment | Prompt Modification |

## Key Experimental Results

### Table 1: Offline Human Evaluation - Overall Comparison and Statistical Tests

| Evaluation Metric | Z-Test Statistic | P-value | LLM Significantly Better? |
|----------|:---:|:---:|:---:|
| Understandability | -4.05 | 5.09e-05 | Yes |
| Contextual Clarity | -3.40 | 0.00067 | Yes |
| Interest Level | -4.32 | 1.59e-05 | Yes |

- Out of 238 valid annotations, the LLM previews were rated better than or equivalent to legacy previews in 81.09% of cases, with a pure win rate of 54.2%.
- The binomial test yielding a p-value of 1.37e-10 demonstrates that the LLM's superiority is highly statistically significant.

### Table 2: Online A/B Test Results

| Metric | Gain | Description |
|------|---------|------|
| User Evaluation Time / User | +4.6% | Statistically significant, week 2 data |
| Evaluation Time per Preview | +4.0% | Statistically significant, week 2 data |
| Processing Efficiency | 5x Gain | 100s → 20s |

- The A/B test covered 67 English-speaking countries over 6 weeks, with LLM previews constituting 34% of the visible set for the treatment groups.

### Key Findings
- The LLM statistically significantly outperforms the legacy system across all three dimensions: understandability, contextual clarity, and interest level.
- Online data validates the offline evaluation conclusions, showing that users indeed engage more with LLM-generated previews.
- Outputs leveraging text-only context surpass those of the legacy system, which required both audio and text modalities.

## Highlights & Insights

1. **Real-world large-scale deployment**: Deployed in Spotify's live production environment serving hundreds of thousands of podcast previews, validated by an A/B test covering 67 countries.
2. **Substantial reduction in engineering complexity**: Simplifying the system from a pipeline of 6+ expert models to a single LLM API call brings a qualitative change to maintenance costs and iteration speed.
3. **5x improvement in processing efficiency**: 20 seconds vs. 100 seconds, without requiring audio signal processing.
4. **Rigorous evaluation framework**: Dual validation via offline human evaluation (20 evaluators, 238 annotations, statistical tests) and online A/B testing (6 weeks, 67 countries).

## Limitations & Future Work

1. **English Only**: Currently relies on metadata language tagging for English filtering; multilingual expansion remains unexplored.
2. **Reliance on commercial LLMs**: Using Gemini 1.5 Pro introduces dependencies on third-party API costs and controllability.
3. **Non-automated prompt iteration**: The manual optimization process is neither reproducible nor scalable.
4. **No audio signals used**: Outstanding clips that require acoustic cues (e.g., tone, laughter) to identify might be missed.
5. **Limited evaluation metrics**: Increased user engagement duration does not necessarily equate to improved content discovery; deeper metrics like conversion rate are still lacking.

## Related Work & Insights

| Method | Mechanism | Data Modality | Requires Multiple Models? | Deployment Scale |
|------|----------|:---:|:---:|:---:|
| Legacy Feature Engineering | Multi-expert model aggregation | Audio+Text | Yes (6+) | Production-scale |
| Unsupervised Highlight Detection | Clustering or graph-based methods | Video/Text | Partial | Research-grade |
| LLM Summarization | Extractive / Abstractive | Text | No | Research-grade |
| PodTile (Chapter Generation) | LLM + Indexing | Text | No | Production-scale |
| **Ours (LLM Preview)** | **Few-shot LLM + Sentence Index** | **Text** | **No (Single LLM)** | **Production-scale (Spotify)** |

## Rating
- Novelty: ⭐⭐⭐ (The idea of replacing traditional pipelines with an LLM is not entirely brand new, but the engineering execution and dual validation are highly valuable)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Offline human evaluation + online A/B testing, complete statistical testing, industry-grade validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, thorough legacy vs. LLM comparison, intuitive tables)
- Value: ⭐⭐⭐⭐ (A benchmark paper for industrial applications, demonstrating the complete deployment path and efficacy of LLMs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)
- [\[ICML 2025\] Expert Evaluation of LLM World Models: A High-Tc Superconductivity Case Study](../../ICML2025/llm_nlp/expert_evaluation_of_llm_world_models_a_high-t_c_superconductivity_case_study.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)

</div>

<!-- RELATED:END -->
