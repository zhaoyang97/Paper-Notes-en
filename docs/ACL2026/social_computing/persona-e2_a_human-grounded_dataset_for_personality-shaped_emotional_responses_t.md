---
title: >-
  [Paper Note] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events
description: >-
  [ACL 2026][Social Computing][personality modeling] Constructs the first large-scale dataset Persona-E2 linking personality traits (MBTI + Big Five) with reader emotional responses, containing 3,111 events × 36 annotators totaling 112K annotations, revealing that LLMs suffer from "personality illusion" when simulating personality-shaped emotional responses, and that Big Five features mitigate this more effectively than MBTI.
tags:
  - ACL 2026
  - Social Computing
  - personality modeling
  - emotional assessment
  - reader perspective
  - MBTI
  - Big Five
date: 2026-05-08
content_hash: b9eca5e3e86e909b
---

# Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events

**Conference**: ACL 2026  
**arXiv**: [2604.09162](https://arxiv.org/abs/2604.09162)  
**Code**: [HuggingFace](https://huggingface.co/datasets/CRIS-Yang/Persona-E2-Dataset)  
**Area**: Social Computing  
**Keywords**: personality modeling, emotional assessment, reader perspective, MBTI, Big Five

## TL;DR

Constructs the first large-scale dataset Persona-E2 linking personality traits (MBTI + Big Five) with reader emotional responses, containing 3,111 events × 36 annotators totaling 112K annotations, revealing that LLMs suffer from "personality illusion" when simulating personality-shaped emotional responses, and that Big Five features mitigate this more effectively than MBTI.

## Background & Motivation

**Background**: Affective computing research primarily focuses on emotions expressed by authors in text, while neglecting reader-perspective emotional assessment. Existing datasets mostly aggregate annotations into single labels, masking emotional diversity arising from personality differences across individuals.

**Limitations of Prior Work**: Role-playing LLMs attempt to simulate personalized responses by injecting personality traits into prompts, but they often exhibit "personality illusion"—merely imitating surface linguistic styles rather than truly adopting personality-based cognitive appraisal patterns. More critically, real human data is lacking to verify whether LLMs genuinely capture personality-driven emotional diversity.

**Key Challenge**: Cognitive appraisal theory indicates that emotions arise from individualized appraisal processes influenced by goals and personality traits, but the NLP field lacks benchmark datasets systematically linking personality traits with emotional responses. LLM-generated pseudo-labels cannot substitute for real human data for validation.

**Goal**: Construct a dataset with real personality annotations for reader emotional responses, enabling (1) analysis of how personality influences emotional appraisal, (2) evaluation of LLM ability to simulate personality-shaped emotions, (3) investigation of whether LLMs can generate psychologically plausible reasoning.

**Key Insight**: Have real annotators with measured personality traits (MBTI + Big Five) annotate emotional responses to events across news, social media, and life narratives, with 36 annotations per event ensuring dense personality diversity coverage.

**Core Idea**: Through real personality assessment + dense annotation (36 people/event) + cross-domain event coverage, construct the first personality-event-emotion benchmark dataset to systematically evaluate personality's influence on emotional appraisal and LLMs' simulation capability.

## Method

### Overall Architecture

Persona-E2 construction has three phases: (1) Event collection and filtering—collect events from news, social media, and life narratives, filtering through safety checks, LLM multi-dimensional scoring, and expert review from 77K candidates down to 3,111 high-quality events; (2) Personality-aware annotation—recruit 36 annotators who complete MBTI and Big Five questionnaires, each annotating all 3,111 events for genuine emotional responses; (3) Experimental evaluation across three research questions—analyzing emotional disagreement patterns, LLM simulation capability, and cognitive plausibility.

### Key Designs

1. **Multi-Dimensional Event Filtering Pipeline**:

    - Function: Filter from large candidate pools to select high-quality stimuli that effectively trigger personality-differentiated emotional responses
    - Mechanism: Three-stage filtering—(a) NSFW classifier filters harmful content; (b) Qwen3-MAX scores each event on four dimensions: personality variability (V), emotional arousal (A), emotional implicitness (I), and source relevance (R), with weighted computation $Score = 0.35V + 0.30A + 0.20R + 0.15I$; (c) 5-person expert panel final review
    - Design Motivation: Only events where "different personalities would react differently" maximize the dataset's discriminative value

2. **Personality-Aware Annotation Protocol**:

    - Function: Obtain emotional annotation data anchored in real personality traits
    - Mechanism: 36 annotators first complete standardized MBTI and Big Five questionnaires, then annotate each event from a reader perspective ("How would you feel when reading this event?") with Ekman's six basic emotions + neutral for 7 categories total. No role-playing throughout, ensuring data reflects genuine personality. 36 annotations per event ensure dense personality coverage
    - Design Motivation: Distinct from prior approaches of having annotators simulate specific personalities; directly leveraging annotators' own real personalities ensures psychological validity of the data

3. **Personality Alignment Gap Verification (PAG)**:

    - Function: Verify that annotation disagreements in the dataset are genuinely personality-driven rather than random noise
    - Mechanism: K-means clustering on Big Five vectors ($k=6$), computing within-group agreement $Agr_{in}$ and between-group agreement $Agr_{out}$ difference as PAG. Experiments show all clusters have positive PAG (+8.27% to +25.96%), proving that personality-similar individuals respond more consistently to the same events
    - Design Motivation: PAG serves as an intrinsic data quality validation metric, demonstrating that annotation disagreements are structured personality signals rather than noise

### Loss & Training

This is a dataset paper with no model training involved. The experimental section evaluates existing LLMs (GPT-4o, Claude 3.5, Qwen2.5, etc.) on personality-shaped emotion prediction in zero-shot and few-shot settings.

## Key Experimental Results

### Main Results

LLM personality-shaped emotion prediction performance (weighted F1):

| Model | News | Social Media | Life Narratives | Average |
|-------|------|-------------|-----------------|---------|
| GPT-4o (zero-shot) | 0.42 | 0.31 | 0.38 | 0.37 |
| Claude 3.5 | 0.40 | 0.29 | 0.36 | 0.35 |
| Qwen2.5-72B | 0.39 | 0.28 | 0.35 | 0.34 |
| + BFI prompt | 0.45 | 0.34 | 0.41 | 0.40 |
| + MBTI prompt | 0.43 | 0.32 | 0.39 | 0.38 |

### Ablation Study

Effect of personality information on LLM emotion prediction:

| Config | Weighted F1 | Note |
|--------|------------|------|
| No personality info | 0.37 | Baseline |
| + MBTI label | 0.38 | Only providing 4-letter type |
| + BFI vector | 0.40 | Providing continuous personality dimension scores |
| + BFI + cognitive explanation | 0.42 | Also requiring model to explain reasoning process |

### Key Findings

- LLMs perform worst on social media domain (F1 only 0.28–0.31), as social media text is more ambiguous and relies more on personalized interpretation
- Big Five features significantly outperform MBTI in mitigating "personality illusion," possibly because BFI provides continuous dimensions rather than discrete types
- Author-reader emotional divergence is largest in life narratives and smallest in news, demonstrating that first-person projection amplifies individual differences
- PAG validation shows ESTP type has highest personality consistency (+26.98%), ISTJ the lowest (+9.68%)

## Highlights & Insights

- The core insight of the dataset design is profound—"annotation disagreement is not noise but personality signal." The PAG verification method can generalize to any annotation task involving subjective judgment
- The scale of 36 people × 3,111 events = 112K annotations is unprecedented, with each annotation anchored in genuinely measured personality traits, providing a valuable benchmark for personalized AI research
- Systematic verification of the "personality illusion" concept—revealing that LLMs do not truly understand personality's influence on cognitive appraisal, only imitating stereotypes

## Limitations & Future Work

- Only 36 annotators with limited personality coverage; some MBTI types have fewer than 3 people, preventing statistical analysis
- Only 7 basic emotion categories, unable to capture more nuanced emotional dimensions (e.g., mixed emotions, emotion intensity)
- Events primarily from English sources with limited cultural diversity
- Future work could expand to more annotators and cultural backgrounds, exploring how dynamic personality trait changes affect emotional appraisal

## Related Work & Insights

- **vs GoodNewsEveryone**: GNE includes author + reader perspectives but no personality annotations; Persona-E2 is the first to introduce real personality measurements
- **vs Big5-Chat**: Big5-Chat uses LLM-generated personality-shaped dialogue data lacking real human validation; Persona-E2 is based on real annotators' genuine responses

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First large-scale dataset systematically linking real personality measurements with reader emotional annotations
- Experimental Thoroughness: ⭐⭐⭐⭐ Three comprehensive research questions, though 36-person sample size is small for psychology experiments
- Writing Quality: ⭐⭐⭐⭐ Clear structure with solid psychological theory foundations
- Value: ⭐⭐⭐⭐⭐ Provides an urgently needed real benchmark for personalized AI and affective computing

## Related Papers

- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ACL 2025\] Detection of Human and Machine-Authored Fake News in Urdu](../../ACL2025/social_computing/detection_of_human_and_machine-authored_fake_news_in_urdu.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[ACL 2025\] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](../../ACL2025/social_computing/banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](../../NeurIPS2025/social_computing/averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](../../ICCV2025/social_computing/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)

<!-- RELATED:END -->
