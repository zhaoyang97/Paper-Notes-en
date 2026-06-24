---
title: >-
  [Paper Note] Can Third-parties Read Our Emotions?
description: >-
  [ACL 2025][LLM (Other)][Emotion Recognition] Through human subject experiments, this study systematically compares the alignment between third-party annotators (human annotators and LLMs) and the first-party (author self-annotations) in emotion recognition tasks. It reveals a significant gap between third-party annotations and authors' actual emotions. Although LLMs outperform human annotators, their performance remains suboptimal. Demographic similarity is shown to improve a…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Emotion Recognition"
  - "Annotation Quality"
  - "Third-party Annotation"
  - "LLM Annotation"
  - "Demographics"
date: 2026-05-08
content_hash: 2512ca1374b04c62
---

# Can Third-parties Read Our Emotions?

**Conference**: ACL 2025  
**arXiv**: [2504.18673](https://arxiv.org/abs/2504.18673)  
**Code**: None  
**Area**: Other  
**Keywords**: Emotion Recognition, Annotation Quality, Third-party Annotation, LLM Annotation, Demographics

## TL;DR

Through human subject experiments, this study systematically compares the alignment between third-party annotators (human annotators and LLMs) and the first-party (author self-annotations) in emotion recognition tasks. It reveals a significant gap between third-party annotations and authors' actual emotions. Although LLMs outperform human annotators, their performance remains suboptimal. Demographic similarity is shown to improve annotation quality.

## Background & Motivation

Many Natural Language Processing (NLP) tasks involve inferring an author's private state (such as emotions, opinions, etc.), including sentiment classification, sarcasm detection, and stance detection. These tasks commonly rely on "gold standard" datasets annotated by third-party annotators. However, this practice implicitly assumes that third-party annotators can accurately capture the author's internal emotional state, an assumption that remains largely untested.

In reality, subjective language often lacks explicit linguistic cues, requiring annotators to infer emotional states from textual clues that may be implicit, ambiguous, or highly context-dependent. Furthermore, annotators' own socio-demographic backgrounds, cultural factors, and personal beliefs inevitably influence their interpretation of the author's text.

This misalignment is not merely an annotation error; it propagates through training data into models, compromising the reliability of downstream applications. In high-stakes scenarios (such as content moderation, deception detection, and therapeutic chatbots), misinterpreting user emotions can lead to societal harm.

## Method

### Overall Architecture

The study is designed as a two-phase human subject experiment:
1. Collect first-party data: Social media users submit their own posts and self-annotate their emotions.
2. Collect third-party annotations: Human annotators and LLMs annotate the emotions of the same set of posts, which are then compared with the first-party labels.

### Key Designs

1. **Emotion Classification Taxonomy**: The study adopts the fine-grained emotion taxonomy proposed by Demszky et al. (2020), which includes 27 different emotions plus "neutral" (28 categories in total). Additionally, these 28 categories are aggregated into 7 basic emotion groups (joy, love, anger, surprise, fear, sadness, neutral) for coarse-grained analysis.

2. **First-Party Data Collection**: US social media users were recruited via the Connect crowdsourcing platform, covering three age groups (18-27, 28-43, 44-59), two genders (female, male), and three racial groups (Black, White, Asian). An intersectional recruitment strategy was used to ensure demographic balance. A total of 123 participants provided 729 posts (44% text-only, 56% multimodal with image and text), with each user submitting 5-15 posts from the past 12 months and self-annotating their emotions.

3. **Third-Party Human Annotation**: Each post was assigned to 6 annotators—3 ingroup annotators (sharing all three demographic characteristics with the author: age group, gender, race) and 3 outgroup annotators (differing from the author in at least two characteristics). Annotators viewed the screenshots of the original posts and selected the emotions expressed by the author from the 28 categories (a multi-label task).

4. **LLM Annotation**: Five models—GPT-4 Turbo, GPT-4o, Gemini 1.5 Pro, Gemini 1.5 Flash, and Claude 3.5 Sonnet—were utilized, receiving the same post screenshots and instructions as the human annotators (multimodal input).

5. **Demographic Prompting Experiment (RQ3)**: The first-party authors' demographic information (age, gender, race) was incorporated into the LLM prompts to explore whether it could improve annotation alignment.

### Loss & Training

This study does not involve model training. The core evaluation metrics and methods include:
- **Cohen's kappa**: To measure the agreement between annotators and the gold standard.
- **F1 score / Recall / Precision**: For multi-label classification evaluation.
- **Wilcoxon signed-rank test**: To determine the statistical significance of differences between groups.
- **Linear Mixed Models (LMM)**: To compare performance at the annotator level while controlling for the random effects of task ID and annotator ID.

## Key Experimental Results

### Main Results

Alignment between third-party annotations and first-party labels (macro-averaged after majority voting):

| Annotator Type | Precision | Recall | F1 | Cohen's κ Range |
|-----------|-----------|--------|-----|-------------|
| Ingroup Human Annotators | 0.38 | 0.29 | 0.32 | 0-0.45 |
| Outgroup Human Annotators | 0.36 | 0.24 | 0.28 | 0-0.45 |
| LLM (Majority Vote of 5 Models) | 0.38 | 0.50 | **0.40** | 0-0.45 |

Comparison between LLMs and Human Annotators (Wilcoxon test):

| Comparison Dimension | Ingroup vs LLM p-value | Outgroup vs LLM p-value |
|---------|---------------|---------------|
| F1 | 8.34×10⁻¹² *** | 4.62×10⁻²⁵ *** |
| Recall | 4.95×10⁻³¹ *** | 2.76×10⁻³⁹ *** |
| Cohen's κ | 2.75×10⁻⁵ *** | 7.45×10⁻⁹ *** |

### Ablation Study

Comparison between Ingroup and Outgroup Annotators (Post-level, Wilcoxon test):

| Metric | Ingroup Median | Outgroup Median | p-value |
|------|----------|----------|-----|
| F1 | 0.29 | 0.00 | 0.004* |
| Recall | 0.25 | 0.00 | 0.001* |
| Cohen's κ | 0.28 | 0.24 | 0.028* |

Impact of demographic prompting on LLMs: F1 shows visual significance with $p=0.0095$, but the actual median remains unchanged (still 0.4), indicating limited practical performance improvement.

### Key Findings

- Third-party annotations (whether by humans or LLMs) exhibit systematically low alignment with first-party labels ($\kappa$ range 0–0.45, mostly falling into the "slight to moderate" range).
- LLMs outperform human annotators across almost all emotions, but for emotions like *grief*, *sadness*, and *curiosity*, ingroup human annotators perform comparably or even better.
- *realization*, *relief*, and *neutral* consistently perform the worst, indicating these emotions are the most difficult for third parties to identify.
- Ingroup annotators (sharing demographic characteristics) significantly outperform outgroup annotators, particularly in Recall and F1.
- Posts with high alignment typically contain explicit emotional vocabulary cues (e.g., "happy" mapping to *joy*), while low-alignment posts often lack textual cues or heavily depend on context.
- A significant number of authors self-report "neutral" even when their posts contain discernible emotional cues—indicating that the presence of emotional language does not equate to the author's internal emotional state.

## Highlights & Insights

1. **Challenging Basic Assumptions**: This work directly challenges the prevailing third-party annotation paradigm in NLP, especially for tasks involving the inference of private states. This finding has profound implications for data collection practices across the fields of sentiment analysis and opinion mining.
2. **Fundamental Difference Between First-party vs. Third-party Perspectives**: A phrase like "I got a cup of coffee" can express entirely different emotions depending on the speaker and the context—information that may be opaque to third parties. Are we modeling third-party emotion perception, or the author's actual expressed emotions?
3. **Multimodal Experimental Design**: Utilizing screenshots of original posts (rather than text only) as inputs offers a more realistic simulation of the annotation process.
4. **Advantages and Limitations of LLMs as Annotators**: Although LLMs perform better overall than human annotators, they still show significant limitations in tasks requiring nuanced emotional understanding.

## Limitations & Future Work

- Whether first-party labels truly reflect the authors' internal emotions cannot be externally validated (as self-annotations may also be prone to inaccuracies).
- The study is limited to US participants, lacking broader cultural and linguistic diversity.
- Some demographic intersection groups (e.g., Asian participants aged 44–59) have relatively small sample sizes.
- The sample size is somewhat limited (729 posts, 123 participants), which may impact the robustness of the statistical analysis.
- The work only investigates the emotion recognition task; the generalizability of its findings to other private state inference tasks requires further validation.
- Demographic prompting yields negligible practical improvements on LLMs, indicating a need for more effective approaches.
- Temporal and context factors are not considered—an author's emotion self-annotation for the same post might shift over time or with different contextual framing.

## Related Work & Insights

- Similar to the work by Oprea & Magdy (2019) on the discrepancy between intent and perception in sarcasm detection, this paper narrows its focus to emotion, a more fundamental private state.
- Direct implications for data annotation practices: In tasks concerning private states, incorporating first-party feedback should be considered to supplement annotations.
- Sociological and psychological studies regarding how cultural and social factors shape emotional expression and interpretation find empirical validation here within NLP.
- Provides a cautionary note for LLM alignment research: If emotional labels in training data do not accurately reflect users' real emotions, models trained on such data will inevitably suffer from systematic biases in emotional understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ This work is the first to systematically compare first-party and third-party emotion annotations; the angle is novel though the methodology is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-dimensional comparisons (human vs. LLM, ingroup vs. outgroup, with/without demographic prompting) with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ The research questions are clear and the discussion is in-depth, though the paper is relatively long.
- Value: ⭐⭐⭐⭐ Provides a significant cautionary warning for emotion annotation practices, although direct actionable guidance remains somewhat limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] LLMs Can Be Easily Confused by Instructional Distractions](llms_can_be_easily_confused_by_instructional_distractions.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_writing_assessment.md)

</div>

<!-- RELATED:END -->
