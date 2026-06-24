---
title: >-
  [Paper Note] iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News
description: >-
  [ACL 2025][Multimodal VLM][Personalized Affective Response] A personalized affective annotation dataset, iNews, is constructed, containing annotations from 291 UK annotators on 2,899 multimodal Facebook news posts. Annotator profiles (demographics, personality, media trust, etc.) explain 15.2% of the annotation variance, and combining persona information with LLM zero-shot prediction improves accuracy by up to 7%.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Personalized Affective Response"
  - "News Perception"
  - "Multimodal Dataset"
  - "LLM Persona Prompting"
  - "VAD Affect Model"
  - "Annotator Profiles"
date: 2026-05-08
content_hash: 2da5a565deafda51
---

# iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News

**Conference**: ACL 2025  
**arXiv**: [2503.03335](https://arxiv.org/abs/2503.03335)  
**Code**: Open-sourced (Dataset publicly available)  
**Authors**: Tiancheng Hu, Nigel Collier  
**Affiliation**: University of Cambridge  
**Area**: Multimodal / Affective Computing / Personalized Modeling  
**Keywords**: Personalized Affective Response, News Perception, Multimodal Dataset, LLM Persona Prompting, VAD Affect Model, Annotator Profiles  

## TL;DR

A personalized affective annotation dataset, iNews, is constructed, containing annotations from 291 UK annotators on 2,899 multimodal Facebook news posts. Annotator profiles (demographics, personality, media trust, etc.) explain 15.2% of the annotation variance, and combining persona information with LLM zero-shot prediction improves accuracy by up to 7%.

## Background & Motivation

**Background**: Sentiment detection is a long-standing research direction in NLP. However, almost all existing datasets use aggregated labels (gold labels), ignoring individual differences in human emotional reactions. Extensive psychological research shows that individual factors such as personality, demographics, and cultural background profoundly influence emotional perception.

**Limitations of Prior Work**:
- Most emotion detection datasets only focus on the emotional polarity of the text content itself, rather than the readers' actual emotional responses.
- A few datasets containing annotator information (e.g., Diaz et al. 2018) cover only simple emotional dimensions and lack multimodal, multi-aspect annotations.
- Research on personalized LLM capabilities lacks benchmark datasets that combine behavioral data with individual profiles.

**Design Motivation**: News emotional response is an ideal scenario for studying individual differences—mature measurement frameworks (VAD, Ekman) exist for emotional responses, individual differences are well-supported in communication and psychology literature, and news represents real-world stimuli that people encounter daily. iNews aims to simultaneously collect rich annotator profile information and multi-aspect affective annotations, providing data support for directions such as LLM personalization, subjective phenomenon modeling, and affective computing.

## Method

### Overall Architecture

The data collection process of the iNews dataset consists of three stages (see Figure 1 in the paper):

1. **Annotator Recruitment**: 291 UK annotators were recruited via the Prolific platform. Quota sampling was employed to ensure a balanced distribution of gender, age, political orientation, and geographical region (covering 97 out of 124 UK postcode areas).
2. **Persona Profiling Survey** (Stage 1): Each annotator completed a questionnaire containing 47 variables across 5 broad categories:
   - **Demographics and Ideology**: Age, gender, political leaning.
   - **News Consumption and Trust**: Consumption habits and trust ratings for major UK news outlets.
   - **Cognitive Characteristics**: Cognitive Reflection Test (CRT).
   - **Personality Traits**: Big Five Inventory (BFI-10).
   - **Affective Traits**: Perth Emotional Reactivity Scale (PERS) and Positive and Negative Affect Schedule (PANAS).
3. **News Post Annotation** (Stage 2): Each annotator provided 5 categories of annotations for approximately 50 Facebook news post screenshots:
   - **Dimensional Emotion Ratings**: Valence, Arousal, Dominance (VAD), measured using the Self-Assessment Manikin (SAM) on a 7-point scale.
   - **Discrete Emotion Classification**: Ekman's 6 basic emotions.
   - **Modality Influence**: The relative impact of images vs. text on their emotional response.
   - **Personal Relevance**: How relevant the post is to the individual.
   - **Willingness to Share**: Whether they would share the post.

### Key Designs

**News Post Sampling Strategy**:
- Data collection was conducted during three periods around the 2024 UK General Election (April, June, July) to ensure temporal diversity and coverage of news events.
- Phase 1: Random sampling.
- Phases 2 and 3: Proportional stratified sampling based on the media's follower count to enhance ecological validity.
- CrowdTangle was used to collect Facebook news posts, displaying screenshots instead of plain text, keeping interaction data but excluding comments.

**Quality Control**:
- A mandatory stay of at least 2 minutes on the instruction page (averaging 4.67 minutes).
- Comprehension test questions to filter out annotators who did not understand the task.
- Two attention check questions (only 2.4% of annotators failed them).
- Three standardized calibration items from ANET to validate annotator consistency using the SAM scale compared to original norms (average deviation < 0.5).

## Key Experimental Results

### Regression Analysis: Explanatory Power of Individual Differences

The paper uses linear mixed-effects models to analyze the variance sources of arousal ratings:

| Model | Fixed Effects | Random Effects | Marginal $R^2$ | Conditional $R^2$ |
|------|----------|----------|---------|---------|
| Null Model | None | Text | 0.000 | 0.131 |
| Persona Model | 47 Profiling Variables | Text | 0.152 | 0.286 |
| User Model | None | Text + User | 0.000 | 0.317 |

**Key Findings**:
- News content explains only 13.1% of the arousal variance.
- After adding profiling variables, the explanatory power increases to 28.6%, with profiling variables alone explaining **15.2%** of the variance—higher than all previous NLP datasets.
- The User model (including unobserved factors) explains 31.7%, indicating that additional individual differences still exist.

### Zero-Shot LLM Prediction Experiments

Tested 4 input conditions (Text/Image × With/Without Persona) across 7 frontier LLMs:

| Model | Input | MAE↓ | Exact Accuracy↑ | $\pm1$ Accuracy↑ |
|------|------|------|-----------|-----------|
| Gemini 1.5 Pro | Image+Persona | **0.84** | **39.55%** | **82.04%** |
| Gemini 1.5 Pro | Text+Persona | 0.91 | 36.44% | 78.76% |
| Gemini 1.5 Pro | Image | 0.94 | 36.96% | 77.03% |
| Llama-3.1-405B | Text+Persona | 0.89 | 38.00% | 81.17% |
| GPT-4o | Image+Persona | 0.89 | 36.79% | 79.45% |

**Key Findings**:
- Persona information consistently improves the performance of all models. Gemini 1.5 Pro's MAE decreased by 10.1% (image) and 11.6% (text).
- Image inputs generally outperform text inputs in the zero-shot setting (consistent with psychology: visual stimuli evoke stronger emotional reactions).
- The steerability of different models towards persona prompts varies significantly: the Gemini/Grok/Llama families show prominent responses, while GPT-4o is relatively stable.

### Few-Shot Experiments (Gemini 1.5 Pro)

- An **"early dip" phenomenon** is observed: from 0-shot to 4-shot, performance drops initially, but at 8-16 shots, it gradually recovers and surpasses zero-shot.
- 32-shot achieves the best exact accuracy of **44.4%** (Text+Persona).
- Even under 32-shot, persona information still provides additional gain (MAE: 0.812 $\rightarrow$ 0.782).
- The few-shot scaling effect of images is not as good as text—although images are superior in zero-shot, image recovery is slower in few-shot learning.

## Highlights & Insights

1. **NLP dataset with the highest explanatory power from profile variables**: The variance explanation of 15.2% is significantly higher than existing work, validating the importance of collecting rich individual profiles.
2. **Well-designed multi-dimensional annotation framework**: It covers VAD continuous dimensions + Ekman discrete emotions + modality influence + relevance + sharing willingness, making it widely applicable to various downstream tasks.
3. **Supplementary qualitative analysis**: An open-ended questionnaire with 20 annotators revealed individual difference patterns beyond structured data (e.g., the impact of growing up during the Cold War or working-class backgrounds on emotional desensitization).
4. **Complementarity of Persona + Exemplars**: Explicit profile information and behavioral exemplars provide complementary signals, resembling a hybrid of profile-based and collaborative filtering methods in recommender systems.
5. **Early dip phenomenon**: Systems-level verification of the initial performance drop and subsequent rise pattern in few-shot ICL for emotion prediction.

## Limitations & Future Work

1. **Sample population limitations**: Only covers UK annotators and UK news media, which is not representative of global cultures and media ecosystems.
2. **Platform limitations**: Only uses Facebook posts, potential platform-specific biases.
3. **Limited emotion measurement frameworks**: Relies on self-reported emotions, which may be affected by social desirability bias; VAD and Ekman frameworks cannot fully capture the complexity of emotional responses.
4. **Data quality**: Although multiple quality controls were implemented, AI-generated responses cannot be completely ruled out.
5. **Exact accuracy remains low**: The best model only reaches ~44%, demonstrating that individual-level emotional prediction remains an extremely challenging task.

## Related Work & Insights

- **News Emotion and Individual Differences**: Soroka et al. (2019) on negativity bias, Oliver (2002) on communication literature regarding individual exposure differences.
- **NLP Emotion Detection**: Strapparava & Mihalcea (2007), Plaza-del Arco et al. (2024) reviewed resources in this field, but most used aggregated labels.
- **Annotator Background Modeling**: Diaz et al. (2018) provided demographic + emotional annotations for online communities, but on a single dimension.
- **LLM Personalization**: Rescala et al. (2024), Dong et al. (2024) explored the effectiveness of persona prompting.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The first multimodal news dataset to include both rich individual profiles and multi-aspect affective annotations, filling an important gap.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Regression analysis + 7 LLMs in zero-shot + few-shot experiments, comprehensive and systematic.
- **Value** ⭐⭐⭐⭐⭐: Directly drives multiple fields such as LLM personalization, affective computing, and human behavior simulation.
- **Limitations** ⭐⭐⭐: Restricted to the UK and Facebook; exact accuracy remains low, reflecting the inherent difficulty of the task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Aligning VLM Assistants with Personalized Situated Cognition](aligning_vlm_assistants_with_personalized_situated.md)
- [\[ACL 2025\] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[NeurIPS 2025\] SmartWilds: Multimodal Wildlife Monitoring Dataset](../../NeurIPS2025/multimodal_vlm/smartwilds_multimodal_wildlife_monitoring_dataset.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)

</div>

<!-- RELATED:END -->
