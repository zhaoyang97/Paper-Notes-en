---
title: >-
  [Paper Note] KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors
description: >-
  [ACL 2025][Dialogue Systems][psychological counseling dialogue] This paper proposes KokoroChat, a Japanese psychological counseling dialogue dataset collected via role-playing by trained counselors, consisting of 6,589 long sessions and detailed client feedback ratings, designed to enhance the counseling response generation and dialogue evaluation capabilities of LLMs.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "psychological counseling dialogue"
  - "dataset construction"
  - "role-playing"
  - "dialogue evaluation"
  - "Japanese NLP"
date: 2026-05-08
content_hash: 3abbd1c9f488d5a5
---

# KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors

**Conference**: ACL 2025  
**arXiv**: [2506.01357](https://arxiv.org/abs/2506.01357)  
**Code**: [https://github.com/UEC-InabaLab/KokoroChat](https://github.com/UEC-InabaLab/KokoroChat)  
**Area**: Dialogue Systems  
**Keywords**: psychological counseling dialogue, dataset construction, role-playing, dialogue evaluation, Japanese NLP

## TL;DR

This paper proposes KokoroChat, a Japanese psychological counseling dialogue dataset collected via role-playing by trained counselors, consisting of 6,589 long sessions and detailed client feedback ratings, designed to enhance the counseling response generation and dialogue evaluation capabilities of LLMs.

## Background & Motivation

**Background**: Mental health issues present a significant global challenge, but professional counseling resources remain scarce. Consequently, researchers have explored utilizing language models to generate empathetic responses to provide emotional support.

**Limitations of Prior Work**: Crowdsourced data collection requires rigorous professional training and is highly costly. Real-world counseling data involves privacy and ethical concerns. Meanwhile, LLM-augmented datasets (e.g., AugESC, SMILECHAT) suffer from content redundancy and lack of diversity, with average dialogue turns being significantly lower than those in real-world counseling.

**Key Challenge**: The tension between the difficulty of acquiring high-quality, professional psychological counseling dialogue data and the imperative of NLP research for large-scale training data.

**Goal**: To construct a large-scale psychological counseling dialogue dataset that ensures professionalism, authenticity, and privacy security.

**Key Insight**: Adopting a role-playing method where professionally trained counselors play the roles of both the counselor and the client in simulated sessions, thereby guaranteeing high data quality while avoiding privacy risks.

**Core Idea**: Encouraging professional counselors to simulate counseling dialogues through role-playing to balance professionalism, scale, and privacy concerns in the dataset.

## Method

### Overall Architecture

Matching participants via an online platform $\rightarrow$ Role-playing dialogue sessions (approx. 1 hour) $\rightarrow$ Client feedback rating $\rightarrow$ Data cleaning and filtering $\rightarrow$ Construction of training/test sets.

### Key Designs

1. **Role-Playing Data Collection**: The study involved 480 participants (117 male / 360 female), where more than 1/3 held professional certifications and the remaining had completed 6 months to 1 year of systematic course study. All participants underwent 10 hours of structured training. Counselors communicated via PC keyboards while clients used the LINE mobile application, simulating authentic online counseling scenarios in Japan.
2. **20-Dimensional Client Feedback System**: Designed by experts holding national psychological certifications and doctorates, the system covers "Overall Dialogue Impression" (10 items: feeling understood, respected, new insights, hopefulness, etc.) and "Counseling Skill Evaluation" (10 items: empathy, questioning, goal setting, etc.). Each item is rated on a scale of 0 to 5, with a total score of 100.
3. **Data Quality Control**: Dialogue sessions with fewer than 30 turns, lasting less than 30 minutes, or having uniform scores of 3 across all items were filtered out, ultimately retaining 6,589 sessions.

### Loss & Training

Downstream tasks are fine-tuned using standard SFT. For the response generation task, consecutive utterances by the same speaker are merged, and the entire dialogue history is used as input to generate the counselor's next response. For the rating prediction task, the model predicts the 20-dimensional scores given the complete dialogue history.

## Key Experimental Results

### Main Results

Automatic evaluation of response generation (based on Llama-3.1-Swallow-8B):

| Model | BLEU-1 | BLEU-4 | ROUGE-L | Dist-1 | Dist-2 |
|------|:---:|:---:|:---:|:---:|:---:|
| Llama-3.1 (Untuned) | 17.32 | 2.25 | 16.96 | 1.04 | 6.86 |
| GPT-4o | 21.77 | 3.17 | 19.82 | 1.19 | 6.90 |
| Kokoro-Low (Low Score Data) | 25.39 | 5.39 | 27.28 | 2.42 | 12.98 |
| Kokoro-High (High Score Data) | **27.03** | **6.00** | **28.00** | 2.33 | 13.08 |
| Kokoro-Full (Full Data) | 25.69 | 5.83 | 28.10 | **2.48** | **13.24** |

### Ablation Study

Dataset statistical comparison (with existing datasets):

| Dataset | Manual | Rating Items | Language | Dialogues | Avg. Turns |
|--------|:---:|:---:|:---:|:---:|:---:|
| AugESC | ✗ | 0 | English | 65,000 | 26.7 |
| ESConv | ✓ | 2 | English | 1,300 | 29.5 |
| Client-Reactions | ✓ | 4 | Chinese | 2,382 | 78.5 |
| **KokoroChat** | ✓ | **20** | Japanese | **6,589** | **91.2** |

### Key Findings

- Even using only low-scoring data (Kokoro-Low) for fine-tuning yields performance significantly superior to untuned models, demonstrating the intrinsic value of the dataset.
- High-scoring data (Kokoro-High) yields the best results despite its smaller size, emphasizing the critical importance of training data quality.
- Human evaluation reveals that fine-tuned models still lag behind GPT-4o, and there remains a significant gap between GPT-4o and high-performing human counselors.
- Correlation analysis between dialogue features and ratings: The total word count of the client has the strongest positive correlation with ratings ($\rho = 0.42$), whereas the counselor's response time exhibits a negative correlation with ratings ($\rho = -0.21$).

## Highlights & Insights

- The role-playing data collection method elegantly solves the twin issues of privacy risks in real-world data and the low quality of LLM-generated data.
- The 20-dimensional rating system is exceptionally elaborate, providing high-quality annotations for dialogue evaluation.
- The scale of the dataset (6,589 sessions $\times$ 91.2 average turns) far exceeds existing manual counseling datasets.
- This fills a critical gap in Japanese psychological counseling dialogue data.

## Limitations & Future Work

- The dataset is restricted to Japanese, leaving cross-lingual generalizability to be validated.
- Role-playing is ultimately not identical to real-world sessions, and clients may exhibit insufficient emotional depth.
- The subjectivity of ratings cannot be fully eliminated; different raters might assign varied scores to the same dialogue.
- The base model is limited to Llama-3.1 with 8B parameters; the effectiveness of fine-tuning larger models remains unexplored.

## Related Work & Insights

- Comparison with ESConv: KokoroChat dialogues are substantially longer (91 vs. 30 turns) and feature richer rating dimensions (20 vs. 2).
- Comparison with LLM-augmented datasets: Manual data, though smaller in raw size, delivers higher quality and greater diversity.
- Insight: Role-play data collection can be generalized to other dialogue scenarios requiring specialized expertise (e.g., legal consultation, medical inquiry).

## Supplementary Analysis

- Topic distribution indicates that family issues account for 20.2%, workplace issues 17.0%, and mental health 14.7%, covering the primary demands of real-world counseling.
- The rating distribution is unimodal, with a mean of 63.58 and a median of 64.00, slightly right-skewed, indicating that most sessions received above-average feedback.
- High positive correlations exist between rating dimensions ($\rho > 0.6$); in particular, the correlation between "feeling understood", "satisfaction", and "sense of value" is the strongest.
- Rating prediction experiments: The fine-tuned Llama-3.1 achieves 80.10% accuracy on $ACC_{soft}$, outperforming GPT-4o's 75.27% by a large margin.
- The inclusion of 4,900 distinct counselor-client pairs enhances the diversity of the dialogues.

## Rating

- Novelty: ⭐⭐⭐⭐ The role-play data collection approach is novel, though the method itself is not highly technical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-perspective validation including automatic evaluation, human evaluation, and rating prediction.
- Writing Quality: ⭐⭐⭐⭐ The dataset is thoroughly documented with clear statistical analysis.
- Value: ⭐⭐⭐⭐ Represents a high-quality dataset contribution that fills a significant void in Japanese psychological counseling data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Codified Finite-state Machines for Role-playing](../../ICLR2026/dialogue/codified_finite-state_machines_for_role-playing.md)
- [\[ACL 2025\] When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)
- [\[ACL 2025\] SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script](share_shared_memory-aware_open-domain_long-term_dialogue_dataset_constructed_fro.md)
- [\[ACL 2025\] Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System](enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system.md)
- [\[ACL 2025\] Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)

</div>

<!-- RELATED:END -->
