---
title: >-
  [Paper Note] Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice
description: >-
  [ACL 2025][Medical LLM][Mental health screening] This paper proposes a questionnaire-guided mental health screening framework. By leveraging adaptive RAG to retrieve relevant content from users' Reddit posts, LLMs are used to fill out standardized psychometric scales (such as BDI-II) on behalf of users. It matches or outperforms supervised methods without requiring training data, while providing clinically interpretable assessment results.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Mental health screening"
  - "Adaptive RAG"
  - "Psychometric scales"
  - "Depression detection"
  - "Interpretable AI"
date: 2026-05-08
content_hash: dd6581fedc374141
---

# Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice

**Conference**: ACL 2025  
**arXiv**: [2501.00982](https://arxiv.org/abs/2501.00982)  
**Code**: [https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment](https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment)  
**Area**: Medical NLP  
**Keywords**: Mental health screening, Adaptive RAG, Psychometric scales, Depression detection, Interpretable AI

## TL;DR
This paper proposes a questionnaire-guided mental health screening framework. By leveraging adaptive RAG to retrieve relevant content from users' Reddit posts, LLMs are used to fill out standardized psychometric scales (such as BDI-II) on behalf of users. It matches or outperforms supervised methods without requiring training data, while providing clinically interpretable assessment results.

## Background & Motivation

**Background**: Mental health issues are increasingly severe, with a 28% increase in depression cases post-COVID-19. Social media serves as a crucial platform for people to express emotions and seek support, and many NLP studies have leveraged social media content for mental health analysis. However, mainstream approaches mostly adopt end-to-end classification (directly predicting mental states from text), a black-box method that lacks clinical interpretability.

**Limitations of Prior Work**: (1) Zero-shot/few-shot performance of directly using LLMs to classify mental states is suboptimal, making it difficult to match supervised methods. (2) End-to-end classification fails to explain "why a user is deemed to have depressive tendencies", which does not align with psychological diagnostic practices. (3) Standardized tools for mental health assessment (e.g., the BDI-II questionnaire) are widely used in clinical settings, yet computational methods largely bypass these well-validated instruments.

**Key Challenge**: Psychologists systematically assess patient states using standardized questionnaires, where each item has clear clinical significance. In contrast, NLP methods skip this structured assessment step and jump directly from text to diagnosis, losing interpretability and clinical alignment.

**Goal**: To enable LLMs to behave like psychologists by analyzing a user's social media posts to "proxy-fill" standardized psychological questionnaires, decomposing the complex diagnostic task into item-by-item assessments to achieve interpretable and accurate mental health screening.

**Key Insight**: Redefine mental health prediction as $\sum_i f(\text{Text}, \text{Item}_i) \to Y$—retrieving relevant posts and generating responses for each questionnaire item, then aggregating the scores to obtain the final assessment.

**Core Idea**: Use adaptive RAG to retrieve the most relevant user posts for each psychological questionnaire item, and then let the LLM "proxy-answer" the item based on the retrieved content, anchoring the diagnostic logic within standardized clinical tools.

## Method

### Overall Architecture
For each user, the system executes the following process: (1) Vectorize all of the user's Reddit posts; (2) Vectorize each questionnaire item and its options to serve as queries; (3) Retrieve the most relevant posts for each item via adaptive retrieval; (4) Use the retrieved posts as context for the LLM to score the item; (5) Aggregate the scores of all items to obtain the final severity assessment.

### Key Designs

1. **Adaptive Zero-Shot Retrieval Strategy (ABIDE-ZS)**:

    - **Function**: Dynamically determines the optimal number of retrieved posts for each questionnaire item.
    - **Mechanism**: Calculates the semantic similarity between the four options of each item and the user's posts. The ABIDE algorithm automatically determines the optimal retrieval quantity $k^*$ by detecting the boundaries of semantically consistent regions in the embedding space. The value of $k^*$ can vary across items and users—some items may find many relevant posts, while others only a few.
    - **Design Motivation**: A fixed retrieval size is unreasonable—some users may post heavily about a specific topic, while others rarely mention it. Adaptive retrieval ensures that each item receives exactly enough high-quality context.

2. **LLM Scoring Based on Questionnaire Structure**:

    - **Function**: Enables the LLM to score each item of the psychological questionnaire based on the retrieved posts.
    - **Mechanism**: Constructs a prompt for each item containing the item description, option details, and retrieved user posts. The LLM is required to analyze how well the post content matches each option and output the most probable option score (0-3). Both direct scoring and Chain-of-Thought (CoT) strategies are supported; the latter requires the LLM to explain its reasoning before outputting the score.
    - **Design Motivation**: Item-by-item scoring decomposes a complex global assessment into manageable sub-problems with clear clinical definitions, making it easier for the LLM to make accurate judgments.

3. **Multi-Scale Extension**:

    - **Function**: Extends the framework from depression to other psychological conditions such as self-harm, eating disorders, and pathological gambling.
    - **Mechanism**: Applies different standardized questionnaires (BDI-II $\to$ SHI, SCOFF, DSM-5 gambling scale) using the exact same retrieval + scoring pipeline without any modifications.
    - **Design Motivation**: The core of the framework is "questionnaire-guided" rather than "disease-specific", proving the generalizability of this paradigm.

### Loss & Training
Fully unsupervised—no training data is required. The LLM uses zero-shot reasoning, and the retrieval model utilizes pre-trained dense retrievers (10 different retrieval models were tested).

## Key Experimental Results

### Main Results

| Method | eRisk 2019 (RMSE↓) | eRisk 2020 (RMSE↓) | Training Data |
|------|-------------------|-------------------|---------|
| Supervised SOTA | 8.21 | 10.45 | Required |
| GPT-4o-mini Direct Classification | 12.35 | 13.82 | Not Required |
| GPT-4o-mini + Direct Prompting | 10.8 | 12.1 | Not Required |
| **Ours (Claude-3.5 + aRAG)** | **7.89** | **9.95** | **Not Required** |
| Ours (Qwen-2.5-70B + aRAG) | 8.15 | 10.28 | Not Required |
| Ours (DeepSeek-V3 + aRAG) | 8.32 | 10.51 | Not Required |

### Ablation Study

| Configuration | RMSE (eRisk 2020) | Description |
|------|-------------------|------|
| aRAG + Claude-3.5 | 9.95 | Optimal configuration |
| Fixed $k=5$ retrieval | 11.23 | Adaptive $k$ outperforms fixed $k$ |
| No retrieval (LLM only) | 13.82 | Suboptimal performance without context |
| Direct classification (no questionnaire) | 12.10 | Questionnaire-guided outperforms direct diagnosis |
| CoT vs. Direct scoring | 10.12 vs 9.95 | Direct scoring slightly performs better |
| Best retrieval model: sf-e5 | 9.95 | Best among 10 tested retrievers |
| Worst retrieval model: contriever | 11.45 | Significant impact of retrieval quality |

### Key Findings
- The questionnaire-guided approach significantly outperforms direct LLM depression classification (with an RMSE reduction of roughly 28%), validating the effectiveness of the item-by-item decomposition strategy.
- The adaptive retrieval mechanism (ABIDE-ZS) retrieves an average of 9–20 posts per item, which is far fewer than the user's total number of posts, effectively preventing information overload.
- Closed-source models (Claude-3.5) perform the best, but open-source 70B models (such as Qwen-2.5-70B) can achieve performance near the SOTA.
- The identical framework can be seamlessly extended to other psychological conditions like self-harm and eating disorders, showcasing the generalizability of the questionnaire-guided paradigm.
- The choice of retrieval model has a substantial impact, showing a difference of 1.5 RMSE points between the best and worst retrievers.

## Highlights & Insights
- The core insight of "turning LLMs into psychologists" is profound: instead of demanding direct diagnosis, the model is guided to follow a psychologist's diagnosis flow (item-by-item assessment $\to$ score aggregation $\to$ severity determination), substantially improving both interpretability and accuracy.
- The fact that an unsupervised method outperforms supervised counterparts is an important finding, suggesting that structured knowledge from standardized questionnaires can compensate for the lack of training data.
- The framework serves as an excellent paradigm of "knowledge-guided AI"—encoding domain expertise (psychological questionnaires) into reasoning structures is far more efficient than expecting models to learn diagnostics from scratch.

## Limitations & Future Work
- The Reddit user demographic may not be representative of the general population, introducing potential self-selection bias.
- Having LLMs "fill out" questionnaires on behalf of users based on their online posts raises ethical concerns, as users have not consented to such evaluations.
- Posts addressing certain questionnaire items (e.g., suicidal ideation) might be extremely sparse, affecting retrieval quality.
- Future research can incorporate longitudinal analysis (tracking user state changes over time) instead of relying solely on cross-sectional assessments.

## Related Work & Insights
- **vs MentalBERT (Ji et al., 2022)**: MentalBERT is a supervised model requiring training data and is uninterpretable; ours is unsupervised and interpretable.
- **vs Rosenman et al. (2024)**: While they prompt the LLM to "roleplay" as a respondent to fill out questionnaires, ours populates the scale based on the actual content of user posts, making it more evidence-based.
- **vs eRisk Competitions**: This work is the first to match the top competition results under a completely unsupervised setting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulating psychological questionnaires as structured inference frameworks for LLMs is a highly creative design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Tested on 6 LLMs, 10 retrieval models, 4 psychological conditions, and two benchmark datasets.
- Writing Quality: ⭐⭐⭐⭐ The research question is precisely defined, and the methodology is clearly described.
- Value: ⭐⭐⭐⭐⭐ Provides a significant methodological contribution to AI-assisted mental health screening.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)
- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)

</div>

<!-- RELATED:END -->
