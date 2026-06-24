---
title: >-
  [Paper Note] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation
description: >-
  [ACL 2025][Long-context evaluation] This paper systematically studies the automatic evaluation of book-length stories (>100K tokens), constructs the first large-scale long story evaluation benchmark LongStoryEval (600 newly published novels, 340K reader reviews), proposes a hierarchical evaluation criteria framework, compares the effectiveness of three evaluation strategies, and trains a specialized evaluation model NovelCritique-8B, which outperforms GPT-4o in alignment with…
tags:
  - "ACL 2025"
  - "Long-context evaluation"
  - "novel evaluation"
  - "story evaluation"
  - "evaluation criteria"
  - "LLM evaluation"
date: 2026-05-08
content_hash: 9f9cd67d6c5a9109
---

# What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation

**Conference**: ACL 2025  
**arXiv**: [2512.12839](https://arxiv.org/abs/2512.12839)  
**Code**: [github](https://github.com/DingyiYang/LongStoryEval)  
**Area**: Others  
**Keywords**: Long-context evaluation, novel evaluation, story evaluation, evaluation criteria, LLM evaluation

## TL;DR

This paper systematically studies the automatic evaluation of book-length stories (>100K tokens), constructs the first large-scale long story evaluation benchmark LongStoryEval (600 newly published novels, 340K reader reviews), proposes a hierarchical evaluation criteria framework, compares the effectiveness of three evaluation strategies, and trains a specialized evaluation model NovelCritique-8B, which outperforms GPT-4o in alignment with human ratings.

## Background & Motivation

Automatic story evaluation has always been a challenging task in NLP: unlike evaluation tasks such as translation that focus on fluency and accuracy, story evaluation requires comprehensive assessment based on multi-dimensional, human-centered criteria. While some progress has been made in the evaluation of short stories (100-2,000 tokens), the evaluation of book-length stories (over 100K tokens) remains severely underexplored.

Long story evaluation faces three major challenges:

**Data annotation limitations**: Human evaluation is the gold standard, but annotating stories over 100K tokens is practically infeasible in terms of time and cognitive load.

**Inconsistent evaluation criteria**: Existing works use their own predefined criteria for evaluation without a unified standard, and whether these criteria reflect actual reader preferences remains unclear.

**Long text processing challenges**: Book-length stories often exceed the 128K token context limit of most LLMs; even within the limit, processing such long contexts remains a challenge for models.

## Method

### Overall Architecture

The study is divided into four levels:
1. Dataset creation: Collect ratings and reviews for 600 new books.
2. Evaluation criteria analysis: Extract and organize evaluation dimensions from reader reviews.
3. Evaluation method comparison: Compare three processing strategies for long stories.
4. Specialized model training: Train the NovelCritique evaluation model.

### Key Designs

1. **LongStoryEval Dataset Construction**: Collect 600 novels published from 2024 to January 2025, with an average length of 121K tokens (maximum 397K). These books do not appear in the training data of the evaluated LLMs, avoiding data contamination issues. Gather the average rating and 340K reader reviews for each book from the Goodreads platform.

2. **Structured Review Processing**: Raw reviews are usually unstructured and lack clarity. Use LLMs (DeepSeek-v2.5 as the main engine, GPT-4o as backup) to reformat raw reviews into: identifying evaluation aspects mentioned by users $\rightarrow$ extracting opinions on each aspect $\rightarrow$ summarizing into a concise overall evaluation. A quality threshold of 40% lexical overlap is set to filter out low-quality processed reviews.

3. **Hierarchical Evaluation Criteria**: Extract over 1,000 user-mentioned evaluation aspects from reader reviews, analyze the most frequent aspects, and organize them into a hierarchical structure:

    - **Objective aspects (5)**: Plot & Structure, Characters, Writing & Language, World-building & Setting, Themes
    - **Subjective aspects (3)**: Emotional Impact, Overall Enjoyment & Engagement, Expectation Fulfillment
    - A total of 8 top-level dimensions and 20 sub-dimensions.

4. **Comparison of Three Evaluation Methods**:

    - **Aggregation-Based Evaluation**: Evaluate chapter by chapter and take the average. When evaluating each chapter, provide book metadata, the current chapter, and the plot summary of previous parts. The advantage is access to all details; the disadvantage is the high computational overhead.
    - **Incremental-Updated Evaluation**: Simulate the process of a reader reading, updating evaluation opinions and ratings chapter by chapter. Though theoretically sound, its practical performance is limited—the model must simultaneously understand current content and update prior evaluations, and inconsistencies accumulate.
    - **Summary-Based Evaluation**: First condense the entire book into a plot summary, character analysis, and writing excerpts using incremental summarization, then perform evaluation based on the summaries. This is highly efficient and allows reuse of the summaries.

5. **NovelCritique Model**: Based on Llama 3.1-8B, trained using a summary-based framework. Key designs include:

    - **Review Bias Mitigation**: It was found that authors giving moderate ratings write reviews less frequently, leading to a skewed rating distribution in the training data. This bias is corrected by filtering training reviews according to the actual rating distribution of each book.
    - **Rating Standardization**: Rating standards vary greatly among different users. Use $S' = \frac{S - \mu_u}{\sigma_u} \times \sigma_{plat} + \mu_{plat}$ to normalize, where $\mu_u, \sigma_u$ represent the user's personal rating statistics and $\mu_{plat}, \sigma_{plat}$ represent the platform's overall statistics.

### Loss & Training

Cross-entropy loss with instruction tuning is used:

$$-\log P(r_{i \leq m}, R, S' | X_{\text{Instruct, Metadata, Sum, Excerpts}}, a_{i \leq m})$$

Training parameters: learning rate 1e-5, batch size 32, 3 epochs. LoRA parameters r=64, alpha=16. Trained on 4 A6000 GPUs for about 125 hours. The training set uses 176K filtered reviews from the remaining 450 books, with input summaries incrementally generated by GPT-4o.

## Key Experimental Results

### Main Results

System-level Kendall correlation coefficient (model rating vs. human average rating):

| Evaluation Method | Model | Overall Kendall τ |
|---------|------|-------------------|
| One-Pass | GPT-4o | 5.5 |
| Aggregation | GPT-4o | 15.2 |
| Aggregation | DeepSeek-v2.5 | 15.1 |
| Incremental | GPT-4o | 10.9 |
| Summary | GPT-4o | 13.4 |
| Summary | DeepSeek-v2.5 | 14.4 |
| Summary | Llama 3.1-8B | 12.4 |
| **NovelCritique-8B** | **Summary** | **27.7** |

Kendall correlation coefficients across dimensions (NovelCritique-8B):

| Dimension | PLOT | CHA | WRI | WOR | THE | EMO | ENJ | EXP |
|------|------|-----|-----|-----|-----|-----|-----|-----|
| Kendall τ | 27.1 | 27.0 | 24.1 | 18.3 | 24.3 | 27.8 | 21.1 | 25.5 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Remove review structuring | Performance drops significantly | Structured reviews are crucial for training |
| Remove review bias mitigation| Performance drops | Rating distribution skew affects model prediction |
| Remove rating standardization | Performance drops | Calibration is needed due to varying user rating criteria |
| Detailed summary vs. concise summary | Slight improvement | More detailed but requires balancing length |
| GPT-4o-mini summary vs. GPT-4o summary | No significant drop | Cost-effective models can be used for summary generation |

### Key Findings

- **Key aspects influencing final rating**: Among objective aspects, plot and characters are the most influential; world-building and writing quality have the least impact (possibly because most stories do not differ wildly in these aspects). Subjective aspects (emotional impact, enjoyment, expectation fulfillment) are also crucial.
- One-Pass evaluation (directly processing the entire book) has an extremely low Kendall correlation coefficient (5.5), indicating that current LLMs cannot effectively "read" and evaluate a long book all at once.
- Aggregation-based and summary-based methods significantly outperform the incremental-updated approach. The limitations of incremental updating are: (a) high demands on model capabilities (needs to simultaneously understand + update evaluations), and (b) accumulation of inconsistencies.
- A core issue with closed-source LLMs is inconsistency—even at temperature=0, results across multiple evaluations vary significantly.
- Generating summaries with a cost-effective model (GPT-4o-mini) and then evaluating with a stronger model is highly cost-efficient.
- Existing models tend to focus on the strengths of the story and under-represent weaknesses, leading to overly high ratings for poorly reviewed books.

## Highlights & Insights

1. **Data-Driven Evaluation Criteria**: Unlike previous practices of hardcoding evaluation criteria, this paper distills criteria from real reader reviews, ensuring that the evaluation system reflects the actual preferences of readers. This data-driven methodology is highly valuable for other evaluation tasks.
2. **Identification and Treatment of Review Bias**: The paper identifies selection bias in Goodreads reviews (extreme rating users are more likely to write reviews) and mitigates it by filtering according to the actual rating distribution—an insight valuable for all user review-based studies.
3. **Summary Reuse for Efficiency**: A single high-quality summary can serve multiple evaluations and provide pre-evaluation feedback during early writing stages—this writer-oriented "early feedback" application scenario is highly practical.
4. **Large-Scale Evaluation Benchmark**: With 600 uncontaminated new books and 340K structured reviews, this work provides the first large-scale benchmark for book-length story evaluation.

## Limitations & Future Work

- The evaluation is based on rating generation, which is inherently prone to high inconsistency; pairwise comparison might be more stable but comes with higher computational costs.
- Current evaluation leans toward general assessment, without considering personalized preferences (different readers may have completely different evaluations of the same book).
- Main experiments are conducted on English novels; generalization to other languages remains to be validated.
- NovelCritique is based on Llama 3.1-8B, which is constrained by the context window and still relies on summaries rather than directly processing the full text.
- Due to copyright restrictions, only plot and character summaries are put up rather than the full book text, affecting reproducibility.
- The quality of reviews depends on the LLM's structured processing, which might introduce bias.
- The overall Kendall correlation coefficient is still relatively low (even the best is only 27.7), indicating that book-length story evaluation remains a highly open research question.

## Related Work & Insights

- Complementary to long-context evaluation works such as BookScore (Chang et al., 2023), but this paper focuses on establishing evaluation criteria and comparing evaluation methods.
- Short-story evaluation works like HANNA (Chhun et al., 2022) provide methodological references, which this paper extends to the book level.
- Brings direct value to LLM-based writing assistants: NovelCritique can provide early quality feedback to authors.
- The discovery of review bias also offers insights for fields like recommendation systems and review mining.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic study on book-length story evaluation, with significant contributions in the dataset and evaluation framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale dataset of 600 books, comprehensive comparison of 3 evaluation strategies × 6 models, including ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly organized with rich charts, though some analysis could be more in-depth.
- Value: ⭐⭐⭐⭐ Lays the foundation for evaluating book-length stories, and the NovelCritique model holds direct practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)
- [\[ACL 2025\] Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](completing_a_systematic_review_in_hours.md)
- [\[ACL 2025\] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)

</div>

<!-- RELATED:END -->
