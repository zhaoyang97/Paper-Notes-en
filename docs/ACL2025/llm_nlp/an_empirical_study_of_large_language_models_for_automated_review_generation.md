---
title: >-
  [Paper Note] An Empirical Study of Large Language Models for Automated Review Generation
description: >-
  [ACL 2025][LLM (Other)][Automated Review Generation] This paper presents a systematic empirical study evaluating the capabilities of various large language models (LLMs) in automatically generating peer reviews for academic papers. It analyzes the quality, consistency, and utility of the generated reviews, uncovering the strengths, weaknesses, and directions for improvement of current LLMs in the task of review generation.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Automated Review Generation"
  - "Large Language Models"
  - "Academic Peer Review"
  - "Text Generation Quality"
  - "Review Automation"
date: 2026-05-08
content_hash: 8e4067b948a67154
---

# An Empirical Study of Large Language Models for Automated Review Generation

**Conference**: ACL 2025  
**Area**: LLM/NLP  
**Keywords**: Automated Review Generation, Large Language Models, Academic Peer Review, Text Generation Quality, Review Automation

## TL;DR
This paper presents a systematic empirical study evaluating the capabilities of various large language models (LLMs) in automatically generating peer reviews for academic papers. It analyzes the quality, consistency, and utility of the generated reviews, uncovering the strengths, weaknesses, and directions for improvement of current LLMs in the task of review generation.

## Background & Motivation

**Background**: Academic peer review is facing a severe workload crisis—the volume of submissions increases annually while the number of qualified reviewers remains limited, leading to prolonged review cycles and inconsistent review quality. The emergence of LLMs offers the potential to assist or partially automate the reviewing process. While prior research has explored LLM performance on specific reviewing sub-tasks (e.g., summary rating, methodological flaw detection), there lacks a systematic evaluation of complete review generation.

**Limitations of Prior Work**: Automated review generation faces multiple challenges: it requires a deep understanding of the paper's technical content, accurate evaluation of novelty and methodological rigor, provision of specific and actionable improvement suggestions, and maintaining fairness and consistency. Most existing work focuses on "whether LLMs can detect paper flaws" rather than "whether LLMs can generate high-quality complete reviews."

**Key Challenge**: LLMs possess powerful text understanding and generation capabilities, but academic reviewing also demands domain expertise, critical thinking, and constructive feedback capabilities—precisely the weak points of current LLMs.

**Goal**: To systematically evaluate the capability of mainstream LLMs in generating complete academic reviews, analyze the gap between generated reviews and human reviews across multiple quality dimensions, and propose directions for improvement.

**Key Insight**: Selecting papers with existing real human reviews as a test suite and prompting LLMs to generate reviews for the same papers, then comparing the quality differences between the two using automatic metrics and human evaluation.

**Core Idea**: Unveiling the capability boundaries and quality patterns of LLM-based automated reviewing through large-scale comparative experiments.

## Method

### Overall Architecture
Papers alongside their corresponding human reviews from top venues (ICLR, NeurIPS, ACL, etc.) are collected as the evaluation benchmark. For each paper, given its full text, various LLMs (GPT-4o, Claude 3.5, Llama 3-70B, etc.) are employed to generate structured reviews (including summary, strengths, weaknesses, recommendations, and overall score). The generated reviews are then comparatively evaluated across five dimensions: content coverage, specificity, accuracy, consistency, and helpfulness.

### Key Designs

1. **Multi-Dimensional Review Quality Evaluation Framework**:

    - **Function**: Comprehensively evaluate the quality of automated reviews
    - **Mechanism**: Define five orthogonal quality dimensions: (1) content coverage—whether the review covers key aspects of the paper (methodology, experiments, writing); (2) specificity—whether comments are specific to particular parts of the paper rather than generic; (3) accuracy—whether technical comments are correct; (4) consistency—whether the logic between evaluation (strengths/weaknesses) and the final rating is consistent; (5) helpfulness—whether the improvement suggestions are concrete and actionable. Each dimension is evaluated on a 1-5 Likert scale
    - **Design Motivation**: A single "good/bad" evaluation fails to reveal the specific strengths and weaknesses of LLM review generation

2. **Review Counterfactual Test**:

    - **Function**: Detect whether systematic biases exist in LLM reviews
    - **Mechanism**: Generate multiple reviews for the same paper to observe rating variance; check whether LLMs can correctly distinguish paper pairs with distinct quality differences (e.g., accepted vs. rejected); and test whether the reviews are influenced by non-academic factors by modifying paper metadata (such as authors' institutions, citation count)
    - **Design Motivation**: Review bias is a major concern in academia, and automated review systems must demonstrate fairness

3. **Relationship Analysis between Context Length and Review Depth**:

    - **Function**: Understand how paper length/complexity affects LLM review quality
    - **Mechanism**: Group test papers by length and methodological complexity to analyze how LLM review quality varies across different groups. Particular attention is paid to whether LLMs exhibit a "lost in the middle" phenomenon when processing long papers—i.e., focusing only on the beginning and end of the paper while neglecting core methodology sections
    - **Design Motivation**: Real-world paper lengths vary significantly; understanding the relationship between review quality and paper characteristics is crucial for practical deployment

### Loss & Training
This is an evaluation study and does not involve model training. The evaluation utilizes approximately 500 papers, each with 2-4 human reviews as references.

## Key Experimental Results

### Main Results

| Model | Content Coverage | Specificity | Accuracy | Consistency | Helpfulness | Overall Quality |
|------|----------|--------|--------|--------|--------|---------|
| GPT-4o | 3.82 | 3.15 | 3.28 | 3.91 | 2.87 | 3.41 |
| Claude 3.5 | 3.76 | 3.31 | 3.42 | 4.02 | 3.05 | 3.51 |
| Llama 3-70B | 3.41 | 2.86 | 2.95 | 3.55 | 2.53 | 3.06 |
| Human Review | 3.95 | 3.89 | 3.75 | 3.62 | 3.67 | 3.78 |

### Bias and Consistency Analysis

| Evaluation Dimension | GPT-4o | Claude 3.5 | Human | Description |
|---------|--------|-----------|------|------|
| Rating Variance (3 runs/paper) | 0.31 | 0.28 | 0.87 | LLMs are more consistent but perhaps too consistent |
| Accept/Reject Distinction (ACC) | 71.2% | 73.8% | 82.5% | Gap remains in LLM distinguishing capability |
| Institutional Bias (score change after renaming) | 0.12 | 0.08 | - | Slightly affected |
| Quality Decay on Long Papers | -0.34 | -0.21 | -0.05 | LLMs still struggle with long document processing |

### Key Findings
- **Specificity and Helpfulness represent the largest gaps**: LLM reviews approach human-level performance in content coverage and internal consistency, but fall significantly short in delivering specific, paper-section-targeted comments and actionable improvement suggestions. LLMs tend to output generic critiques like "experiments should be more thorough" instead of specifying which baseline is missing or what exact experiment is needed.
- **LLM rating consistency is "too high"**: The rating variance of multiple runs by the same LLM on the same paper is far lower than that of human reviewers (0.28-0.31 vs. 0.87). While seemingly an advantage, this actually reflects the lack of diverse perspectives in LLM reviews.
- **Claude leads in accuracy**: Claude 3.5 demonstrates the highest accuracy in evaluating technical content, which might be attributed to its more rigorous reasoning pre-training.
- **Long papers significantly degrade review quality**: For papers exceeding 20 pages, LLM review quality drops markedly, confirming that long-context processing remains a bottleneck.

## Highlights & Insights
- This study presents the first systematic, multi-dimensional evaluation framework for automated review quality, establishing a standardized evaluation methodology for future work.
- The finding of "excessive consistency" reveals an interesting paradox: although high variance in human reviews is often deemed a flaw, a certain degree of diversity reflects complementary expert viewpoints. LLM reviews risk being overly "mediocre."

## Limitations & Future Work
- The evaluation is confined to the computer science domain; review patterns in fields like biomedicine and social sciences may differ.
- LLM reviews may suffer from data leakage issues if they have already seen the papers in their training data, potentially causing evaluation bias.
- The human evaluators' judgments are inherently subjective, introducing potential inter-rater bias into the evaluation results.
- Iterative review scenarios are not considered; in practice, peer reviews involve multiple rounds of author feedback and reviewer revisions.
- Future work could explore a collaborative "AI-assisted reviewing" paradigm that integrates LLM reviews with human expertise.
- Future endeavors could combine LLM reviews with human reviews to form an "AI-assisted peer review" model, capitalizing on LLMs' coverage advantages and humans' depth of expertise.

## Related Work & Insights
- **vs ReviewerGPT**: ReviewerGPT focuses on detecting paper flaws, whereas this work evaluates complete review generation.
- **vs MARG**: MARG uses retrieval augmentation to improve review quality, while this work serves as a benchmark evaluation that reveals directions for improvement.
- **vs DeepCRCEval**: DeepCRCEval focuses on code review, whereas this work addresses academic paper reviewing.
- The five-dimensional quality evaluation framework can be directly utilized to train reward models for specialized review-generation models.

## Rating
- Novelty: ⭐⭐⭐ Primarily an empirical study; while methodological innovation is limited, the evaluation framework is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing 500 papers across multiple models and dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured analysis.
- Value: ⭐⭐⭐⭐ Provides significant reference value for research on academic review automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)
- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] A Systematic Study of Compositional Syntactic Transformer Language Models](a_systematic_study_of_compositional_syntactic_transformer_language_models.md)

</div>

<!-- RELATED:END -->
