---
title: >-
  [Paper Note] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?
description: >-
  [ACL 2025][LLM (Other)][Psychometrics] Evaluating the "human-likeness" of 18 instruction-tuned LLMs in educational assessment from a psychometric perspective (Classical Test Theory, CTT and Item Response Theory, IRT), this study finds that even after temperature scaling calibration, LLM response distributions are inherently different from humans—large models are overconfident and fail to predict patterns of humans being attracted by distractors…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Psychometrics"
  - "Item Response Theory"
  - "Classical Test Theory"
  - "Educational Assessment"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 165d255903c19991
---

# Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?

**Conference**: ACL 2025  
**arXiv**: [2506.09796](https://arxiv.org/abs/2506.09796)  
**Code**: [https://github.com/mainlp/llm-psychometrics](https://github.com/mainlp/llm-psychometrics)  
**Area**: LLM/NLP, Educational Assessment  
**Keywords**: Psychometrics, Item Response Theory, Classical Test Theory, Educational Assessment, LLM Evaluation

## TL;DR

Evaluating the "human-likeness" of 18 instruction-tuned LLMs in educational assessment from a psychometric perspective (Classical Test Theory, CTT and Item Response Theory, IRT), this study finds that even after temperature scaling calibration, LLM response distributions are inherently different from humans—large models are overconfident and fail to predict patterns of humans being attracted by distractors, suggesting that zero-shot LLMs are not suitable to replace humans in test piloting.

## Background & Motivation

**Background**: Developing educational assessments (e.g., SAT, GRE) is a long and expensive process, involving expert item drafting and repeated piloting with hundreds to thousands of human participants to evaluate item quality. Recent research has explored using LLMs to simulate participants to accelerate this process.

**Limitations of Prior Work**: For LLMs to effectively replace human respondents, their responses must exhibit "human-likeness" within psychometric frameworks (CTT and IRT)—meaning items that LLMs find difficult should also be difficult for humans, and LLMs' distractor distributions should resemble those of humans. However, there is a lack of systematic studies evaluating the psychometric plausibility of LLMs under both CTT and IRT frameworks.

**Key Challenge**: LLMs are already highly capable in terms of accuracy, but accuracy $\neq$ human-likeness. A model that always selects the correct answer (being overconfident) cannot provide useful item analysis information, as it fails to reveal which items are "good" (i.e., highly discriminating items).

**Goal**: Propose a psychometric plausibility evaluation method based on CTT and IRT, and systematically benchmark it across 18 LLMs $\times$ 2 datasets $\times$ 3 subjects.

**Key Insight**: Treat the first-token probability distribution of LLMs as "the response distribution of a group of virtual human test-takers" and compare it with the response statistics of real human test-takers.

**Core Idea**: Quantitatively measure the psychometric plausibility of LLM responses using CTT item difficulty correlation and IRT item characteristic curve fitting.

## Method

### Overall Architecture

A three-dimensional evaluation method for psychometric plausibility is proposed: (1) Response distribution comparison—using KL divergence to measure the similarity between LLM and human option probability distributions; (2) CTT analysis—using Pearson correlation coefficients to measure the consistency between LLM's correct answer probability and human item facility (difficulty); (3) IRT analysis—comparing LLM responses with Item Characteristic Curves (ICC) fitted from human data.

### Key Designs

1. **Response Probability Extraction and Temperature Scaling**:

    - **Function**: Extract the probabilities of the four options A/B/C/D from the first-token logits of the LLM, use cyclic permutation to eliminate option position bias, and calibrate the distribution via temperature scaling.
    - **Mechanism**: Generate 4 responses for each item (one for each option permutation) and take the average probability. Then, minimize KL divergence to search for the optimal temperature parameter $T$, making the LLM response distribution as close to human as possible. LLMs are typically overconfident (almost all probability is concentrated on a single option); temperature scaling can mitigate this.
    - **Design Motivation**: Directly using raw logits makes the models appear extremely overconfident (with 90%+ probability on the correct option), making it impossible to compare with the human distribution (which is typically more dispersed).

2. **CTT Analysis—Item Difficulty Correlation**:

    - **Function**: Calculate the Pearson correlation coefficient between the LLM's correct response probability and human item facility (proportion of humans answering correctly).
    - **Mechanism**: If an LLM is psychometrically plausible, then for items that humans find easy, the LLM should also assign a higher probability to the correct option. A strong positive correlation indicates that the LLM aligns with humans along the item difficulty dimension.
    - **Design Motivation**: CTT is the most fundamental analytical framework in educational assessment, and item facility is the most intuitive item characteristic.

3. **IRT Analysis—ICC Goodness-of-Fit**:

    - **Function**: Conduct correlation analysis between the LLM's correct probability and the expected response probability ($\theta=0$, i.e., average-ability test-taker) of the 3PL (three-parameter logistic) IRT model: $P(X=1) = c + \frac{1-c}{1+e^{-a(\theta-b)}}$.
    - **Mechanism**: The ICC of IRT includes the item discrimination parameter $a$, difficulty parameter $b$, and guessing parameter $c$, providing a more granular description of item characteristics than CTT. If the LLM aligns with these characteristics, it indicates that its response behavior can simulate a human respondent with a specific ability level.
    - **Design Motivation**: The advantage of IRT lies in item parameters being independent of specific respondent populations, providing a more theoretically grounded evaluation standard.

### Datasets

- **NAEP** (National Assessment of Educational Progress): 549 four-option multiple-choice items (Reading 252, US History 204, Economics 93), covering grades 4/8/12, with human response distributions and IRT parameters.
- **CMCQRD** (Cambridge Multiple Choice Reading Dataset): 504 four-option multiple-choice items covering four CEFR levels from B1 to C2, targeting English as a Second Language (ESL) learners, with human response distributions.

## Key Experimental Results

### Main Results: Response Distribution Similarity (KL Divergence)

| Model Family | Size | NAEP Reading | NAEP History | CMCQRD B1 |
|--------|------|---------|---------|----------|
| Llama 3 | 8B | Medium-High | High | Medium |
| Llama 3 | 70B | Medium | Medium-High | Low |
| Qwen 2.5 | 72B | Medium | Medium-High | Low |
| OracleBaseline | — | Reference | Reference | Reference |

As the model size increases, the KL divergence decreases (responses become closer to humans), but only a very few large models significantly outperform the OracleBaseline (a simple baseline with high probability only for the correct option and equal probabilities for distractors) on CMCQRD B1.

### Ablation Study: CTT Correlation Analysis

| Domain | Grade/Level | Max Correlation | Min Correlation | Significant Prop. |
|------|---------|---------|---------|--------|
| CMCQRD B1 (Reading) | B1 | 0.56 | 0.32 | High |
| NAEP Reading | Grade 8 | ~0.4 | ~0.2 | Medium |
| NAEP History | Grade 8 | ~0.3 | ~0.1 | Low |
| NAEP Economics | Grade 12 | ~0.2 | ~0.1 | Very Low |

Reading comprehension shows the highest correlation (where LLMs are most "human-like"), while history and economics are very low, with some even showing significant negative correlations (where LLMs are more confident on difficult items).

### Key Findings

- LLMs are not easily "fooled" by distractors—they excel at identifying correct answers but fail completely to predict which incorrect options are attractive to humans. Temperature scaling cannot resolve this fundamental issue.
- Results across all model families and sizes are highly consistent, indicating that "human-likeness" cannot be solved simply by model scaling.
- Significant cross-disciplinary differences—reading comprehension is the most human-like, while history/economics are the least. Possible reasons: (1) Reading comprehension relies more on text comprehension (which LLMs are relatively good at) rather than long-term memory retrieval (factual knowledge required for history/economics); (2) History/Economics items more frequently contain images, which text-only LLMs cannot fully comprehend.
- Significant negative correlations appear on certain IRT scales in Grade 4 History—LLMs are more confident on difficult items, completely violating psychometric expectations.

## Highlights & Insights

- First study to systematically evaluate the psychometric plausibility of LLMs as "virtual respondents" under both CTT and IRT frameworks, offering outstanding methodological contributions.
- The finding that "LLMs are not easily fooled by distractors" carries profound implications—suggesting that LLM "errors" and human "errors" are driven by different mechanisms; LLMs do not fail because they are misled by surface features.
- The discovery of cross-disciplinary differences provides clues for understanding the cognitive traits of LLMs—their reading comprehension is closer to humans, while their knowledge-based reasoning differs substantially from humans.
- Rigorous experimental design—large-scale systematic evaluation of 18 models $\times$ 2 datasets $\times$ 3 subjects, with all code and data publicly released.

## Limitations & Future Work

- Evaluated only in zero-shot scenarios; fine-tuning (e.g., on human response distributions) might significantly improve psychometric plausibility.
- The NAEP dataset contains visual items (represented by alt text), which may have affected results in history/economics.
- Used only publicly aggregated data, lacking individual-level response data for more fine-grained analyses.
- Temperature scaling is optimized on the evaluation data itself (upper-bound estimation); practical applications would require an independent calibration set.
- Did not explore whether multimodal LLMs would exhibit more human-like behavior on items containing images.

## Related Work & Insights

- **vs Hayakawa & Saggion (2024)**: This work also used CTT to compare LLM and human item difficulty but focused solely on reading tasks; Ours extends to multiple disciplines and introduces the IRT framework.
- **vs Lalor et al. (2019)**: Adopted an "artificial population" (training multiple models on partial/corrupted data) to simulate test-takers of different ability levels; Ours explores a more modern single-LLM + temperature scaling approach.
- **vs Zotos et al. (2025)**: Predicted student response distributions using LLM uncertainty, yielding similar conclusions—zero-shot LLMs are not human-like enough.

## Rating

- Novelty: ⭐⭐⭐⭐ Evaluated from a psychometric perspective, this is a novel and valuable interdisciplinary study.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly systematic with 18 models $\times$ 2 datasets $\times$ 3 subjects $\times$ 3 evaluation dimensions (KL/CTT/IRT).
- Writing Quality: ⭐⭐⭐⭐ Psychometric concepts are clearly explained and friendly to NLP readers.
- Value: ⭐⭐⭐⭐ Provides an important negative result for the educational AI field—zero-shot LLMs cannot replace human piloting, benchmarking the current performance boundaries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_writing_assessment.md)
- [\[ACL 2025\] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)
- [\[ACL 2025\] LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_test_case_gen_bugs.md)
- [\[ACL 2025\] Which Demographics Do LLMs Default to During Annotation?](which_demographics_do_llms_default_to_during_annotation.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)

</div>

<!-- RELATED:END -->
