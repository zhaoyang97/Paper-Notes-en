---
title: >-
  [Paper Note] A Little Human Data Goes A Long Way
description: >-
  [ACL 2025][Synthetic Data] Through large-scale experiments on 8 fact verification and QA datasets, it is demonstrated that mixing a very small amount of human-annotated data (even as few as 125 samples) into synthetic data significantly improves model performance. Replacing the final 10% of human data leads to severe performance degradation, and the performance gain from just 200 human samples requires orders of magnitude more synthetic data to match.
tags:
  - "ACL 2025"
  - "Synthetic Data"
  - "Human Annotation"
  - "Fact Verification"
  - "Question Answering"
  - "Data Efficiency"
date: 2026-05-08
content_hash: 24debb6e8833b4a0
---

# A Little Human Data Goes A Long Way

**Conference**: ACL 2025  
**arXiv**: [2410.13098](https://arxiv.org/abs/2410.13098)  
**Code**: [GitHub](https://github.com/DhananjayAshok/LittleHumanData)  
**Area**: Others  
**Keywords**: Synthetic Data, Human Annotation, Fact Verification, Question Answering, Data Efficiency

## TL;DR

Through large-scale experiments on 8 fact verification and QA datasets, it is demonstrated that mixing a very small amount of human-annotated data (even as few as 125 samples) into synthetic data significantly improves model performance. Replacing the final 10% of human data leads to severe performance degradation, and the performance gain from just 200 human samples requires orders of magnitude more synthetic data to match.

## Background & Motivation

**Background**: Modern LLMs have been widely used to generate synthetic training data to alleviate the high cost of human annotation. Synthetic data generation methods have been applied in tasks such as QA, NLI, text classification, and instruction tuning.

**Limitations of Prior Work**:
- Human annotated data is expensive and time-consuming to obtain, especially for tasks requiring comprehension of complex "evidence texts" (such as fact verification and question answering).
- Whether synthetic data can completely replace human annotation remains unclear, as existing research conclusions vary by task.
- There is a lack of systematic research on the substitution capability of synthetic data in fact verification and evidence-based question answering.

**Key Challenge**: The fundamental tension between the convenience of synthetic data generation and its inability to fully replicate the quality of human-annotated data—no amount of synthetic data can easily compensate for the unique value of human data.

**Goal**: To systematically quantify the limits of substituting human-annotated data with synthetic data, particularly in Fact Verification (FV) and evidence-based Question Answering (QA) tasks, addressing questions such as "what proportion of synthetic data is safe" and "how much is a small amount of human data worth."

**Key Insight**: Holding the training set size constant while progressively increasing the proportion of synthetic data (from 0% to 100%) to conduct controlled experiments across 8 cross-domain FV and QA datasets.

**Core Idea**: Even when large-scale human annotation is unfeasible, retaining a very small proportion (2.5% to 10%) of human data can yield disproportionately massive performance gains, proving that synthetic data cannot fully replace human data.

## Method

### Overall Architecture

Few-Shot In-Context Learning is utilized to generate synthetic data from evidence texts. While keeping the total number of training samples fixed, models are trained and evaluated using mixed human and synthetic data at various ratios:

1. Sample few-shot exemplars from the human training set.
2. Use GPT-3.5-Turbo to generate synthetic (claim, label) or (question, answer) pairs from the evidence texts.
3. Replace human data with synthetic data progressively at specific ratios (0%, 10%, 25%, 50%, 75%, 90%, 95%, 97.5%, 100%).
4. Fine-tune Llama3-8B using LoRA and evaluate on the human-annotated test set.

### Key Designs

1. **Progressive Synthetic Data Substitution Experiment**:
    - **Function**: To quantify the performance variation curve as synthetic data replaces human data.
    - **Mechanism**: Keeping the training set size constant while varying only the mixing ratio of synthetic and human data.
    - **Design Motivation**: To eliminate the confounding factor of changing training size, thereby precisely measuring the effect of data source.

2. **Zoomed-in Analysis on the Extreme Interval (90%-100%)**:
    - **Function**: To verify the disproportionate value of the "last 10% of human data".
    - **Mechanism**: Fixing $n=5000$ and conducting fine-grained experiments at synthetic ratios of 95%, 97.5%, and 100%.
    - **Design Motivation**: This revealed that a mere 2.5% (125 samples) of human-annotated data can significantly improve models trained on purely synthetic data.

3. **Quantifying Performance-Cost Trade-offs**:
    - **Function**: To estimate the equivalent exchange rate between human and synthetic data.
    - **Mechanism**: Fitting the synthetic data performance curve using $y = a_0 + a_1 \log(x)$ to calculate how much synthetic data is required to substitute for 200 human data points.
    - **Design Motivation**: To provide an actionable cost-benefit analysis framework for practical decision-making.

### Loss & Training

- The FV tasks employ the standard classification cross-entropy loss, while the QA tasks use the sequence generation loss.
- Models are fine-tuned on Llama3-8B using LoRA, with robustness verified on other models like Mistral and MPT.
- GPT-3.5-Turbo serves as the primary synthetic data generator, with GPT-4 and Claude-3.5-Sonnet used for validation.
- Chain-of-Thought prompting strategies are evaluated as an additional robustness check.

## Key Experimental Results

### Main Results

| Synthetic Data Ratio | Typical Performance Trend (Relative to All-Human) |
|---|---|
| 0% → 90% | Small, gradual decline |
| 90% → 100% | Sharp drop in performance, often exceeding the total decline of 0-90% |
| 97.5% → 100% | Removing just 125 human samples causes substantial performance loss |

### Equivalency of Synthetic Data in Replacing Human Data

| Dataset | Additional Synthetic Data Needed to Match 200 Human Samples (Mean) |
|---|---|
| WANLI | 17,671 |
| ROPES | 17,333 |
| FairyTaleQA | 281,951 |
| FEVER | 1,155 |

### Ablation Study

| Dimension | Conclusion |
|---|---|
| Cross-lingual (Arabic/Georgian/Indonesian) | Trends remain consistent |
| Cross-model (Mistral/MPT) | Trends remain consistent |
| Cross-generator (GPT-4/Claude-3.5) | Trends remain consistent |
| OOD evaluation (Cross-dataset Train-Test) | Ruling out spurious correlations; the value of human data is genuine |
| CoT prompting strategy | Trends remain unchanged |

### Key Findings

- Replacing 0-90% of human data results in only marginal performance loss, but replacing the last 10% leads to a severe drop.
- A mere 125 human samples (2.5%) can significantly boost models trained on purely synthetic data.
- The performance gain of 200 human data points typically requires 1 to 2 orders of magnitude more synthetic data to match.
- On FairyTaleQA, the equivalent exchange ratio reaches as high as ~2e5:200, implying that human data might unlock performance ceilings that synthetic data cannot achieve.
- OOD experiments rule out explanations based on spurious correlations (such as annotation artifacts) between training and test sets.

## Highlights & Insights

- **Deep Insights**: Uncovered a non-linear substitution relationship between synthetic and human data—essentially lossless up to 90%, with the final 10% being extremely critical.
- **High Practical Value**: Offers clear guidance for real-world engineering decisions—even with a tight budget, a small fraction of human annotations should be preserved.
- **Analysis of Synthetic vs. Human Data**: Reveals that synthetic data tends to be longer and more extractive (sharing higher n-gram overlap with evidence texts), while human data demonstrates better paraphrasing and more diverse vocabulary usage.
- **Counter-Intuitive Finding**: Synthetic data samples from more diverse locations within the evidence text, whereas human annotators disproportionately focus on the beginning sections of the evidence text.
- **Cost-Benefit Framework**: Proposes an actionable method to estimate price-ratio thresholds (e.g., on WANLI, human data is more cost-effective as long as the unit cost of synthetic data exceeds 1/73 that of human data).

## Limitations & Future Work

- The study primarily focuses on English; although multi-lingual trends are consistent, the exact amount of human data required may vary.
- Limited control over data leakage—only a subset of datasets is confirmed not to be included in GPT-3.5 training data.
- No actionable modeling improvements were direct outputs of the error analysis.
- Spans only two types of tasks (FV and QA); more complex generative tasks remain unexplored.
- Did not investigate optimal human data selection strategies (e.g., active learning to filter the most valuable human samples).

## Related Work & Insights

- **vs. Training on Purely Synthetic Data**: Consistent with model collapse research, though in this context, purely synthetic training still achieves reasonable performance, possibly because the diversity of evidence texts provides grounding.
- **vs. Li et al. (2023) Subjectivity Analysis**: Complementary; while Li et al. found synthetic data performs worse on subjective tasks, this work focuses on relatively objective tasks like FV and QA.
- **vs. Image/Multimodal Domains**: Concepts from Fan et al., He et al., etc., align with findings here—synthetic data is useful but must be combined with human data.
- **vs. Bisbee et al. (2024)**: Purely synthetic data is unreliable for replacing political survey respondents, consistent with the conclusion that synthetic data "cannot fully replace human data."

## Rating

- **Novelty**: ⭐⭐⭐ The method itself is not brand new (synthetic replacement experiments), but the experimental design is highly elegant and yields high-value insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 datasets × multiple models × multiple generators × multilingual × OOD, exceptionally thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rigorous logical arguments, and intuitive visualizations.
- **Value**: ⭐⭐⭐⭐ Directly impacts data annotation strategies in NLP with concise and powerful conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Forecasts of Suicide Attempts for Patients with Little Data](../../NeurIPS2025/others/improving_forecasts_of_suicide_attempts_for_patients_with_little_data.md)
- [\[ACL 2025\] HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling](helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)
- [\[ACL 2025\] Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs](anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)

</div>

<!-- RELATED:END -->
