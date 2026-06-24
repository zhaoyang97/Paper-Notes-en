---
title: >-
  [Paper Note] Call for Rigor in Reporting Quality of Instruction Tuning Data
description: >-
  [ACL 2025][LLM Alignment][Instruction Tuning] Through systematic experiments across 16 hyperparameter combinations, this study reveals a severe issue in evaluating the quality of instruction tuning data: researchers' arbitrary choice of training hyperparameters can lead to entirely opposite conclusions (e.g., "Data A is superior to Data B"). It calls for the mandatory use of validated hyperparameter configurations when reporting data quality.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Instruction Tuning"
  - "Data Quality"
  - "Hyperparameter Selection"
  - "Experimental Rigor"
  - "Reproducibility"
date: 2026-05-08
content_hash: 6559f44c1c75f16f
---

# Call for Rigor in Reporting Quality of Instruction Tuning Data

**Conference**: ACL 2025  
**arXiv**: [2503.04807](https://arxiv.org/abs/2503.04807)  
**Code**: None  
**Area**: LLM Alignment  
**Keywords**: Instruction Tuning, Data Quality, Hyperparameter Selection, Experimental Rigor, Reproducibility

## TL;DR

Through systematic experiments across 16 hyperparameter combinations, this study reveals a severe issue in evaluating the quality of instruction tuning data: researchers' arbitrary choice of training hyperparameters can lead to entirely opposite conclusions (e.g., "Data A is superior to Data B"). It calls for the mandatory use of validated hyperparameter configurations when reporting data quality.

## Background & Motivation

**Background**: Instruction tuning (IT) is a key technique for aligning LLMs with user intent, and extensive research highlights the importance of IT data quality. The standard practice for evaluating data quality is: "train a model with the data under evaluation, and then use the model's performance to represent data quality" — i.e., "good data yields a good model."

**Limitations of Prior Work**: A systematic survey by the authors (Table 1) reveals that under the exact same setup of training Llama-2-7B with 1K IT data, hyperparameters vary drastically across different studies: learning rates range from 1e-5 to 5e-5 (a 5x difference), epochs from 3 to 15 (a 5x difference), and batch sizes from 8 to 128 (a 16x difference), with most studies lacking sufficient justification for their hyperparameter choices.

**Key Challenge**: If the choice of hyperparameters can alter the conclusion that "Data A is better than Data B," are the conclusions of numerous current studies on IT data quality actually reliable?

**Goal**: To prove the existence and severity of this issue through rigorous experimentation and to propose recommendations for improvement.

**Key Insight**: Using LIMA and Alpaca-Longest, two classic IT datasets, as case studies to systematically vary hyperparameters and observe whether conclusions flip.

**Core Idea**: Arbitrary choices of hyperparameters can make almost any conclusion hold — data quality research must report validated hyperparameter configurations.

## Method

### Overall Architecture

This paper does not propose a new method, but rather designs a rigorous experiment to expose flaws in current research practices. Core experimental design: train LLMs with two IT datasets (LIMA and Alpaca-Longest) respectively, systematically vary 4 hyperparameters (each taking 2 common values) to generate 16 combinations of configurations, and compare whether the findings on data quality remain consistent across these configurations.

### Key Designs

1. **Exam-taker Dataset Selection**:

    - **Function**: Select two well-investigated IT datasets that have contradictory conclusions in the literature.
    - **Specific Datasets**: (1) LIMA — 1,000 highly curated, high-quality instruction-following samples; (2) Alpaca-Longest — 1,000 samples with the longest token count selected from Alpaca (the original paper Zhao et al. 2024a reports its training effect defeats LIMA).
    - **Design Motivation**: If even the relative quality of these two widely studied datasets can flip due to hyperparameters, the ubiquity of this issue is self-evident.

2. **Systematic Hyperparameter Combinations**:

    - **Function**: Vary the 4 most common training hyperparameters in a controlled manner, resulting in $2^4 = 16$ combinations.
    - **Variable Settings**: Learning rate {1e-5, 2e-5}, scheduler {Linear, Cosine}, batch size {64, 256}, epoch {3, 15}.
    - **Design Motivation**: These 4 parameters cover the dimensions with the largest discrepancies in the survey of Table 1, and each value is taken from actual settings used in published work.

3. **Multi-Benchmark Evaluation**:

    - **Function**: Conduct pairwise comparisons on 3 LLM alignment benchmarks (Koala, MT-Bench, Self-Instruct) using GPT-4o as the judge.
    - **Design Motivation**: Avoid bias from a single benchmark to ensure the robustness of the conclusions.

## Key Experimental Results

### Main Results: LIMA vs Alpaca-Longest (Llama-2-7B)

| Hyperparameter Setting | Conclusion on Koala | On MT-Bench | On Self-Instruct |
|-----------|-----------------|---------------|-------------------|
| Setting 4, 5, 10, 12, 13 | Alpaca-Longest outperforms LIMA | Mostly consistent | Mostly consistent |
| Setting 8, 16 | LIMA outperforms Alpaca-Longest | Mostly consistent | Mostly consistent |
| Other settings | Inconsistent or benchmark-dependent | — | — |

**Core Conclusion**: Simply by choosing different hyperparameters, researchers can reach completely opposite conclusions regarding data quality.

### Ablation Study: Impact of Hyperparameters Within the Same Dataset

| Comparison (Koala dataset, LIMA) | Setting x Wins | Tie | Setting 1 Wins |
|--------------------------|---------------|-----|---------------|
| Setting 1 vs Setting 7 | 89 | 8 | 83 |
| Setting 1 vs Setting 12 | 27 | 7 | 146 |
| Setting 1 vs Setting 15 | 91 | 7 | 82 |

The performance of models trained on the same dataset under different hyperparameters varies dramatically — Setting 12 vs Setting 1 has a win rate of 146:27, whereas Setting 15 vs Setting 1 is 91:82.

### Finding the Optimal Settings

| Key Findings | Details |
|---------|------|
| Locally optimal setting | Setting 7, 15 (2e-5 LR / 256 Batch / 15 Epochs) |
| Most critical hyperparameter | Epoch (the difference between 3 vs 15 is most significant) |
| Impact on existing studies | Most studies choose 3 epochs, but 15 epochs performs significantly better |
| Potential conclusion | Many published results might stem from undertrained models |

### Key Findings

- The number of epochs is the most influential hyperparameter. Most existing studies choose to train for 3 epochs, which may lead to model undertraining and fail to fully exploit the data potential.
- Hyperparameters exhibit interaction effects; optimizing one parameter individually does not guarantee an optimal combination.
- Replicated experiments on Mistral-7B yielded the same pattern of conclusions, validating the cross-model generality of the issue.

## Highlights & Insights

- Unveils an overlooked yet consequential methodological issue: the arbitrariness of hyperparameter choices in data quality research may have already caused confusion in the research direction as a whole.
- Highly empirical — instead of relying on conjecture, the study clearly demonstrates the problem with experimental evidence across 16 settings: 16 settings $\times$ 3 benchmarks = 48 group experiments, which is highly convincing.
- Expresses clear and actionable recommendations: (1) Report at least a locally optimal configuration from a hyperparameter pool; (2) Clearly state the rationale for hyperparameter selection; (3) Adopt established standard configurations such as LIMA's settings.
- The paper itself stands as an exemplar of "scientific rigor," exposing a crucial issue through highly concise experimental design.

## Limitations & Future Work

- Only considers 4 hyperparameters (learning rate, scheduler, batch size, epoch), leaving weight decay, dropout, warmup steps, etc., unexplored.
- Only uses 2 IT datasets (LIMA and Alpaca-Longest), which is sufficient to demonstrate the issue but offers limited coverage.
- Did not attempt automated hyperparameter optimization (HPO), but only explored discrete grid search.
- Evaluated on 7B-scale models only; patterns might vary on larger or smaller models.
- No "standardized" hyperparameter search protocol is provided for direct adoption by the community.

## Related Work & Insights

- **vs LIMA (Zhou et al. 2023)**: LIMA claims that 1,000 highly curated samples can align LLMs, but this conclusion may be restricted by its own selected hyperparameter configuration.
- **vs Alpaca-Longest (Zhao et al. 2024)**: This prior work reports that Alpaca-Longest outperforms LIMA, but this study proves that this conclusion can flip under different hyperparameters.
- **vs Hyperparameter Optimization (HPO) Research**: Although HPO is common practice in smaller models (like BERT, BART), it has been largely neglected in the LLM era due to cost considerations — a trade-off issue that demands attention.

## Rating

- Novelty: ⭐⭐⭐⭐ Uncovers an important yet overlooked methodological issue; though the experiments themselves are not complex, the insights are deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ Highly systematic, covering 16 configurations $\times$ 3 benchmarks $\times$ 2 datasets $\times$ 2 models.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning with a highly persuasive three-part structure (Problem-Evidence-Recommendation).
- Value: ⭐⭐⭐⭐⭐ Has profound methodological implications for the entire research direction of IT data quality, warranting widespread attention from the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[ACL 2025\] Federated Data-Efficient Instruction Tuning for Large Language Models](federated_data-efficient_instruction_tuning_for_large_language_models.md)
- [\[ACL 2025\] Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection](beyond_similarity_a_gradient-based_graph_method_for_instruction_tuning_data_sele.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)

</div>

<!-- RELATED:END -->
