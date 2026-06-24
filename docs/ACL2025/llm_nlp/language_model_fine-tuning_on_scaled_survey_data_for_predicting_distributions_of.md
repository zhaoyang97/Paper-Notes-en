---
title: >-
  [Paper Note] Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions
description: >-
  [ACL 2025][LLM (Other)][Public Opinion] Proposes directly fine-tuning LLMs on large-scale public opinion survey data (SubPOP, containing 3,362 questions and 70K subpopulation-response pairs) to predict the opinion distributions of different demographic subpopulations, reducing Wasserstein distance by 32-46% compared to prompt engineering baselines while generalizing to unseen surveys and subpopulations.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Public Opinion"
  - "Distribution Prediction"
  - "Survey Data"
  - "LoRA Fine-Tuning"
  - "Wasserstein Distance"
date: 2026-05-08
content_hash: 74ef758623b531aa
---

# Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions

**Conference**: ACL 2025  
**arXiv**: [2502.16761](https://arxiv.org/abs/2502.16761)  
**Code**: [GitHub](https://github.com/JosephJeesungSuh/subpop)  
**Area**: Public Opinion Prediction / LLM Fine-Tuning  
**Keywords**: Public Opinion, Distribution Prediction, Survey Data, LoRA Fine-Tuning, Wasserstein Distance

## TL;DR

Proposes directly fine-tuning LLMs on large-scale public opinion survey data (SubPOP, containing 3,362 questions and 70K subpopulation-response pairs) to predict the opinion distributions of different demographic subpopulations, reducing Wasserstein distance by 32-46% compared to prompt engineering baselines while generalizing to unseen surveys and subpopulations.

## Background & Motivation

**Background**: Public opinion surveys are crucial tools for probing public sentiments, but they are costly and time-consuming. LLMs are expected to assist in questionnaire design and pilot testing by predicting response distributions for different subpopulations during the early stages of survey design.

**Limitations of Prior Work**: Existing methods mostly prompt LLMs to simulate the responses of specific groups, but their effectiveness is limited—out-of-the-box LLMs tend to reflect the views of wealthy, educated groups, exhibit stereotypes or biases against minority groups, and fail to capture intra-group opinion variance.

**Key Challenge**: Prompting methods cannot effectively "steer" LLMs to the opinion distributions of specific subpopulations; existing fine-tuning datasets are too small (OpinionQA contains only about 500 questions), which is insufficient for effective fine-tuning.

**Goal**: Enable LLMs to accurately predict opinion **distributions** (rather than just the most likely option) for diverse subpopulations.

**Key Insight**: Large-scale, high-quality survey data is naturally suited for fine-tuning because (1) it provides clear subpopulation-response pairs; (2) it is representative after post-stratification calibration; and (3) it allows direct modeling of distributions.

**Core Idea**: Construct a large-scale survey dataset, SubPOP, and fine-tune LLMs using Forward KL divergence as the loss function, enabling the token probability distribution output by the model to match the human survey response distribution.

## Method

### Overall Architecture

Given a multiple-choice question $q$ and a subpopulation $g$, the fine-tuned LLM outputs probabilities for each option token (A/B/C/D/E). This probability distribution should match the response distribution of that subpopulation in the real human survey.

### Key Designs

1. **SubPOP Dataset**: Collected 3,229 training questions from Pew's ATP (American Trends Panel) waves 61-132 (excluding waves already in OpinionQA), and 133 questions from NORC's GSS (General Social Survey) for cross-survey-family OOD evaluation. This yields a total of 70K subpopulation-response distribution pairs, which is 6.5 times the size of OpinionQA. It covers 22 subpopulation dimensions (age, education, political ideology, etc.).
2. **Forward KL Loss**: Employs Forward KL divergence $D_{\text{KL}}(p_H \| p_\theta)$ as the training objective, where $p_H$ represents the empirical distribution from the human survey, and $p_\theta$ represents the model's output probability for each option token. The "mean-seeking" behavior of Forward KL ensures that the model does not ignore options with high human probabilities.
3. **LoRA Fine-Tuning**: Fine-tunes Llama-2-7B/13B, Mistral-7B, and Llama-3-70B using LoRA. Pre-trained base models are used instead of instruction-tuned versions (experiments show pre-trained models perform better).
4. **Distribution Modeling vs. Alternatives**: Compares (1) one-hot encoding—which only focuses on the most likely option and loses distribution information; and (2) replicating data by frequency—which is computationally inefficient. Direct distribution modeling is shown to be superior.

### Evaluation Metrics

- **Wasserstein Distance (WD)**: Considers the ordinal relationship between options and measures the "earth mover's distance" between the predicted distribution and the human distribution. This is more fine-grained than one-hot accuracy.
- Lower Bound: Estimated via bootstrap sampling to capture the sampling variance of the human survey itself.

## Key Experimental Results

### Main Results (Table 1, WD ↓)

| Method | OpinionQA (Llama-2-7B) | OpinionQA (Mistral-7B) | SubPOP-Eval (Llama-2-7B) |
|------|------------------------|------------------------|--------------------------|
| Upper Bound (Uniform) | 0.178 | 0.178 | 0.208 |
| Lower Bound (Human) | 0.031 | 0.031 | 0.033 |
| Zero-shot QA | 0.173 | 0.153 | 0.206 |
| Few-shot | 0.186 | 0.174 | 0.217 |
| Modular Pluralism | 0.285 | 0.279 | — |
| **SubPOP-FT (Ours)** | **0.106** | **0.096** | **0.121** |

- After fine-tuning, WD reduces by 32-46%, consistently outperforming all baselines across all models and evaluation sets.
- It remains highly effective on SubPOP-Eval (GSS survey family), demonstrating cross-organization generalization capability.

### Subpopulation Consistency (Figure 3)

- The relative improvement across 22 subpopulations ranges from 38% to 54% (mean 46.7%, standard deviation 4.4%), which is highly consistent.
- The improvement is not biased toward any specific demographic group.

### Generalization to Unseen Subpopulations (Table 2)

- For subpopulations not seen during training (e.g., specific age groups like 18-29, 65+), the relative improvement averages 44.7%, which is comparable to the groups seen during training.
- The model successfully learns "steerability" based on demographic prompts—cross-group disagreement patterns align well with human data.

### Data Scaling Effect (Figure 5)

- Using only 25% of the data achieves 72-78% of the total improvement.
- Performance continues to scale with data size, and different models exhibit similar scaling slopes.
- It is estimated that approximately 25 times the current data volume would be required to approach the human lower bound.

## Highlights & Insights

1. **Distribution Modeling over Point Prediction**: Opinions are naturally distributed; even within the same group, there is diversity. One-hot approaches inherently ignore this property.
2. **Rationality of Forward KL**: Ensures that the model covers high-probability regions of the human distribution, aligning with maximum likelihood training.
3. **Cross-Survey-Family Generalization**: Though SubPOP-Train is based on ATP, it performs well on GSS, indicating that the model learns general subpopulation-to-opinion mappings.
4. **Steerability Verification**: Elegantly demonstrates via a "cross-group disagreement matrix" that the model indeed shifts its output based on demographic labels in the prompt, rather than simply regressing to the overall mean.
5. **Training Data Design**: Uniformly allocating training samples to each subpopulation is a key design decision for consistent improvement.

## Limitations & Future Work

1. Only focuses on US survey data; cross-cultural generalization remains unverified.
2. Subpopulations are defined using coarse-grained demographic labels, failing to capture finer-grained individual differences.
3. The ordinal option mapping assumes a natural order among options (a prerequisite for WD evaluation), but not all survey questions satisfy this.
4. Forward KL encourages mode covering but may lead to "over-smoothed" distribution predictions.
5. Ethical risks—the fine-tuned models could be misused to serialize "fake survey results."

## Related Work & Insights

- **Public Opinion Datasets**: OpinionQA, GlobalOpinionQA, PRISM, etc.
- **LLMs Predicting Human Opinions**: Ranging from rule-based prompting to personal narrative prompting, up to direct fine-tuning.
- **Pluralistic Alignment**: Preference learning methods like DPO focus on pairwise preference ranking, whereas this work focuses on distribution matching.

## Rating

⭐⭐⭐⭐ — Clear problem definition, prominent dataset contribution (6.5x expansion), and a simple yet effective method. The experimental design for validating "steerability" is highly clever. A limitation is the reliance solely on US survey scenarios, and the methodological novelty is moderate (inherently LoRA + KL fine-tuning).

## Supplementary Details

- Using pre-trained models performs better than instruction-tuned models, potentially because instruction-tuned models have overly sharp token probability distributions, which is unfavorable for distribution modeling.
- The distribution shift between ATP and GSS includes differences in respondent pools, calibration techniques, and methodology. Generalization under these conditions suggests that the model learns general subpopulation-to-opinion mapping capabilities.
- Uniformly distributing training samples for each subpopulation is a critical design decision for consistent improvement.
- The similar scaling slopes across different models in data scaling experiments imply that performance gains are primarily driven by data and task characteristics rather than model architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](algorithmic_fidelity_german_opinion.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] SDD: Self-Degraded Defense against Malicious Fine-tuning](sdd_self-degraded_defense_against_malicious_fine-tuning.md)

</div>

<!-- RELATED:END -->
