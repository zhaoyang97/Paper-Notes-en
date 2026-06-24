---
title: >-
  [Paper Note] Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality
description: >-
  [ACL 2025][Social Computing][Social bias] Proposes attention-weighted prediction quality proxy metrics $\Delta\text{pa}$ and CRRA to evaluate social bias in MLMs under Iterative Masking Experiments (IME), and introduces a model comparison function BSRT to estimate bias introduced by retraining. The proposed methods are found to be more accurate and sensitive than existing methods like CSPS, AUL, and AULA.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Social bias"
  - "Masked Language Models"
  - "Attention weighting"
  - "Bias evaluation"
  - "Iterative masking experiments"
date: 2026-05-08
content_hash: 159568a0f77f5971
---

# Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality

**Conference**: ACL 2025  
**arXiv**: [2402.13954](https://arxiv.org/abs/2402.13954)  
**Code**: None (open-source Python package provided for bias computation)  
**Area**: Social Computing  
**Keywords**: Social bias, Masked Language Models, Attention weighting, Bias evaluation, Iterative masking experiments

## TL;DR

Proposes attention-weighted prediction quality proxy metrics $\Delta\text{pa}$ and CRRA to evaluate social bias in MLMs under Iterative Masking Experiments (IME), and introduces a model comparison function BSRT to estimate bias introduced by retraining. The proposed methods are found to be more accurate and sensitive than existing methods like CSPS, AUL, and AULA.

## Background & Motivation

Masked Language Models (MLMs) such as BERT and RoBERTa have achieved state-of-the-art performance on various NLP tasks, but have also been found to encode undesirable social biases against marginalized groups. Evaluating these biases is crucial, yet existing methods suffer from the following issues:

**Limitations of Pseudo-Likelihood Methods**: Pseudo-likelihood-based methods like CrowS-Pairs Score (CSPS) and StereoSet Score (SSS) assume statistical independence among masked tokens and suffer from selection bias towards high-frequency words.

**Conceptual Divergence of AUL/AULA**: AUL and AULA proposed by Kaneko & Bollegala (2022) attempt to eliminate mask-related bias by predicting all tokens simultaneously without using masks; however, the authors argue that when evaluating the core pre-training objective of MLMs (masked language modeling), mask-related bias information is precisely what is valuable.

**Gap in Retraining Bias Evaluation**: There is a lack of effective methods to evaluate bias changes introduced after retraining MLMs under the masked language modeling objective. Existing methods may underestimate bias in retraining scenarios.

**Disregard of Attention Weights**: Existing methods fail to fully leverage the attention mechanism of MLMs to measure the contribution weights of different tokens to the prediction.

## Method

### Overall Architecture

Based on Iterative Masking Experiments (IME): one by one, each token in the sentence is masked until all tokens have been masked, obtaining the model's prediction quality for each token. Bias is evaluated by comparing the model's prediction preferences for paired sentences that favor the disadvantaged group (Sdis) and the advantaged group (Sadv).

### Key Designs

1. **Modified Probability Difference $\Delta p$ (Eq. 3)**:
   $\Delta P(t|s_{\setminus t_m};\theta) = \log P(t_p|s_{\setminus t_m};\theta) - \log P(t_m|s_{\setminus t_m};\theta)$
   
   Using a logarithmic transformation instead of the original probability difference reduces sensitivity in high-probability regions and enhances discrimination in low-probability regions, which is more suitable for the typically long-tailed distribution of MLMs.

2. **Attention-Weighted Complementary Reciprocal Rank Association CRRA (Eq. 4)**:
   $\text{CRRA}(t|s_{\setminus t_m};\theta) = a_m(1 - \log \rho(t_m|s_{\setminus t_m};\theta)^{-1})$
   
   Where $a_m$ is the average of multi-head attention related to the ground truth token, and $\rho$ is the rank of the masked token. Through attention-weighting, the importance of different tokens to the MLM's predictions is taken into account.

3. **Attention-Weighted Probability Difference $\Delta\text{pa}$ (Eq. 5)**:
   $\Delta\text{pa}(t|s_{\setminus t_m};\theta) = a_m(\log P(t_p|s_{\setminus t_m};\theta) - \log P(t_m|s_{\setminus t_m};\theta))$
   
   Combining attention weights with logarithmic probability differences.

4. **Pre-training Bias Score BSPT (Eq. 9)**:
   $\text{BSPT}(f) = \frac{100}{N}\sum_{i=1}^{N}\mathbb{1}(\Delta f_T(i) > 0)$
   
   Represents the percentage of sentences with higher bias towards the disadvantaged group. A score above 50 indicates that the model exhibits greater bias against the disadvantaged group.

5. **Retraining Bias Score BSRT (Eq. 10)**:
   $\text{BSRT}(f) = \frac{100}{N}\sum_{i=1}^{N}\mathbb{1}(\Delta f_{T_1}(i) > \Delta f_{T_2}(i))$
   
   Compares the relative changes in bias of the retrained model $T_1$ versus the pre-trained model $T_2$. This is the core innovation of this work, designed to evaluate bias introduced by retraining.

### Loss & Training

Retraining experimental setup:
- Retrained MLMs on either Sdis (sentences biased toward disadvantaged groups) or Sadv (sentences biased toward advantaged groups) of the CPS dataset.
- Used PyTorch, P100/T4 GPUs, training for 30 epochs.
- 4 MLMs evaluated: BERT-base-uncased, RoBERTa-base, distilBERT-base-uncased, distilRoBERTa-base.

## Key Experimental Results

### Main Results

**Pre-trained Model Bias Score BSPT (CPS Dataset)**:

| Metric | RoBERTa | BERT_unc | D-RoBERTa | D-BERT_unc |
|------|---------|----------|-----------|------------|
| CSPS | 59.35 | 60.48 | 59.35 | 56.83 |
| AUL | 58.75 | 48.34 | 53.32 | 51.59 |
| AULA | 58.09 | 48.21 | 51.86 | 52.65 |
| CRR | 58.89 | 61.07 | 57.76 | 56.23 |
| **CRRA** | **60.68** | 58.89 | **61.94** | **60.08** |
| Δp | 59.88 | 60.08 | 59.75 | 57.49 |
| **Δpa** | 60.15 | **60.81** | 59.81 | 58.02 |

All models across all metrics exhibit a bias score >50, indicating that all MLMs encode social biases against disadvantaged groups.

### Ablation Study

| Configuration | Key Indicator | Description |
|------|---------|------|
| Retraining Bias Alignment Error Rate (Sdis Retraining) | Δp, Δpa: 0% error | Correctly detects the direction of bias in 100% of cases |
| Retraining Bias Alignment Error Rate (Sadv Retraining) | Δp, Δpa: 0% error | CSPS: 5.6%, AUL: 8.3% error |
| CRR Error Rate | 0% | Same perfect performance as Δp and Δpa |
| CRRA Error Rate | 2.8% (only 1 case) | Near perfect |
| CSPS Error Rate | 5.6% | Significantly higher than the proposed methods |
| AULA Error Rate | 12.5% | Worst performing |

**McNemar Test Results**:
- Δp, Δpa: Significant ($p < 0.05$) for all MLMs and all bias categories.
- CSPS: 11 non-significant results (Sdis retraining).
- AULA: 12 non-significant results (Sadv retraining).

### Key Findings

1. **All MLMs encode social biases**: All four transformer models exhibit bias against disadvantaged groups on both CPS and StereoSet.
2. **AUL/AULA Underestimate BERT**: AUL and AULA estimate BERT's bias on CPS to be below 50 (48.34, 48.21), which contradicts other metrics.
3. **Retraining Bias Detection**: The proposed metrics are the most sensitive to bias introduced by retraining, while CSPS, AUL, and AULA produce worrying underestimations in retraining bias evaluation.
4. **Religious bias is generally high**: High religious bias scores are observed across all MLMs in CPS.
5. **Gender bias is relatively low on CPS**: Gender bias in CPS is lower than in StereoSet, possibly due to differences in dataset construction.
6. **Alignment with Human Annotations**: CRRA and Δpa outperform AUL and AULA in aligning with human annotated bias judgments.

## Highlights & Insights

- **Focus on the Core Objective**: Returning to the core pre-training objective of MLMs (masked language modeling) to evaluate bias, rather than using indirect proxies, which is logically robust.
- **Rational Utilization of Attention Weights**: Weighing token importance by averaging multi-head attention provides a more rational sentence-level bias estimation than uniform weighting.
- **Practical Value of BSRT**: The model comparison function can directly evaluate the effect of retraining on bias, which is highly practical for bias mitigation research.
- **Rigorous Statistical Verification**: Confirms the reliability of results using statistical methods such as the Shapiro-Wilk normality test and the McNemar test.

## Limitations & Future Work

1. **English Only**: CPS and StereoSet are both English datasets; bias analysis in other languages requires new benchmarks.
2. **Limited to MLM Architectures**: Only masked language models (BERT/RoBERTa) are evaluated, which are not applicable to autoregressive models (GPT series).
3. **Binary Bias Categorization**: Simplifies bias into a dichotomy of disadvantaged/advantaged groups, failing to capture more nuanced levels of bias.
4. **Limited Model Scale**: Only base-level models are tested, without exploring large or larger-scale models.
5. **Single Retraining Setting**: Fixed at 30 epochs without exploring the effect of different amounts of training on bias variations.

## Related Work & Insights

- **CrowS-Pairs (Nangia et al., 2020)**: A paired sentence benchmark containing 9 categories of social biases, serving as the core evaluation dataset in this work.
- **StereoSet (Nadeem et al., 2021)**: An intrasentence and intersentence paired dataset covering 4 categories of bias.
- **AUL/AULA (Kaneko & Bollegala, 2022)**: A mask-free bias evaluation method, serving as the primary baseline for comparison in this work.
- **Salutari et al. (2023)**: Proposed the original versions of CRR and Δp, upon which this work introduces attention-weighted extensions.
- Insight: The choice of evaluation method has a massive impact on bias estimation results; particularly in retraining scenarios, different metrics can yield completely opposite conclusions.

## Rating

- Novelty: ⭐⭐⭐ Attention-weighting is a reasonable but incremental improvement; the BSRT function is the main innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison across two benchmarks, four models, and seven metrics, with rigorous statistical testing.
- Writing Quality: ⭐⭐⭐ Formulas are dense, with a complex notation system, making the barrier to entry relatively high.
- Value: ⭐⭐⭐⭐ Significant contribution to the methodology of MLM bias evaluation; BSRT is highly practical for bias mitigation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[ICLR 2026\] Measuring and Mitigating Rapport Bias of Large Language Models under Multi-Agent Social Interactions](../../ICLR2026/social_computing/measuring_and_mitigating_rapport_bias_of_large_language_models_under_multi-agent.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](../../ACL2026/social_computing/the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)

</div>

<!-- RELATED:END -->
