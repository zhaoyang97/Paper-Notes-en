---
title: >-
  [Paper Note] Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions
description: >-
  [ACL 2025][Object Detection][Anchored Bias] This work provides the first mechanistic analysis of "anchored bias" (consistently choosing "A") in the GPT-2 family within multiple-choice questions (MCQs) from a failure-case perspective. It localizes specific value vectors storing the "A" preference in MLPs using Logit Lens, and achieves an average MCQ accuracy improvement of 70%+ through minimal intervention (updating the value vectors).
tags:
  - "ACL 2025"
  - "Object Detection"
  - "Anchored Bias"
  - "GPT-2"
  - "Mechanistic Interpretability"
  - "Logit Lens"
  - "MLP Value Vectors"
date: 2026-05-08
content_hash: 9bd7788ec69d8a95
---

# Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions

**Conference**: ACL 2025  
**arXiv**: [2405.03205](https://arxiv.org/abs/2405.03205)  
**Code**: [GitHub](https://github.com/ruizheliUOA/Anchored_Bias_GPT2)  
**Area**: Object Detection  
**Keywords**: Anchored Bias, GPT-2, Mechanistic Interpretability, Logit Lens, MLP Value Vectors

## TL;DR

This work provides the first mechanistic analysis of "anchored bias" (consistently choosing "A") in the GPT-2 family within multiple-choice questions (MCQs) from a failure-case perspective. It localizes specific value vectors storing the "A" preference in MLPs using Logit Lens, and achieves an average MCQ accuracy improvement of 70%+ through minimal intervention (updating the value vectors).

## Background & Motivation

**Background**: LLMs exhibit positional bias in MCQs, where the position of the correct answer affects prediction accuracy.  
**Limitations of Prior Work**: Existing research mitigates bias via prompt engineering but fails to deeply analyze internal mechanisms; mechanistic interpretability works (e.g., Lieberum et al.) focus only on successful cases.  
**Key Challenge**: The GPT-2 family exhibits extreme anchored bias—consistently favoring "A"—which shows high regularity across datasets, yet its internal mechanism remains unknown.  
**Goal**: To localize and repair the specific internal modules in GPT-2 that cause anchored bias from a failure-case perspective.  
**Key Insight**: View the MLP as a key-value memory and trace the sources of bias layer-by-layer using Logit Lens.  
**Core Idea**: MLP value vectors store the preference for "A" during pre-training, which manifests as anchored bias under MCQ formats and can be eliminated through direct editing.

## Method

### Overall Architecture

Three stages: (1) Quantitatively confirming the ubiquity of anchored bias across 5 MCQ datasets; (2) Localizing bias sources in MLP layers, dimensions, and attention heads using Logit Lens; (3) Intervening via value vector updates and attention weight swapping.

### Key Designs

1. **MLP Bias Localization**:
    - **Function**: Localizes specific MLP layers and dimensions storing the "A" preference.
    - **Mechanism**: Computes logit difference $\text{logit}_T^\ell[\text{A}](\mathbf{m}_T^\ell) - \text{logit}_T^\ell[\text{B/C/D/E}](\mathbf{m}_T^\ell)$ to find key layers, localizes dimensions using MLP Contribution $|\mathbf{k}_T^{\ell,n}| \|\mathbf{v}_T^{\ell,n}\|$, and projects the value vectors via unembedding to verify the top-10 tokens.
    - **Design Motivation**: Treating MLPs as key-value memory, if the top tokens of the value vectors are words related to "A", it directly demonstrates that these are the storage locations of the bias.

2. **MLP Value Vector Update**:
    - **Function**: Directly modifies value vectors to eliminate the preference for "A".
    - **Mechanism**: $\mathbf{v}^{\ell,n} = \mathbf{v}^{\ell,n} - \lambda_1 W_U[\text{A}] + \lambda_2 W_U[\text{B/C/D/E}]$, where $\lambda_1=1, \lambda_2=8$.
    - **Design Motivation**: "Rewrites" the bias information directly at the knowledge storage level without model retraining.

3. **Attention Weight Swapping**:
    - **Function**: Swaps the weighted attention values corresponding to "A" and the correct answer position.
    - **Mechanism**: $\mathbf{r}_{T,p(\text{A})}^{\ell,h} \leftrightarrow \mathbf{r}_{T,p(\text{B/C/D/E})}^{\ell,h}$.
    - **Design Motivation**: Attention heads allocate extra focus on the "A" position; swapping can further eliminate this bias.

### Loss & Training

No training involved. $\lambda_2$ is determined through ablation studies, where accuracy consistently improves as it scales from 2 to 8.

## Key Experimental Results

### Main Results

MCQ accuracy after MLP value vector updates (%):

| GPT-2 | Modified Vector | IOI (2-choice) | LD (3-choice) | Greater (4-choice) | ARC (4-choice) | CSQA (5-choice) |
|-------|---------|:---:|:---:|:---:|:---:|:---:|
| Small | v9,1853 | 100 | 100 | 100 | 100 | 100 |
| Large | v34,1541 | 100 | 100 | 100 | 96.7 | 99.7 |
| XL | v44,4967 | 98.2 | 100 | 100 | 90.7 | 94.8 |

### Ablation Study

Occurrence rate of anchored bias across the GPT-2 family (%):

| Dataset | Small | Medium | Large | XL |
|-------|:---:|:---:|:---:|:---:|
| IOI (2-choice) | 45.5 | 97.4 | **100** | 85.8 |
| ARC (4-choice) | 54.6 | 91.6 | 97.6 | 69.9 |
| CSQA (5-choice) | 34.8 | 81.5 | 99.6 | 97.7 |

### Key Findings

1. **MLPs are the primary source of bias**: Value vectors in specific layers directly store knowledge of the preference for "A".
2. **Modifying only 1-2 value vectors eliminates the bias**: Updating v9,1853 in GPT-2 Small achieves 100% accuracy across all 5 datasets.
3. **Attention heads play a supporting role**: Effective on IOI (reaching 92.47% on GPT-2 Medium), but have limited effect on other datasets.
4. **Interventions have a controllable impact on general capabilities**: The original IOI task still maintains an accuracy of 85.8%.

## Highlights & Insights

- **Analyzing failure cases** is complementary to analyzing success cases—understanding weaknesses is more practical than understanding strengths.
- **Logit Lens precisely localizes to specific dimensions**—offering high actionable feasibility.
- **Minimal intervention with high reward**: Modifying just one value vector shifts performance from 0% to 100%.
- **Complete bias circuit tracing**: Visualization maps of MLPs and attention heads across the entire GPT-2 family.

## Limitations & Future Work

- Limited to the GPT-2 family (124M–1.5B); larger models may exhibit different mechanisms.
- Attention swapping requires prior knowledge of the correct answer's position.
- Value vector updates incur some degradation in general capabilities.
- Bias storage locations might differ entirely in different architectures (e.g., MoE/Mamba).

## Related Work & Insights

- **vs Lieberum et al. 2023**: Analyzes the "correct letter heads" of successful cases—while this work analyzes failure cases, creating a complementary relationship.
- **vs PriDe (Zheng et al. 2024)**: Performs debiasing during inference—whereas this work directly modifies parameters.
- **Insight**: MLP value vectors store not only factual knowledge but also biases—representing a double-edged sword for model editing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First mechanistic analysis of MCQ bias from a failure-case perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Entire GPT-2 family × 5+ datasets × multiple interventions.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth, layer-by-layer analysis.
- Value: ⭐⭐⭐⭐ Methodological contributions to LLM interpretability and bias mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multiple Object Tracking as ID Prediction](../../CVPR2025/object_detection/multiple_object_tracking_as_id_prediction.md)
- [\[ECCV 2024\] Rectify the Regression Bias in Long-Tailed Object Detection](../../ECCV2024/object_detection/rectify_the_regression_bias_in_long-tailed_object_detection.md)
- [\[AAAI 2026\] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](../../AAAI2026/object_detection/towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)
- [\[CVPR 2026\] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](../../CVPR2026/object_detection/ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)
- [\[AAAI 2026\] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](../../AAAI2026/object_detection/when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)

</div>

<!-- RELATED:END -->
