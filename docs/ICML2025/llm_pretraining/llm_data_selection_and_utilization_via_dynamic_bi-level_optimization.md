---
title: >-
  [Paper Note] LLM Data Selection and Utilization via Dynamic Bi-level Optimization
description: >-
  [ICML2025][LLM Pretraining][Data Selection] A Dynamic Data Weighting Model (DWM) is proposed to adjust the weight of each batch of data in real time during LLM training via bi-level optimization. This captures the dynamically changing data preferences of the model, bringing consistent performance improvements over static data selection methods while scaling across different model sizes.
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Data Selection"
  - "Bi-level Optimization"
  - "Data Weighting"
  - "LLM Pre-training"
  - "Dynamic Preference"
date: 2026-05-08
content_hash: edaeb16168b2d564
---

# LLM Data Selection and Utilization via Dynamic Bi-level Optimization

**Conference**: ICML2025  
**arXiv**: [2507.16178](https://arxiv.org/abs/2507.16178)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Data Selection, Bi-level Optimization, Data Weighting, LLM Pre-training, Dynamic Preference

## TL;DR
A Dynamic Data Weighting Model (DWM) is proposed to adjust the weight of each batch of data in real time during LLM training via bi-level optimization. This captures the dynamically changing data preferences of the model, bringing consistent performance improvements over static data selection methods while scaling across different model sizes.

## Background & Motivation

### Limitations of Static Data Selection
Methods like DoReMi and DSIR select data prior to training, thereby ignoring the evolution of model preferences during training. Furthermore, selecting samples independently overlooks the joint effects of samples within the same batch.

### Goals of DWM
1. Dynamically adjust data weights during training.
2. Account for the joint effects of samples within a batch.
3. Transferable to models of different scales.

### Mechanism

**Goal**: ### Bi-level Optimization Framework
**Lower level**: Train the LLM for one step using data weighted by DWM.
**Upper level**: Optimize DWM parameters based on the validation performance of the updated LLM from the lower level.
Gradient backpropagation is achieved through the chain rule.


## Method

### Bi-level Optimization Framework
**Lower level**: Train the LLM for one step using data weighted by DWM.
**Upper level**: Optimize DWM parameters based on the validation performance of the updated LLM from the lower level.
Gradient backpropagation is achieved through the chain rule.

### Dynamic Learning
The bi-level optimization is re-run at different stages of training (every specific number of steps) to capture the continuously evolving data preferences of the model.

### DWM Design
Inputting the features of each sample, DWM outputs its relative weight within the current batch. It is plug-and-play and orthogonal to existing data selection methods.

### Loss & Training
The model is trained in an end-to-end manner, with the optimization objective considering both task loss and regularization terms.


## Key Experimental Results

### 370M Model Pre-training


### Main Results

| Method | Data Selection | +DWM | Improvement |
|------|---------|------|------|
| Random | Baseline | **Outperforms Baseline** | Significant |
| DSIR | Good | **Better** | Consistent |
| DsDM | Good | **Better** | Consistent |

### Cross-Model Transfer


### Ablation Study

| Training Scale | DWM Trained On | Performance |
|---------|---------|------|
| 370M | 370M | Baseline |
| 1.3B | 370M (Transfer) | **Consistent Gain** |

### Analysis of Data Preference Evolution

| Training Phase | Preferred Features |
|---------|---------|
| Early Stage | Prefers highly diverse data |
| Middle Stage | Prefers moderately difficult data |
| Late Stage | Prefers data similar to the validation set |

### Key Findings
1. DWM is effective for both randomly selected and meticulously selected data.
2. DWM can be transferred from small models to large models.
3. The model's data preferences indeed change dynamically during training.
4. The joint effect of samples within a batch has a substantial impact on the direction of updates.

## Highlights & Insights

1. The core difference between dynamic and static selection: different data are needed at different stages of model training.
2. Bi-level optimization enables the model to "tell you what data it needs."
3. Cross-scale transfer makes DWM highly practical, as weights learned from tiny models can guide large models.
4. The data preference evolution analysis provides novel insights into the training process.
5. The plug-and-play design ensures compatibility with existing data selection methods.

## Limitations & Future Work

1. Bi-level optimization increases training computational overhead (requiring additional forward/backward passes every $N$ steps).
2. The selection of the validation set affects the preferences learned by DWM.
3. Validation on larger scales (>10B) is currently lacking.
4. Comparison with curriculum learning approaches is insufficient.

## Related Work & Insights

- Difference from DoReMi: DoReMi performs static domain reweighting, while DWM conducts dynamic sample-level weighting.
- Relationship with MATES: MATES performs online selection but does not perform intra-batch weighting.
- Insight: Bi-level optimization can be applied to other training scenarios that require dynamic adaptation.

## Rating
- Novelty: 4.0/5 — Application of bi-level optimization to data weighting
- Experimental Thoroughness: 4.5/5 — Evaluation across multiple methods, transfers, and analyses
- Writing Quality: 4.0/5
- Value: 4.5/5 — Introduces a new dimension to data utilization

## Supplement

### Practical Implications of Data Preference Evolution
In the early stage, the model prefers highly diverse data, while in later stages, it prefers data similar to the validation set. This aligns with curriculum learning paradigms.

### Connection to Curriculum Learning
DWM can be viewed as an automated paradigm of curriculum learning: the model self-determines its data preference at each stage, rather than relying on human design.

### Computational Efficiency Trade-offs
Bi-level optimization requires additional forward and backward passes every $N$ steps, increasing training time by approximately 10-15%. However, the resulting performance gain justifies this overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[ACL 2025\] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](../../NeurIPS2025/llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)
- [\[ICML 2025\] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)

</div>

<!-- RELATED:END -->
