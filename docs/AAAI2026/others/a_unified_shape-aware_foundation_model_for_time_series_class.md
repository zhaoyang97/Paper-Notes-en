---
title: >-
  [Paper Note] UniShape: A Unified Shape-Aware Foundation Model for Time Series Classification
description: >-
  [AAAI 2026][Time Series Classification] This paper proposes UniShape — the first shape-aware foundation model for time series classification (TSC). It captures class-discriminative temporal patterns via a shape-aware adapter that adaptively aggregates multi-scale subsequences (shapes), and jointly learns transferable shapelet representations at both instance and shape levels through a prototype-based pretraining module. Pretrained on 1.89M samples, UniShape achieves an average accuracy of 0.8708 across 128 UCR datasets, surpassing all baselines.
tags:
  - AAAI 2026
  - Time Series Classification
  - Foundation Model
  - Shapelet
  - Multi-scale
  - Prototype Learning
date: 2026-05-08
content_hash: d8380e09dfe4b051
---

# UniShape: A Unified Shape-Aware Foundation Model for Time Series Classification

**Conference**: AAAI 2026
**arXiv**: [2601.06429](https://arxiv.org/abs/2601.06429)
**Code**: [https://github.com/qianlima-lab/UniShape](https://github.com/qianlima-lab/UniShape)
**Area**: Others
**Keywords**: Time Series Classification, Foundation Model, Shapelet, Multi-scale, Prototype Learning

## TL;DR

This paper proposes UniShape — the first shape-aware foundation model for time series classification (TSC). It captures class-discriminative temporal patterns via a shape-aware adapter that adaptively aggregates multi-scale subsequences (shapes), and jointly learns transferable shapelet representations at both instance and shape levels through a prototype-based pretraining module. Pretrained on 1.89M samples, UniShape achieves an average accuracy of 0.8708 across 128 UCR datasets, surpassing all baselines.

## Background & Motivation

Existing time series foundation models are primarily designed for forecasting tasks, which differ fundamentally from classification: forecasting focuses on the continuous extrapolation of trends and seasonality, whereas classification requires identifying discriminative local patterns (shapelets) within fixed-length samples. Forecasting-oriented foundation models therefore perform poorly when directly applied to TSC. Meanwhile, most existing TSC methods are trained on small-scale, single-domain datasets with limited cross-domain generalizability. Furthermore, shapelets — the most interpretable features for classification — exhibit multi-scale characteristics (discriminative subsequences may appear at varying lengths and positions), a property that has not been effectively modeled in prior foundation models.

## Method

### Overall Architecture

UniShape follows a pretrain-then-finetune paradigm: (1) a Shape-Aware Adapter encodes variable-length subsequences into shape tokens and aggregates them into class tokens via attention pooling; (2) a prototype pretraining module performs contrastive learning jointly at the instance and shape levels; (3) during finetuning, the class token is passed through a classification head to produce predictions.

### Key Designs

1. **Shape-Aware Adapter**: Multi-scale subsequences are extracted from the input time series using $Q$ sliding windows of different scales ($W_q \in \{64, 32, 16, 8, 4\}$). Each subsequence is normalized and encoded into a shape token via a 1D CNN, then adaptively aggregated into a class token through attention pooling. A coarse-to-fine hierarchical fusion strategy is adopted, where the class token from the previous scale is prepended to the token sequence of the next scale, enabling cross-scale information transfer.
2. **Prototype Pretraining Module**: A set of learnable prototype vectors, one per class, is maintained and updated dynamically via exponential moving average. Instance-level contrastive learning (class token ↔ class prototype) captures global discriminative features, while shape-level contrastive learning (high-confidence shape tokens ↔ class prototype) models local discriminative patterns. Pseudo-labels are assigned to unlabeled samples using the nearest prototype.
3. **Multi-scale Interpretability**: The attention pooling weights $\alpha$ directly reflect the discriminative importance of each shape, providing shapelet-level interpretability. On the ECGFiveDays dataset, the model correctly highlights the delayed T-wave interval; on GunPoint, it localizes the motion overshoot interval.

### Loss & Training

- **Pretraining loss** = Prototype contrastive loss (instance-level + shape-level) + MoCo v3 self-supervised contrastive loss
- Shape-level loss weight $\lambda = 0.01$; temperature $\tau$ controls the sharpness of contrastive learning
- Pretraining with only 10% labeled data achieves performance statistically comparable to full supervision
- Pretraining: 30 epochs, batch size 2048; Finetuning: 300 epochs, cross-entropy + shape contrastive auxiliary loss ($\mu = 0.01$)

## Key Experimental Results

### Main Results (128 UCR Datasets, Fully Supervised)

| Method | Type | Params | Avg. Accuracy | Avg. Rank |
|--------|------|--------|---------------|-----------|
| UniShape | FM | 3.1M | **0.8708** | **2.71** |
| Mantis | FM | 8.7M | 0.8441 | 5.21 |
| NuTime | FM | 2.4M | 0.8353 | 6.68 |
| MR-H | NDL | - | 0.8621 | 3.97 |
| SoftShape | DS | 472K | 0.8388 | 5.89 |
| MOMENT | FM | 341M | 0.7020 | 12.10 |

### Zero-shot Feature Extraction (30 Additional Datasets)

| Method | Avg. Accuracy | Avg. Rank |
|--------|---------------|-----------|
| UniShape | **0.7262** | **3.07** |
| Mantis | 0.7052 | 3.67 |
| NuTime | 0.6917 | 3.53 |
| RandomForest | 0.6930 | 3.77 |

### Ablation Study

- Performance consistently improves with larger pretraining data scale (UCR 60K → ALL 1.89M)
- The difference between 10% and 100% labeled pretraining data is statistically insignificant ($P = 0.20$), indicating that a small label budget suffices
- Both the shape-aware adapter and the prototype pretraining module contribute independently; removing either component leads to significant performance degradation

### Key Findings

- Forecasting-oriented foundation models (GPT4TS, MOMENT, UniTS) substantially underperform non-deep-learning methods on TSC, demonstrating the critical importance of task-specific design
- UniShape with only 3.1M parameters surpasses MOMENT with 341M parameters, exhibiting exceptional parameter efficiency
- Interpretability analysis shows that attention weights align closely with shapelet intervals identified by domain experts

## Highlights & Insights

- This is the first work to explicitly identify the unsuitability of forecasting-oriented foundation models for classification and to provide a targeted solution
- The multi-scale design of the shape-aware adapter is elegant and computationally efficient, with shared parameters handling all scales
- Prototype learning captures class structure with very few labels, making it particularly valuable for semi-supervised and few-shot scenarios
- Attention weights serve as an interpretability mechanism for shapelets, offering practical utility in domains such as medical time series analysis

## Limitations & Future Work

- Only univariate time series classification is addressed; multivariate settings require additional design considerations
- The fixed five scales ($4$–$64$) may not be optimal for shapelet lengths in all domains
- Pretraining sequences are uniformly interpolated to length 512, potentially discarding information from very long sequences
- Zero-shot accuracy still has room for improvement (0.73 vs. 0.87 under full supervision)

## Related Work & Insights

- The evolution of shapelet learning — from exhaustive search to gradient-based optimization to foundation model pretraining — represents a trajectory worth following
- Prototype contrastive learning can be generalized to other domains requiring class-aware pretraining, such as few-shot image classification
- The multi-scale attention pooling design is transferable to time series forecasting foundation models
- The momentum contrastive learning framework of MoCo v3 proves effective on time series data as well
- Non-deep-learning methods such as Rocket/MiniRocket remain extremely strong baselines (0.85+); foundation models must demonstrate clear improvements to justify their adoption

## Pretraining Data Construction

| Source | # Samples | Notes |
|--------|-----------|-------|
| UCR Archive | ~60K | 128 univariate classification datasets |
| UEA Archive | ~1.39M | Multivariate → channel-independent splitting |
| Additional Data | ~0.44M | 8 commonly used time series datasets |
| **Total** | **1.89M** | Uniformly interpolated to length 512 |

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | First shape-aware foundation model for TSC |
| Technical Depth | 4 | Multi-scale adapter + prototype learning, elegantly designed |
| Experimental Thoroughness | 5 | 158 datasets, 16 baselines, comprehensive ablation |
| Writing Quality | 4 | Clear motivation, fluent method presentation |
| Value | 4 | Parameter-efficient, interpretable, cross-domain transferable |

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[NeurIPS 2025\] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](../../NeurIPS2025/others/a_unified_framework_for_variable_selection_in_modelbased_clu.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](measuring_model_performance_in_the_presence_of_an_intervention.md)

<!-- RELATED:END -->
