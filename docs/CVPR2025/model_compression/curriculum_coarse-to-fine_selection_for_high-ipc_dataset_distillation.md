---
title: >-
  [Paper Note] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation
description: >-
  [CVPR 2025][Model Compression][Dataset Distillation] This work proposes CCFS, which progressively selects suitable real samples from the original dataset via a curriculum learning framework to supplement distilled data. This addresses the incompatibility between distilled and real data in high-IPC scenarios, outperforming state-of-the-art methods significantly on CIFAR-10/100 and Tiny-ImageNet (by up to +6.6%).
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Curriculum Learning"
  - "High IPC"
  - "Real Data Selection"
  - "Coarse-to-Fine"
date: 2026-05-08
content_hash: ed40f4f1635b9ba6
---

# Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation

**Conference**: CVPR 2025  
**arXiv**: [2503.18872](https://arxiv.org/abs/2503.18872)  
**Code**: [https://github.com/CYDaaa30/CCFS](https://github.com/CYDaaa30/CCFS)  
**Area**: Model Compression  
**Keywords**: Dataset Distillation, Curriculum Learning, High IPC, Real Data Selection, Coarse-to-Fine

## TL;DR
This work proposes CCFS, which progressively selects suitable real samples from the original dataset via a curriculum learning framework to supplement distilled data. This addresses the incompatibility between distilled and real data in high-IPC scenarios, outperforming state-of-the-art methods significantly on CIFAR-10/100 and Tiny-ImageNet (by up to +6.6%).

## Background & Motivation

**Background**: Dataset distillation aims to compress large-scale training data into a small set of synthetic images, enabling models trained on them to achieve performance close to that of the full dataset. Existing methods perform exceptionally well at extremely low IPC (e.g., 1 or 5 images per class), but their performance drops sharply as IPC increases, occasionally even underperforming random selection.

**Limitations of Prior Work**: SelMatch, a recent co-distillation paradigm method, proposes merging distilled data $\mathcal{D}_{\text{distill}}$ and real data $\mathcal{D}_{\text{real}}$ to construct the synthetic dataset. While achieving state-of-the-art results, its real data selection suffers from two defects: (1) a one-time fixed selection that may pick unsuitable real images; (2) the selection of $\mathcal{D}_{\text{real}}$ is independent of $\mathcal{D}_{\text{distill}}$, leading to a lack of complementarity.

**Key Challenge**: SelMatch selects real data before distillation, but the distillation process alters the feature distribution of $\mathcal{D}_{\text{distill}}$, leading to incompatibility between pre-selected real data and post-distill data. The authors verify this via three sets of comparative experiments, showing that performing distillation before selection yields an average 1.7% improvement over selecting before distilling.

**Goal**: How to progressively select real samples that are most complementary to the current distilled data within the co-distillation paradigm to resolve the incompatibility issue.

**Key Insight**: Modeling real data selection as a curriculum learning problem, where real samples are introduced progressively from easy to hard. A filter model is trained on the current synthetic dataset to identify its "blind spots" (misclassified samples), from which the easiest samples are selected to serve as supplements.

**Core Idea**: Distill first, and then employ a curriculum learning framework to select real data over multiple rounds. In each round, the optimal supplement is acquired via a "coarse filtering of misclassified samples + fine selection of the easiest ones" strategy.

## Method

### Overall Architecture
CCFS consists of two phases: first, distilling to obtain $\mathcal{D}_{\text{distill}}$ using an existing DD method (e.g., CDA), then progressively selecting real samples from the original dataset over $J$ curriculum stages to add to the synthetic dataset. In each curriculum stage, a filter model is first trained on the current synthetic dataset. Then, a coarse-to-fine strategy is employed to select the optimal real samples. Finally, these are combined to form $\mathcal{S} = \mathcal{D}_{\text{distill}} \cup \mathcal{D}_{\text{real}}^{1} \cup ... \cup \mathcal{D}_{\text{real}}^{J}$.

### Key Designs

1. **Curriculum Selection Framework**:

    - **Function**: Structures the real data selection process into multi-stage progressive selection.
    - **Mechanism**: The initial synthetic set is set to $\mathcal{S}_0 = \mathcal{D}_{\text{distill}}$. In each round, $k_j = \lfloor \text{IPC} \times (1-\alpha) / J \rfloor$ real samples per class are selected for inclusion, excluding previously selected samples. Crucially, distilling first and then selecting ensures that the selection is based on final distillation results rather than initialized data.
    - **Design Motivation**: A one-time selection cannot adapt to changes in distilled data. Multi-round selection progressively increases the selection difficulty as the synthetic dataset gets enriched, achieving progressive coverage from easy to hard.

2. **Coarse-to-Fine Selection**:

    - **Function**: Pinpoints the supplementary samples most needed by the current synthetic dataset in each curriculum round.
    - **Mechanism**: Operates in two steps—the coarse filtering stage trains a filter model on the current $\mathcal{S}$ and evaluates the original training set, filtering out correctly classified samples (whose features are already covered by the synthetic set). The fine selection stage ranks the remaining misclassified samples in ascending order based on pre-computed Forgetting scores, selecting the easiest $k$ samples per class. Choosing the easiest misclassified samples is because they represent features that the model "just missed learning but are not too difficult," carrying the highest training value.
    - **Design Motivation**: Correctly classified samples indicate that their features are already covered by the synthetic set, rendering redundant inclusion unnecessary; selecting the easiest among misclassified samples avoids introducing overly complex features that interfere with training.

3. **Difficulty Scores**:

    - **Function**: Provides global sample difficulty ordering for the fine selection stage.
    - **Mechanism**: Employs pre-computed Forgetting scores to measure sample difficulty. Experiments compare three difficulty dimensions: Forgetting, C-score, and Logits, with Forgetting performing best across all datasets (outperforming Logits by 2.8% on CIFAR-100).
    - **Design Motivation**: A global difficulty metric independent of the current filter is necessary to provide stable selection guidance. The Forgetting score, which is based on how many times a sample is forgotten during training, accurately reflects the intrinsic complexity of samples.

### Loss & Training
The distillation phase utilizes the MTT matching loss from the CDA method. The curriculum selection phase does not involve additional loss functions, in which the filter model is trained using standard cross-entropy. By default, 3 curriculum stages are used, and the distillation ratio $\alpha$ needs to be optimized for different IPCs.

## Key Experimental Results

### Main Results

| Dataset | IPC (Ratio) | CCFS | SelMatch | CDA | Gain (vs SelMatch) |
|--------|-------------|------|----------|-----|-------------------|
| CIFAR-10 | 500 (10%) | 92.5% | 85.9% | 84.4% | +6.6% |
| CIFAR-100 | 50 (10%) | 71.5% | 54.5% | 59.7% | +5.8%* |
| Tiny-ImageNet | 100 (20%) | 60.2% | 50.4% | 52.4% | +3.4%* |

*Note: On CIFAR-100, the performance exceeds the previous state-of-the-art CUDD (65.7%) to reach 71.5% (+5.8%); on Tiny-ImageNet, it is close to full-dataset training (60.5%), with only a 0.3% gap.

### Ablation Study

| Configuration | CIFAR-100 IPC=50 | Description |
|------|-----------------|------|
| Misclassified + Easiest Selection | 71.5% | Full CCFS strategy |
| Misclassified + Random Selection | 70.1% | Drops 1.4% without fine selection |
| Misclassified + Hardest Selection | 65.0% | Selecting overly hard samples is detrimental |
| Correct + Easiest Selection | 66.8% | Incorrect coarse filtering direction |
| 1-round curriculum | 67.9% | Lacks curriculum progression |
| 3-round curriculum | 71.5% | Optimal trade-off |
| Forgetting score | 71.5% | Best difficulty metric |
| Logits score | 68.7% | Drops 2.8% due to coarse metric |

### Key Findings
- Within the coarse-to-fine strategy, the combination of "selecting the easiest of the misclassified samples" significantly outperforms the other 5 combinations, indicating that simple features that the model just missed learning offer the most value as supplements.
- Increasing the number of curriculum rounds from 1 to 3 yields a notable improvement (67.9% $\rightarrow$ 71.5%), but the gain nearly saturates at 4–5 rounds.
- Strong cross-architecture generalization: Data selected using a ResNet-18 filter remains effective on ResNet-50/101, DenseNet-121, and RegNet.
- As the curriculum progresses, the difficulty of selected real samples strictly increases monotonically (visualized as migrating from simple backgrounds $\rightarrow$ complex poses/occlusion).

## Highlights & Insights
- **Reverse thinking of distilling before selecting**: Breaks the existing pipeline of "first select real data $\rightarrow$ then distill." By distilling first and then selecting real data based on the distilled outcome, a strong correlation between the two is established. This "first do A, then select B based on A" mindset can be generalized to various scenarios requiring complementarity between two data groups.
- **Using misclassified samples as a "demand detector"**: Using a model trained on the synthetic set to identify what it "fails to learn" accurately targets the shortcomings of the dataset. This idea is simple yet elegant.
- **Adaptive positive feedback of the curriculum framework**: Richer synthetic set $\rightarrow$ stronger filter $\rightarrow$ harder misclassified samples $\rightarrow$ naturally introducing harder samples. This forms a virtuous cycle without requiring manual design of the curriculum difficulty curve.

## Limitations & Future Work
- Requires pre-computing Forgetting scores for the entire dataset, increasing initial pre-computation overhead.
- Re-training the filter model from scratch in each curriculum stage implies $J$ additional full-model training runs.
- The distillation ratio $\alpha$ needs to be tuned for each IPC, increasing the hyperparameter search burden.
- Validation is limited to CIFAR-10/100 and Tiny-ImageNet, lacking experiments on larger scale datasets like ImageNet-1K.
- Selecting the "easiest misclassified samples" might not be optimal under certain data distributions (e.g., class-imbalanced scenarios).

## Related Work & Insights
- **vs SelMatch**: SelMatch selects real data once using a sliding window, where selection is independent of distillation results. CCFS performs distillation prior to multi-round selection, completely eliminating the incompatibility problem and bringing a massive boost (+5.8% on CIFAR-100).
- **vs DATM**: DATM tackles the problem from an optimization perspective, generating diverse data using trajectories across different training stages. CCFS approaches from a data composition perspective. The two methodologies are orthogonal and can theoretically be combined.
- **vs CUDD**: CUDD also expands the distilled set using a curriculum, but CCFS features a more granular coarse-to-fine selection strategy, surpassing CUDD's 65.7% to reach 71.5% on CIFAR-100.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of curriculum and coarse-to-fine selection is innovative, although the core idea (distillation prior to selecting real data) was already observed in preliminary analyses.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Thorough evaluations across three datasets and multiple IPC settings, along with cross-architecture tests, comprehensive ablations, and clear visualizations.
- Writing Quality: ⭐⭐⭐⭐ Well-analyzed problems with a logical and natural progression from analysis experiments to methodology design.
- Value: ⭐⭐⭐⭐ Achieves performance close to full-dataset training in the highly practical and important setting of high-IPC dataset distillation, presenting high engineering utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](../../CVPR2026/model_compression/hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)
- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [\[CVPR 2025\] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)

</div>

<!-- RELATED:END -->
