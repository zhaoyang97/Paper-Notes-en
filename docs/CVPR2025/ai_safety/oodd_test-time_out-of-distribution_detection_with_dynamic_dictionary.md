---
title: >-
  [Paper Note] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary
description: >-
  [CVPR 2025][AI Safety][Out-of-Distribution Detection] OODD is proposed to maintain a dynamic OOD dictionary via a priority queue to collect potential OOD sample features in real-time during test time for calibrating OOD scores. Compared to SOTA methods, it reduces FPR95 by 26.0% on CIFAR-100 Far OOD without requiring any fine-tuning.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Out-of-Distribution Detection"
  - "Dynamic Dictionary"
  - "Priority Queue"
  - "Test-Time Adaptation"
  - "KNN"
date: 2026-05-08
content_hash: f03d879396ddd0b2
---

# OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary

**Conference**: CVPR 2025  
**arXiv**: [2503.10468](https://arxiv.org/abs/2503.10468)  
**Code**: [GitHub](https://github.com/zxk1212/OODD)  
**Area**: LLM Evaluation  
**Keywords**: Out-of-Distribution Detection, Dynamic Dictionary, Priority Queue, Test-Time Adaptation, KNN

## TL;DR

OODD is proposed to maintain a dynamic OOD dictionary via a priority queue to collect potential OOD sample features in real-time during test time for calibrating OOD scores. Compared to SOTA methods, it reduces FPR95 by 26.0% on CIFAR-100 Far OOD without requiring any fine-tuning.

## Background & Motivation

Out-of-distribution (OOD) detection is a critical issue for the safe deployment of deep learning models. Existing methods fall into three categories: (1) post-processing methods (designing score functions based on logits or features); (2) fine-tuning methods (using auxiliary datasets to expose outliers); and (3) test-time calibration methods.

The core limitation of fine-tuning methods is that their performance drops drastically when there is a significant distribution shift between the outliers used during training and the actual OOD samples encountered at test time. Test-time adaptation methods (e.g., RTL++, AUTO) perform calibration via batch-level parameter updates, but this can lead to catastrophic forgetting and unstable detection.

**Core Motivation**: The key to test-time OOD detection lies in utilizing potential OOD features. If a dictionary of OOD samples can be collected and maintained in real-time during the testing process, these features can be utilized to calibrate detection scores without requiring any parameter updates.

## Method

### Overall Architecture

OODD consists of three core components: (1) Informative In-distribution Sampling (IIS) — filtering highly confident, representative samples from training ID data to construct an ID dictionary; (2) Dynamic OOD Dictionary — collecting features of low-OOD-score samples in real-time during testing via a priority queue; and (3) Double OOD Stabilization (DOS) — initializing the priority queue and a memory bank using outliers generated from ID data. The final OOD score is the sum of the similarity to the ID dictionary and the similarity to the OOD dictionary: $S(\mathbf{x}^*) = S_{in}(\mathbf{x}^*) + S_{out}(\mathbf{x}^*)$.

### Key Design 1: Informative In-distribution Sampling (IIS)

**Function**: Construct a high-quality, compact ID dictionary as an alternative to using the entire training dataset in KNN.

**Mechanism**: Perform multiple random crops on each training sample and select the crop patch with the highest confidence. Then, sort them by confidence within each class and select the top $\alpha\%$ patch features to be stored in the ID dictionary $\mathcal{K}_{n'}^{id}$. During testing, compute the cosine similarity between the query feature and its $\mathbb{K}$-th nearest neighbor in the ID dictionary as $S_{in}$.

**Design Motivation**: Using the entire training set as an ID dictionary includes redundant and noisy samples, which decreases both efficiency and detection quality. Through dual-level filtering at both the patch and class levels, the most representative ID features are retained while significantly reducing the dictionary size.

### Key Design 2: Priority Queue Dynamic OOD Dictionary

**Function**: Collect and maintain the most representative potential OOD sample features in real-time during testing.

**Mechanism**: Maintain a priority queue where priority is based on the OOD score, with the sample having the highest OOD score at the head of the queue. When the OOD score of a new test sample is lower than that of the queue head (indicating it is more likely to be OOD), it is enqueued, and the head of the queue is dequeued (when the queue is full). During testing, the negative similarity of the $\hat{\mathbb{K}}$-th nearest neighbor to the OOD dictionary is computed as $S_{out}$. The dictionary size is a hyperparameter and is decoupled from the batch size.

**Design Motivation**: OOD samples encountered during test time provide the most direct calibration signals. The priority queue ensures that the dictionary always retains features of samples with the lowest OOD scores (most likely to be OOD), avoiding the catastrophic forgetting issues caused by batch-level parameter updates. The dictionary update requires only a single matrix multiplication, incurring negligible computational overhead.

### Key Design 3: Double OOD Stabilization (DOS)

**Function**: Address the instability of detection scores during the early stages of testing when the OOD dictionary is empty.

**Mechanism**: Outliers generated from ID data (C-Out) are split into two parts: one part initializes the priority queue, and the other is stored in a fixed memory bank $\mathcal{K}_{mb}^{ood}$. The final OOD dictionary is represented as $\mathcal{K}_{total}^{ood} = \mathcal{K}_l^{ood} \cup \mathcal{K}_{mb}^{ood}$. The C-Out method generates low-confidence patches through random cropping to act as outliers without requiring external data.

**Design Motivation**: An empty OOD dictionary at the start of testing makes $S_{out}$ unavailable, causing severe fluctuations in calibrated scores. The fixed memory bank provides a stable baseline, while the priority queue gradually replaces initialized features with real OOD features.

### Loss & Training

No training loss. The method runs entirely at test time, involving only cosine similarity computation and priority queue maintenance.

## Key Experimental Results

### Main Results (OpenOOD Benchmark, CIFAR-100)

| Method | Near OOD AUROC↑ | Near OOD FPR95↓ | Far OOD AUROC↑ | Far OOD FPR95↓ |
|------|----------------|----------------|----------------|----------------|
| KNN | — | — | — | — |
| TULIP (SOTA) | — | — | — | 58.17 |
| **OODD** | — | — | — | **24.74** |

### CIFAR-10 Benchmark

| Method | Near OOD AUROC↑ | Far OOD AUROC↑ | Far OOD FPR95↓ |
|------|----------------|----------------|----------------|
| KNN | 90.64 | 92.96 | 24.27 |
| ViM | 88.68 | 93.48 | 25.05 |
| TULIP | 89.67 | 92.55 | 24.43 |
| **OODD** | **90.96** | **95.77** | **17.44** |

### Key Findings

1. **Significant improvement in Far OOD detection**: On CIFAR-100 Far OOD, FPR95 decreases from 58.17% to 24.74% (TULIP to OODD), representing a reduction of over 33 percentage points.
2. **C-Out is highly effective**: Strong performance is achieved using outliers generated solely from ID data via random cropping (without introducing external data).
3. **Complementary to post-processing methods**: OODD can be integrated with methods like KNN and ViM to further boost performance.
4. **KNN Acceleration**: By replacing Euclidean distance with cosine similarity, which is theoretically proven to have equivalent discriminative power, a 3x speedup is achieved.
5. **No fine-tuning overhead**: The method does not modify any model parameters and only requires a single matrix multiplication to calculate the similarity with the OOD dictionary.

## Highlights & Insights

- **Utilization of test-time signals**: Treating OOD samples encountered during test time as valuable resources rather than merely objects to be detected.
- **Elegance of priority queue**: Ensures dictionary quality (retaining the most likely OOD samples) while preventing catastrophic forgetting.
- **Zero external data**: The C-Out method relies solely on ID training data, acquiring initialized outliers via "reverse" filtering (selecting low-confidence instead of high-confidence samples).

## Limitations & Future Work

- **Assumptions on OOD score distribution**: Relies on the assumption that OOD samples lie in the left tail of the OOD score distribution; this might show limited effectiveness for hard OOD cases (where scores heavily overlap with ID).
- **Sensitivity to dictionary size**: The size of the priority queue and parameters $\mathbb{K}$, $\hat{\mathbb{K}}$ need to be carefully tuned.
- **Sequence dependency in streaming scenarios**: If the proportion of OOD samples in early test samples is extremely low, the quality of the dictionary may remain poor for a long time.
- Future directions could explore adaptive dictionary sizes, multi-scale feature dictionaries, etc.

## Related Work & Insights

- **KNN OOD Detection**: OODD is essentially a bidirectional extension of KNN methods — maintaining both ID and OOD dictionaries simultaneously.
- **MoCo Momentum Dictionary**: The idea of maintaining a dictionary via a priority queue shares similarities with MoCo's queue design.
- **Insight**: Utilizing available signals during test time is an underestimated direction, which is applicable to any scenario requiring adaptation to new distributions.

## Rating

⭐⭐⭐⭐ — The method is simple yet highly efficient, and test-time OOD detection without fine-tuning is extremely valuable for real-world deployment. The priority queue design is elegant, delivering a significant 26% reduction in FPR95. The theoretical analysis (equivalence of cosine similarity and Euclidean distance) is a strong plus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/ai_safety/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)

</div>

<!-- RELATED:END -->
