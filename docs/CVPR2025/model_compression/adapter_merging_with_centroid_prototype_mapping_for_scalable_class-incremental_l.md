---
title: >-
  [Paper Note] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning
description: >-
  [CVPR 2025][Model Compression][Class-Incremental Learning] This paper proposes the ACMap framework, which incrementally averages and merges independently trained task adapters into a single adapter (maintaining $O(1)$ inference complexity). Combined with centroid prototype mapping to align the representation of old task prototypes in the new subspace, it achieves comparable accuracy to the SOTA method EASE on five benchmarks while being 39 times faster in inference.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Class-Incremental Learning"
  - "Adapter Merging"
  - "Prototype Mapping"
  - "Exemplar-Free"
  - "Scalable Inference"
date: 2026-05-08
content_hash: 875427b48d4bb478
---

# Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning

**Conference**: CVPR 2025  
**arXiv**: [2412.18219](https://arxiv.org/abs/2412.18219)  
**Code**: [https://github.com/tf63/ACMap](https://github.com/tf63/ACMap)  
**Area**: Model Compression / Continual Learning  
**Keywords**: Class-Incremental Learning, Adapter Merging, Prototype Mapping, Exemplar-Free, Scalable Inference

## TL;DR

This paper proposes the ACMap framework, which incrementally averages and merges independently trained task adapters into a single adapter (maintaining $O(1)$ inference complexity). Combined with centroid prototype mapping to align the representation of old task prototypes in the new subspace, it achieves comparable accuracy to the SOTA method EASE on five benchmarks while being 39 times faster in inference.

## Background & Motivation

**Background**: Pre-trained models combined with parameter-efficient fine-tuning (adapters/prompts) have become the mainstream paradigm for class-incremental learning (CIL). EASE trains independent adapters for each task and extracts prototypes with each adapter for cosine classification. Although achieving SOTA accuracy, it requires forward passes through all adapters during inference, resulting in $O(T)$ complexity.

**Limitations of Prior Work**: Existing methods suffer from an accuracy-speed trade-off—SimpleCIL/APER operate with $O(1)$ inference but have limited accuracy (utilizing only the first-task adapter), while EASE achieves SOTA accuracy but suffers from a 39x slowdown in inference with 40 tasks. The inability to store old task exemplars (exemplar-free) in privacy-sensitive scenarios further increases the difficulty.

**Key Challenge**: After merging adapters (weight averaging), the prototypes of old tasks shift in the new subspace, making the prototypes inaccurate. However, without old task data, these prototypes cannot be recomputed.

**Core Idea**: Leverage the prototype shift of current task data (the only available data) between the old and new subspaces as a clue, and approximate the mapping of old prototypes to the new subspace using an affine transformation.

## Method

### Overall Architecture

Train independent adapters for each task (starting from shared initial weights) $\rightarrow$ Incrementally average and merge them into a single merged adapter $\rightarrow$ Compute prototypes for the current task in the merged adapter subspace $\rightarrow$ Correct old task prototypes using centroid prototype mapping $\rightarrow$ Perform inference with a cosine classifier utilizing only the merged adapter ($O(1)$).

### Key Designs

1. **Incremental Average Merging of Adapters**:

    - **Function**: Merges the adapters of all tasks into a single adapter.
    - **Mechanism**: $\bar{\theta}_t = (1 - 1/t) \bar{\theta}_{t-1} + (1/t) \theta_t$
    - **Key Condition - Initial Weight Substitution**: After training the first task, the random initial weights are replaced with $\theta_1$, and all subsequent tasks are trained starting from $\theta_1$. This encourages all adapters to follow similar optimization paths in the parameter space, forming a shared low-loss basin.
    - **Loss Surface Analysis**: Visualizing the linear interpolation loss surface of three consecutive adapters verifies the existence of a low-loss basin (red region), where the averaged point (asterisk) lies in the center.

2. **Centroid Prototype Mapping**:

    - **Function**: Maps old prototypes to the current merged adapter subspace without access to old task data.
    - **Mechanism**: $P_i(\bar{A}_t) \approx P_i(\bar{A}_i) + \Delta P$, where $\Delta P = \mathbb{E}[P_t(\bar{A}_t) - P_t(\bar{A}_i)]$ represents the centroid shift of the current task's prototype between the old and new subspaces.
    - **Key Assumption**: The shift pattern of the current task's prototype can be transferred to old task prototypes, which is validated through cosine similarity in experiments.
    - **Interesting Finding**: Recent tasks (e.g., task 5 in the task 6 subspace) exhibit high cosine similarity (small shift), whereas earlier tasks (e.g., task 1 in the task 10 subspace) show significant shifts, which also justifies the early stopping strategy.

3. **Early Stopping Strategy**:

    - **Function**: Stops merging adapters after the number of tasks exceeds a threshold $L$.
    - **Design Motivation**: The term $1/t$ in the merging formula approaches 0 as $t$ increases, making the contribution of newly added adapters negligible; moreover, prototype shifts of recent tasks are smaller, yielding more accurate mapping.
    - **Effect**: The results of $L=10$ vs. $L=\infty$ are almost identical, while training costs are reduced.

## Key Experimental Results

### Main Results: Average/Final Accuracy on 5 Benchmarks

| Method | CIFAR $\bar{A}/A_T$ | CUB $\bar{A}/A_T$ | IN-R $\bar{A}/A_T$ | IN-A $\bar{A}/A_T$ | VTAB $\bar{A}/A_T$ |
|------|:-:|:-:|:-:|:-:|:-:|
| SimpleCIL | 87.6/81.3 | 92.2/86.7 | 62.6/54.6 | 59.8/48.9 | 86.0/84.4 |
| APER | 90.7/85.2 | 92.2/86.7 | 72.4/64.3 | 60.5/49.4 | 86.0/84.4 |
| EASE | 91.5/85.8 | 92.2/86.8 | **78.3/70.6** | 65.3/55.0 | **93.6/93.6** |
| **ACMap** | **92.0/87.8** | 91.6/86.7 | 77.3/70.5 | **65.2/56.2** | 91.2/87.6 |

### Ablation Study: Inference Time (IN-R Task 40)

| Method | Inference Time (s) | Relative Speedup vs. ACMap |
|------|:-:|:-:|
| SimpleCIL | 22.6 | ×0.96 |
| APER | 44.1 | ×1.88 |
| EASE | **916.5** | **×39.0** |
| **ACMap** | 23.5 | 1.0 |

### Key Findings

- ACMap achieves comparable accuracy to EASE on 4 out of 5 datasets ($\pm1\%$) but is **39x faster in inference**.
- It outperforms SimpleCIL in final accuracy by $16\%+$ (IN-R: 70.5 vs. 54.6) with almost identical inference time.
- Initial weight substitution yields a consistent improvement of around $0.5\%$ (CIFAR: 91.54 $\rightarrow$ 92.04).
- Early stopping with $L=10$ vs. $L=\infty$ shows almost no difference, confirming diminishing returns of merging under a large number of tasks.
- On VTAB, ACMap underperforms EASE because the 5 domains in VTAB are highly diverse, making a single merged adapter difficult to fit; conversely, the 5 independent adapters in EASE naturally match these domains.

## Highlights & Insights

- **A New Paradigm of Weight Averaging for CIL**: By leveraging the theory from model merging ("shared initialization $\rightarrow$ low-loss basin $\rightarrow$ safe to average"), it elegantly resolves the inference overhead caused by adapter accumulation.
- **Simplicity and Elegance of Centroid Prototype Mapping**: It requires zero learnable parameters, estimating the prototype shifts of all old tasks using only the centroid shift of the current task data. The assumption is simple yet empirically effective.
- **Accuracy-Speed Pareto Optimality**: On the accuracy-speed curve in Fig. 1, ACMap lies on the Pareto frontier—achieving higher accuracy than methods of similar speed, and faster speed than methods of similar accuracy.
- **Exemplar-Free Design**: It does not store old task data, which is privacy-friendly and suitable for real-world scenarios.

## Limitations & Future Work

- In scenarios with extremely large domain gaps like VTAB, the representation capacity of a single merged adapter is insufficient.
- The affine assumption of centroid prototype mapping may not hold when subspaces change drastically (e.g., for early tasks).
- It has not been validated in combination with other PEFT methods besides LoRA (e.g., prefix tuning).
- The theoretical analysis is primarily validated through visualization, lacking formal convergence guarantees.

## Related Work & Insights

- **vs. EASE**: EASE concatenates features from $T$ independent adapters. While its accuracy is slightly better, its $O(T)$ inference is unscalable. ACMap achieves $O(1)$ inference through merging and mapping, offering a superior trade-off between accuracy and speed.
- **vs. RAPF**: RAPF also performs adapter merging but relies on SVD (computationally intensive) and CLIP text features. ACMap uses simple averaging and is vision-only, making it much more lightweight.
- **Insights from Model Merging**: This work represents a successful application of model merging theories to CIL. The core insight is "shared starting point $\rightarrow$ similar optimization paths $\rightarrow$ safe to average."

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of adapter merging and centroid mapping is novel, though the individual techniques themselves are relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets, thorough ablation studies (loss surface visualization, prototype shift analysis, and time comparison), rendering solid conclusions.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, beautiful figures (loss surfaces, cosine similarity curves), and each design choice is fully motivated.
- Value: ⭐⭐⭐⭐ Resolves the core pain point of adapter-based CIL (inference scalability), achieving a Pareto-optimal trade-off between accuracy and speed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICML 2025\] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning](../../ICML2025/model_compression/semantic_shift_estimation_via_dual-projection_and_classifier_reconstruction_for_.md)
- [\[CVPR 2025\] Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)

</div>

<!-- RELATED:END -->
