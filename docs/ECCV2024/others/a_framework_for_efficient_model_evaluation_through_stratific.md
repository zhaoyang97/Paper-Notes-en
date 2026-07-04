---
title: >-
  [Paper Note] A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation
description: >-
  [ECCV 2024][Stratified Sampling] A statistical framework is proposed that synergistically designs three components—stratification, sampling design, and estimation—to accurately estimate Computer Vision (CV) model accuracy with only a small number of annotated test samples, achieving up to a 10x efficiency gain (i.e., reaching equivalent accuracy with 1/10 of the annotations).
tags:
  - "ECCV 2024"
  - "Stratified Sampling"
  - "Horvitz-Thompson Estimation"
  - "Difference Estimator"
  - "k-means Clustering"
  - "Annotation Cost"
date: 2026-05-08
content_hash: 8ff0b4fe23e82957
---

# A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation

**Conference**: ECCV 2024  
**arXiv**: [2406.07320](https://arxiv.org/abs/2406.07320)  
**Code**: [https://github.com/amazon-science/ssepy](https://github.com/amazon-science/ssepy)  
**Area**: Other  
**Keywords**: Stratified Sampling, Horvitz-Thompson Estimation, Difference Estimator, k-means Clustering, Annotation Cost  

## TL;DR
A statistical framework is proposed that synergistically designs three components—stratification, sampling design, and estimation—to accurately estimate Computer Vision (CV) model accuracy with only a small number of annotated test samples, achieving up to a 10x efficiency gain (i.e., reaching equivalent accuracy with 1/10 of the annotations).

## Background & Motivation

### Background

**Background**: Evaluating CV model accuracy requires high-quality annotated test sets, which are expensive to obtain. The existing practice is to draw a simple random sample (SRS) of a subset from the dataset for annotation and average the results—this naive approach fails to exploit the model's own predictive information, resulting in low estimation precision under a limited annotation budget. While the field of survey sampling in statistics has long addressed similar problems, the CV community has lacked systematic guidelines and comprehensive empirical comparisons, leaving these highly efficient sampling techniques unadopted.

### Proposed Approach

**Goal**: **How to accurately estimate model prediction accuracy with the minimal number of annotated test samples?** This question is particularly crucial when models need to be evaluated across multiple datasets and metrics (such as benchmarking CLIP on dozens of classification tasks). Reducing the annotation volume not only lowers costs but also accelerates model iteration.

## Method
The overall idea is: rather than blindly sampling randomly, the framework leverages the model's own predictive confidence to "smartly" select which samples are most worthy of annotation, while utilizing predictive information from unannotated samples during the estimation stage to boost accuracy.

### Overall Architecture
The input consists of a large-scale test set $\mathcal{D}$ ($N$ samples) and predictions from a model $f$, with an annotation budget of $n \ll N$. The framework comprises four steps:
1. **Prediction**: Build a proxy $\hat{Z}_i$ of model correctness for each sample (e.g., the model's confidence in its top-1 prediction).
2. **Stratification**: Partition the dataset into $H$ strata based on $\hat{Z}$ or feature representations.
3. **Sampling**: Draw samples within each stratum according to proportional or Neyman allocation.
4. **Estimation**: Compute the accuracy using Horvitz-Thompson (HT) or difference estimators (DF).

### Key Designs
1. **Construction of proxy variable $\hat{Z}$**: One can directly use the confidence scores of the evaluated model $f$, or employ a stronger surrogate model $f^*$ (e.g., using ViT-L/14 to assist the evaluation of ViT-B/32). Experiments demonstrate that predictions from a stronger surrogate model yield higher efficiency; calibrating the confidence via isotonic regression further improves the performance.
2. **Optimality of stratification strategy (Proposition 2 & Corollary 3)**: It is proved that under proportional allocation, minimizing the MSE of the HT estimator is equivalent to performing $k$-means clustering on $\hat{Z}$. This theoretical result is highly elegant—it reduces the estimator precision optimization problem to a standard clustering problem, providing a clear and actionable stratification method.
3. **Difference Estimator (DF)**: $\hat{\theta}_{DF} = \frac{1}{N}\sum_{i\in\mathcal{D}}\hat{Z}_i + \frac{1}{N}\sum_{i\in\mathcal{S}}\frac{Z_i - \hat{Z}_i}{\pi_i}$. The first term leverages predictions from the entire dataset, while the second term corrects bias on the annotated subset. When $\hat{Z}$ is accurate, the variance of the residual $Z_i - \hat{Z}_i$ is small, significantly boosting estimation precision. Proposition 4 proves that the efficiency gain ratio of DF relative to HT under SRS is $\mathbb{E}[\text{Var}(Z|X)] / \text{Var}(Z)$, meaning the better the model predictions (i.e., the smaller the proportion of conditional variance to total variance), the larger the gain.

### Loss & Training
This work does not involve model training. The key "training strategies" are: (1) calibrating the proxy $\hat{Z}$ using isotonic regression, fitted on a random half of the dataset and evaluated on the other half; (2) setting the number of strata to 10 (more strata theoretically yield higher precision) and performing $k$-means clustering.

## Key Experimental Results

| Dataset | Method | Relative Efficiency (vs HT+SRS) | Equivalent Annotation Savings |
|--------|------|----------------------|------------|
| CIFAR-10 | SSRS_p + HT (Calibrated surrogate model) | ~0.1 | **~10x** |
| Stanford Cars | SSRS_p + HT (Calibrated surrogate model) | ~0.2 | **~5x** |
| Dmlab Frames | SSRS_p + HT | ~1.0 | No significant gain |
| Median across datasets | SSRS_p + HT (f* calibrated) | ~0.3-0.5 | 2-3x |
| SRS + DF (Calibrated) | No stratification, DF only | Comparable to SSRS_p+HT | 2-3x |

### Ablation Study
- **Stratification Variable Selection**: Stratifying based on $\hat{Z}$ (model predictions) > stratifying based on image embeddings, as the former directly aligns with the target metric of model correctness.
- **Surrogate Model Accuracy**: Using ViT-L/14 > ViT-B/32 as the surrogate model; a more accurate $\hat{Z}$ yields higher efficiency.
- **Effect of Calibration**: Calibration has a minimal impact on proportional allocation (as stratification already absorbs it) but is crucial for Neyman allocation and DF estimators—without calibration, Neyman allocation can even perform worse than SRS.
- **When is Efficiency Gain Maximized?**: The gain is larger for tasks where model accuracy is higher (since high accuracy $\rightarrow$ most $Z_i=1$ $\rightarrow$ $\hat{Z}$ is easier to predict $\rightarrow$ small residual variance).
- **Out-of-Distribution (OOD) Data**: Efficiency gains are larger on in-distribution data; under OOD scenarios, the quality of proxy $\hat{Z}$ degrades, thus limiting the gains.

## Highlights & Insights
- **An Elegant Bridge Between Theory and Practice**: Proposition 2 + Corollary 3 reduce MSE minimization to $k$-means, transforming a statistical theory problem into a highly practical tool for everyone—this is the most "aha!" moment of the paper.
- **High Practical Utility**: The recommended "backpocket method" is extremely simple—run $k$-means on model confidence to partition into 10 strata $\rightarrow$ perform proportional allocation sampling $\rightarrow$ estimate with HT. It can be used without any complex mathematical derivations.
- **DF Estimator Remains Applicable Post-Hoc**: If SRS annotation has already been completed, one can still leverage the DF estimator post-hoc to boost estimate precision without needing to re-sample.
- **Comprehensive Empirical Coverage**: Systematic comparisons are conducted over 26+ classification tasks from the LAION CLIP-Benchmark, covering MSE/cross-entropy metrics, linear probing, different backbones, and OOD scenarios.

## Limitations & Future Work
- **Limited to One-shot Sampling**: The framework does not handle sequential or iterative annotation scenarios. The authors acknowledge that this choice is driven by practical considerations (annotations are usually outsourced in a single batch), though sequential sampling is theoretically more efficient.
- **Fixed Stratum Count of 10**: While more strata could theoretically be beneficial, $k$-means and Neyman allocation might become unstable when the number of strata is very large.
- **Validated Only on Classification Tasks**: Structured prediction tasks, such as object detection and segmentation, remain unexplored. Defining "model correctness" and constructing proxy variables in those domains are inherently more complex.
- **Calibration Depends on Partially Annotated Data**: Isotonic regression requires half of the data to fit the calibration function, and the annotation budget for this subset was not accounted for in the efficiency comparisons.
- **Multi-model Co-evaluation Not Considered**: Real-world benchmarks typically evaluate and compare multiple models simultaneously; jointly optimizing the evaluation efficiency for multiple models is a natural future extension.

## Related Work & Insights

| Aspect | Ours | Active Testing (Kossen et al., NeurIPS 2022) | PPI++ (Angelopoulos et al., 2023) |
|------|------|----------------------------------------------|-----------------------------------|
| Sampling Method | One-shot | Iterative | One-shot |
| Stratification | $k$-means on $\hat{Z}$ | Surrogate-guided | None |
| Estimator | HT + DF | Surrogate estimator | Prediction-powered inference |
| Theoretical Contribution | Proof of $k$-means optimality | Active learning convergence | Statistical inference guarantees |
| Key Difference | More systematic framework comparison | Requires multiple interaction rounds | DF under SRS is a special case of PPI |

The core advantage of this work lies in its systematicity and practicality—rather than chasing state-of-the-art complexity, it identifies the best open-of-the-box practices from existing statistical tools for the CV community.

## Related Work & Insights
- The core idea of this paper—**leveraging a model's own predictions to optimize evaluation efficiency**—is highly instructive for any research requiring extensive benchmark evaluations.
- **Potential Idea**: Extending this framework to the evaluation of object detection/segmentation tasks—requiring the definition of appropriate loss proxy variables and stratification strategies (e.g., stratification based on predicted IoU or confidence).
- The crucial role of calibration across the entire framework implies: researching model calibration itself $\rightarrow$ improvements in evaluation efficiency can be directly quantified.

## Rating
- Novelty: ⭐⭐⭐ The individual methodological components originate from classical statistics; the novelty lies in their systematic integration and the theoretical-empirical bridge built for the CV field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage across 26+ tasks, multiple metrics, diverse backbones, OOD scenarios, and linear probing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear, systematic, with thorough theoretical derivations, supported by illustrative examples and diagrams.
- Value: ⭐⭐⭐⭐ Holds direct, practical value for CV research involving large-scale benchmarking, though currently restricted to classification scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Exploring Guided Sampling of Conditional GANs](exploring_guided_sampling_of_conditional_gans.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](../../NeurIPS2025/others/active_measurement_efficient_estimation_at_scale.md)
- [\[ECCV 2024\] An Incremental Unified Framework for Small Defect Inspection](an_incremental_unified_framework_for_small_defect_inspection.md)
- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](attnzero_efficient_attention_discovery_for_vision_transformers.md)
- [\[ECCV 2024\] ADMap: Anti-disturbance Framework for Vectorized HD Map Construction](admap_anti-disturbance_framework_for_vectorized_hd_map_construction.md)

</div>

<!-- RELATED:END -->
